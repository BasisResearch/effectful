import ast
import collections.abc
import inspect
import linecache
import logging
import textwrap
import types
import typing

import pydantic

import effectful.handlers.llm.harness.execution.hooks
import effectful.handlers.llm.harness.validation.hooks
from effectful.handlers.llm.harness.serialization import (
    _IS_FINAL_KEY,
    _TYPE_CHECK_ANCHOR_KEY,
    EncodedFunction,
    TypeToPydanticType,
    _inline_refs,
    _serialize_callable,
)

# The shared output of the three splicers (`splice_into_source`,
# `splice_skill_body`, `splice_repl_code_into_body`): the module ``source`` to
# type-check and the inclusive ``[lo, hi]`` line span within it to report
# diagnostics from -- exactly the leading arguments of `type_check`. ``None`` (not
# this type) is returned when the anchor's source can't be recovered.
type SplicedRegion = tuple[str, int, int]

logger = logging.getLogger(__name__)


def _reject_param_count_mismatch(fn: collections.abc.Callable, ty: typing.Any) -> None:
    """Raise ``ValueError`` if the synthesized ``fn``'s positional arity does not
    match the expected ``Callable[[...], ret]`` type.

    The mypy signature check only runs when a type-check anchor is in scope; this
    structural check runs unconditionally, so a wrong parameter count is still
    rejected on the anchorless argument-decoding path.
    """
    args = typing.get_args(ty)
    if not args or args[0] is ...:
        return  # bare ``Callable`` or ``Callable[..., R]``: any arity is acceptable
    expected = len(args[0])
    params = list(inspect.signature(fn).parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return  # ``*args`` accepts any number of positional arguments
    positional = sum(
        p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )
    if positional != expected:
        raise ValueError(
            f"synthesized function takes {positional} positional parameter(s), "
            f"but the expected signature has {expected}"
        )


# Reserved key under which the type-check anchor -- the enclosing `Skill`
# itself -- rides in the Pydantic decoding context, alongside the lexical
# environment. `decode` reads it to type-check a synthesized function against the
# Skill's source (recovered from the Skill via `inspect.unwrap`); absent
# (tool-argument decoding) means skip. Deliberately not a valid identifier so
# `LexicalReaders` skips it (no tool leak) and it can never collide with a lexical
# name.
def _def_nodes(
    module: ast.Module,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """All function definitions in ``module``, in a stable order that an
    ``ast.unparse`` -> ``ast.parse`` round-trip preserves (so a def keeps its
    index across it)."""
    return [
        n
        for n in ast.walk(module)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _find_def_at_lineno(
    module: ast.Module, lineno: int
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Locate the function definition whose definition site is ``lineno``.

    Matches ``fn.__code__.co_firstlineno`` -- the first decorator line, or the
    ``def`` line when undecorated -- which identifies the def directly and
    unambiguously (no name matching, and nesting-agnostic). Returns None only if
    no def starts there: a dynamically generated ``fn`` with no source def, or
    source that has drifted since import.
    """
    for node in _def_nodes(module):
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        if start == lineno:
            return node
    return None


def _recover_skill_def(
    anchor: collections.abc.Callable[..., typing.Any],
) -> tuple[ast.Module, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Locate the anchor Skill's own ``def`` in its real module source.

    Returns the parsed module AST and the def node, or ``None`` when the source can't
    be recovered (REPL/exec/notebook Skill with no linecache entry -- the caller
    skips rather than guesses). Raises ``RuntimeError`` on source drift (source
    recovered but the def no longer sits where ``fn`` was compiled from).
    """
    # `anchor` is the enclosing `Skill` (an `Operation`), a bound method, or a
    # plain function; `inspect.unwrap` follows the `__wrapped__` chain that
    # `Operation`/method binding sets up, resolving all of them to the original
    # source-backed function (staticmethod/classmethod included).
    fn = inspect.unwrap(anchor)
    # Recover the module source via fn's own filename -- a real path or a
    # linecache-registered synthetic name (e.g. <synthesis:...>) for REPL/exec/
    # notebook skills; linecache.getlines reads real files from disk too.
    try:
        source_file = inspect.getsourcefile(fn)
    except TypeError:
        source_file = None
    module_source = "".join(linecache.getlines(source_file)) if source_file else ""
    if not module_source:
        logger.warning("skipping type check: cannot recover source for %r", fn)
        return None
    module_ast = ast.parse(module_source)
    skill_def = _find_def_at_lineno(module_ast, fn.__code__.co_firstlineno)
    if skill_def is None:
        raise RuntimeError(
            f"cannot locate {getattr(fn, '__qualname__', fn)!r} in its module "
            f"source (source drifted since import?)"
        )
    return module_ast, skill_def


def _splice_function(
    generated: ast.Module,
    module_ast: ast.Module,
    skill_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> SplicedRegion:
    """Splice `generated` into the anchor Skill's own function body, in its real
    module source.

    Returns the modified module source and the ``[lo, hi]`` line span of the
    spliced body within it, or ``None`` when the anchor's source can't be recovered
    (the caller skips rather than guesses). Raises ``RuntimeError`` if the source is
    recovered but the anchor's def can't be located in it (source drift) -- a real
    error, not a silent pass.

    The generated function -- and any helpers it defines alongside -- becomes the
    body of the Skill's own function at its real (possibly nested) position, so
    the generated code is checked in its real lexical scope with no synthesized
    type stubs.

    This is the splice for a Skill whose *return type* is a callable (the model
    writes a function and the Skill returns it). Example. For the Skill ::

        @Skill.define
        def make_adder(n: int) -> Callable[[int], int]:
            '''Return a function that adds {n}.'''

    a model that submits this ``generated`` (its last statement is the function to
    return) ::

        def adder(x: int) -> int:
            return x + n

    becomes the whole Skill body followed by ``return <its name>`` ::

        @Skill.define
        def make_adder(n: int) -> Callable[[int], int]:
            def adder(x: int) -> int:
                return x + n
            return adder

    so mypy checks that ``adder`` satisfies ``Callable[[int], int]`` and that its
    body may reference the Skill's ``n``. Contrast `splice_skill_body`, which
    grafts the model's function *body* under the Skill's own header (for a
    Skill whose body -- not return value -- is synthesized). The returned
    ``[lo, hi]`` spans the generated statements only, not the ``def`` header.
    """
    last = generated.body[-1]
    assert isinstance(last, ast.FunctionDef | ast.AsyncFunctionDef)

    # Splice in place: replace the body with the generated body and bind the
    # target against the (source) return annotation via `return`. Decorators are
    # left untouched -- mypy checks a function's body against its declared return
    # type regardless of decorators (even an unresolvable / `Any` one), and the
    # decorator application itself doesn't spuriously fail, so touching the
    # surrounding source as little as possible keeps the splice robust.
    skill_def.body = [
        *generated.body,
        ast.Return(ast.Name(last.name, ast.Load())),
    ]

    # mypy reports line numbers in the coordinates of `checked_source`, so we need
    # the spliced *body's* span there. ast.unparse reassigns line numbers but
    # preserves def order, so the def keeps its index in walk order -- take the def
    # at that same index in the re-parsed source.
    #
    # The region is the body (the generated code) only, NOT the def header: the
    # signature and decorators are the Skill author's own pre-existing source,
    # which we must not attribute to synthesis. This matters for skills whose
    # module source can't be fully recovered -- notably notebook/REPL cells, which
    # share a runtime namespace but whose recovered source is a single cell missing
    # the other cells' imports, so the signature's own annotations (e.g. `Literal`,
    # `Callable`) look undefined to mypy. Flagging only the body keeps those
    # spurious signature-line diagnostics out of the gate.
    def_index = _def_nodes(module_ast).index(skill_def)
    checked_source = ast.unparse(ast.fix_missing_locations(module_ast))
    spliced = _def_nodes(ast.parse(checked_source))[def_index]
    lo = spliced.body[0].lineno  # first generated statement (body is non-empty)
    hi = spliced.end_lineno or lo
    return checked_source, lo, hi


class SynthesizedFunction(EncodedFunction):
    """
    Structured output for function synthesis.
    """

    module_code: str = pydantic.Field(
        ...,
        description=textwrap.dedent("""
        A string containing the complete Python source code for the function.
        The code MUST satisfy the following constraints, or it will fail validation:

        <constraints>
        1. The code MUST be one complete syntactically valid Python module.
        2. The code MUST NOT use star imports or ``__future__`` imports.
        3. The function definition MUST be the LAST statement - do not add any code after it.
        4. The function MUST have type annotations for all parameters and the return type.
        5. You may include doctest examples (lines starting with >>>) inside the function's
        docstring to demonstrate and verify its behavior; these examples are run as tests.
        </constraints>
        """),
    )

    # A general `Callable` is type-checked against the requested signature, so it must
    # be fully annotated. A Skill *body* is instead checked against the enclosing
    # Skill's own signature (`splice_skill_body`), which already carries the
    # annotations -- so its subclasses waive this and may omit the `self` receiver.
    _require_annotations: typing.ClassVar[bool] = True

    @pydantic.field_validator("module_code")
    @classmethod
    def _validate_module_code(cls, value: str) -> str:
        module: ast.AST = ast.parse(value)

        if not isinstance(module, ast.Module) or not module.body:
            raise ValueError(
                "decode() requires module code with at least one statement."
            )

        last_stmt = module.body[-1]
        if not isinstance(last_stmt, ast.FunctionDef):
            raise ValueError(
                f"decode() requires the last statement to be a function definition, "
                f"got {type(last_stmt).__name__}"
            )

        if cls._require_annotations:
            for arg in last_stmt.args.args:
                if arg.annotation is None:
                    raise ValueError(
                        f"decode() requires all parameters to have type annotations, "
                        f"parameter '{arg.arg}' is missing an annotation"
                    )
            if last_stmt.returns is None:
                raise ValueError(
                    "decode() requires the function to have a return type annotation"
                )

        for stmt in module.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                raise ValueError(
                    "decode() does not allow __future__ imports in the module code"
                )

        for stmt in module.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.names:
                for alias in stmt.names:
                    if alias.name == "*":
                        raise ValueError(
                            "decode() does not allow star imports in the module code"
                        )

        return value

    @classmethod
    def _create_model_from_callable_type(
        cls, typ: type[collections.abc.Callable]
    ) -> type[typing.Self]:
        """Create a SynthesizedFunction subclass carrying the requested signature in
        the model-facing description.

        Uses ``pydantic.create_model`` so the rendered signature (and any
        subclass-specific instructions) ride in the JSON schema ``description`` sent
        to the model. Subclasses customize the receiver rendering via `_param_names`
        and add guidance via `_extra_instructions`.
        """
        doc = (
            f"Python function with signature "
            f"<signature>{cls._signature_str(typ)}</signature>"
            f"{cls._extra_instructions()}"
        )
        return pydantic.create_model(
            "TypedSynthesizedFunction",
            __base__=cls,
            __doc__=doc,
        )

    @classmethod
    def _signature_str(cls, typ: type[collections.abc.Callable]) -> str:
        """Render a ``Callable[[...], ...]`` signature by type *name* (not its
        fully-qualified ``repr``), so the model sees ``Callable[[State], int]`` rather
        than ``collections.abc.Callable[[pkg.mod.State], builtins.int]``."""
        args = typing.get_args(typ)
        if not args:
            return "Callable"
        param_types, return_type = args
        params_str = (
            "..." if param_types is ... else ", ".join(cls._param_names(param_types))
        )
        return_str = getattr(return_type, "__name__", str(return_type))
        return f"Callable[[{params_str}], {return_str}]"

    @classmethod
    def _param_names(cls, param_types: typing.Iterable[typing.Any]) -> list[str]:
        return [getattr(t, "__name__", str(t)) for t in param_types]

    @classmethod
    def _extra_instructions(cls) -> str:
        return ""


@TypeToPydanticType.register(collections.abc.Callable)
def _pydantic_callable(ty: typing.Any) -> typing.Any:
    """Pydantic-compatible Annotated type for a parameterized `Callable` value.

    The model *produces* a function (as ``module_code``); it is synthesized,
    type-checked in the enclosing Skill's scope, and its own doctests are run.
    Skill-body synthesis (`submit_solution`) has its own encoding,
    `_pydantic_skill_body`.
    """
    typed_enc = SynthesizedFunction._create_model_from_callable_type(
        collections.abc.Callable[..., typing.Any] if not typing.get_args(ty) else ty  # type: ignore[arg-type]
    )

    def _validate(
        value: SynthesizedFunction | dict | str | collections.abc.Callable,
        info: pydantic.ValidationInfo,
    ) -> collections.abc.Callable:
        if isinstance(value, str):
            value = typed_enc.model_validate({"module_code": value})
        elif isinstance(value, dict):
            value = typed_enc.model_validate(value)
        elif isinstance(value, EncodedFunction):
            value = typed_enc.model_validate(value.model_dump())
        elif callable(value):
            return value

        assert isinstance(value, typed_enc)

        ctx = info.context or {}
        anchor = ctx.get(_TYPE_CHECK_ANCHOR_KEY)

        filename = f"<synthesis:{id(value)}>"
        module: ast.Module = effectful.handlers.llm.harness.execution.hooks.parse(
            value.module_code, filename
        )

        if anchor is not None and _recover_skill_def(anchor) is not None:
            anchor_asts = _recover_skill_def(anchor)
            assert anchor_asts is not None
            module_ast, skill_def = anchor_asts
            spliced = _splice_function(module, module_ast, skill_def)
            # use _IS_FINAL_KEY to determine if this is a return value or a tool argument
            is_final = ctx.get(_IS_FINAL_KEY, False)
            effectful.handlers.llm.harness.validation.hooks.type_check(
                *spliced, lenient=not is_final
            )

        bytecode: types.CodeType = (
            effectful.handlers.llm.harness.execution.hooks.compile(module, filename)
        )
        g: dict[str, typing.Any] = {k: v for k, v in ctx.items() if k.isidentifier()}
        effectful.handlers.llm.harness.execution.hooks.exec(bytecode, g)
        result = g[module.body[-1].name]  # type: ignore
        _reject_param_count_mismatch(result, ty)
        effectful.handlers.llm.harness.validation.hooks.run_doctests(result, g)
        return result

    # Distinct schemas per direction: validation (the model *produces* a function)
    # carries the synthesis instructions; serialization (the model *reads* an
    # encoded function) shows only the `module_code` shape `_serialize_synthesized`
    # emits, with no synthesis prose.
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.BeforeValidator(_validate),
        pydantic.PlainSerializer(_serialize_callable),
        pydantic.WithJsonSchema(
            _inline_refs(pydantic.TypeAdapter(typed_enc).json_schema()),
            mode="validation",
        ),
        pydantic.WithJsonSchema(
            EncodedFunction.model_json_schema(), mode="serialization"
        ),
    ]
