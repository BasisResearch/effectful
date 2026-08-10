import ast
import collections.abc
import functools
import inspect
import textwrap
import types
import typing
from collections.abc import Callable

import pydantic

import effectful.handlers.llm.harness.execution.hooks
from effectful.handlers.llm.harness.hooks import call_assistant, call_system
from effectful.handlers.llm.harness.serialization import (
    _TYPE_CHECK_ANCHOR_KEY,
    EncodedFunction,
    TypeToPydanticType,
    _inline_refs,
    _serialize_callable,
)
from effectful.handlers.llm.harness.synthesis.function import (
    SplicedRegion,
    SynthesizedFunction,
    _def_nodes,
    _recover_template_def,
)
from effectful.handlers.llm.types import FinalTool, Template
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements


def _splice_body(
    generated: ast.Module,
    module_ast: ast.Module,
    template_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> SplicedRegion:
    """Splice a synthesized function in as the anchor Template's *own body*.

    Unlike `splice_into_source` (which appends ``return <fn>`` and checks that the
    Template returns the synthesized *function*), this treats the synthesized
    function as the Template's implementation: the Template keeps its own
    authoritative signature and its body becomes ``[<helpers/imports the model
    wrote>, *<the synthesized function's body>]``.  mypy then checks that body
    against the Template's declared parameter and return types -- so a body that
    fails to return the declared type is rejected.  The synthesized function's own
    parameter list (including any ``self``) is intentionally discarded: the
    Template's real signature is the contract.

    ``generated`` is the model's whole ``module_code`` parsed to a module; its
    *last* statement is the implementation, any earlier statements are helper
    definitions/imports.  For example, given the Template ::

        @Template.define
        def parity(numbers: Sequence[int]) -> bool:
            '''True iff the sum of {numbers} is odd.
            >>> parity([1, 2])  # doctest: +SKIP
            True
            '''

    a model that submits this ``generated`` (note the header on its final ``def``
    -- ``numbers: list`` -- is discarded) ::

        import math
        def _odd(n: int) -> bool:
            return n % 2 == 1
        def parity(numbers: list) -> bool:
            return _odd(sum(numbers))

    is spliced into the Template's real source as ::

        @Template.define
        def parity(numbers: Sequence[int]) -> bool:   # authoritative header kept
            import math
            def _odd(n: int) -> bool:
                return n % 2 == 1
            return _odd(sum(numbers))                  # from the final def's body

    so mypy checks the grafted body against ``numbers: Sequence[int]`` and
    ``-> bool``.  The helper ``_odd`` and ``import math`` (everything before the
    final ``def``) become locals at the top of the body; only the final ``def``'s
    *body* is taken, under the Template's own header.

    Returns the modified module source and the ``[lo, hi]`` line span from the
    ``def`` line through the last body line, or ``None`` when the anchor's source
    can't be recovered (REPL/notebook template -- the caller skips rather than
    guesses). Raises ``RuntimeError`` on source drift, via `_recover_template_def`.
    """
    last = generated.body[-1]
    assert isinstance(last, ast.FunctionDef | ast.AsyncFunctionDef)

    # Keep the Template's real header (authoritative annotations, `self` for
    # methods); replace only its body with the model's helpers/imports followed by
    # the synthesized function's body statements, so the declared return type is
    # enforced. Any docstring/doctests in the recovered source are dropped.
    template_def.body = [*generated.body[:-1], *last.body]

    # Report the def line through the end of the body. Unlike `splice_into_source`,
    # the region starts at the `def` line (not the first body statement): mypy
    # anchors "Missing return statement"/"empty-body" there, and a body that doesn't
    # return the Template's declared type is a real defect we want to catch. The
    # header is the Template's own (recovered, resolvable) signature -- sourceless
    # templates return `None` above and skip -- so including it adds no spurious
    # signature diagnostics. Decorator lines sit above `spliced.lineno` and stay out.
    # `template_def` is still a node in `module_ast` (only its body changed), so its
    # walk-order index is stable across the unparse round-trip.
    def_index = _def_nodes(module_ast).index(template_def)
    checked_source = ast.unparse(ast.fix_missing_locations(module_ast))
    spliced = _def_nodes(ast.parse(checked_source))[def_index]
    lo = spliced.lineno
    hi = spliced.body[-1].end_lineno or lo
    return checked_source, lo, hi


class TemplateBody:
    """The synthesized *body* of a `Template`, as opposed to a general `Callable`.

    Used only as the type of `submit_solution`'s ``implementation`` parameter (see
    `effectful.handlers.llm.completions.SynthesizeAndCall`).  A `TemplateBody[[P],
    R]` carries the Template's parameter and return types exactly like a
    `Callable`, but gets its own `TypeToPydanticType` case (`_pydantic_template_body`)
    so the synthesized function is type-checked against the enclosing Template's
    source and its doctests run with self/recursive calls routed to the synthesized
    implementation.  The enclosing `Template` is recovered from the decode context
    (the ``anchor``), so no state rides on the type itself.
    """

    def __class_getitem__(cls, item):
        return types.GenericAlias(cls, item)


class SynthesizedTemplateBody(SynthesizedFunction):
    """Structured output for synthesizing a `Template`'s body (`submit_solution`).

    Decoded through `_pydantic_template_body`: the function is type-checked against
    the enclosing Template's source and its doctests are run with self/recursive
    calls routed to the synthesized implementation.

    Unlike `SynthesizedFunction`, the parameter and return *annotations* are not
    required: a Template body is type-checked against the Template's own signature
    (see `splice_template_body`), so the model may omit or vary them -- in
    particular it need not annotate the ``self`` receiver of an instance-method
    Template.
    """

    module_code: str = pydantic.Field(
        ...,
        description=textwrap.dedent("""
        The complete Python source implementing the Template shown in its spec.
        The code MUST satisfy the following constraints, or it will fail validation:

        <constraints>
        1. The code MUST be one complete syntactically valid Python module.
        2. The code MUST NOT use star imports or ``__future__`` imports.
        3. The function definition MUST be the LAST statement - do not add any code after it.
        4. Write the function with the Template's signature; parameter and return
        annotations are optional.
        5. Do not include a docstring or doctests; the Template's are supplied automatically.
        </constraints>
        """),
    )

    # A Template body is checked against the Template's own (already-annotated)
    # signature, so the synthesized body's annotations are optional.
    _require_annotations: typing.ClassVar[bool] = False


@TypeToPydanticType.register(TemplateBody)
def _pydantic_template_body(ty: typing.Any) -> typing.Any:
    """`TypeToPydanticType` case for a free-function `Template` body.

    Like `_pydantic_callable`, but the synthesized function is checked against the
    enclosing Template's source (the ``anchor`` in the decode context) and its
    doctests are run with the Template's own name/op routed back to the synthesized
    implementation, so a doctest that calls the Template (including for recursion)
    exercises the freshly synthesized code rather than re-invoking the model.
    """
    typed_enc = SynthesizedTemplateBody._create_model_from_callable_type(
        ty if typing.get_args(ty) else Callable[..., typing.Any],  # type: ignore[arg-type]
    )

    def _validate(
        value: SynthesizedTemplateBody | dict | str, info: pydantic.ValidationInfo
    ) -> Callable:
        if isinstance(value, str):
            value = typed_enc.model_validate_json(value)
        if isinstance(value, dict):
            value = typed_enc.model_validate(value)
        ctx = info.context or {}
        anchor = ctx.get(_TYPE_CHECK_ANCHOR_KEY)
        if anchor is not None:
            # template bodies should not have access to call-local variables
            assert isinstance(anchor, Template)
            ctx = anchor.__context__

        filename = f"<synthesis:{id(value.module_code)}>"
        module: ast.Module = effectful.handlers.llm.harness.execution.hooks.parse(
            value.module_code, filename
        )

        # `None` means the Template's source can't be recovered (REPL/exec/notebook
        # template): skip the type check rather than guess, but still route the
        # doctests below -- that only needs the anchor op, not its source.
        anchor_asts = _recover_template_def(anchor) if anchor is not None else None
        if anchor_asts is not None:
            spliced = _splice_body(module, *anchor_asts)
            effectful.handlers.llm.harness.execution.hooks.type_check(*spliced)

        bytecode: types.CodeType = (
            effectful.handlers.llm.harness.execution.hooks.compile(module, filename)
        )
        g: dict[str, typing.Any] = {k: v for k, v in ctx.items() if k.isidentifier()}
        effectful.handlers.llm.harness.execution.hooks.exec(bytecode, g)
        result = g[module.body[-1].name]  # type: ignore

        if anchor is None:
            effectful.handlers.llm.harness.execution.hooks.run_doctests(result, g)
            return result
        # Shadow the global name the doctests call and route the Template op back
        # into the synthesized function.
        result = functools.wraps(anchor)(result)
        g.update({anchor.__name__: result})
        with handler({anchor: result}):
            effectful.handlers.llm.harness.execution.hooks.run_doctests(result, g)
        return result

    # Distinct schemas per direction: validation (the model *produces* a function)
    # carries the synthesis instructions; serialization (the model *reads* an
    # encoded function) shows only the `module_code` shape `_serialize_synthesized`
    # emits, with no synthesis prose.
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate),
        pydantic.PlainSerializer(lambda value: _serialize_callable(value)),
        pydantic.WithJsonSchema(
            _inline_refs(pydantic.TypeAdapter(typed_enc).json_schema()),
            mode="validation",
        ),
        pydantic.WithJsonSchema(
            EncodedFunction.model_json_schema(), mode="serialization"
        ),
    ]


class MethodTemplateBody(TemplateBody):
    """A `TemplateBody` for an *instance-method* Template.

    Carries the method/free distinction on the type's origin (context-free schema
    generation reads it) so `submit_solution`'s description names the leading
    receiver ``self`` and the receiver is exempt from the annotation requirement --
    the model no longer has to reverse-engineer that the first parameter is ``self``.
    The Template's real signature (which includes the receiver) remains the
    type-check contract; see `splice_template_body`.
    """


class SynthesizedMethodTemplateBody(SynthesizedTemplateBody):
    """Structured output for synthesizing an *instance-method* `Template`'s body.

    Decoded through `_pydantic_template_body`: the function is type-checked against
    the enclosing Template's source and its doctests are run with self/recursive
    calls routed to the synthesized implementation.

    Unlike `SynthesizedFunction`, the parameter and return *annotations* are not
    required: a Template body is type-checked against the Template's own signature
    (see `splice_template_body`), so the model may omit or vary them -- in
    particular it need not annotate the ``self`` receiver of an instance-method
    Template.
    """

    module_code: str = pydantic.Field(
        ...,
        description=textwrap.dedent("""
        The complete Python source implementing the instance-method Template shown in
        its spec. The code MUST satisfy the following constraints, or it will fail
        validation:

        <constraints>
        1. The code MUST be one complete syntactically valid Python module.
        2. The code MUST NOT use star imports or ``__future__`` imports.
        3. The function definition MUST be the LAST statement - do not add any code after it.
        4. Write the function with the Template's signature: its FIRST parameter is the
        instance receiver ``self`` (which you may leave unannotated); all other parameter
        and return annotations are optional too.
        5. Do not include a docstring or doctests; the Template's are supplied automatically.
        </constraints>
        """),
    )

    @classmethod
    def _param_names(cls, param_types: typing.Iterable[typing.Any]) -> list[str]:
        # The method's callable type already carries the receiver as its first
        # parameter (with an uninformative Agent-class type); relabel it ``self`` so
        # the model reproduces it rather than inventing one -- do NOT prepend a receiver.
        names = super()._param_names(param_types)
        if names:
            names[0] = "self"
        return names

    @classmethod
    def _extra_instructions(cls) -> str:
        return (
            "\n\nThis implements an instance method: the first parameter is the "
            "instance receiver `self`. Include it as the first parameter; you may "
            "leave it unannotated."
        )


def _class_template_of(op: typing.Any) -> typing.Any | None:
    """The class-level `Template` underlying an Agent-method Template ``op``.

    Returns ``None`` for a free-function template (whose ``__default__`` is a plain
    function rather than a bound method).
    """
    default = getattr(op, "__default__", None)
    if isinstance(default, types.MethodType):
        return default.__func__.__wrapped__  # type: ignore[attr-defined]
    return None


def _method_instance(op: typing.Any, class_template: typing.Any) -> typing.Any | None:
    """The instance ``op`` is bound to, if ``op`` is ``class_template`` on *some*
    instance; otherwise ``None``.
    """
    if class_template is not None and _class_template_of(op) is class_template:
        return op.__default__.__self__
    return None


@TypeToPydanticType.register(MethodTemplateBody)
def _pydantic_method_template_body(ty: typing.Any) -> typing.Any:
    """`TypeToPydanticType` case for an instance-method `Template` body.

    Registered separately from `TemplateBody` (rather than reached via subclass
    MRO) so the method/free distinction is an explicit dispatch: it surfaces the
    leading ``self`` receiver in the signature hint, and its doctests -- which build
    their own instances -- route ``agent.method(...)`` on *any* instance to the
    synthesized implementation.
    """
    typed_enc = SynthesizedMethodTemplateBody._create_model_from_callable_type(
        ty if typing.get_args(ty) else Callable[..., typing.Any],  # type: ignore[arg-type]
    )

    def _validate(
        value: SynthesizedMethodTemplateBody | dict | str, info: pydantic.ValidationInfo
    ) -> Callable:
        if isinstance(value, str):
            value = typed_enc.model_validate_json(value)
        if isinstance(value, dict):
            value = typed_enc.model_validate(value)
        ctx = info.context or {}
        anchor = ctx.get(_TYPE_CHECK_ANCHOR_KEY)
        if anchor is not None:
            # template bodies should not have access to call-local variables
            assert isinstance(anchor, Template)
            ctx = anchor.__context__

        filename = f"<synthesis:{id(value.module_code)}>"
        module: ast.Module = effectful.handlers.llm.harness.execution.hooks.parse(
            value.module_code, filename
        )
        anchor_asts = _recover_template_def(anchor) if anchor is not None else None
        if anchor_asts is not None:
            spliced = _splice_body(module, *anchor_asts)
            effectful.handlers.llm.harness.execution.hooks.type_check(*spliced)

        bytecode: types.CodeType = (
            effectful.handlers.llm.harness.execution.hooks.compile(module, filename)
        )
        g: dict[str, typing.Any] = {k: v for k, v in ctx.items() if k.isidentifier()}
        effectful.handlers.llm.harness.execution.hooks.exec(bytecode, g)
        result = g[module.body[-1].name]  # type: ignore

        class_template = _class_template_of(anchor) if anchor is not None else None
        if class_template is None:
            effectful.handlers.llm.harness.execution.hooks.run_doctests(result, g)
            return result
        # A fresh instance's `agent.method(...)` dispatches through
        # `Template.__apply__`, which we intercept and redirect to the synthesized
        # implementation.
        result = functools.wraps(class_template)(result)

        def _doctest_apply(op, *args, **kwargs):
            instance = _method_instance(op, class_template)
            if instance is None:
                return fwd()
            return class_template(instance, *args, **kwargs)

        with handler({Template.__apply__: _doctest_apply, class_template: result}):
            effectful.handlers.llm.harness.execution.hooks.run_doctests(result, g)
        return result

    # Distinct schemas per direction: validation (the model *produces* a function)
    # carries the synthesis instructions; serialization (the model *reads* an
    # encoded function) shows only the `module_code` shape `_serialize_synthesized`
    # emits, with no synthesis prose.
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate),
        pydantic.PlainSerializer(_serialize_callable),
        pydantic.WithJsonSchema(
            _inline_refs(pydantic.TypeAdapter(typed_enc).json_schema()),
            mode="validation",
        ),
        pydantic.WithJsonSchema(
            EncodedFunction.model_json_schema(), mode="serialization"
        ),
    ]


def _callable_type_from_signature(
    signature: inspect.Signature,
) -> type[types.FunctionType]:
    """Construct a `Callable` type from a signature.

    Raises if the signature is recursive (e.g. a Template that returns itself)
    or contains variadic parameters (which cannot be expressed in a `Callable`
    type).
    """
    param_types = []
    for pname, param in signature.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise NotImplementedError(
                f"Cannot synthesize a function for parameter "
                f"'{pname}' of kind {param.kind.description}: variadic parameters "
                "cannot be expressed as a Callable type signature."
            )
        param_types.append(
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else typing.Any
        )
    return_type = signature.return_annotation
    return collections.abc.Callable[param_types, return_type]  # type: ignore


class FinalBodySynthesizer(ObjectInterpretation):
    """Answer a Template by synthesizing a function and calling it.

    Instead of asking the LLM to generate an instance of the Template's return
    type directly, this handler exposes a :class:`FinalTool` that lets the model
    "answer" by writing a Python function with the Template's signature.  The
    harness applies that function to the original arguments and its return value
    becomes the Template's result.  This is the declarative "CodeAdapt" workflow:
    the LLM writes code implementing the body of the Template rather than
    reasoning out the answer itself.

    The synthesis tool is offered *alongside* the Template's normal completion
    paths rather than replacing them: across turns the model may freely call any
    other tool in scope (their results are fed back as usual), and it may still
    answer the return type directly via structured output.  The loop terminates
    when it either answers directly or calls the synthesis :class:`FinalTool`.
    To force the synthesis path, pass ``tool_choice="required"`` (handler config
    is forwarded to the model request).  The function is synthesized by reusing
    the existing ``Callable`` synthesis machinery: the tool's argument is typed
    as ``Callable[[params], ret]``, so :func:`call_assistant`'s tool-call
    decoding parses, type-checks, compiles and executes the model's code into a
    real function before it is applied.

    Failures compose with :class:`RetryLLMHandler`: a function that fails to
    synthesize surfaces as a :class:`ToolCallDecodingError`, and one that raises
    when applied to the inputs as a :class:`ToolCallExecutionError`; both are fed
    back to the model as a tool message and the loop continues so it can revise::

        with (
            handler(LiteLLMProvider(model="gpt-5-mini")),
            handler(SynthesizeAndCall()),
            handler(RetryLLMHandler()),
        ):
            ...

    Requires an eval provider (e.g. :class:`UnsafeEvalProvider` or
    :class:`RestrictedEvalProvider`) to be installed so the synthesized code can
    be compiled and executed.
    """

    @typing.final
    class _SynthesisFinalTool[T](FinalTool[[collections.abc.Callable[..., T]], T]):
        """## Code synthesis

        You may "answer" a Template by writing code instead of producing the value
        directly. A final tool (typically `submit_solution`) accepts a single
        argument: a Python function whose signature matches the Template's signature
        (see its spec below). The harness applies that function to the original
        inputs and its return value becomes the answer, so write the function body
        as a drop-in implementation of the Template. The function may reference
        names from the lexical scope (see the *Lexical scope* table).

        You do not need to write a docstring or doctests: on submission the harness
        attaches the Template's own docstring to your function and runs *its*
        doctests (with recursive calls to the Template routed to your
        implementation). A solution whose doctests fail — or that errors when
        applied — is rejected and fed back to you to revise, so the answer only
        stands once the Template's doctests pass. Write just the implementation;
        any docstring you add is replaced and ignored. Calling this tool terminates
        the completion.

        This answers the *current* call only. Each call is a fresh, independent
        task: even if you already submitted a working solution earlier in this
        conversation, a prior submission is not a standing answer — you must call
        `submit_solution` again to answer the current call. Never end a turn with
        a prose summary in place of the answer; a plain message is not a valid
        response and will be rejected.
        """

        __toolname__: typing.ClassVar[typing.Literal["submit_solution"]] = (
            "submit_solution"
        )

        @classmethod
        def define(
            cls,
            template: Template[..., T],
            bound_args: inspect.BoundArguments,
        ) -> FinalTool[[collections.abc.Callable[..., T]], T]:
            if isinstance(template.__default__, types.MethodType):
                signature = inspect.signature(template.__default__.__func__)
                args, kwargs = (
                    (template.__default__.__self__,) + bound_args.args,
                    bound_args.kwargs,
                )
                body_type = MethodTemplateBody[  # type: ignore
                    typing.get_args(_callable_type_from_signature(signature))
                ]
                return_type = signature.return_annotation
            else:
                signature = inspect.signature(template)
                args, kwargs = bound_args.args, bound_args.kwargs
                body_type = TemplateBody[  # type: ignore
                    typing.get_args(_callable_type_from_signature(signature))
                ]
                return_type = signature.return_annotation

            def submit_solution(implementation: body_type) -> return_type:  # type: ignore
                """
                Answer this Template by submitting a Python function that implements
                it (see the "Code synthesis" section); its return value on the
                original arguments becomes the answer.
                """
                return implementation(*args, **kwargs)  # type: ignore

            return super().define(submit_solution, name=cls.__toolname__)

    @implements(call_system)
    def _call_system(self, template, tool_types=frozenset()):
        return fwd(template, tool_types=tool_types | {self._SynthesisFinalTool})

    @implements(Template.__apply__)
    def _apply[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        tool = self._SynthesisFinalTool.define(template, bound_args)

        def _add_synthesis_tool(
            env, response_type, tools=frozenset(), anchor=None, force_tool=False
        ):
            if any(isinstance(t, self._SynthesisFinalTool) for t in tools):
                return fwd()
            return fwd(
                env, response_type, tools | {tool}, anchor=anchor, force_tool=force_tool
            )

        with handler({call_assistant: _add_synthesis_tool}):
            return fwd()
