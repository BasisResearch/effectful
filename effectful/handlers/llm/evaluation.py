import ast
import builtins
import collections.abc
import copy
import inspect
import keyword
import linecache
import random
import string
import sys
import types
import typing
from collections.abc import Mapping
from types import CodeType
from typing import Any, TypeAliasType

import autoflake
from mypy import api as mypy_api
from RestrictedPython import (
    Eval,
    Guards,
    RestrictingNodeTransformer,
    compile_restricted,
    safe_globals,
)

from effectful.internals.unification import nested_type
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import Operation


@defop
def parse(source: str, filename: str) -> ast.Module:
    """
    Parse source text into an AST.

    source: The Python source code to parse.
    filename: The filename recorded in the resulting AST for tracebacks and tooling.

    Returns the parsed AST.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to parse code."
    )


@defop
def type_check(
    module: ast.Module,
    ctx: typing.Mapping[str, Any],
    expected_params: list[type] | None,
    expected_return: type,
) -> None:
    """
    Type check a parsed module against an expected signature and lexical context.

    module: The parsed Python source code module to type check.
    ctx: The lexical context under which the file should be type checked
    expected_params, expected_return: The expected signature of the last function in the module.

    Returns None, raises TypeError on typechecking failure.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to parse code."
    )


@defop
def compile(module: ast.Module, filename: str) -> CodeType:
    """
    Compile an AST into a Python code object.

    module: The AST to compile (typically produced by parse()).
    filename: The filename recorded in the resulting code object (CodeType.co_filename), used in tracebacks and by inspect.getsource().

    Returns the compiled code object.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to compile code."
    )


@defop
def exec(
    bytecode: CodeType,
    env: dict[str, Any],
) -> None:
    """
    Execute a compiled code object.

    bytecode: A code object to execute (typically produced by compile()).
    env: The namespace mapping used during execution.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to execute code."
    )


# Type checking implementation
def type_to_ast(typ: Any) -> ast.expr:
    """Convert a Python type to an AST expression.

    Takes a type (e.g. from nested_type(value).value) and converts to AST.
    """
    # Handle None type (both None value and NoneType)
    if typ is None or typ is type(None):
        return ast.Constant(value=None)

    # Handle typing.Any (via its metaclass _AnyMeta)
    if isinstance(typ, type(typing.Any)):
        return ast.Attribute(
            value=ast.Name(id="typing", ctx=ast.Load()), attr="Any", ctx=ast.Load()
        )

    # Handle function type -> Callable (before builtins check since FunctionType is in builtins)
    if typ is types.FunctionType:
        return ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="collections", ctx=ast.Load()),
                attr="abc",
                ctx=ast.Load(),
            ),
            attr="Callable",
            ctx=ast.Load(),
        )

    # Handle basic builtin types
    if isinstance(typ, type) and typ.__module__ == "builtins":
        return ast.Name(id=typ.__name__, ctx=ast.Load())

    # Handle runtime-only types (__main__ or <locals> in qualname)
    # These can't be imported, so we just use the simple name
    if isinstance(typ, type) and (
        typ.__module__ == "__main__" or "<locals>" in typ.__qualname__
    ):
        return ast.Name(id=typ.__name__, ctx=ast.Load())

    # Handle types from other modules (e.g., collections.abc.Callable)
    if isinstance(typ, type) and typ.__module__:
        parts = typ.__module__.split(".")
        node: ast.expr = ast.Name(id=parts[0], ctx=ast.Load())
        for part in parts[1:]:
            node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
        return ast.Attribute(value=node, attr=typ.__name__, ctx=ast.Load())

    # Handle typing special forms (Union, Optional, etc.)
    if isinstance(typ, typing._SpecialForm):
        return ast.Attribute(
            value=ast.Name(id="typing", ctx=ast.Load()),
            attr=typ._name,  # type: ignore[attr-defined]
            ctx=ast.Load(),
        )

    # Handle TypeVars, ParamSpecs, TypeVarTuples
    if isinstance(typ, typing.TypeVar | typing.ParamSpec | typing.TypeVarTuple):
        return ast.Name(id=typ.__name__, ctx=ast.Load())

    # Handle TypeAliasType (PEP 695 type aliases)
    if isinstance(typ, TypeAliasType):
        return ast.Name(id=typ.__name__, ctx=ast.Load())

    # Handle union types (int | str)
    if isinstance(typ, types.UnionType):
        args = typing.get_args(typ)
        result = type_to_ast(args[0])
        for arg in args[1:]:
            result = ast.BinOp(left=result, op=ast.BitOr(), right=type_to_ast(arg))
        return result

    # Handle typing.Annotated: use the first argument (the actual type) for typecheck stubs
    origin = typing.get_origin(typ)
    if origin is typing.Annotated:
        args = typing.get_args(typ)
        if args:
            return type_to_ast(args[0])
        raise TypeError("type_to_ast: Annotated must have at least one type argument")

    # Handle generic aliases (e.g., Callable[[int], str], list[int])
    origin = typing.get_origin(typ)
    if origin is not None:
        args = typing.get_args(typ)
        origin_ast = type_to_ast(origin)

        if not args:
            return origin_ast

        # Handle Callable-like types: Callable[[arg_types], return_type]
        # Also handles Operation subclasses which have the same structure
        is_callable_like = isinstance(origin, type) and issubclass(  # noqa: UP038
            origin,
            (collections.abc.Callable, Operation),  # type: ignore[arg-type]
        )
        if is_callable_like:
            if len(args) != 2:
                raise TypeError(
                    f"type_to_ast: Callable-like type {typ} must have exactly 2 args, got {len(args)}"
                )
            param_types, return_type = args
            # Handle varargs: Callable[..., T] or ParamSpec: Callable[P, T]
            params_ast: ast.expr
            if param_types is ...:
                params_ast = ast.Constant(value=...)
            elif isinstance(param_types, typing.ParamSpec):
                params_ast = type_to_ast(param_types)
            elif isinstance(param_types, list | tuple):
                params_ast = ast.List(
                    elts=[type_to_ast(p) for p in param_types], ctx=ast.Load()
                )
            else:
                raise TypeError(
                    f"type_to_ast: Callable param_types must be list, tuple, ParamSpec, or ..., got {type(param_types)}"
                )
            return ast.Subscript(
                value=origin_ast,
                slice=ast.Tuple(
                    elts=[params_ast, type_to_ast(return_type)], ctx=ast.Load()
                ),
                ctx=ast.Load(),
            )

        # Handle single type argument
        if len(args) == 1:
            return ast.Subscript(
                value=origin_ast, slice=type_to_ast(args[0]), ctx=ast.Load()
            )

        # Handle multiple type arguments
        return ast.Subscript(
            value=origin_ast,
            slice=ast.Tuple(elts=[type_to_ast(arg) for arg in args], ctx=ast.Load()),
            ctx=ast.Load(),
        )

    raise TypeError(f"type_to_ast: unhandled type {typ}")


# globals always present in python runtime, so we should not stub, or import for
SKIPPED_GLOBALS = (
    "__builtins__",
    "__annotations__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_frozen_importlib",
)


def collect_imports(ctx: Mapping[str, Any]) -> list[ast.stmt]:
    """Collect module imports and symbol imports from context.

    - Modules in context (e.g. ``import math``) produce ``import math``.
    - Symbols from a module that is not in context (e.g. ``from typing import Any``
      gives context ``{"Any": typing.Any}`` but no ``typing`` module) produce
      ``from <module> import <name>`` or ``from <module> import <name> as <alias>``.
    """
    # (module_name, asname_in_context) for plain imports; asname is None when same as module_name
    modules: set[tuple[str, str | None]] = set(
        (k, None)
        for k in sys.modules.keys()
        if k not in SKIPPED_GLOBALS and not k.startswith("_") and k[0].isalpha()
    )
    # module -> list of (name_in_module, name_in_context) for from-imports
    symbol_imports: dict[str, list[tuple[str, str]]] = {}

    for name_in_ctx, value in ctx.items():
        if name_in_ctx.startswith("@") or name_in_ctx in SKIPPED_GLOBALS:
            # pytest adds @pytest_ar, @py_builtins, etc into the context, we skip
            continue

        if isinstance(value, types.ModuleType):
            mod_name = value.__name__
            asname = name_in_ctx if name_in_ctx != mod_name else None
            modules.add((mod_name, asname))
            continue
        module = getattr(value, "__module__", None)
        if module in (None, "", "__main__", "builtins"):
            continue
        if not isinstance(module, str):
            continue
        if isinstance(value, type) and _is_runtime_only_type(value):
            continue
        # module not in runtime context
        if not sys.modules.get(module):
            continue
        name_in_module = getattr(value, "__name__", name_in_ctx)
        if not hasattr(sys.modules[module], name_in_module):
            continue
        modules.add((module, None))
        symbol_imports.setdefault(module, []).append((name_in_module, name_in_ctx))

    stmts: list[ast.stmt] = []
    # Module imports: list of (module_name, asname_in_context)
    module_pairs = sorted((m, asname or "") for m, asname in modules)
    if module_pairs:
        stmts.append(
            ast.Import(
                names=[
                    ast.alias(name=m, asname=asname or None)
                    for m, asname in module_pairs
                ]
            )
        )
    for module in sorted(symbol_imports):
        names = [
            ast.alias(
                name=name_in_module,
                asname=name_in_ctx if name_in_ctx != name_in_module else None,
            )
            for name_in_module, name_in_ctx in sorted(
                symbol_imports[module], key=lambda t: (t[0], t[1])
            )
        ]
        stmts.append(ast.ImportFrom(module=module, names=names, level=0))
    return stmts


def collect_variable_declarations(ctx: Mapping[str, Any]) -> list[ast.stmt]:
    """Generate variable declarations with type annotations from context.

    For each variable, calls nested_type(value).value to get the type,
    then type_to_ast to convert to AST.
    """
    nodes: list[ast.stmt] = []

    for name, value in sorted(ctx.items()):
        if name in SKIPPED_GLOBALS:
            continue
        # Skip modules (handled by collect_imports)
        if isinstance(value, types.ModuleType):
            continue

        # Skip types (classes) - they don't need variable declarations
        if isinstance(value, type):
            continue

        # skip values that can be imported from
        # somewhere, mypy can just use the imports
        if (
            hasattr(value, "__qualname__")
            and hasattr(value, "__module__")
            and "<locals>" not in value.__qualname__
            and value.__module__ != "__main__"
        ):
            continue

        try:
            inferred_type = nested_type(value).value
            type_ast = type_to_ast(inferred_type)
        except (TypeError, AttributeError):
            # Use Any for values we can't infer
            type_ast = type_to_ast(typing.Any)

        ann_assign = ast.AnnAssign(
            target=ast.Name(id=name, ctx=ast.Store()),
            annotation=type_ast,
            value=None,
            simple=1,
        )
        nodes.append(ann_assign)

    return nodes


def _is_runtime_only_type(typ: type) -> bool:
    """Check if a type is runtime-only (can't be imported)."""
    return typ.__module__ == "__main__" or "<locals>" in typ.__qualname__


def signature_to_ast(name: str, sig: inspect.Signature) -> ast.FunctionDef:
    """Convert an inspect.Signature to an AST function definition stub."""
    # Build arguments
    args_list: list[ast.arg] = []
    for param_name, param in sig.parameters.items():
        annotation: ast.expr | None = None
        if param.annotation is not inspect.Parameter.empty:
            try:
                annotation = type_to_ast(param.annotation)
            except TypeError:
                annotation = type_to_ast(typing.Any)
        args_list.append(ast.arg(arg=param_name, annotation=annotation))

    # Build return annotation
    returns: ast.expr | None = None
    if (
        sig.return_annotation is not inspect.Signature.empty and name != "__init__"
    ):  # mypy complains that __init__ must not return anything
        try:
            returns = type_to_ast(sig.return_annotation)
        except TypeError:
            returns = type_to_ast(typing.Any)

    node = ast.FunctionDef(  # type: ignore
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=args_list,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
                cause=None,
            )
        ],
        decorator_list=[],
        returns=returns,
    )
    return ast.fix_missing_locations(node)


def collect_runtime_type_stubs(ctx: Mapping[str, Any]) -> list[ast.stmt]:
    """Generate class stubs for runtime-only types.

    For types defined at runtime (in __main__ or local scopes), generates
    class definitions with proper inheritance, typed attributes, and method stubs.
    """
    nodes: list[ast.stmt] = []

    for name, value in sorted(ctx.items()):
        # Only process types (classes)
        if not isinstance(value, type):
            continue

        # Only process runtime-only types
        if not _is_runtime_only_type(value):
            continue

        # Build base classes (use __orig_bases__ to preserve generic params)
        bases: list[ast.expr] = []
        orig_bases = getattr(value, "__orig_bases__", value.__bases__)
        for base in orig_bases:
            if base is object:
                continue
            try:
                bases.append(type_to_ast(base))
            except TypeError:
                pass

        # Build class body
        body: list[ast.stmt] = []

        # Add typed attributes from __annotations__
        annotations = getattr(value, "__annotations__", {})
        for attr_name, attr_type in annotations.items():
            try:
                type_ast = type_to_ast(attr_type)
            except TypeError:
                type_ast = type_to_ast(typing.Any)
            body.append(
                ast.AnnAssign(
                    target=ast.Name(id=attr_name, ctx=ast.Store()),
                    annotation=type_ast,
                    value=None,
                    simple=1,
                )
            )

        # Add method stubs
        for method_name in dir(value):
            attr = getattr(value, method_name, None)
            if attr is None:
                continue
            if not isinstance(attr, types.FunctionType):
                continue
            try:
                sig = inspect.signature(attr)
            except (ValueError, TypeError):
                continue
            method_ast = signature_to_ast(method_name, sig)
            body.append(method_ast)

        # Use ellipsis if body is empty
        if not body:
            body = [ast.Expr(value=ast.Constant(value=...))]

        class_def = ast.ClassDef(
            name=name,
            bases=bases,
            keywords=[],
            body=body,
            decorator_list=[],
            type_params=[],
        )
        nodes.append(class_def)

    return nodes


def _generate_unique_name(existing_names: set[str]) -> str:
    """Generate a random valid Python identifier that is not in existing_names.

    Produces names like ``_synth_a3f7b2`` that are valid identifiers,
    not Python keywords, and not in the given set of existing names.
    """
    while True:
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        candidate = f"_synth_{suffix}"
        if (
            candidate not in existing_names
            and candidate.isidentifier()
            and not keyword.iskeyword(candidate)
        ):
            return candidate


class _RenameTransformer(ast.NodeTransformer):
    """Rename function definitions and their references in a module AST.

    Given a mapping ``{old_name: new_name}``, renames:
    - ``FunctionDef.name`` for matching definitions
    - ``ast.Name.id`` references throughout the entire AST

    The rename is applied uniformly because it only targets module-level
    function definitions that collide with context variable declarations.
    Local assignments inside function bodies are in their own scope and
    cannot cause the mypy ``[no-redef]`` error, so they need no special
    handling.
    """

    def __init__(self, rename_map: dict[str, str]):
        self.rename_map = rename_map

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name in self.rename_map:
            node.name = self.rename_map[node.name]
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node


def mypy_type_check(
    module: ast.Module,
    ctx: typing.Mapping[str, Any],
    expected_params: list[type] | None,
    expected_return: type,
) -> None:
    """Type-check a module with mypy against expected signature and context.

    Builds a stub module from ctx (imports, runtime type stubs, variable declarations),
    appends the module body, then a postlude that assigns the last function to a
    variable annotated with Callable[expected_params, expected_return]. Runs mypy
    on the combined source; raises TypeError with the mypy report on failure.

    If the synthesized function name clashes with a name already in the context,
    the function is renamed to a unique random identifier for type-checking only.
    """
    if not module.body:
        raise TypeError("mypy_type_check: module.body is empty")
    last = module.body[-1]
    if not isinstance(last, ast.FunctionDef | ast.ClassDef):
        raise TypeError(
            f"mypy_type_check: last statement must be a function or class definition, "
            f"got {type(last).__name__}"
        )

    imports = collect_imports(ctx)
    # Ensure annotations in the postlude can be resolved (e.g. collections.abc.Callable, typing)
    baseline_imports: list[ast.stmt] = [
        ast.Import(names=[ast.alias(name="collections", asname=None)]),
        ast.Import(names=[ast.alias(name="collections.abc", asname=None)]),
        ast.Import(names=[ast.alias(name="typing", asname=None)]),
        ast.Import(names=[ast.alias(name="types", asname=None)]),
    ]
    stubs = collect_runtime_type_stubs(ctx)
    variables = collect_variable_declarations(ctx)

    # Collect names already declared in the type-checking preamble
    # (variable declarations and class stubs) that could collide with
    # function definitions in the synthesized module.
    declared_names = {
        stmt.target.id
        for stmt in variables
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    } | {stmt.name for stmt in stubs if isinstance(stmt, ast.ClassDef)}

    # Find all function names in the synthesized module that collide
    synthesized_func_names = {
        stmt.name
        for stmt in module.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    colliding_names = synthesized_func_names & declared_names

    if colliding_names:
        # Build a rename map for every colliding function name
        all_reserved = declared_names | synthesized_func_names
        rename_map: dict[str, str] = {}
        for name in colliding_names:
            unique = _generate_unique_name(all_reserved)
            rename_map[name] = unique
            all_reserved.add(unique)

        # Deep-copy the module body so we don't mutate the caller's AST,
        # then rename definitions and all references to them.
        module_body = copy.deepcopy(list(module.body))
        stub_module_body = ast.Module(body=module_body, type_ignores=[])
        _RenameTransformer(rename_map).visit(stub_module_body)
        module_body = stub_module_body.body
    else:
        module_body = list(module.body)

    postlude: list[ast.stmt] = []
    if isinstance(last, ast.FunctionDef):
        func_name = last.name
        tc_func_name = (
            rename_map.get(func_name, func_name)
            if colliding_names
            else func_name
        )
        param_types = expected_params
        expected_callable_type: type = typing.cast(
            type,
            collections.abc.Callable[param_types, expected_return]
            if expected_params is not None
            else collections.abc.Callable[..., expected_return],
        )
        expected_callable_ast = type_to_ast(expected_callable_type)
        postlude = [
            ast.AnnAssign(
                target=ast.Name(id="_synthesized_check", ctx=ast.Store()),
                annotation=expected_callable_ast,
                value=ast.Name(id=tc_func_name, ctx=ast.Load()),
                simple=1,
            )
        ]
    # For ClassDef: no postlude needed, mypy checks the class body directly.

    full_body = (
        baseline_imports
        + list(imports)
        + list(stubs)
        + list(variables)
        + module_body
        + postlude
    )
    stub_module = ast.Module(body=full_body, type_ignores=[])
    source = ast.unparse(ast.fix_missing_locations(stub_module))
    # Drop unused imports/vars

    source = autoflake.fix_code(
        source,
        additional_imports=None,
        expand_star_imports=True,
        remove_all_unused_imports=True,
        remove_duplicate_keys=True,
        remove_unused_variables=True,
    )

    stdout, stderr, status = mypy_api.run(
        [
            "--command",
            source,
            "--no-error-summary",
            "--no-pretty",
            "--ignore-missing-imports",
            "--disable-error-code=import-untyped",
        ]
    )
    if status != 0:
        report = (stdout or "") + (stderr or "")
        raise TypeError(f"mypy type check failed:\n{report}\n{source}")
    return None


# Eval Providers


class UnsafeEvalProvider(ObjectInterpretation):
    """UNSAFE provider that handles parse, comple and exec operations
    by shelling out to python *without* any further checks. Only use for testing."""

    @implements(type_check)
    def type_check(
        self,
        module: ast.Module,
        ctx: typing.Mapping[str, Any],
        expected_params: list[type] | None,
        expected_return: type,
    ) -> None:
        mypy_type_check(module, ctx, expected_params, expected_return)

    @implements(parse)
    def parse(self, source: str, filename: str) -> ast.Module:
        # Cache source under `filename` so inspect.getsource() can retrieve it later.
        # inspect uses f.__code__.co_filename -> linecache.getlines(filename)
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )

        return ast.parse(source, filename=filename, mode="exec")

    @implements(compile)
    def compile(self, module: ast.AST, filename: str) -> CodeType:
        return builtins.compile(typing.cast(typing.Any, module), filename, "exec")

    @implements(exec)
    def exec(
        self,
        bytecode: CodeType,
        env: dict[str, Any],
    ) -> None:
        # Ensure builtins exist in the execution environment.
        env.setdefault("__builtins__", __builtins__)

        # Execute module-style so top-level defs land in `env`.
        builtins.exec(bytecode, env, env)


class RestrictedEvalProvider(ObjectInterpretation):
    """
    Safer provider using RestrictedPython.

    RestrictedPython is not a complete sandbox, but it enforces a restricted
    language subset and expects you to provide a constrained exec environment.

    policy : dict[str, Any], optional
        RestrictedPython compile_restricted policy for compilation
    """

    policy: type[RestrictingNodeTransformer] | None = None

    def __init__(
        self,
        *,
        policy: type[RestrictingNodeTransformer] | None = None,
    ):
        self.policy = policy

    @implements(type_check)
    def type_check(
        self,
        module: ast.Module,
        ctx: typing.Mapping[str, Any],
        expected_params: list[type] | None,
        expected_return: type,
    ) -> None:
        mypy_type_check(module, ctx, expected_params, expected_return)

    @implements(parse)
    def parse(self, source: str, filename: str) -> ast.Module:
        # Keep inspect.getsource() working for dynamically-defined objects.
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )
        return ast.parse(source, filename=filename, mode="exec")

    @implements(compile)
    def compile(self, module: ast.Module, filename: str) -> CodeType:
        # RestrictedPython can compile from an AST directly.
        return compile_restricted(
            module,
            filename=filename,
            mode="exec",
            policy=self.policy or RestrictingNodeTransformer,
        )

    @implements(exec)
    def exec(
        self,
        bytecode: CodeType,
        env: dict[str, Any],
    ) -> None:
        # Build restricted globals from RestrictedPython's defaults
        rglobals: dict[str, Any] = safe_globals.copy()

        # Enable class definitions (required for Python 3)
        rglobals["__metaclass__"] = type
        rglobals["__name__"] = "restricted"

        # Layer `env` on top (without letting callers replace the restricted builtins).
        rglobals.update({k: v for k, v in env.items() if k != "__builtins__"})

        # Enable for loops and comprehensions
        rglobals["_getiter_"] = Eval.default_guarded_getiter
        # Enable sequence unpacking in comprehensions and for loops
        rglobals["_iter_unpack_sequence_"] = Guards.guarded_iter_unpack_sequence

        rglobals["getattr"] = Guards.safer_getattr
        rglobals["setattr"] = Guards.guarded_setattr
        rglobals["_write_"] = lambda x: x

        # Track keys before execution to identify new definitions
        keys_before = set(rglobals.keys())

        builtins.exec(bytecode, rglobals, rglobals)

        # Copy newly defined items back to env so caller can access them
        for key in rglobals:
            if key not in keys_before:
                env[key] = rglobals[key]
