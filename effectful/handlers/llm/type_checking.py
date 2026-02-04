"""Type-check synthesized code with mypy. Supports Python 3.12 and 3.13."""

import ast
import collections.abc
import types
import typing
from typing import Any

import mypy.api

from effectful.internals.unification import nested_type

# PEP 695 (3.12+); use getattr for forward compatibility
_TypeAliasType = getattr(typing, "TypeAliasType", type(None))


def _qualname(t: type) -> str:
    """Given a type, return the fully qualified name."""
    # Prefer builtins without module prefix
    mod = getattr(t, "__module__", "")
    name = getattr(t, "__qualname__", getattr(t, "__name__", str(t)))
    if mod == "builtins":
        return name
    return f"{mod}.{name}"


def _collect(
    typ: type,
    imports: list[str],
    type_alias_lines: list[str] | None = None,
) -> str:
    """Recursively convert a type to annotation string; appends required imports and type alias definitions."""
    if typ is ...:
        return "..."
    # Tuple of types (e.g. Operation's params (DecodedToolCall,)) – must not fall through to str(typ)
    if isinstance(typ, tuple):
        return (
            "(" + ", ".join(_collect(a, imports, type_alias_lines) for a in typ) + ")"
        )
    # PEP 695 type alias (type X = Y; Python 3.12+)
    if isinstance(typ, _TypeAliasType) and getattr(typ, "__value__", None) is not None:
        alias_name = getattr(typ, "__name__", None)
        alias_value = getattr(typ, "__value__", None)
        if alias_name and alias_value is not None and type_alias_lines is not None:
            expansion = _collect(alias_value, imports, type_alias_lines)
            type_alias_lines.append(f"type {alias_name} = {expansion}")
            return alias_name
        # fallback: expand inline
        typ = getattr(typ, "__value__", typ)

    origin = typing.get_origin(typ)
    args = typing.get_args(typ)

    if origin is not None and args:
        # Generic type
        if hasattr(types, "UnionType") and origin is types.UnionType:
            # X | Y (Python 3.10+)
            return " | ".join(_collect(a, imports, type_alias_lines) for a in args)
        if origin is typing.Union:
            return " | ".join(_collect(a, imports, type_alias_lines) for a in args)
        if origin is typing.Callable or origin is collections.abc.Callable:
            # Callable[[p1, p2], ret]
            if args[0] is ...:
                params = "..."
            else:
                params = ", ".join(
                    _collect(a, imports, type_alias_lines) for a in args[0]
                )
            ret = _collect(args[1], imports, type_alias_lines)
            return f"typing.Callable[[{params}], {ret}]"
        # Generic alias: list[int], tuple[int, str], Operation[(P), R], etc.
        if getattr(origin, "__module__", "") in ("builtins", "typing", ""):
            base = getattr(origin, "__name__", _qualname(origin))
        else:
            base = _qualname(origin)
            imports.append(f"import {getattr(origin, '__module__', '')}")
        if base == "tuple" and args == ((),):
            return "tuple[()]"
        args_str = ", ".join(_collect(a, imports, type_alias_lines) for a in args)
        return f"{base}[{args_str}]"
    # Simple type or unparameterized
    if typ is typing.Any or typ is type(None):
        if typ is typing.Any:
            imports.append("from typing import Any")
            return "Any"
        return "None"
    if isinstance(typ, type):
        mod = getattr(typ, "__module__", "")
        name = getattr(typ, "__name__", getattr(typ, "__qualname__", str(typ)))
        if mod == "builtins" or mod == "typing":
            return name
        if mod == "collections.abc":
            imports.append("from collections.abc import " + name)
            return name
        # Other modules: use qualified name and add import
        q = _qualname(typ)
        if mod and mod != "__main__":
            imports.append(f"import {mod}")
            return q
        return name
    return str(typ)


def _referenced_globals(module: ast.Module) -> set[str]:
    """Return names that are loaded (read) at module level or in the last function."""
    referenced: set[str] = set()
    defined: set[str] = set()

    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            for a in node.args.args + node.args.posonlyargs:
                defined.add(a.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and isinstance(
                node.value.ctx, ast.Load
            ):
                referenced.add(node.value.id)

    return referenced - defined


def _prelude_from_ctx(
    lexical_ctx: typing.Mapping[str, Any] | None,
    only_names: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Build prelude lines, import lines, and type alias lines from context.

    If only_names is provided, only include ctx keys that are in that set
    (e.g. names referenced by the synthesized module). This avoids emitting
    stubs for the entire lexical scope that mypy cannot type-check.
    """
    ctx = lexical_ctx or {}
    prelude_lines: list[str] = []
    all_imports: list[str] = []
    all_type_aliases: list[str] = []
    for name, value in ctx.items():
        if name.startswith("__"):
            continue
        if only_names is not None and name not in only_names:
            continue
        if isinstance(value, types.ModuleType):
            mod_name = getattr(value, "__name__", name)
            if name.startswith("@"):
                continue
            prelude_lines.append(
                f"import {mod_name}" + (f" as {name}" if name != mod_name else "")
            )
        elif isinstance(value, _TypeAliasType):
            ann_str, imps, type_aliases = _type_to_annotation_str(value)
            all_imports.extend(imps)
            all_type_aliases.extend(type_aliases)
            alias_name = getattr(value, "__name__", name)
            if alias_name != name:
                prelude_lines.append(f"{name}: {alias_name}")
        else:
            try:
                inferred = nested_type(value).value
            except Exception:
                continue
            ann_str, imps, type_aliases = _type_to_annotation_str(
                typing.cast(type, inferred)
            )
            all_imports.extend(imps)
            all_type_aliases.extend(type_aliases)
            prelude_lines.append(f"{name}: {ann_str}")
    return prelude_lines, all_imports, all_type_aliases


def _type_to_annotation_str(ty: type) -> tuple[str, list[str], list[str]]:
    """Convert a runtime type to a Python annotation string, import lines, and type alias definitions.

    Returns:
        (annotation_source, import lines e.g. ["from typing import Any"], type alias lines e.g. ["type X = int"])
    """
    imports: list[str] = []
    type_alias_lines: list[str] = []
    ann = _collect(ty, imports, type_alias_lines)
    return ann, list(dict.fromkeys(imports)), type_alias_lines


def typecheck_source(
    module: ast.AST,
    ctx: typing.Mapping[str, Any],
    expected_params: list[type] | None,
    expected_return: type,
) -> None:
    """Type-check synthesized module code against expected signature and context.

    Builds a full source with prelude (ctx bindings as type stubs), the module body,
    and a postlude that assigns the function to an expected Callable type so mypy
    validates the signature. Raises TypeError with mypy output on type errors.
    """
    if not isinstance(module, ast.Module) or not module.body:
        raise TypeError(
            "typecheck_source requires a Module AST with at least one statement"
        )
    last_stmt = module.body[-1]
    if not isinstance(last_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise TypeError(
            "typecheck_source requires the last statement to be a function definition"
        )
    func_name = last_stmt.name

    # 1. Prelude: only stub ctx names that the synthesized code actually references
    refs = _referenced_globals(module)
    prelude_lines, all_imports, all_type_aliases = _prelude_from_ctx(
        ctx, only_names=refs
    )

    # Expected callable annotation for postlude
    ret_ann, ret_imps, ret_aliases = _type_to_annotation_str(expected_return)
    all_imports.extend(ret_imps)
    all_type_aliases.extend(ret_aliases)
    all_imports.append("import typing")
    if expected_params is not None:
        params_ann = ", ".join(
            _type_to_annotation_str(typing.cast(type, p))[0] for p in expected_params
        )
        callable_ann = f"typing.Callable[[{params_ann}], {ret_ann}]"
    else:
        callable_ann = f"typing.Callable[..., {ret_ann}]"

    # 2. Module source from AST
    module_src = ast.unparse(module)

    # 3. Postlude: force mypy to check func_name matches expected signature (params + return).
    # Assignment to Callable[[...], ret_ann] enforces both; for zero-arg we also assert return type by calling.
    postlude_lines_list = [f"_check: {callable_ann} = {func_name}"]
    if (
        expected_params is not None
        and len(expected_params) == 0
        and expected_return is not type(None)
    ):
        postlude_lines_list.append(f"_return_check: {ret_ann} = {func_name}()")
    postlude = "\n".join(postlude_lines_list)

    imports_block = "\n".join(dict.fromkeys(all_imports))
    type_alias_block = "\n".join(dict.fromkeys(all_type_aliases))
    prelude_parts = [type_alias_block] if type_alias_block else []
    prelude_parts.append("\n".join(prelude_lines))
    prelude_block = "\n\n".join(prelude_parts)
    full_src = f"{imports_block}\n\n{prelude_block}\n\n{module_src}\n\n{postlude}\n"

    stdout, stderr, status = mypy.api.run(
        ["--command", full_src, "--show-error-codes", "--no-error-summary"]
    )
    if status != 0:
        report = (stdout or "").strip() + (
            "\n" + (stderr or "").strip() if stderr else ""
        )
        raise TypeError(f"Type check failed for synthesized function:\n{report}")
