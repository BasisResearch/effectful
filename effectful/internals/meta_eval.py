import ast
import builtins
import inspect
import linecache
import sys
from collections import ChainMap
from collections.abc import Callable, Generator, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any


class _ClassCell:
    """Mutable container for class forward reference in method definitions."""

    __slots__ = ("cell_contents",)

    def __init__(self) -> None:
        self.cell_contents: Any = None


RESTRICTED_GLOBALS = {
    "hasattr",
    "__import__",
    "quit",
    "__build_class__",
    "__package__",
    "exit",
    "__loader__",
    "compile",
    "exec",
    "copyright",
    "locals",
    "eval",
    "breakpoint",
    "__doc__",
    "globals",
    "input",
    "open",
}

MAX_WHILE_ITERATIONS = 1_000_000


def get_source() -> str:
    if "__file__" in globals():
        return Path(__file__).read_text()
    try:
        module = inspect.getmodule(get_source)
        if module is None:
            raise RuntimeError("Source not available in this context")
        return inspect.getsource(module)
    except (OSError, TypeError) as e:
        raise RuntimeError("Source not available in this context") from e


def install_synthetic_module(
    source_text: str, module_name: str | None = None
) -> tuple[str, str]:
    """
    Registers a synthetic module in sys.modules and installs its full text into linecache,
    so inspect.getsource() can locate definitions by filename + lineno.
    """
    if module_name is None:
        module_name = f"__mci__{abs(hash(source_text))}_{id(source_text)}"
    filename = f"<mci:{module_name}>"

    lines = source_text.splitlines(keepends=True)
    linecache.cache[filename] = (len(source_text), None, lines, filename)

    mod = sys.modules.get(module_name)
    if mod is None:
        mod = ModuleType(module_name)
        sys.modules[module_name] = mod
    mod.__file__ = filename
    mod.__package__ = None
    mod.__loader__ = None
    return module_name, filename


class InterpreterError(RuntimeError):
    pass


class BreakException(BaseException):
    pass


class ContinueException(BaseException):
    pass


class ReturnException(BaseException):
    def __init__(self, value: Any):
        self.value = value


@dataclass
class ScopeDirectives:
    globals: set[str]
    nonlocals: set[str]


@dataclass
class EvaluatorState:
    """
    bindings is a ChainMap stack:
      bindings.maps[0] is current local scope
      bindings.maps[1:] outer scopes + module globals + builtins

    qualname_stack tracks the current lexical nesting for __qualname__.
    Inside a function body, Python inserts a '<locals>' marker in qualnames.
    """

    bindings: ChainMap[str, Any]
    modules: dict[str, ModuleType]
    module_globals: dict[
        str, Any
    ]  # Reference to module-level globals for `global` stmt
    allowed_modules: Mapping[str, ModuleType]
    scope_directives: list[ScopeDirectives]
    allowed_dunder_attrs: set[str]

    module_name: str
    module_filename: str

    qualname_stack: list[str]
    exception_stack: list[BaseException]  # Track active exceptions for bare raise
    class_cell: _ClassCell | None  # Cell for __class__ in method definitions

    @classmethod
    def fresh(
        cls,
        allowed_modules: dict[str, ModuleType] | None = None,
        allowed_dunder_attrs: list[str] | None = None,
        *,
        module_name: str = "__main__",
        module_filename: str = "<mci:__main__>",
    ) -> "EvaluatorState":
        if not allowed_dunder_attrs:
            allowed_dunder_attrs = ["__init__", "__str__", "__repr__"]

        if not allowed_modules:
            allowed_modules = {}
            current_frame = inspect.currentframe()
            current_globals = (
                current_frame.f_back.f_globals
                if current_frame and current_frame.f_back
                else globals()
            )
            for k, v in current_globals.items():
                if isinstance(v, ModuleType):
                    allowed_modules[k] = v
                elif (
                    hasattr(v, "__module__") and getattr(v, "__module__") in sys.modules
                ):
                    allowed_modules[getattr(v, "__module__")] = sys.modules[
                        getattr(v, "__module__")
                    ]

        safe_builtins = {
            k: builtins.__dict__[k]
            for k in builtins.__dict__
            if k not in RESTRICTED_GLOBALS
        }

        module_globals: dict[str, Any] = dict(allowed_modules)
        module_globals.setdefault("__name__", module_name)
        module_globals.setdefault("__file__", module_filename)
        module_globals.setdefault("__package__", None)

        # At module level, maps[0] IS module_globals (no separate local scope)
        return cls(
            bindings=ChainMap(module_globals, safe_builtins),
            modules=module_globals,
            module_globals=module_globals,
            allowed_modules=MappingProxyType(allowed_modules),
            scope_directives=[],
            allowed_dunder_attrs=set(allowed_dunder_attrs),
            module_name=module_name,
            module_filename=module_filename,
            qualname_stack=[],
            exception_stack=[],
            class_cell=None,
        )

    # ----- scope stack -----

    def push_scope(self) -> None:
        self.bindings = self.bindings.new_child({})
        self.scope_directives.append(ScopeDirectives(set(), set()))

    def pop_scope(self) -> None:
        if not self.scope_directives:
            raise RuntimeError("Interpreter Scope stack underflow")
        self.scope_directives.pop()
        # ChainMap.parents returns the parent ChainMap (maps[1:])
        # This is a ChainMap attribute that exists at runtime
        parents = getattr(self.bindings, "parents", None)
        if parents is None:
            raise RuntimeError("ChainMap.parents not available")
        self.bindings = parents

    def current_directives(self) -> ScopeDirectives | None:
        return self.scope_directives[-1] if self.scope_directives else None

    def resolve_store_target_map(self, name: str) -> MutableMapping[str, Any]:
        directives = self.current_directives()
        if directives is None:
            # At module level (no function scope), write to local scope (which is module globals)
            return self.bindings.maps[0]

        if name in directives.globals:
            # Global writes to module globals - find it in the chain
            for m in self.bindings.maps:
                if m is self.module_globals:
                    return m
            # Fallback to maps[1] if module_globals not found
            return (
                self.bindings.maps[1]
                if len(self.bindings.maps) > 1
                else self.bindings.maps[0]
            )

        if name in directives.nonlocals:
            for m in self.bindings.maps[1:]:
                if m is self.module_globals:
                    continue  # Skip module globals for nonlocal
                if name in m:
                    return m
            raise NameError(f"nonlocal '{name}' not found in enclosing scope")

        return self.bindings.maps[0]

    # ----- qualname tracking (NEW) -----

    def current_qual_prefix(self) -> str:
        return ".".join(self.qualname_stack) if self.qualname_stack else ""

    def push_qual(self, name: str, *, add_locals_marker: bool = False) -> None:
        self.qualname_stack.append(name)
        if add_locals_marker:
            self.qualname_stack.append("<locals>")

    def pop_qual(self, *, had_locals_marker: bool = False) -> None:
        if had_locals_marker:
            if not self.qualname_stack or self.qualname_stack[-1] != "<locals>":
                raise RuntimeError("qualname stack mismatch (missing <locals>)")
            self.qualname_stack.pop()
        if not self.qualname_stack:
            raise RuntimeError("qualname stack underflow")
        self.qualname_stack.pop()

    def make_qualname(self, name: str) -> str:
        prefix = self.current_qual_prefix()
        return f"{prefix}.{name}" if prefix else name


# -------------------------
# assignment helpers
# -------------------------


def assign_target(target: ast.AST, value: Any, state: EvaluatorState) -> None:
    match target:
        case ast.Name(id=name, ctx=_):
            dst = state.resolve_store_target_map(name)
            if name in state.bindings.maps[-1]:
                raise InterpreterError(f"Cannot assign to builtin name '{name}'")
            dst[name] = value

        case ast.Tuple(elts=elts, ctx=_):
            seq = list(value)
            if len(seq) != len(elts):
                raise InterpreterError("Tuple unpacking mismatch")
            for t, v in zip(elts, seq):
                assign_target(t, v, state)

        case ast.List(elts=elts, ctx=_):
            seq = list(value)
            if len(seq) != len(elts):
                raise InterpreterError("List unpacking mismatch")
            for t, v in zip(elts, seq):
                assign_target(t, v, state)

        case ast.Subscript(value=base, slice=s, ctx=_):
            obj = eval_expr(base, state)
            idx = eval_expr(s, state)
            obj[idx] = value

        case ast.Attribute(value=base, attr=attr, ctx=_):
            if (
                attr.startswith("__")
                and attr.endswith("__")
                and attr not in state.allowed_dunder_attrs
            ):
                raise InterpreterError(f"Forbidden dunder attribute set: {attr}")
            obj = eval_expr(base, state)
            setattr(obj, attr, value)

        case ast.Starred(value=inner, ctx=_):
            assign_target(inner, value, state)

        case _:
            raise InterpreterError(
                f"Unsupported assignment target: {type(target).__name__}"
            )


def assign_extended_unpack(target: ast.AST, value: Any, state: EvaluatorState) -> Any:
    if not isinstance(target, (ast.Tuple, ast.List)):
        raise InterpreterError("Extended unpacking requires tuple/list target")
    elts = target.elts
    seq = list(value)

    star_i = next(i for i, e in enumerate(elts) if isinstance(e, ast.Starred))
    before = elts[:star_i]
    star = elts[star_i]
    after = elts[star_i + 1 :]

    if len(seq) < (len(before) + len(after)):
        raise InterpreterError("Extended unpacking mismatch")

    for t, v in zip(before, seq[: len(before)]):
        assign_target(t, v, state)

    mid = seq[len(before) : len(seq) - len(after)]
    assign_target(star, mid, state)

    tail = seq[len(seq) - len(after) :]
    for t, v in zip(after, tail):
        assign_target(t, v, state)

    return value


def delete_target(target: ast.AST, state: EvaluatorState) -> None:
    match target:
        case ast.Name(id=name):
            for m in state.bindings.maps:
                if name in m:
                    del m[name]
                    return
            raise InterpreterError(f"Cannot delete '{name}': not found")

        case ast.Subscript(value=base, slice=s):
            obj = eval_expr(base, state)
            idx = eval_expr(s, state)
            del obj[idx]

        case ast.Attribute(value=base, attr=attr):
            if (
                attr.startswith("__")
                and attr.endswith("__")
                and attr not in state.allowed_dunder_attrs
            ):
                raise InterpreterError(f"Forbidden dunder attribute delete: {attr}")
            obj = eval_expr(base, state)
            delattr(obj, attr)

        case _:
            raise InterpreterError(
                f"Unsupported delete target: {type(target).__name__}"
            )


# -------------------------
# imports
# -------------------------


def eval_import(stmt: ast.Import, state: EvaluatorState):
    for mod in stmt.names:
        if mod.name not in state.allowed_modules:
            raise ImportError(
                f"Import of '{mod.name}' is not allowed in this evaluator. "
                f"Allowed modules: {list(state.allowed_modules)}"
            )
        state.modules[mod.asname or mod.name] = state.allowed_modules[mod.name]


def eval_import_from(stmt: ast.ImportFrom, state: EvaluatorState) -> None:
    if stmt.level and stmt.level > 0:
        raise ImportError("Relative imports are not supported in this evaluator")

    if stmt.module is None:
        eval_import(ast.Import(stmt.names), state)
        return

    if stmt.module not in state.allowed_modules:
        raise ImportError(
            f"Import of '{stmt.module}' is not allowed. Allowed: {list(state.allowed_modules.keys())}"
        )

    base_module = state.allowed_modules[stmt.module]
    for mod in stmt.names:
        if mod.name == "*":
            for name in dir(base_module):
                if not name.startswith("_"):
                    state.bindings[name] = getattr(base_module, name)
            continue
        state.modules[mod.asname or mod.name] = getattr(base_module, mod.name)


# -------------------------
# decorators
# -------------------------


def apply_decorators(
    obj: Any, decorator_exprs: list[ast.expr], state: EvaluatorState
) -> Any:
    decorated = obj
    for dec in reversed(decorator_exprs):
        dec_fn = eval_expr(dec, state)
        new_obj = dec_fn(decorated)

        # help inspect.unwrap() find original, best-effort
        if (
            callable(new_obj)
            and callable(decorated)
            and not hasattr(new_obj, "__wrapped__")
        ):
            try:
                new_obj.__wrapped__ = decorated
            except Exception:
                pass

        decorated = new_obj
    return decorated


# -------------------------
# expressions
# -------------------------


def eval_expr(node: ast.AST, state: EvaluatorState) -> Any:
    """Evaluate an expression in a non-generator context. Yields are not supported and will raise InterpreterError."""
    match node:
        case ast.Constant(value=v):
            return v

        case ast.Name(id=name, ctx=_):
            if name in state.bindings:
                val = state.bindings[name]
                return val
            raise NameError(f"Name '{name}' is not defined")

        case ast.Tuple(elts=elts, ctx=_):
            return tuple(eval_expr(e, state) for e in elts)

        case ast.List(elts=elts, ctx=_):
            return [eval_expr(e, state) for e in elts]

        case ast.Set(elts=elts):
            return {eval_expr(e, state) for e in elts}

        case ast.Dict(keys=keys, values=values):
            base_dict = {}
            for k, v in zip(keys, values):
                if k is not None:
                    base_dict[eval_expr(k, state)] = eval_expr(v, state)
                else:
                    res = eval_expr(v, state)
                    if not isinstance(res, dict):
                        raise InterpreterError(
                            "** mapping in dict literal must be a dict"
                        )
                    base_dict = {**base_dict, **res}
            return base_dict

        case ast.JoinedStr(values=values):
            return "".join(str(eval_expr(v, state)) for v in values)

        case ast.FormattedValue(value=v, conversion=conv, format_spec=fs):
            val = eval_expr(v, state)
            match conv:
                case 115:
                    val = str(val)
                case 114:
                    val = repr(val)
                case 97:
                    val = ascii(val)
                case -1:
                    pass
                case _:
                    pass
            if fs is None:
                return val
            return format(val, eval_expr(fs, state))

        case ast.UnaryOp(op=op, operand=operand):
            v = eval_expr(operand, state)
            match op:
                case ast.UAdd():
                    return +v
                case ast.USub():
                    return -v
                case ast.Not():
                    return not v
                case ast.Invert():
                    return ~v
                case _:
                    raise InterpreterError(f"Unsupported unary op: {type(op).__name__}")

        case ast.BinOp(left=l, op=op, right=r):
            a = eval_expr(l, state)
            b = eval_expr(r, state)
            match op:
                case ast.Add():
                    return a + b
                case ast.Sub():
                    return a - b
                case ast.Mult():
                    return a * b
                case ast.MatMult():
                    return a @ b
                case ast.Div():
                    return a / b
                case ast.FloorDiv():
                    return a // b
                case ast.Mod():
                    return a % b
                case ast.Pow():
                    return a**b
                case ast.LShift():
                    return a << b
                case ast.RShift():
                    return a >> b
                case ast.BitAnd():
                    return a & b
                case ast.BitOr():
                    return a | b
                case ast.BitXor():
                    return a ^ b
                case _:
                    raise InterpreterError(f"Unsupported bin op: {type(op).__name__}")

        case ast.BoolOp(op=op, values=values):
            match op:
                case ast.And():
                    last = True
                    for v in values:
                        last = eval_expr(v, state)
                        if not last:
                            return last
                    return last
                case ast.Or():
                    last = False
                    for v in values:
                        last = eval_expr(v, state)
                        if last:
                            return last
                    return last
                case _:
                    raise InterpreterError(f"Unsupported bool op: {type(op).__name__}")

        case ast.Compare(left=left, ops=ops, comparators=comps):
            lval = eval_expr(left, state)
            for cmp_op, comp in zip(ops, comps):
                rval = eval_expr(comp, state)
                match cmp_op:
                    case ast.Eq():
                        ok = lval == rval
                    case ast.NotEq():
                        ok = lval != rval
                    case ast.Lt():
                        ok = lval < rval
                    case ast.LtE():
                        ok = lval <= rval
                    case ast.Gt():
                        ok = lval > rval
                    case ast.GtE():
                        ok = lval >= rval
                    case ast.Is():
                        ok = lval is rval
                    case ast.IsNot():
                        ok = lval is not rval
                    case ast.In():
                        ok = lval in rval
                    case ast.NotIn():
                        ok = lval not in rval
                    case _:
                        raise InterpreterError(
                            f"Unsupported compare op: {type(cmp_op).__name__}"
                        )
                if not ok:
                    return False
                lval = rval
            return True

        case ast.IfExp(test=t, body=b, orelse=o):
            return eval_expr(b, state) if eval_expr(t, state) else eval_expr(o, state)

        case ast.Attribute(value=base, attr=attr, ctx=_):
            if (
                attr.startswith("__")
                and attr.endswith("__")
                and attr not in state.allowed_dunder_attrs
            ):
                raise InterpreterError(f"Forbidden dunder attribute access: {attr}")
            obj = eval_expr(base, state)
            return getattr(obj, attr)

        case ast.Subscript(value=base, slice=s, ctx=_):
            obj = eval_expr(base, state)
            idx = eval_expr(s, state)
            return obj[idx]

        case ast.Slice(lower=lo, upper=up, step=st):
            return slice(
                eval_expr(lo, state) if lo is not None else None,
                eval_expr(up, state) if up is not None else None,
                eval_expr(st, state) if st is not None else None,
            )

        case ast.Call(func=f, args=args, keywords=keywords):
            fn = eval_expr(f, state)

            # Handle super() without arguments - provide __class__ dynamically
            if fn is super and len(args) == 0 and not keywords:
                # Get __class__ from state (set during class definition)
                class_obj: Any = state.bindings.get("__class__")

                # Get self from state (set when method is called) or from local scope
                self_obj = state.bindings.get("__self__")

                # If __class__ not set, get it from type(self)
                if class_obj is None:
                    class_obj = type(self_obj)

                return super(class_obj, self_obj)

            # check for dunder methods
            if hasattr(fn, "__name__"):
                nm = fn.__name__
                if (
                    nm.startswith("__")
                    and nm.endswith("__")
                    and nm not in state.allowed_dunder_attrs
                ):
                    raise InterpreterError(f"Forbidden dunder call: {nm}")

            pos = []
            for a in args:
                if isinstance(a, ast.Starred):
                    pos.extend(list(eval_expr(a.value, state)))
                else:
                    pos.append(eval_expr(a, state))

            kw: dict[str, Any] = {}
            for keyword_node in keywords:
                if keyword_node.arg is None:
                    d = eval_expr(keyword_node.value, state)
                    if not isinstance(d, dict):
                        raise InterpreterError("**kwargs must be a dict")
                    kw.update(d)
                else:
                    kw[keyword_node.arg] = eval_expr(keyword_node.value, state)

            return fn(*pos, **kw)

        case ast.Lambda(args=a, body=b):
            arg_names = [x.arg for x in a.args]
            defaults = [eval_expr(d, state) for d in a.defaults]
            default_map = (
                dict(zip(arg_names[-len(defaults) :], defaults)) if defaults else {}
            )
            captured_maps = list(state.bindings.maps)
            lambda_has_yield = _has_yield_direct(b)

            def _lambda(*vals, **kws):
                local_state = EvaluatorState(
                    bindings=ChainMap({}, *captured_maps),
                    modules=state.modules,
                    module_globals=state.module_globals,
                    allowed_modules=state.allowed_modules,
                    scope_directives=[],
                    allowed_dunder_attrs=state.allowed_dunder_attrs,
                    module_name=state.module_name,
                    module_filename=state.module_filename,
                    qualname_stack=state.qualname_stack + ["<lambda>"],
                    exception_stack=state.exception_stack.copy(),
                    class_cell=None,
                )
                local_state.push_scope()
                local_state.push_qual("<lambda>", add_locals_marker=True)
                try:
                    local_scope = local_state.bindings.maps[0]
                    for n, v in zip(arg_names, vals):
                        local_scope[n] = v
                    for n, v in kws.items():
                        local_scope[n] = v
                    # Apply defaults only if not provided as arg/kwarg
                    for n, v in default_map.items():
                        if n not in local_scope:
                            local_scope[n] = v
                    if lambda_has_yield:
                        return eval_expr_generator(b, local_state)
                    else:
                        return eval_expr(b, local_state)
                finally:
                    local_state.pop_qual(had_locals_marker=True)
                    local_state.pop_scope()

            _lambda.__name__ = "<lambda>"
            _lambda.__qualname__ = state.make_qualname("<lambda>")
            _lambda.__module__ = state.module_name
            _lambda.__code__ = _lambda.__code__.replace(
                co_filename=state.module_filename,
                co_firstlineno=getattr(node, "lineno", 1),
            )
            return _lambda

        case ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp():
            return eval_comprehension(node, state)

        case ast.Yield(value=_):
            raise InterpreterError(
                "yield expressions are not supported in non-generator context. Use eval_expr_generator instead."
            )

        case ast.YieldFrom(value=_):
            raise InterpreterError(
                "yield from expressions are not supported in non-generator context. Use eval_expr_generator instead."
            )

        case _:
            raise InterpreterError(f"Unsupported expression: {type(node).__name__}")


def eval_expr_generator(
    node: ast.AST, state: EvaluatorState
) -> Generator[Any, None, Any]:
    """Evaluate an expression in a generator context. Yields values when encountering yield/yield from."""
    match node:
        case ast.Constant(value=v):
            return v

        case ast.Name(id=name, ctx=_):
            if name in state.bindings:
                return state.bindings[name]
            raise NameError(f"Name '{name}' is not defined")

        case ast.Tuple(elts=elts, ctx=_):
            vls = []
            for e in elts:
                result = yield from eval_expr_generator(e, state)
                vls.append(result)
            return tuple(vls)

        case ast.List(elts=elts, ctx=_):
            vls = []
            for e in elts:
                result = yield from eval_expr_generator(e, state)
                vls.append(result)
            return vls

        case ast.Set(elts=elts):
            vls = []
            for e in elts:
                result = yield from eval_expr_generator(e, state)
                vls.append(result)
            return set(vls)

        case ast.Dict(keys=keys, values=values):
            base_dict = {}
            for k, v in zip(keys, values):
                if k is not None:
                    key_val = yield from eval_expr_generator(k, state)
                    val = yield from eval_expr_generator(v, state)
                    base_dict[key_val] = val
                else:
                    res = yield from eval_expr_generator(v, state)
                    if not isinstance(res, dict):
                        raise InterpreterError(
                            "** mapping in dict literal must be a dict"
                        )
                    base_dict = {**base_dict, **res}
            return base_dict

        case ast.JoinedStr(values=values):
            args = []
            for v in values:
                result = yield from eval_expr_generator(v, state)
                args.append(str(result))
            return "".join(args)

        case ast.FormattedValue(value=v, conversion=conv, format_spec=fs):
            val = yield from eval_expr_generator(v, state)
            match conv:
                case 115:
                    val = str(val)
                case 114:
                    val = repr(val)
                case 97:
                    val = ascii(val)
                case -1:
                    pass
                case _:
                    pass
            if fs is None:
                return val
            fs_val = yield from eval_expr_generator(fs, state)
            return format(val, fs_val)

        case ast.UnaryOp(op=op, operand=operand):
            v = yield from eval_expr_generator(operand, state)
            match op:
                case ast.UAdd():
                    return +v
                case ast.USub():
                    return -v
                case ast.Not():
                    return not v
                case ast.Invert():
                    return ~v
                case _:
                    raise InterpreterError(f"Unsupported unary op: {type(op).__name__}")

        case ast.BinOp(left=l, op=op, right=r):
            a = yield from eval_expr_generator(l, state)
            b = yield from eval_expr_generator(r, state)
            match op:
                case ast.Add():
                    return a + b
                case ast.Sub():
                    return a - b
                case ast.Mult():
                    return a * b
                case ast.MatMult():
                    return a @ b
                case ast.Div():
                    return a / b
                case ast.FloorDiv():
                    return a // b
                case ast.Mod():
                    return a % b
                case ast.Pow():
                    return a**b
                case ast.LShift():
                    return a << b
                case ast.RShift():
                    return a >> b
                case ast.BitAnd():
                    return a & b
                case ast.BitOr():
                    return a | b
                case ast.BitXor():
                    return a ^ b
                case _:
                    raise InterpreterError(f"Unsupported bin op: {type(op).__name__}")

        case ast.BoolOp(op=op, values=values):
            match op:
                case ast.And():
                    last = True
                    for v in values:
                        last = yield from eval_expr_generator(v, state)
                        if not last:
                            return last
                    return last
                case ast.Or():
                    last = False
                    for v in values:
                        last = yield from eval_expr_generator(v, state)
                        if last:
                            return last
                    return last
                case _:
                    raise InterpreterError(f"Unsupported bool op: {type(op).__name__}")

        case ast.Compare(left=left, ops=ops, comparators=comps):
            lval = yield from eval_expr_generator(left, state)
            for cmp_op, comp in zip(ops, comps):
                rval = yield from eval_expr_generator(comp, state)
                match cmp_op:
                    case ast.Eq():
                        ok = lval == rval
                    case ast.NotEq():
                        ok = lval != rval
                    case ast.Lt():
                        ok = lval < rval
                    case ast.LtE():
                        ok = lval <= rval
                    case ast.Gt():
                        ok = lval > rval
                    case ast.GtE():
                        ok = lval >= rval
                    case ast.Is():
                        ok = lval is rval
                    case ast.IsNot():
                        ok = lval is not rval
                    case ast.In():
                        ok = lval in rval
                    case ast.NotIn():
                        ok = lval not in rval
                    case _:
                        raise InterpreterError(
                            f"Unsupported compare op: {type(cmp_op).__name__}"
                        )
                if not ok:
                    return False
                lval = rval
            return True

        case ast.IfExp(test=t, body=b, orelse=o):
            test_val = yield from eval_expr_generator(t, state)
            if test_val:
                return (yield from eval_expr_generator(b, state))
            else:
                return (yield from eval_expr_generator(o, state))

        case ast.Attribute(value=base, attr=attr, ctx=_):
            if (
                attr.startswith("__")
                and attr.endswith("__")
                and attr not in state.allowed_dunder_attrs
            ):
                raise InterpreterError(f"Forbidden dunder attribute access: {attr}")
            obj = yield from eval_expr_generator(base, state)
            return getattr(obj, attr)

        case ast.Subscript(value=base, slice=s, ctx=_):
            obj = yield from eval_expr_generator(base, state)
            idx = yield from eval_expr_generator(s, state)
            return obj[idx]

        case ast.Slice(lower=lo, upper=up, step=st):
            return slice(
                (yield from eval_expr_generator(lo, state)) if lo is not None else None,
                (yield from eval_expr_generator(up, state)) if up is not None else None,
                (yield from eval_expr_generator(st, state)) if st is not None else None,
            )

        case ast.Call(func=f, args=args, keywords=keywords):
            fn = yield from eval_expr_generator(f, state)

            # Handle super() without arguments - provide __class__ dynamically
            if fn is super and len(args) == 0 and not keywords:
                class_obj: Any = state.bindings.get("__class__")
                self_obj = state.bindings.get("__self__")

                if class_obj is None:
                    class_obj = type(self_obj)

                return super(class_obj, self_obj)

            if isinstance(fn, Generator):
                raise InterpreterError(
                    f"Cannot call a generator object {fn}. Did you mean to iterate over it or use 'yield from'?"
                )

            if hasattr(fn, "__name__"):
                nm = fn.__name__
                if (
                    nm.startswith("__")
                    and nm.endswith("__")
                    and nm not in state.allowed_dunder_attrs
                ):
                    raise InterpreterError(f"Forbidden dunder call: {nm}")

            pos = []
            for a in args:
                if isinstance(a, ast.Starred):
                    val = yield from eval_expr_generator(a.value, state)
                    pos.extend(list(val))
                else:
                    val = yield from eval_expr_generator(a, state)
                    pos.append(val)

            kw: dict[str, Any] = {}
            for keyword_node in keywords:
                if keyword_node.arg is None:
                    d = yield from eval_expr_generator(keyword_node.value, state)
                    if not isinstance(d, dict):
                        raise InterpreterError("**kwargs must be a dict")
                    kw.update(d)
                else:
                    val = yield from eval_expr_generator(keyword_node.value, state)
                    kw[keyword_node.arg] = val

            return fn(*pos, **kw)

        case ast.Lambda(args=a, body=b):
            arg_names = [x.arg for x in a.args]
            defaults = []
            for d in a.defaults:
                result = yield from eval_expr_generator(d, state)
                defaults.append(result)
            default_map = (
                dict(zip(arg_names[-len(defaults) :], defaults)) if defaults else {}
            )
            captured_maps = list(state.bindings.maps)
            lambda_has_yield = _has_yield_direct(b)

            def _lambda(*vals, **kws):
                local_state = EvaluatorState(
                    bindings=ChainMap({}, *captured_maps),
                    modules=state.modules,
                    module_globals=state.module_globals,
                    allowed_modules=state.allowed_modules,
                    scope_directives=[],
                    allowed_dunder_attrs=state.allowed_dunder_attrs,
                    module_name=state.module_name,
                    module_filename=state.module_filename,
                    qualname_stack=state.qualname_stack + ["<lambda>"],
                    exception_stack=state.exception_stack.copy(),
                    class_cell=None,
                )
                local_state.push_scope()
                local_state.push_qual("<lambda>", add_locals_marker=True)
                try:
                    local_scope = local_state.bindings.maps[0]
                    for n, v in zip(arg_names, vals):
                        local_scope[n] = v
                    for n, v in kws.items():
                        local_scope[n] = v
                    # Apply defaults only if not provided as arg/kwarg
                    for n, v in default_map.items():
                        if n not in local_scope:
                            local_scope[n] = v
                    if lambda_has_yield:
                        return eval_expr_generator(b, local_state)
                    else:
                        return eval_expr(b, local_state)
                finally:
                    local_state.pop_qual(had_locals_marker=True)
                    local_state.pop_scope()

            _lambda.__name__ = "<lambda>"
            _lambda.__qualname__ = state.make_qualname("<lambda>")
            _lambda.__module__ = state.module_name
            _lambda.__code__ = _lambda.__code__.replace(
                co_filename=state.module_filename,
                co_firstlineno=getattr(node, "lineno", 1),
            )
            return _lambda

        case ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp():
            return eval_comprehension(node, state)

        case ast.Yield(value=v):
            if v is not None:
                val = yield from eval_expr_generator(v, state)
            else:
                val = None
            yield val
            return val

        case ast.YieldFrom(value=v):
            gen = yield from eval_expr_generator(v, state)
            if gen is None:
                raise InterpreterError("yield from requires a generator, got None")
            if not isinstance(gen, Generator):
                raise InterpreterError(
                    f"yield from requires a generator, got {type(gen).__name__}"
                )
            result = yield from gen
            return result

        case _:
            raise InterpreterError(f"Unsupported expression: {type(node).__name__}")


def eval_comprehension(node: ast.AST, state: EvaluatorState) -> Any:
    def gen_items(
        generators: list[ast.comprehension], i: int
    ) -> Generator[None, None, None]:
        if i >= len(generators):
            yield None
            return
        gen = generators[i]
        it = eval_expr(gen.iter, state)
        for v in it:
            state.push_scope()
            try:
                assign_target(gen.target, v, state)
                if all(eval_expr(cond, state) for cond in gen.ifs):
                    yield from gen_items(generators, i + 1)
            finally:
                state.pop_scope()

    if isinstance(node, ast.ListComp):
        # Use list comprehension would require restructuring gen_items, so keep loop
        list_out: list[Any] = []
        for _ in gen_items(node.generators, 0):
            list_out.append(eval_expr(node.elt, state))  # noqa: PERF401
        return list_out

    if isinstance(node, ast.SetComp):
        set_out: set[Any] = set()
        for _ in gen_items(node.generators, 0):
            set_out.add(eval_expr(node.elt, state))
        return set_out

    if isinstance(node, ast.DictComp):
        dict_out: dict[Any, Any] = {}
        for _ in gen_items(node.generators, 0):
            k = eval_expr(node.key, state)
            v = eval_expr(node.value, state)
            dict_out[k] = v
        return dict_out

    if isinstance(node, ast.GeneratorExp):

        def _g():
            for _ in gen_items(node.generators, 0):
                yield eval_expr(node.elt, state)

        return _g()

    raise InterpreterError(f"Unsupported comprehension node: {type(node).__name__}")


# -------------------------
# functions + classes (nested qualnames + inspect support)
# -------------------------


def is_generator_function(node: ast.FunctionDef) -> bool:
    """Check if a function definition contains yield or yield from, making it a generator.

    Only yields directly in the function body count - yields in nested functions/lambdas don't
    make the outer function a generator.
    """
    for stmt in node.body:
        if _has_yield_direct(stmt):
            return True
    return False


def _has_yield_direct(node: ast.AST) -> bool:
    """Recursively check if an AST node contains yield or yield from.

    Stops at nested function definitions (FunctionDef, Lambda) - yields inside
    those don't count for the outer function.
    """
    if isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
        return True

    # Don't recurse into nested function definitions
    if isinstance(node, (ast.FunctionDef, ast.Lambda)):
        return False

    for child in ast.iter_child_nodes(node):
        if _has_yield_direct(child):
            return True
    return False


def make_function(fn: ast.FunctionDef, state: EvaluatorState) -> Callable[..., Any]:
    captured_maps = list(state.bindings.maps)

    # definition-time defaults (Python semantics)
    defaults = [eval_expr(d, state) for d in fn.args.defaults]
    kw_defaults = [
        eval_expr(d, state) if d is not None else None for d in fn.args.kw_defaults
    ]

    arg_names = [a.arg for a in fn.args.args]
    posonly_names = [a.arg for a in getattr(fn.args, "posonlyargs", [])]
    kwonly_names = [a.arg for a in fn.args.kwonlyargs]
    vararg_name = fn.args.vararg.arg if fn.args.vararg else None
    kwarg_name = fn.args.kwarg.arg if fn.args.kwarg else None

    fn_qualname = state.make_qualname(fn.name)
    fn_lineno = getattr(fn, "lineno", 1)

    is_gen = is_generator_function(fn)

    # Capture class cell for super() support (None if not in a class)
    class_cell = state.class_cell
    first_param_name = arg_names[0] if arg_names else None

    # Extract docstring if present (first statement is a string constant)
    docstring = None
    if fn.body and isinstance(fn.body[0], ast.Expr):
        c = getattr(fn.body[0], "value", None)
        if isinstance(c, ast.Constant) and isinstance(c.value, str):
            docstring = c.value

    def setup_args(*args, **kwargs):
        """Common setup for both generator and regular functions."""
        local_state = EvaluatorState(
            bindings=ChainMap({}, *captured_maps),
            modules=state.modules,
            module_globals=state.module_globals,
            allowed_modules=state.allowed_modules,
            scope_directives=[],
            allowed_dunder_attrs=state.allowed_dunder_attrs,
            module_name=state.module_name,
            module_filename=state.module_filename,
            # Inside the function body, nested defs/classes should be qualified as:
            # outer.<locals>.inner
            qualname_stack=fn_qualname.split(".") + ["<locals>"],
            exception_stack=state.exception_stack.copy(),
            class_cell=None,
        )

        local_state.push_scope()
        local_scope = local_state.bindings.maps[0]

        # Set up __class__ from captured cell if we're in a method
        if class_cell is not None:
            local_scope["__class__"] = class_cell.cell_contents

        all_pos_params = posonly_names + arg_names
        for name, val in zip(all_pos_params, args):
            local_scope[name] = val

        extra_pos = args[len(all_pos_params) :]
        if extra_pos:
            if vararg_name:
                local_scope[vararg_name] = tuple(extra_pos)
            else:
                raise TypeError(
                    f"{fn.name}() takes {len(all_pos_params)} positional arguments but more were given"
                )
        elif vararg_name:
            local_scope[vararg_name] = tuple()

        for k, v in kwargs.items():
            if k in posonly_names:
                raise TypeError(
                    f"{fn.name}() got positional-only arguments passed as keyword: {k}"
                )
            if k in all_pos_params or k in kwonly_names:
                local_scope[k] = v
            elif kwarg_name:
                local_scope.setdefault(kwarg_name, {})[k] = v
            else:
                raise TypeError(f"{fn.name}() got an unexpected keyword argument '{k}'")

        if kwarg_name and kwarg_name not in local_scope:
            local_scope[kwarg_name] = {}

        # Apply defaults only if not provided as arg/kwarg (check local scope only)
        if defaults:
            trailing = all_pos_params[-len(defaults) :]
            for name, val in zip(trailing, defaults):
                if name not in local_scope:
                    local_scope[name] = val

        for name, dval in zip(kwonly_names, kw_defaults):
            if name not in local_scope and dval is not None:
                local_scope[name] = dval

        # Set up __self__ for super() support
        if first_param_name and first_param_name in local_scope:
            local_scope["__self__"] = local_scope[first_param_name]

        return local_state

    if is_gen:

        def _call(*args, **kwargs):
            local_state = setup_args(*args, **kwargs)
            try:

                def _gen():
                    try:
                        for stmt in fn.body:
                            stmt_gen = eval_stmt_generator(stmt, local_state)
                            result = None
                            try:
                                result = yield from stmt_gen
                            except StopIteration:
                                pass
                            if isinstance(result, ReturnException):
                                return result.value
                    except ReturnException as r:
                        return r.value
                    finally:
                        local_state.pop_scope()

                gen = _gen()
                return gen
            except ReturnException as r:
                local_state.pop_scope()
                return r.value

    else:

        def _call(*args, **kwargs):
            local_state = setup_args(*args, **kwargs)
            try:
                for stmt in fn.body:
                    eval_stmt(stmt, local_state)
                return None
            except ReturnException as r:
                return r.value
            finally:
                local_state.pop_scope()

    _call.__name__ = fn.name
    _call.__qualname__ = fn_qualname
    _call.__module__ = state.module_name
    _call.__doc__ = docstring
    _call.__code__ = _call.__code__.replace(
        co_filename=state.module_filename,
        co_firstlineno=fn_lineno,
    )

    # Register source with linecache for inspect.getsource()
    if state.module_filename not in linecache.cache:
        function_source = ast.unparse(fn)
        if fn_lineno > 1:
            padding = "\n" * (fn_lineno - 1)
            module_source = padding + function_source
        else:
            module_source = function_source
        lines = module_source.splitlines(keepends=True)
        if not lines:
            lines = [module_source] if module_source else [""]
        linecache.cache[state.module_filename] = (
            len(module_source),
            None,
            lines,
            state.module_filename,
        )

    return _call


def eval_classdef(node: ast.ClassDef, state: EvaluatorState) -> type:
    bases = tuple(eval_expr(b, state) for b in node.bases)
    keywords = {
        kw.arg: eval_expr(kw.value, state) for kw in node.keywords if kw.arg is not None
    }
    metaclass = keywords.pop("metaclass", type)

    if hasattr(metaclass, "__prepare__"):
        ns = metaclass.__prepare__(node.name, bases, **keywords)
    else:
        ns = {}

    ns["__module__"] = state.module_name
    ns["__qualname__"] = state.make_qualname(node.name)
    ns["__firstlineno__"] = getattr(node, "lineno", 1)

    if node.body and isinstance(node.body[0], ast.Expr):
        c = getattr(node.body[0], "value", None)
        if isinstance(c, ast.Constant) and isinstance(c.value, str):
            ns["__doc__"] = c.value

    # Create cell for __class__ before executing body
    class_cell = _ClassCell()
    old_cell = state.class_cell
    state.class_cell = class_cell

    state.push_scope()
    state.push_qual(node.name, add_locals_marker=False)
    try:
        local_ns = state.bindings.maps[0]
        local_ns.clear()
        local_ns.update(ns)

        annotations: dict[str, Any] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.annotation is not None:
                    annotations[stmt.target.id] = eval_expr(stmt.annotation, state)
            eval_stmt(stmt, state)

        if annotations:
            local_ns["__annotations__"] = annotations

        cls = metaclass(node.name, bases, dict(local_ns), **keywords)

        # Now fill the cell
        class_cell.cell_contents = cls
    finally:
        state.class_cell = old_cell
        state.pop_qual(had_locals_marker=False)
        state.pop_scope()

    for key, value in vars(cls).items():
        if hasattr(value, "__set_name__"):
            try:
                value.__set_name__(cls, key)
            except Exception:
                pass

    return cls


# -------------------------
# statements
# -------------------------


def eval_match_pattern(pattern: ast.pattern, value: Any, state: EvaluatorState) -> bool:
    """Evaluate a match pattern against a value."""
    if isinstance(pattern, ast.MatchValue):
        pattern_val = eval_expr(pattern.value, state)
        return value == pattern_val
    elif isinstance(pattern, ast.MatchSingleton):
        return value is pattern.value
    elif isinstance(pattern, ast.MatchAs):
        if pattern.pattern is None:
            # case _: or case name: (catch-all pattern)
            if pattern.name:
                assign_target(ast.Name(id=pattern.name, ctx=ast.Store()), value, state)
            return True
        # case name as pat: (pattern with binding)
        if pattern.name:
            assign_target(ast.Name(id=pattern.name, ctx=ast.Store()), value, state)
        return eval_match_pattern(pattern.pattern, value, state)
    elif isinstance(pattern, ast.MatchOr):
        return any(eval_match_pattern(p, value, state) for p in pattern.patterns)
    elif isinstance(pattern, ast.MatchClass):
        # Match against a class with attributes
        # Resolve the class name
        cls = eval_expr(pattern.cls, state)
        if not isinstance(value, cls):
            return False

        # Match positional patterns using __match_args__ if available
        if pattern.patterns:
            if hasattr(cls, "__match_args__"):
                match_args = cls.__match_args__
                if len(pattern.patterns) != len(match_args):
                    return False
                for pat, attr_name in zip(pattern.patterns, match_args):
                    if not hasattr(value, attr_name):
                        return False
                    attr_value = getattr(value, attr_name)
                    if not eval_match_pattern(pat, attr_value, state):
                        return False
            else:
                # No __match_args__, can't match positional patterns
                return False

        # Match keyword patterns
        if pattern.kwd_attrs:
            for attr_name, pat in zip(pattern.kwd_attrs, pattern.kwd_patterns):
                if not hasattr(value, attr_name):
                    return False
                attr_value = getattr(value, attr_name)
                if not eval_match_pattern(pat, attr_value, state):
                    return False

        return True
    elif isinstance(pattern, ast.MatchSequence):
        # Match against sequences (tuples, lists)
        if not isinstance(value, (tuple, list)):
            return False
        patterns = pattern.patterns
        value_list = list(value)

        # Handle MatchStar patterns (e.g., [a, *rest, b])
        star_indices = [
            i for i, p in enumerate(patterns) if isinstance(p, ast.MatchStar)
        ]

        if not star_indices:
            # No star pattern - exact match required
            if len(patterns) != len(value_list):
                return False
            for pat, item in zip(patterns, value_list):
                if not eval_match_pattern(pat, item, state):
                    return False
            return True
        else:
            # Has star pattern(s) - more complex matching
            if len(star_indices) > 1:
                # Multiple stars not supported in simple implementation
                raise InterpreterError("Multiple MatchStar patterns not supported")

            star_idx = star_indices[0]
            # Patterns before star
            before_patterns = patterns[:star_idx]
            # Patterns after star
            after_patterns = patterns[star_idx + 1 :]

            # Minimum length required
            min_len = len(before_patterns) + len(after_patterns)
            if len(value_list) < min_len:
                return False

            # Match patterns before star
            for i, pat in enumerate(before_patterns):
                if not eval_match_pattern(pat, value_list[i], state):
                    return False

            # Match patterns after star (from the end)
            for i, pat in enumerate(reversed(after_patterns)):
                if not eval_match_pattern(pat, value_list[-(i + 1)], state):
                    return False

            # Extract the star pattern's value
            star_pattern = patterns[star_idx]
            start_idx = len(before_patterns)
            end_idx = len(value_list) - len(after_patterns)
            star_value = value_list[start_idx:end_idx]

            if isinstance(star_pattern, ast.MatchStar) and star_pattern.name:
                assign_target(
                    ast.Name(id=star_pattern.name, ctx=ast.Store()), star_value, state
                )

            return True
    elif isinstance(pattern, ast.MatchMapping):
        # Match against mappings (dicts)
        if not isinstance(value, dict):
            return False
        # Match required keys
        for key_expr, pat in zip(pattern.keys, pattern.patterns):
            key = eval_expr(key_expr, state)
            if key not in value:
                return False
            if not eval_match_pattern(pat, value[key], state):
                return False
        # Check rest pattern if present
        if pattern.rest:
            # Extract remaining keys
            matched_keys = {eval_expr(k, state) for k in pattern.keys}
            remaining = {k: v for k, v in value.items() if k not in matched_keys}
            assign_target(ast.Name(id=pattern.rest, ctx=ast.Store()), remaining, state)
        return True
    else:
        raise InterpreterError(f"Unsupported match pattern: {type(pattern).__name__}")


def eval_match(stmt: ast.Match, state: EvaluatorState) -> Any:
    """Evaluate a match statement.

    In Python, match cases do NOT create a new scope. All variables assigned in a case
    body (both pattern bindings and regular assignments) are in the outer scope.
    """
    subject_val = eval_expr(stmt.subject, state)
    for case in stmt.cases:
        # Handle None pattern (case _:)
        matched = (
            eval_match_pattern(case.pattern, subject_val, state)
            if case.pattern
            else True
        )

        if matched:
            # Check guard condition if present
            if case.guard:
                if not eval_expr(case.guard, state):
                    continue

            # Execute case body in the outer scope (no new scope for match cases)
            out = None
            for s in case.body:
                out = eval_stmt(s, state)
            return out
    return None


def eval_aug_op(cur: Any, op: ast.operator, rhs: Any) -> Any:
    match op:
        case ast.Add():
            return cur + rhs
        case ast.Sub():
            return cur - rhs
        case ast.Mult():
            return cur * rhs
        case ast.MatMult():
            return cur @ rhs
        case ast.Div():
            return cur / rhs
        case ast.FloorDiv():
            return cur // rhs
        case ast.Mod():
            return cur % rhs
        case ast.Pow():
            return cur**rhs
        case ast.LShift():
            return cur << rhs
        case ast.RShift():
            return cur >> rhs
        case ast.BitAnd():
            return cur & rhs
        case ast.BitOr():
            return cur | rhs
        case ast.BitXor():
            return cur ^ rhs
        case _:
            raise InterpreterError(f"Unsupported augassign op: {type(op).__name__}")


def eval_stmt_generator(
    node: ast.stmt, state: EvaluatorState
) -> Generator[Any, None, Any | ReturnException]:
    """
    Evaluate a statement in a generator context.
    Yields values when expressions contain yields, returns ReturnException for returns.
    """
    match node:
        case ast.Expr(value=v):
            yield from eval_expr_generator(v, state)
            return None

        case ast.Assign(targets=targets, value=v):
            val = yield from eval_expr_generator(v, state)
            if len(targets) == 1:
                t = targets[0]
                if isinstance(t, (ast.Tuple, ast.List)) and any(
                    isinstance(e, ast.Starred) for e in t.elts
                ):
                    assign_extended_unpack(t, val, state)
                else:
                    assign_target(t, val, state)
                return val
            for t in targets:
                assign_target(t, val, state)
            return val

        case ast.AnnAssign(target=target, value=v, simple=_):
            if v is None:
                return None
            val = yield from eval_expr_generator(v, state)
            assign_target(target, val, state)
            return val

        case ast.AugAssign(target=t, op=op, value=v):
            cur = yield from eval_expr_generator(t, state)
            rhs = yield from eval_expr_generator(v, state)
            tmp = eval_aug_op(cur, op, rhs)
            assign_target(t, tmp, state)
            return tmp

        case ast.Return(value=v):
            if v is not None:
                val = yield from eval_expr_generator(v, state)
            else:
                val = None
            return ReturnException(val)

        case ast.Import():
            return eval_import(node, state)

        case ast.ImportFrom():
            eval_import_from(node, state)
            return None

        case ast.If(test=t, body=body, orelse=orelse):
            test_val = yield from eval_expr_generator(t, state)
            branch = body if test_val else orelse
            out = None
            for s in branch:
                result = yield from eval_stmt_generator(s, state)
                if isinstance(result, ReturnException):
                    return result
                out = result
            return out

        case ast.While(test=t, body=body, orelse=orelse):
            it = 0
            out = None
            while True:
                test_val = yield from eval_expr_generator(t, state)
                if not test_val:
                    break
                it += 1
                if it > MAX_WHILE_ITERATIONS:
                    raise InterpreterError("While loop iteration limit exceeded")
                try:
                    for s in body:
                        result = yield from eval_stmt_generator(s, state)
                        if isinstance(result, ReturnException):
                            return result
                        out = result
                except BreakException:
                    break
                except ContinueException:
                    continue
            else:
                for s in orelse:
                    result = yield from eval_stmt_generator(s, state)
                    if isinstance(result, ReturnException):
                        return result
                    out = result
            return out

        case ast.For(target=target, iter=it, body=body, orelse=orelse):
            iterable = yield from eval_expr_generator(it, state)
            out = None
            broke = False
            for item in iterable:
                try:
                    assign_target(target, item, state)
                    for s in body:
                        result = yield from eval_stmt_generator(s, state)
                        if isinstance(result, ReturnException):
                            return result
                        out = result
                except BreakException:
                    broke = True
                    break
                except ContinueException:
                    continue
            if not broke:
                for s in orelse:
                    result = yield from eval_stmt_generator(s, state)
                    if isinstance(result, ReturnException):
                        return result
                    out = result
            return out

        case ast.Break():
            raise BreakException()

        case ast.Continue():
            raise ContinueException()

        case ast.Pass():
            return None

        case ast.FunctionDef():
            fn_obj = make_function(node, state)
            fn_obj = apply_decorators(fn_obj, list(node.decorator_list), state)
            assign_target(ast.Name(id=node.name, ctx=ast.Store()), fn_obj, state)
            return fn_obj

        case ast.ClassDef():
            cls_obj = eval_classdef(node, state)
            cls_obj = apply_decorators(cls_obj, list(node.decorator_list), state)
            assign_target(ast.Name(id=node.name, ctx=ast.Store()), cls_obj, state)
            return cls_obj

        case ast.Global(names=names):
            d = state.current_directives()
            if d is None:
                raise InterpreterError("global statement outside of function scope")
            d.globals.update(names)
            return None

        case ast.Nonlocal(names=names):
            d = state.current_directives()
            if d is None:
                raise InterpreterError("nonlocal statement outside of function scope")
            d.nonlocals.update(names)
            return None

        case ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody):
            out = None
            try:
                for s in body:
                    result = yield from eval_stmt_generator(s, state)
                    if isinstance(result, ReturnException):
                        return result
                    out = result
            except BaseException as e:
                handled = False
                for h in handlers:
                    if h.type is None:
                        matched = True
                    else:
                        exc_type = yield from eval_expr_generator(h.type, state)
                        matched = isinstance(e, exc_type)
                    if matched:
                        handled = True
                        # Push exception onto stack for bare raise support
                        state.exception_stack.append(e)
                        try:
                            state.push_scope()
                            try:
                                if h.name:
                                    assign_target(
                                        ast.Name(id=h.name, ctx=ast.Store()), e, state
                                    )
                                for s in h.body:
                                    result = yield from eval_stmt_generator(s, state)
                                    if isinstance(result, ReturnException):
                                        return result
                                    out = result
                            finally:
                                state.pop_scope()
                        finally:
                            # Pop exception from stack when exiting handler
                            if state.exception_stack and state.exception_stack[-1] is e:
                                state.exception_stack.pop()
                        break
                if not handled:
                    raise
            else:
                for s in orelse:
                    result = yield from eval_stmt_generator(s, state)
                    if isinstance(result, ReturnException):
                        return result
                    out = result
            finally:
                for s in finalbody:
                    yield from eval_stmt_generator(s, state)
            return out

        case ast.Raise(exc=exc, cause=cause):
            if exc is None:
                # Bare raise - re-raise the current exception from the exception stack
                if not state.exception_stack:
                    raise InterpreterError(
                        "Re-raise without active exception is not supported"
                    )
                ex = state.exception_stack[-1]
                raise ex
            ex = yield from eval_expr_generator(exc, state)
            if cause is not None:
                ca = yield from eval_expr_generator(cause, state)
                raise ex from ca
            raise ex

        case ast.Assert(test=t, msg=m):
            test_val = yield from eval_expr_generator(t, state)
            if not test_val:
                if m is not None:
                    msg_val = yield from eval_expr_generator(m, state)
                    raise AssertionError(msg_val)
                else:
                    raise AssertionError("Assertion failed")
            return None

        case ast.With(items=items, body=body, type_comment=_):
            entered = []
            caught_exc: BaseException | None = None
            out = None
            try:
                for item in items:
                    ctx = yield from eval_expr_generator(item.context_expr, state)
                    val = ctx.__enter__()
                    entered.append(ctx)
                    if item.optional_vars is not None:
                        assign_target(item.optional_vars, val, state)
                for s in body:
                    result = yield from eval_stmt_generator(s, state)
                    if isinstance(result, ReturnException):
                        # Handle return in finally
                        for ctx_inner in reversed(entered):
                            ctx_inner.__exit__(None, None, None)
                        return result
                    out = result
            except BaseException as e:
                caught_exc = e
            finally:
                # Call __exit__ on all context managers in reverse order
                suppressed = False
                for ctx in reversed(entered):
                    if caught_exc is not None:
                        if ctx.__exit__(
                            type(caught_exc), caught_exc, caught_exc.__traceback__
                        ):
                            suppressed = True
                    else:
                        ctx.__exit__(None, None, None)
                # Re-raise if not suppressed
                if caught_exc is not None and not suppressed:
                    raise caught_exc
            return out

        case ast.Delete(targets=targets):
            for t in targets:
                delete_target(t, state)
            return None

        case ast.Match(subject=subject, cases=cases):
            subject_val = yield from eval_expr_generator(subject, state)
            for case in cases:
                if case.pattern is None:
                    matched = True
                else:
                    matched = yield from eval_match_pattern_generator(
                        case.pattern, subject_val, state
                    )

                if matched:
                    if case.guard:
                        guard_val = yield from eval_expr_generator(case.guard, state)
                        if not guard_val:
                            continue

                    out = None
                    for s in case.body:
                        result = yield from eval_stmt_generator(s, state)
                        if isinstance(result, ReturnException):
                            return result
                        out = result
                    return out
            return None

        case _:
            raise InterpreterError(f"Unsupported statement: {type(node).__name__}")


def eval_match_pattern_generator(
    pattern: ast.pattern, value: Any, state: EvaluatorState
) -> Generator[Any, None, bool]:
    """Evaluate a match pattern in generator context."""
    if isinstance(pattern, ast.MatchValue):
        pattern_val = yield from eval_expr_generator(pattern.value, state)
        return value == pattern_val
    elif isinstance(pattern, ast.MatchSingleton):
        return value is pattern.value
    elif isinstance(pattern, ast.MatchAs):
        if pattern.pattern is None:
            if pattern.name:
                assign_target(ast.Name(id=pattern.name, ctx=ast.Store()), value, state)
            return True
        if pattern.name:
            assign_target(ast.Name(id=pattern.name, ctx=ast.Store()), value, state)
        return (yield from eval_match_pattern_generator(pattern.pattern, value, state))
    elif isinstance(pattern, ast.MatchOr):
        for p in pattern.patterns:
            matched = yield from eval_match_pattern_generator(p, value, state)
            if matched:
                return True
        return False
    elif isinstance(pattern, ast.MatchClass):
        # Match against a class with attributes
        # Resolve the class name
        cls = eval_expr(pattern.cls, state)
        if not isinstance(value, cls):
            return False
        # Match positional patterns using __match_args__ if available
        if pattern.patterns:
            if hasattr(cls, "__match_args__"):
                match_args = cls.__match_args__
                if len(pattern.patterns) != len(match_args):
                    return False
                for pat, attr_name in zip(pattern.patterns, match_args):
                    if not hasattr(value, attr_name):
                        return False
                    attr_value = getattr(value, attr_name)
                    matched = yield from eval_match_pattern_generator(
                        pat, attr_value, state
                    )
                    if not matched:
                        return False
            else:
                # No __match_args__, can't match positional patterns
                return False

        # Match keyword patterns
        if pattern.kwd_attrs:
            for attr_name, pat in zip(pattern.kwd_attrs, pattern.kwd_patterns):
                if not hasattr(value, attr_name):
                    return False
                attr_value = getattr(value, attr_name)
                if not eval_match_pattern(pat, attr_value, state):
                    return False

        return True
    elif isinstance(pattern, ast.MatchSequence):
        if not isinstance(value, (tuple, list)):
            return False
        patterns = pattern.patterns
        value_list = list(value)

        # Handle MatchStar patterns (e.g., [a, *rest, b])
        star_indices = [
            i for i, p in enumerate(patterns) if isinstance(p, ast.MatchStar)
        ]

        if not star_indices:
            # No star pattern - exact match required
            if len(patterns) != len(value_list):
                return False
            for pat, item in zip(patterns, value_list):
                matched = yield from eval_match_pattern_generator(pat, item, state)
                if not matched:
                    return False
            return True
        else:
            # Has star pattern(s) - more complex matching
            if len(star_indices) > 1:
                # Multiple stars not supported in simple implementation
                raise InterpreterError("Multiple MatchStar patterns not supported")

            star_idx = star_indices[0]
            # Patterns before star
            before_patterns = patterns[:star_idx]
            # Patterns after star
            after_patterns = patterns[star_idx + 1 :]

            # Minimum length required
            min_len = len(before_patterns) + len(after_patterns)
            if len(value_list) < min_len:
                return False

            # Match patterns before star
            for i, pat in enumerate(before_patterns):
                matched = yield from eval_match_pattern_generator(
                    pat, value_list[i], state
                )
                if not matched:
                    return False

            # Match patterns after star (from the end)
            for i, pat in enumerate(reversed(after_patterns)):
                matched = yield from eval_match_pattern_generator(
                    pat, value_list[-(i + 1)], state
                )
                if not matched:
                    return False

            # Extract the star pattern's value
            star_pattern = patterns[star_idx]
            start_idx = len(before_patterns)
            end_idx = len(value_list) - len(after_patterns)
            star_value = value_list[start_idx:end_idx]

            if isinstance(star_pattern, ast.MatchStar) and star_pattern.name:
                assign_target(
                    ast.Name(id=star_pattern.name, ctx=ast.Store()), star_value, state
                )

            return True
    elif isinstance(pattern, ast.MatchMapping):
        if not isinstance(value, dict):
            return False
        for key_expr, pat in zip(pattern.keys, pattern.patterns):
            key = yield from eval_expr_generator(key_expr, state)
            if key not in value:
                return False
            matched = yield from eval_match_pattern_generator(pat, value[key], state)
            if not matched:
                return False
        if pattern.rest:
            matched_keys = set()
            for k in pattern.keys:
                key = yield from eval_expr_generator(k, state)
                matched_keys.add(key)
            remaining = {k: v for k, v in value.items() if k not in matched_keys}
            assign_target(ast.Name(id=pattern.rest, ctx=ast.Store()), remaining, state)
        return True
    else:
        raise InterpreterError(f"Unsupported match pattern: {type(pattern).__name__}")


def eval_stmt(node: ast.stmt, state: EvaluatorState) -> Any:
    match node:
        case ast.Import():
            return eval_import(node, state)

        case ast.ImportFrom():
            eval_import_from(node, state)
            return None

        case ast.Expr(value=v):
            return eval_expr(v, state)

        case ast.Assign(targets=targets, value=v):
            val = eval_expr(v, state)
            if len(targets) == 1:
                t = targets[0]
                if isinstance(t, (ast.Tuple, ast.List)) and any(
                    isinstance(e, ast.Starred) for e in t.elts
                ):
                    return assign_extended_unpack(t, val, state)
                assign_target(t, val, state)
                return val
            for t in targets:
                assign_target(t, val, state)
            return val

        case ast.AnnAssign(target=target, value=v, simple=_):
            if v is None:
                return None
            val = eval_expr(v, state)
            assign_target(target, val, state)
            return val

        case ast.AugAssign(target=t, op=op, value=v):
            cur = eval_expr(t, state)
            rhs = eval_expr(v, state)
            tmp = eval_aug_op(cur, op, rhs)
            assign_target(t, tmp, state)
            return tmp

        case ast.If(test=t, body=body, orelse=orelse):
            branch = body if eval_expr(t, state) else orelse
            out = None
            for s in branch:
                out = eval_stmt(s, state)
            return out

        case ast.While(test=t, body=body, orelse=orelse):
            it = 0
            out = None
            while eval_expr(t, state):
                it += 1
                if it > MAX_WHILE_ITERATIONS:
                    raise InterpreterError("While loop iteration limit exceeded")
                try:
                    for s in body:
                        out = eval_stmt(s, state)
                except BreakException:
                    break
                except ContinueException:
                    continue
            else:
                for s in orelse:
                    out = eval_stmt(s, state)
            return out

        case ast.For(target=target, iter=it, body=body, orelse=orelse):
            iterable = eval_expr(it, state)
            out = None
            broke = False
            for item in iterable:
                try:
                    assign_target(target, item, state)
                    for s in body:
                        out = eval_stmt(s, state)
                except BreakException:
                    broke = True
                    break
                except ContinueException:
                    continue
            if not broke:
                for s in orelse:
                    out = eval_stmt(s, state)
            return out

        case ast.Break():
            raise BreakException()

        case ast.Continue():
            raise ContinueException()

        case ast.Return(value=v):
            raise ReturnException(eval_expr(v, state) if v is not None else None)

        case ast.Pass():
            return None

        case ast.FunctionDef():
            fn_obj = make_function(node, state)
            fn_obj = apply_decorators(fn_obj, list(node.decorator_list), state)
            assign_target(ast.Name(id=node.name, ctx=ast.Store()), fn_obj, state)
            return fn_obj

        case ast.ClassDef():
            cls_obj = eval_classdef(node, state)
            cls_obj = apply_decorators(cls_obj, list(node.decorator_list), state)
            assign_target(ast.Name(id=node.name, ctx=ast.Store()), cls_obj, state)
            return cls_obj

        case ast.Global(names=names):
            d = state.current_directives()
            if d is None:
                raise InterpreterError("global statement outside of function scope")
            d.globals.update(names)
            return None

        case ast.Nonlocal(names=names):
            d = state.current_directives()
            if d is None:
                raise InterpreterError("nonlocal statement outside of function scope")
            d.nonlocals.update(names)
            return None

        case ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody):
            out = None
            try:
                for s in body:
                    out = eval_stmt(s, state)
            except BaseException as e:
                handled = False
                for h in handlers:
                    if h.type is None:
                        matched = True
                    else:
                        exc_type = eval_expr(h.type, state)
                        matched = isinstance(e, exc_type)
                    if matched:
                        handled = True
                        # Push exception onto stack for bare raise support
                        state.exception_stack.append(e)
                        try:
                            # Exception handlers don't create a new scope - they use the same scope as the try block
                            if h.name:
                                # Bind exception variable in the try block's scope
                                assign_target(
                                    ast.Name(id=h.name, ctx=ast.Store()), e, state
                                )
                            # Execute handler body in the same scope as try block (no new scope)
                            for s in h.body:
                                out = eval_stmt(s, state)
                        finally:
                            # Pop exception from stack when exiting handler
                            if state.exception_stack and state.exception_stack[-1] is e:
                                state.exception_stack.pop()
                        break
                if not handled:
                    raise
            else:
                for s in orelse:
                    out = eval_stmt(s, state)
            finally:
                for s in finalbody:
                    eval_stmt(s, state)
            return out

        case ast.Raise(exc=exc, cause=cause):
            if exc is None:
                # Bare raise - re-raise the current exception from the exception stack
                if not state.exception_stack:
                    raise InterpreterError(
                        "Re-raise without active exception is not supported"
                    )
                ex = state.exception_stack[-1]
                raise ex
            ex = eval_expr(exc, state)
            if cause is not None:
                ca = eval_expr(cause, state)
                raise ex from ca
            raise ex

        case ast.Assert(test=t, msg=m):
            if not eval_expr(t, state):
                raise AssertionError(
                    eval_expr(m, state) if m is not None else "Assertion failed"
                )
            return None

        case ast.With(items=items, body=body, type_comment=_):
            entered = []
            caught_exc: BaseException | None = None
            try:
                for item in items:
                    ctx = eval_expr(item.context_expr, state)
                    val = ctx.__enter__()
                    entered.append(ctx)
                    if item.optional_vars is not None:
                        assign_target(item.optional_vars, val, state)
                out = None
                for s in body:
                    out = eval_stmt(s, state)
            except BaseException as e:
                caught_exc = e
            finally:
                # Call __exit__ on all context managers in reverse order
                suppressed = False
                for ctx in reversed(entered):
                    if caught_exc is not None:
                        if ctx.__exit__(
                            type(caught_exc), caught_exc, caught_exc.__traceback__
                        ):
                            suppressed = True
                    else:
                        ctx.__exit__(None, None, None)
                # Re-raise if not suppressed
                if caught_exc is not None and not suppressed:
                    raise caught_exc
            return out

        case ast.Delete(targets=targets):
            for t in targets:
                delete_target(t, state)
            return None

        case ast.Match():
            return eval_match(node, state)

        case _:
            raise InterpreterError(f"Unsupported statement: {type(node).__name__}")


def eval_module(module: ast.Module, state: EvaluatorState):
    source_text = ast.unparse(module)
    # Re-parse to get correct line numbers matching the unparsed source
    module = ast.parse(source_text)
    state.module_name, state.module_filename = install_synthetic_module(source_text)

    for stmt in module.body:
        eval_stmt(stmt, state)
