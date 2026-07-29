import ast
import builtins
import collections.abc
import contextlib
import copy
import doctest
import linecache
import operator
import string
import sys
import types
import typing
import warnings

from RestrictedPython import (
    Eval,
    Guards,
    RestrictingNodeTransformer,
    compile_restricted,
    safe_globals,
)
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.transformer import (
    FORBIDDEN_FUNC_NAMES,
    INSPECT_ATTRIBUTES,
    copy_locations,
)

from effectful.handlers.llm.harness.executing import (
    _mypy_check_region,
    _run_doctests,
    compile,
    exec,
    parse,
    run_doctests,
    type_check,
)
from effectful.ops.syntax import ObjectInterpretation, implements

# ----------------------------------------------------------------------------
# The RestrictedPython policy: what generated code may name, touch and import
#
# RestrictedPython's job is to keep generated code away from the interpreter's
# internals. The names below draw that line once, and both halves of the sandbox
# read from it: the compile-time policy (`RestrictedPythonPolicy`) and the runtime
# guards installed in the exec environment (`_guarded_getattr`, `_guarded_import`).
# ----------------------------------------------------------------------------


@contextlib.contextmanager
def _doctest_compiled_with(
    compiler: collections.abc.Callable[..., types.CodeType],
):
    """Run the examples inside this block through ``compiler`` instead of the
    built-in `compile`.

    `doctest.DocTestRunner` compiles and runs every example with a bare
    ``exec(compile(source, filename, "single", flags, True), test.globs)``, with no
    hook to supply a different compiler. Both names resolve as globals of the
    `doctest` module before falling back to builtins, so binding ``compile`` there
    for the duration of the run redirects example compilation without
    reimplementing (and having to track) the runner. ``compiler`` is called with
    `compile`'s positional signature, which `RestrictedPython.compile_restricted`
    already matches.

    This rebinds module state, so it is not safe against another thread running
    doctests concurrently under a *different* compiler; the block is short and
    holds only for the examples of one synthesized object.
    """
    sentinel = object()
    original = doctest.__dict__.get("compile", sentinel)
    doctest.compile = compiler  # type: ignore[attr-defined]
    try:
        yield
    finally:
        # Restore `doctest` exactly as we found it -- normally by *removing* the
        # global again, so `compile` resolves to the builtin as it did before.
        if original is sentinel:
            del doctest.compile  # type: ignore[attr-defined]
        else:
            doctest.compile = original  # type: ignore[attr-defined]


class _StdoutPrintCollector(PrintCollector):
    """`_print_` factory whose `print(...)` writes to the real `sys.stdout`
    (so output-capturing callers see it) rather than accumulating into the
    collector's discarded `printed` buffer."""

    def _call_print(self, *objects, **kwargs):
        kwargs.setdefault("file", sys.stdout)
        builtins.print(*objects, **kwargs)


# Names the RestrictedPython transformer itself emits into compiled code -- the
# guarded accessors (`_getattr_(x, "y")`), the print collector (`_print`), the
# class metaclass. Generated code must not bind or read them, or it could hand
# itself an unguarded accessor (or quietly disable one).
_GUARD_NAMES = frozenset(
    {
        "_apply_",
        "_getattr_",
        "_getitem_",
        "_getiter_",
        "_inplacevar_",
        "_iter_unpack_sequence_",
        "_print",
        "_print_",
        "_unpack_sequence_",
        "_write_",
        "__builtins__",
        "__metaclass__",
    }
)

# Dunder attributes generated code may read, call and define: the operator/context
# protocols plus a few purely descriptive names. Every *other* ``__dunder__`` --
# `__class__`, `__bases__`, `__subclasses__`, `__mro__`, `__globals__`, `__code__`,
# `__dict__`, `__getattribute__`, `__reduce__`, ... -- is the road from sandboxed
# code back to the interpreter, and stays closed.
_SAFE_DUNDER_ATTRS = frozenset(
    {
        "__abs__",
        "__add__",
        "__bool__",
        "__call__",
        "__contains__",
        "__doc__",
        "__enter__",
        "__eq__",
        "__exit__",
        "__ge__",
        "__getitem__",
        "__gt__",
        "__hash__",
        "__init__",
        "__iter__",
        "__le__",
        "__len__",
        "__lt__",
        "__mul__",
        "__name__",
        "__ne__",
        "__neg__",
        "__next__",
        "__post_init__",
        "__radd__",
        "__repr__",
        "__reversed__",
        "__rmul__",
        "__setitem__",
        "__str__",
        "__sub__",
    }
)

# Modules generated code may import. Pure computation and data structures only:
# nothing that reaches the filesystem, the network, the process, or the import
# system itself (`os`, `sys`, `subprocess`, `socket`, `importlib`, `builtins`,
# `inspect`, `types`, `pickle`, `ctypes`, ...). Submodules must be listed in full,
# since the check is on the imported name.
#
# Two near-misses worth recording, so they don't get added later by analogy:
# `io` is not here because `io.open` *is* `builtins.open`, which would hand back
# the filesystem that omitting `open` closes; and `numpy`/`matplotlib` are not here
# because `numpy.load(..., allow_pickle=True)` executes arbitrary code. Synthesized
# code that genuinely needs those belongs under `UnsafeEvalProvider`, not behind a
# widened allowlist.
_ALLOWED_MODULES = frozenset(
    {
        "abc",
        "array",
        "base64",
        "binascii",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "collections.abc",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "enum",
        "fractions",
        "functools",
        "graphlib",
        "hashlib",
        "heapq",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "queue",
        "random",
        "re",
        "statistics",
        "string",
        "struct",
        "textwrap",
        "typing",
        "unicodedata",
        "uuid",
    }
)

# `n += 1` compiles to `n = _inplacevar_("+=", n, 1)`, so the environment has to
# supply the operators; without this every augmented assignment is a NameError.
_INPLACE_OPS: dict[
    str, collections.abc.Callable[[typing.Any, typing.Any], typing.Any]
] = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "//=": operator.ifloordiv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
    "&=": operator.iand,
    "^=": operator.ixor,
    "|=": operator.ior,
    "@=": operator.imatmul,
}

# Builtins beyond RestrictedPython's deliberately minimal `safe_builtins`. Each is
# pure and reaches nothing outside its arguments; the omissions are the point --
# `open`, `input`, `compile`, `eval`, `exec`, `globals`, `locals`, `vars`, `dir`
# and `breakpoint` are all absent, and `__import__`/`getattr`/`setattr`/`delattr`
# are installed as guarded wrappers rather than taken from `builtins`.
_EXTRA_SAFE_BUILTIN_NAMES = frozenset(
    {
        "all",
        "any",
        "ascii",
        "bin",
        "bytearray",
        "classmethod",
        "dict",
        "enumerate",
        "filter",
        "format",
        "frozenset",
        "iter",
        "list",
        "map",
        "max",
        "memoryview",
        "min",
        "next",
        "object",
        "property",
        "reversed",
        "set",
        "staticmethod",
        "sum",
        "super",
        "type",
        # Exceptions `safe_builtins` happens to omit but recursive/brute-force
        # generated code routinely names.
        "RecursionError",
        "StopAsyncIteration",
        "TimeoutError",
        "NotImplemented",
    }
)


# Dunder *bindings* a class or module body may make. Binding these declares
# something about the code being written; it is not a way to read a dunder
# attribute off an object, which `_is_allowed_attribute` governs separately.
_SAFE_DUNDER_BINDINGS = frozenset({"__all__", "__slots__"})


def _is_allowed_name(name: str) -> bool:
    """Whether generated code may bind or read ``name`` as an identifier.

    RestrictedPython rejects *every* name starting with ``_``, which costs a great
    deal (``_helper``, ``_memo``, ``_solve`` are how Python spells "private") and
    buys little: an identifier is only dangerous when it is one of the guard names
    the transformer emits, or a dunder. Both of those stay rejected; a plain
    single-underscore private does not.
    """
    if name in _GUARD_NAMES or name in FORBIDDEN_FUNC_NAMES:
        return False
    if name == "_":  # the conventional throwaway
        return True
    if name in _SAFE_DUNDER_BINDINGS:
        return True
    if not name.startswith("_"):
        return True
    # `_private` is fine; `__dunder__`-shaped and `_guard_`-shaped names are not.
    return not name.startswith("__") and not name.endswith("_")


def _is_allowed_attribute(name: str) -> bool:
    """Whether generated code may read, write or define the attribute ``name``.

    Same trade as `_is_allowed_name`, one level in: single-underscore attributes
    (``self._items``) are ordinary Python, while dunders outside
    `_SAFE_DUNDER_ATTRS` -- and the frame/code/traceback attributes RestrictedPython
    lists in ``INSPECT_ATTRIBUTES`` -- are how sandboxed code climbs out.
    """
    if name in INSPECT_ATTRIBUTES or name.endswith("__roles__"):
        return False
    if name.startswith("__"):
        return name in _SAFE_DUNDER_ATTRS
    return True


class RestrictedPythonPolicy(RestrictingNodeTransformer):
    """RestrictedPython's policy, relaxed where it rejects ordinary modern Python.

    `RestrictingNodeTransformer` predates a good deal of the language a model
    writes today, and its rejections are not all security-carrying. Four in
    particular make it unusable as-is for synthesized code:

    * **annotated assignments** (``total: int = 0``) are rejected outright, and with
      them every ``@dataclass`` and every class-level field -- while this library
      *asks* models for annotated code and type-checks what it gets;
    * **any name starting with ``_``**, so a helper called ``_solve`` fails to
      compile;
    * **``nonlocal``**, so a closure cannot update the variable it closes over;
    * **``counts[k] += 1``**, so nothing can accumulate into a container in place.

    This subclass allows those four (plus ``type`` aliases and the dunder methods
    in `_SAFE_DUNDER_ATTRS`, so a model can define ``__len__``/``__repr__`` on its
    own classes) and changes nothing else. Everything the sandbox actually rests on
    is inherited untouched: ``exec``/``eval`` calls, star imports, ``async``,
    ``except*``, ``match`` and any other unreviewed syntax are still rejected, and
    attribute access, subscripting and iteration are still rewritten to the guarded
    accessors that `RestrictedEvalProvider` installs.
    """

    def check_name(
        self, node: typing.Any, name: str | None, allow_magic_methods: bool = False
    ) -> None:
        if name is None or _is_allowed_name(name):
            return
        # Defining a dunder *method* (never at column 0, i.e. always inside a class
        # or function body) is how you write `__len__`; reading one is governed by
        # `_is_allowed_attribute`, which agrees on the same set.
        if (
            allow_magic_methods
            and name in _SAFE_DUNDER_ATTRS
            and getattr(node, "col_offset", 0) != 0
        ):
            return
        super().check_name(node, name, allow_magic_methods)

    def visit_Attribute(self, node: ast.Attribute) -> typing.Any:
        """``a.b`` -> ``_getattr_(a, 'b')``, ``a.b = c`` -> ``_write_(a).b = c``.

        Identical to the base transform except that the name is checked against
        `_is_allowed_attribute` rather than "does it start with an underscore".
        """
        if not _is_allowed_attribute(node.attr):
            self.error(node, f'"{node.attr}" is a restricted attribute name.')
        node = self.node_contents_visit(node)
        if isinstance(node.ctx, ast.Load):
            new_node: ast.expr = ast.Call(
                func=ast.Name("_getattr_", ast.Load()),
                args=[node.value, ast.Constant(node.attr)],
                keywords=[],
            )
            copy_locations(new_node, node)
            return new_node
        # Store/Del: the base returns the attribute node itself with its *object*
        # routed through the write guard.
        new_value = ast.Call(
            func=ast.Name("_write_", ast.Load()), args=[node.value], keywords=[]
        )
        copy_locations(new_value, node.value)
        node.value = new_value
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> typing.Any:
        """Allow annotated assignment (``x: int = 1``).

        It carries no capability a plain assignment doesn't: there is a single
        target and no unpacking, and an attribute or subscript target is rewritten
        to the write guard by the visitors for those nodes, exactly as in
        ``visit_Assign``. The annotation is just another expression.
        """
        return self.node_contents_visit(node)

    def visit_TypeAlias(self, node: ast.AST) -> typing.Any:
        """Allow ``type X = ...`` aliases: a binding of a lazily-evaluated
        annotation expression, with no more reach than the expression itself."""
        return self.node_contents_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> typing.Any:
        """Allow ``nonlocal``.

        Like the ``global`` the base already allows, it only rebinds a name in an
        *enclosing scope of the same generated code* -- and `check_name` still
        governs which names those can be."""
        return self.node_contents_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> typing.Any:
        """Allow augmented assignment to an item or attribute (``counts[k] += 1``).

        The base allows it only for a plain name (rewritten to ``_inplacevar_``) and
        rejects ``a[i] += x`` / ``a.b += x`` outright, which rules out most code that
        accumulates into a container. Left alone, those compile to a subscript/attribute
        read, the in-place operation, and a store -- and the target still goes through
        the ordinary `visit_Subscript`/`visit_Attribute` rewriting, so the store lands on
        ``_write_(a)`` and an attribute name is still checked. The read is the one thing
        not routed through ``_getitem_``/``_getattr_``; that costs nothing here, where
        `RestrictedEvalProvider` reads are unrestricted, but a policy paired with a
        *restricting* ``_getitem_`` should not use this class.
        """
        if isinstance(node.target, ast.Name):
            return super().visit_AugAssign(node)
        return self.node_contents_visit(node)

    # -- `match` -------------------------------------------------------------
    #
    # RestrictedPython has no visitor for any of the match nodes, so its
    # `generic_visit` rejects the whole statement. Supporting it takes more than
    # forwarding to `node_contents_visit`, for two reasons.
    #
    # First, a pattern is not an expression, even where it holds expression nodes:
    # the value of a `case Color.RED` and the class of a `case Point(...)` must stay
    # a literal or a dotted name. Rewriting them into `_getattr_(...)` calls -- what
    # `visit_Attribute` does everywhere else -- produces a program CPython refuses to
    # compile ("patterns may only match literals and attribute lookups"). So those
    # positions are *checked* against the same name policy and left alone.
    #
    # Second, a pattern reads attributes and items of the subject without any of the
    # rewriting the guards rely on: a class pattern reads the attributes named in
    # `kwd_attrs`, and a mapping pattern subscripts. The `kwd_attrs` names are in the
    # AST, so they get the same `_is_allowed_attribute` check as `obj.attr`. The
    # subscripting is unguarded, exactly as in `visit_AugAssign` above, and costs
    # nothing while this provider's `_getitem_`/`_getiter_` are unrestricted.
    #
    # The residual: a *positional* class pattern (`case Point(x, y)`) reads whatever
    # attribute names the class's own `__match_args__` lists, which is not in the AST
    # and cannot be checked here. Generated code cannot supply those names -- binding
    # `__match_args__` is refused by `_is_allowed_name` and setting it by
    # `_guarded_setattr` -- so they can only come from a class the host program put
    # in scope, which would have to name an interpreter internal there deliberately.

    def visit_Match(self, node: ast.Match) -> typing.Any:
        """Allow ``match``; the subject is an ordinary expression."""
        return self.node_contents_visit(node)

    def visit_match_case(self, node: ast.match_case) -> typing.Any:
        """Allow a ``case``: its guard and body are ordinary code (and are rewritten
        as such); its pattern dispatches to the visitors below."""
        return self.node_contents_visit(node)

    def _check_pattern_expression(self, node: ast.AST) -> None:
        """Check the names and attributes of an expression sitting in a pattern
        position, without rewriting it.

        Every such expression is a literal or a dotted name, and is only ever
        *compared* against the subject -- never bound -- so checking it under the
        same policy as an ordinary read is enough.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if not _is_allowed_attribute(child.attr):
                    self.error(child, f'"{child.attr}" is a restricted attribute name.')
            elif isinstance(child, ast.Name):
                self.check_name(child, child.id)

    def visit_MatchValue(self, node: ast.MatchValue) -> typing.Any:
        """``case 3:`` / ``case Color.RED:`` -- checked, deliberately not rewritten."""
        self._check_pattern_expression(node.value)
        return node

    def visit_MatchSingleton(self, node: ast.MatchSingleton) -> typing.Any:
        """``case None:`` / ``case True:`` -- a bare constant, nothing to check."""
        return node

    def visit_MatchSequence(self, node: ast.MatchSequence) -> typing.Any:
        """``case [a, b]:`` -- sub-patterns are visited; see the note on unguarded
        iteration above."""
        return self.node_contents_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> typing.Any:
        """``case [first, *rest]:`` -- ``rest`` is a binding."""
        self.check_name(node, node.name)
        return node

    def visit_MatchAs(self, node: ast.MatchAs) -> typing.Any:
        """``case x:`` / ``case [1] as pair:`` / ``case _:`` -- a binding plus an
        optional sub-pattern."""
        self.check_name(node, node.name)
        return self.node_contents_visit(node)

    def visit_MatchOr(self, node: ast.MatchOr) -> typing.Any:
        """``case 1 | 2:`` -- alternatives are themselves patterns."""
        return self.node_contents_visit(node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> typing.Any:
        """``case {"k": v, **rest}:`` -- keys are checked in place (they are pattern
        expressions), sub-patterns are visited, ``rest`` is a binding."""
        for key in node.keys:
            self._check_pattern_expression(key)
        self.check_name(node, node.rest)
        node.patterns = [self.visit(pattern) for pattern in node.patterns]
        return node

    def visit_MatchClass(self, node: ast.MatchClass) -> typing.Any:
        """``case Point(x=0, y=y):`` -- the class is checked in place, and each
        keyword names an attribute read off the subject, so it gets the attribute
        policy."""
        self._check_pattern_expression(node.cls)
        for attr in node.kwd_attrs:
            if not _is_allowed_attribute(attr):
                self.error(node, f'"{attr}" is a restricted attribute name.')
        node.patterns = [self.visit(pattern) for pattern in node.patterns]
        node.kwd_patterns = [self.visit(pattern) for pattern in node.kwd_patterns]
        return node


_RAISE = object()

# `str.format`'s field names, the part before any `!conversion` or `:spec`. A field
# that *reaches* -- `{0.__class__}`, `{0[1]}` -- is the whole reason RestrictedPython
# refuses these methods, since the traversal happens inside CPython's formatter,
# where the `_getattr_`/`_getitem_` rewriting cannot see it.
_FORMAT_REACHING_CHARS = (".", "[")


def _reject_reaching_format_fields(template: str) -> None:
    """Raise unless every replacement field in ``template`` names its argument and
    stops there -- no attribute or item traversal from it.

    Nested fields inside a format spec (``"{0:{1}}"``) are checked too; the
    recursion terminates because a spec with no braces yields a single field of
    ``None``, and CPython caps spec nesting at one level regardless.
    """
    for _, field_name, format_spec, _ in string.Formatter().parse(template):
        if field_name is not None and any(
            char in field_name for char in _FORMAT_REACHING_CHARS
        ):
            raise NotImplementedError(
                f"format field {'{'}{field_name}{'}'} reaches into an attribute or "
                f"item of its argument, which is not allowed in restricted code; "
                f"index or attribute it yourself, or use an f-string"
            )
        if format_spec:
            _reject_reaching_format_fields(format_spec)


def _checked_str_format(
    target: typing.Any, name: str
) -> collections.abc.Callable[..., str]:
    """``str.format``/``str.format_map``, wrapped to check the template first.

    RestrictedPython refuses these outright because ``"{0.__class__.__mro__}"``
    walks attributes inside CPython's formatter, out of reach of the guards -- but
    refusing them wholesale is expensive and, worse, invisible: the call compiles and
    type-checks and only fails when the code runs, possibly inside a doctest. What is
    actually dangerous is the *traversal*, and that is written in the template, where
    it can be read off before anything is formatted. So check the template and allow
    the rest, which is nearly all real uses.
    """
    # Reached either bound (``"...".format(x)``, template in hand) or unbound
    # (``str.format("...", x)``, template passed as the first argument).
    unbound = isinstance(target, type)

    def formatter(*args: typing.Any, **kwargs: typing.Any) -> str:
        if unbound and not args:
            return typing.cast(str, getattr(str, name)())  # let Python raise TypeError
        _reject_reaching_format_fields(args[0] if unbound else target)
        return typing.cast(str, getattr(target, name)(*args, **kwargs))

    return formatter


def _guarded_getattr(
    obj: typing.Any, name: str, default: typing.Any = _RAISE
) -> typing.Any:
    """The runtime half of `_is_allowed_attribute`: the ``_getattr_`` every
    attribute access in restricted code compiles to, and the ``getattr`` builtin
    generated code sees (so a dynamically-computed name is checked too)."""
    if type(name) is not str:
        raise TypeError("type(name) must be str")
    if name in ("format", "format_map") and (
        isinstance(obj, str) or (isinstance(obj, type) and issubclass(obj, str))
    ):
        # The one place this guard is *more* permissive than RestrictedPython's, and
        # only because it checks what that one could not. `string.Formatter` and its
        # methods stay blocked by the delegation below: there the template is not in
        # hand at attribute-access time, so there is nothing to check.
        return _checked_str_format(obj, name)
    if not name.startswith("_"):
        # Public name: defer to RestrictedPython's own guard, which additionally
        # blocks `string.Formatter` (it can reach attributes of whatever is
        # interpolated) and the `inspect` attributes.
        if default is _RAISE:
            return Guards.safer_getattr_raise(obj, name)
        return Guards.safer_getattr(obj, name, default)
    if not _is_allowed_attribute(name):
        raise AttributeError(f'"{name}" is a restricted attribute name.')
    if default is _RAISE:
        return getattr(obj, name)
    return getattr(obj, name, default)


def _guarded_hasattr(obj: typing.Any, name: str) -> bool:
    """`hasattr` that agrees with `_guarded_getattr`: a restricted attribute reads
    as absent rather than as a way to probe for one."""
    try:
        _guarded_getattr(obj, name)
    except (AttributeError, TypeError, NotImplementedError):
        return False
    return True


def _guarded_setattr(obj: typing.Any, name: str, value: typing.Any) -> None:
    """`setattr` under `_is_allowed_attribute`, matching what ``obj.name = value``
    compiles to (RestrictedPython checks *that* name at compile time; this is the
    same rule for a name computed at run time)."""
    if type(name) is not str:
        raise TypeError("type(name) must be str")
    if not _is_allowed_attribute(name):
        raise AttributeError(f'"{name}" is a restricted attribute name.')
    setattr(obj, name, value)


def _guarded_delattr(obj: typing.Any, name: str) -> None:
    """`delattr` under `_is_allowed_attribute`; see `_guarded_setattr`."""
    if type(name) is not str:
        raise TypeError("type(name) must be str")
    if not _is_allowed_attribute(name):
        raise AttributeError(f'"{name}" is a restricted attribute name.')
    delattr(obj, name)


def _guarded_import(
    name: str,
    globals: typing.Any = None,
    locals: typing.Any = None,
    fromlist: collections.abc.Sequence[str] = (),
    level: int = 0,
) -> types.ModuleType:
    """The ``__import__`` restricted code sees: `_ALLOWED_MODULES` only.

    Without an ``__import__`` at all, ``import math`` fails and most generated code
    with it; with the real one, ``import os`` hands back the process. Relative
    imports are refused outright -- there is no package for generated code to be
    relative to."""
    if level != 0:
        raise ImportError("relative imports are not allowed in restricted code")
    if name not in _ALLOWED_MODULES:
        raise ImportError(
            f"import of {name!r} is not allowed in restricted code; "
            f"the allowed modules are: {', '.join(sorted(_ALLOWED_MODULES))}"
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


def _guarded_inplacevar(op: str, x: typing.Any, y: typing.Any) -> typing.Any:
    """``x <op>= y``, as `_INPLACE_OPS` spells it."""
    try:
        return _INPLACE_OPS[op](x, y)
    except KeyError:
        raise NotImplementedError(f"augmented assignment {op!r} is not supported")


def _guarded_apply(
    func: collections.abc.Callable, *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """``f(*args, **kwargs)`` -- the form the transformer routes starred calls
    through. Argument *values* need no guarding here: whatever built them was
    itself compiled under the same policy."""
    return func(*args, **kwargs)


class RestrictedPythonExecutor(ObjectInterpretation):
    """
    Safer provider using RestrictedPython.

    RestrictedPython is not a complete sandbox, but it enforces a restricted
    language subset and expects you to provide a constrained exec environment.
    This provider supplies that environment: a `RestrictedPythonPolicy` at compile
    time, and at run time the guarded accessors that policy's output calls into
    (`_guarded_getattr`, `_guarded_import`, ...) over a builtins namespace with no
    I/O, no introspection and no way back to `compile`/`eval`/`exec`.

    Doctests are executed under the same policy as the code they exercise, so a
    model cannot smuggle past the sandbox in a docstring.

    policy : type[RestrictingNodeTransformer], optional
        RestrictedPython compile_restricted policy for compilation. Defaults to
        `RestrictedPythonPolicy`.
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
        source: str,
        lo: int | None = None,
        hi: int | None = None,
        *,
        lenient: bool = False,
    ) -> None:
        _mypy_check_region(source, lo, hi, lenient)

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

    def _compile_restricted(
        self, source: typing.Any, filename: str, mode: str = "exec", *args: typing.Any
    ) -> types.CodeType:
        """`compile_restricted` under this provider's policy, with `compile`'s
        positional signature so it can stand in for the built-in (which is how
        `_doctest_compiled_with` uses it)."""
        # RestrictedPython's transformer rewrites its argument *in place*, so after
        # this call `tree` is the checked, guard-injected program -- which we then
        # compile ourselves. That second compile is not redundant: RestrictedPython
        # calls `compile()` without `dont_inherit`, from a module that begins with
        # `from __future__ import annotations`, so its future flag leaks into every
        # program it compiles and turns generated code's annotations into strings
        # (which breaks, among others, every `@dataclass`). Compiling the same tree
        # here, from a module with no future imports and with `dont_inherit`, gives
        # generated code the semantics its source actually asks for.
        #
        # Transform a *copy*, so the caller's AST is left as it passed it: it is the
        # `parse` op's output, which callers also read (and could compile again --
        # re-transforming an already-transformed tree would reject its own guards).
        tree = (
            copy.deepcopy(source)
            if isinstance(source, ast.AST)
            else ast.parse(source, filename, mode)
        )
        with warnings.catch_warnings():
            # RestrictedPython warns ("Prints, but never reads 'printed'") whenever
            # the code prints, because its protocol expects the collected text to be
            # read back; we route `print` to stdout instead, so the warning is noise.
            warnings.filterwarnings(
                "ignore", ".*Prints, but never reads", SyntaxWarning
            )
            compile_restricted(  # raises SyntaxError if the policy rejects `tree`
                typing.cast(typing.Any, tree),
                filename=filename,
                mode=typing.cast(typing.Any, mode),
                policy=self.policy or RestrictedPythonPolicy,
            )
        return builtins.compile(
            typing.cast(typing.Any, tree),
            filename,
            mode,
            dont_inherit=True,
        )

    def _restricted_globals(
        self, env: collections.abc.Mapping[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """The namespace restricted code runs in: RestrictedPython's safe builtins
        (extended by `_EXTRA_SAFE_BUILTIN_NAMES` and the guarded
        ``__import__``/``getattr``/``hasattr``), the guarded accessors compiled code
        calls into, then ``env`` layered on top."""
        rglobals: dict[str, typing.Any] = safe_globals.copy()

        # `safe_globals` is module-level shared state and its `__builtins__` is one
        # dict, so extend a *copy* rather than mutating everyone's builtins.
        restricted_builtins = dict(rglobals["__builtins__"])
        for name in _EXTRA_SAFE_BUILTIN_NAMES:
            restricted_builtins.setdefault(name, getattr(builtins, name))
        restricted_builtins["__import__"] = _guarded_import
        restricted_builtins["getattr"] = _guarded_getattr
        restricted_builtins["hasattr"] = _guarded_hasattr
        restricted_builtins["setattr"] = _guarded_setattr
        restricted_builtins["delattr"] = _guarded_delattr
        rglobals["__builtins__"] = restricted_builtins

        # Enable class definitions (required for Python 3)
        rglobals["__metaclass__"] = type
        rglobals["__name__"] = "restricted"

        # The accessors compiled code is rewritten to call. Without every one of
        # these, perfectly ordinary code -- indexing, unpacking, `+=` -- dies with a
        # `NameError` on a guard name the model never wrote.
        rglobals["_getattr_"] = _guarded_getattr
        rglobals["_getitem_"] = Eval.default_guarded_getitem
        rglobals["_getiter_"] = Eval.default_guarded_getiter
        rglobals["_iter_unpack_sequence_"] = Guards.guarded_iter_unpack_sequence
        rglobals["_unpack_sequence_"] = Guards.guarded_unpack_sequence
        rglobals["_inplacevar_"] = _guarded_inplacevar
        rglobals["_apply_"] = _guarded_apply
        # Attribute/item *writes* land on objects the generated code was handed or
        # built itself; the names it may write are already fixed at compile time by
        # `RestrictedPythonPolicy.visit_Attribute`.
        rglobals["_write_"] = lambda x: x

        # RestrictedPython rewrites `print(...)` into its `_print_` collector
        # protocol; route it to the real stdout so output-capturing callers
        # (e.g. redirect_stdout) see it instead of a discarded collector.
        rglobals["_print_"] = _StdoutPrintCollector
        # The transformer injects `_print = _print_(_getattr_)` at the top of each
        # module and function that prints -- but not into `single`-mode code, which
        # is what a doctest example is. Seed one so `print` works there too.
        rglobals["_print"] = _StdoutPrintCollector(_guarded_getattr)

        # Layer `env` on top (without letting callers replace the restricted
        # builtins, or any guard, with something of their choosing).
        rglobals.update(
            {
                k: v
                for k, v in env.items()
                if k != "__builtins__" and k not in _GUARD_NAMES
            }
        )
        return rglobals

    @implements(compile)
    def compile(self, module: ast.Module, filename: str) -> types.CodeType:
        # RestrictedPython can compile from an AST directly.
        return self._compile_restricted(module, filename, "exec")

    @implements(exec)
    def exec(
        self,
        bytecode: types.CodeType,
        env: dict[str, typing.Any],
    ) -> None:
        rglobals = self._restricted_globals(env)

        # Snapshot value identities before execution so we can copy back every
        # *binding effect* — both new names and rebindings of seeded names.
        before = dict(rglobals)
        builtins.exec(bytecode, rglobals, rglobals)

        sentinel = object()
        env.update(
            {
                key: value
                for key, value in rglobals.items()
                # The guards (and the `_print` the transformer injects) are the
                # sandbox's own furniture, not a binding effect of the code.
                if key != "__builtins__"
                and key not in _GUARD_NAMES
                and before.get(key, sentinel) is not value
            }
        )

    @implements(run_doctests)
    def run_doctests(
        self,
        obj: collections.abc.Callable | type | types.ModuleType,
        globs: collections.abc.Mapping[str, typing.Any],
    ) -> None:
        # A docstring's examples are as much model output as the code they
        # document, so run them under the same policy and in the same guarded
        # namespace -- otherwise `>>> __import__("os").system(...)` in a synthesized
        # docstring would execute with nothing restricting it at all.
        with _doctest_compiled_with(self._compile_restricted):
            _run_doctests(obj, self._restricted_globals(globs))
