"""Desugar generator expressions into :meth:`Monoid.reduce` calls.

A generator expression is a concrete Python object, but its *syntax* describes a
loop nest that a monoid can interpret for itself. This module recovers that
syntax with :func:`effectful.internals.disassembly.disassemble` and replays it
against a monoid, turning

.. code-block:: python

    Sum(f(x) * g(x, y) for x in xs for y in ys(x))

into

.. code-block:: python

    Sum.reduce(f(x()) * g(x(), y()), {x: xs, y: ys(x())})

where ``x`` and ``y`` are fresh :class:`Operation` s standing for "an element of
``xs``" and "an element of ``ys(x())``".

The interesting part is not the rewriting but the *typing*. Each loop target
becomes an :class:`Operation` whose return type is the element type of its
stream, and that type has to be known before the target can be applied to
anything in the body. Streams may also depend on earlier targets, as ``ys(x())``
does, so element types are inferred one generator at a time, left to right:
evaluate a stream, infer its element type, mint the target operation, bind it,
and only then move on to the next generator.

Some of what a comprehension can contain is syntax rather than operations.
``and``, ``or``, ``not`` and conditional expressions all ask their operands for
a concrete :class:`bool`, which a :class:`Term` cannot supply, so they are
rewritten into :func:`~effectful.ops.syntax.ite`, which yields one of its arms
outright when the condition is concrete and a term when it is not. Comparisons
are left alone: numeric terms already implement ``==`` and ``<`` symbolically.

.. warning::

   Equality on a *non-numeric* term is not symbolic -- ``c() == "a"`` is
   ``False``, not a term -- so a filter comparing non-numeric elements silently
   reduces to nothing. Numeric streams, which are what a monoid reduces, are
   unaffected.
"""

import ast
import collections.abc
import copy
import functools
import inspect
import itertools
import operator
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from effectful.internals.disassembly import CompExp, disassemble, ensure_ast
from effectful.internals.unification import Box, unify
from effectful.ops.monoid import And, Max, Min, Monoid, Or, Sum
from effectful.ops.semantics import apply, evaluate, typeof
from effectful.ops.syntax import defop, ite, iter_, range_
from effectful.ops.types import Expr, NotHandled, Operation, Term

REDUCTIONS: Mapping[Any, Monoid] = {
    sum: Sum,
    any: Or,
    all: And,
    max: Max,
    min: Min,
}
"""Builtins that mean a reduction when applied to a generator expression.

Only these have a monoid whose ``plus`` agrees with them elementwise. Every
other callable in a comprehension is called as written, since introducing a
monoid where none was meant would change what the comprehension computes.

``max`` and ``min`` differ from their builtins on an empty stream, where the
monoid identity (an infinity) stands in for :class:`ValueError`.
"""

SUBSTITUTIONS: Mapping[Any, Any] = {range: range_}
"""Stream constructors replaced by symbolic-friendly counterparts.

A dependent stream such as ``range(x)`` is built from a term, which the builtin
cannot accept. :func:`~effectful.ops.syntax.range_` yields an ordinary
:class:`range` when its bounds are concrete, so this substitution only changes
the outcome where the builtin would have failed outright.
"""


# ============================================================================
# LIFTING SYNTAX INTO OPERATIONS
# ============================================================================


def _conjoin(*conditions: Any) -> Any:
    """``and`` over conditions, as a conditional a term can survive."""
    return functools.reduce(lambda a, b: ite(a, b, False), conditions, True)


def _disjoin(*conditions: Any) -> Any:
    """``or`` over conditions, as a conditional a term can survive."""
    return functools.reduce(lambda a, b: ite(a, True, b), conditions, False)


def _as_condition(cond: Any) -> Any:
    """Coerce a filter to a boolean, as ``if`` does.

    A filter may be any expression -- ``if x``, ``if some_list`` -- and Python
    takes its truthiness. A mask needs an actual condition, and a term cannot be
    asked for its truthiness later, so anything not already boolean is put
    through a conditional now.
    """
    return cond if typeof(cond) is bool else ite(cond, True, False)


def _negate(cond: Any) -> Any:
    """``not cond``, as a conditional a term can survive."""
    return ite(cond, False, True)


# Names bound in the evaluation namespace for the rewritten syntax. These are
# deliberately not valid identifiers, so they cannot collide with anything the
# comprehension itself refers to.
_CONJUNCTION = ".conjunction"
_DISJUNCTION = ".disjunction"
_NEGATION = ".negation"
_CONDITIONAL = ".conditional"
_CALL = ".call"
_STREAM_CALL = ".stream_call"

_RESERVED = frozenset(
    {_CONJUNCTION, _DISJUNCTION, _NEGATION, _CONDITIONAL, _CALL, _STREAM_CALL}
)


class LiftOperators(ast.NodeTransformer):
    """Rewrite the parts of a comprehension that are syntax rather than calls.

    ``and``, ``or``, ``not`` and conditional expressions become calls to
    reserved names, so the transformed tree is only meaningful in the namespace
    :func:`desugar_comprehension` builds. Rewriting ``and`` and ``or`` costs
    short-circuiting -- both operands are evaluated -- which is unavoidable if
    the result is to be a term rather than a decision made at desugaring time.

    Every remaining call is routed through a dispatcher as well. Substituting
    a stream constructor or reinterpreting an inner reduction has to key on the
    callable itself rather than on the name it was reached by, so that an
    aliased builtin is still recognized and a shadowed one is still left alone.

    Which dispatcher depends on where the call sits. A stream may be symbolic,
    so a constructor in that position is replaced by a counterpart that accepts
    terms. Anywhere else -- a body, a filter, or inside a nested comprehension
    or lambda, all of which Python evaluates eagerly -- it is left alone: a
    symbolic iterable has no end, so eagerly consuming one would never finish,
    whereas the builtin rejects a term outright.
    """

    def __init__(self, substituting: bool = False):
        self.substituting = substituting

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.Call:
        self.generic_visit(node)
        name = _CONJUNCTION if isinstance(node.op, ast.And) else _DISJUNCTION
        return _reserved_call(name, node.values)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        self.generic_visit(node)
        if not isinstance(node.op, ast.Not):
            return node
        return _reserved_call(_NEGATION, [node.operand])

    def visit_IfExp(self, node: ast.IfExp) -> ast.Call:
        self.generic_visit(node)
        return _reserved_call(_CONDITIONAL, [node.test, node.body, node.orelse])

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in _RESERVED:
            return node
        dispatcher = _STREAM_CALL if self.substituting else _CALL
        return ast.Call(
            func=ast.Name(id=dispatcher, ctx=ast.Load()),
            args=[node.func, *node.args],
            keywords=node.keywords,
        )

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return LiftOperators().generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        # A generator expression is lazy, and may itself become a stream or be
        # reduced in its own right, so its outermost iterable is a stream
        # position too. The rest of it is evaluated by whoever consumes it.
        inner = LiftOperators()
        generators = [
            ast.comprehension(
                target=generator.target,
                iter=(LiftOperators(True) if index == 0 else inner).visit(
                    generator.iter
                ),
                ifs=[inner.visit(condition) for condition in generator.ifs],
                is_async=generator.is_async,
            )
            for index, generator in enumerate(node.generators)
        ]
        return ast.GeneratorExp(elt=inner.visit(node.elt), generators=generators)

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return LiftOperators().generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        return LiftOperators().generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        return LiftOperators().generic_visit(node)


class BindTargets(ast.NodeTransformer):
    """Rewrite each loop target into a call to the operation standing for it.

    Substitution is scope-aware. A nested comprehension, lambda or assignment
    expression may bind a name the enclosing comprehension already uses as a
    loop target, and inside that binder the name means the inner binding: its
    own target stays a target, and its body refers to its own variable rather
    than to the operation minted for the outer one.
    """

    def __init__(self, bound: collections.abc.Set[str]):
        self.bound = frozenset(bound)

    def _without(self, names: collections.abc.Set[str]) -> "BindTargets":
        return BindTargets(self.bound - set(names))

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if isinstance(node.ctx, ast.Load) and node.id in self.bound:
            return ast.Call(func=node, args=[], keywords=[])
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        # Defaults are evaluated where the lambda is written; its body is not.
        arguments = ast.arguments(
            posonlyargs=node.args.posonlyargs,
            args=node.args.args,
            vararg=node.args.vararg,
            kwonlyargs=node.args.kwonlyargs,
            kw_defaults=[
                None if default is None else self.visit(default)
                for default in node.args.kw_defaults
            ],
            kwarg=node.args.kwarg,
            defaults=[self.visit(default) for default in node.args.defaults],
        )
        inner = self._without(_parameter_names(node.args))
        return ast.Lambda(args=arguments, body=inner.visit(node.body))

    def visit_GeneratorExp(self, node):
        return self._visit_comprehension(node)

    def visit_ListComp(self, node):
        return self._visit_comprehension(node)

    def visit_SetComp(self, node):
        return self._visit_comprehension(node)

    def visit_DictComp(self, node):
        return self._visit_comprehension(node)

    def _visit_comprehension(self, node: CompExp) -> CompExp:
        """Visit a nested comprehension, shadowing as Python scopes it.

        The outermost iterable is evaluated where the comprehension is written;
        everything else sees the targets bound to its left.
        """
        scope = self
        generators = []
        for index, generator in enumerate(node.generators):
            iterable = (self if index == 0 else scope).visit(generator.iter)
            scope = scope._without(_bound_names(generator.target))
            generators.append(
                ast.comprehension(
                    target=generator.target,
                    iter=iterable,
                    ifs=[scope.visit(condition) for condition in generator.ifs],
                    is_async=generator.is_async,
                )
            )

        if isinstance(node, ast.DictComp):
            return ast.DictComp(
                key=scope.visit(node.key),
                value=scope.visit(node.value),
                generators=generators,
            )
        return type(node)(elt=scope.visit(node.elt), generators=generators)


def _parameter_names(arguments: ast.arguments) -> set[str]:
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    for variadic in (arguments.vararg, arguments.kwarg):
        if variadic is not None:
            names.add(variadic.arg)
    return names


def _bound_names(target: ast.expr) -> set[str]:
    """Every name a loop target binds, whatever its shape."""
    return {node.id for node in ast.walk(target) if isinstance(node, ast.Name)}


def _reserved_call(name: str, args: Sequence[ast.expr]) -> ast.Call:
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[])


def _dispatch_stream_call(callee: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Interpret a call that builds a stream, where a term is admissible."""
    try:
        callee = SUBSTITUTIONS.get(callee, callee)
    except TypeError:  # an unhashable callee matches nothing
        pass
    return _dispatch_call(callee, *args, **kwargs)


def _dispatch_call(callee: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Interpret a call in a comprehension, given what is being called.

    ``sum(f(x, y) for y in ys)`` in a body describes an inner loop nest, so it
    is desugared against its own monoid rather than run. Anything else -- an
    extra argument, a non-generator operand, a callable with no monoid of its
    own -- is called as written.

    Dispatching on the callable rather than on the name it was reached by is
    what makes shadowing work in both directions: a locally defined ``sum`` is
    not the builtin and so is called normally, while a builtin reached under
    another name is still recognized.
    """
    try:
        monoid = REDUCTIONS.get(callee)
    except TypeError:  # an unhashable callee matches nothing
        monoid = None

    if not kwargs and len(args) == 1 and inspect.isgenerator(args[0]):
        # A monoid applied to a comprehension means its own reduction, whether
        # it was reached as a builtin or written out.
        if monoid is None and isinstance(callee, Monoid):
            monoid = callee
        if monoid is not None:
            return desugar_comprehension(args[0], monoid)
        if isinstance(args[0].gi_frame.f_locals.get(".0"), Term):
            # Consuming this would not terminate: a symbolic iterable yields
            # terms forever rather than stopping. Only a reduction can take a
            # comprehension over one, by reducing it rather than running it.
            raise NotImplementedError(
                f"{getattr(callee, '__name__', callee)} cannot consume a "
                "comprehension over a symbolic iterable"
            )

    return callee(*args, **kwargs)


# ============================================================================
# OPAQUE VALUES
# ============================================================================

# Values that cannot be written down as source, keyed by the name standing in
# for them. Entries are removed by the desugaring that created them.
_OPAQUE_VALUES: dict[str, Any] = {}
_OPAQUE_COUNTER = itertools.count()


def opaque(value: Any) -> ast.Name:
    """Return a name standing for ``value``, which has no source representation.

    :func:`ensure_ast` reconstructs a stream by writing it out as an expression,
    which works for containers and iterator adaptors but not for a symbolic
    term. Such a stream is instead bound to a fresh name and passed through the
    namespace untouched.
    """
    name = f".opaque_{next(_OPAQUE_COUNTER)}"
    _OPAQUE_VALUES[name] = value
    return ast.Name(id=name, ctx=ast.Load())


@ensure_ast.register(Term)
def _ensure_ast_term(value: Term) -> ast.expr:
    """Pass a symbolic stream through by reference rather than by reconstruction.

    A symbolic iterable reaches the disassembler already wrapped in ``iter_``,
    since creating the generator applied :func:`iter` to it. The wrapper is
    dropped here so that the stream handed to the monoid is the iterable the
    comprehension named, matching what :func:`ensure_ast` does for a concrete
    iterator over a concrete container.
    """
    if value.op is iter_ and len(value.args) == 1 and not value.kwargs:
        value = value.args[0]
    return opaque(value)


# ============================================================================
# ELEMENT TYPE INFERENCE
# ============================================================================

_T = typing.TypeVar("_T")
_ITERABLE_OF_T: Any = collections.abc.Iterable[_T]  # type: ignore[valid-type]


def annotation_of(term: Term) -> Any:
    """The type of ``term``, keeping the parameters :func:`typeof` erases.

    :func:`~effectful.ops.semantics.typeof` reduces its answer to something
    dispatchable, which turns ``Iterable[int]`` into ``Iterable`` and so loses
    exactly the part that says what a stream yields.
    """

    def _apply(op, *args, **kwargs):
        return Box(op.__type_rule__(*args, **kwargs))

    from effectful.internals.runtime import interpreter

    with interpreter({apply: _apply}):
        boxed = evaluate(term)

    return boxed.value if isinstance(boxed, Box) else type(boxed)


def element_type(stream: Any) -> Any:
    """Infer the type of the elements of ``stream``.

    A symbolic stream carries its element type in its own type, recovered by
    unifying that type against ``Iterable[T]``. A concrete stream usually does
    not: Python erases the element type of a list at runtime. Where the elements
    are available without consuming the stream they are inspected directly;
    otherwise this falls back to :class:`object`, which is honest but leaves the
    loop target untyped.
    """
    if isinstance(stream, Term):
        try:
            return unify(_ITERABLE_OF_T, annotation_of(stream)).get(_T, object)
        except (TypeError, ValueError):
            return object

    if isinstance(stream, range):
        return int
    if isinstance(stream, str):
        return str
    if isinstance(stream, bytes | bytearray):
        return int

    # A parameterized generic instance records its arguments; an ordinary
    # container does not.
    orig_class = getattr(stream, "__orig_class__", None)
    if orig_class is not None:
        try:
            return unify(_ITERABLE_OF_T, orig_class).get(_T, object)
        except (TypeError, ValueError):
            pass

    # Peeking is only safe on a stream that can be iterated more than once.
    if isinstance(stream, collections.abc.Collection):
        for element in stream:
            return _value_type(element)

    return object


def _value_type(value: Any) -> Any:
    """The type of a stream element, keeping tuple structure for unpacking."""
    if isinstance(value, Term):
        return annotation_of(value)
    if isinstance(value, tuple):
        return tuple[*(_value_type(element) for element in value)]  # type: ignore[misc]
    return type(value)


def _component_types(elem_type: Any, arity: int) -> list[Any]:
    """Split the element type of a stream over an ``arity``-way tuple target."""
    args = typing.get_args(elem_type)
    if len(args) != arity or Ellipsis in args:
        return [object] * arity
    return list(args)


# ============================================================================
# DESUGARING
# ============================================================================


def desugar_comprehension[W](
    comprehension: collections.abc.Generator[Any, None, None],
    monoid: Monoid[W],
) -> Expr[W]:
    """Desugar a generator expression into a call to :meth:`Monoid.reduce`.

    :param comprehension: A generator that has not yet been started.
    :param monoid: The monoid the comprehension is reduced over.
    :returns: The reduced expression, usually a :class:`Term`.

    Each loop target becomes a fresh operation whose type is the element type of
    its stream, and the body is expressed in terms of those operations. A filter
    becomes a :meth:`Monoid.mask` on the body rather than a filtered stream, so
    that it stays meaningful when the stream is symbolic.

    **Example usage**:

    >>> from effectful.ops.monoid import EvaluateIntp, NormalizeIntp
    >>> from effectful.ops.semantics import coproduct, evaluate, handler
    >>> term = Sum(x * 2 for x in range(4) if x != 1)
    >>> with handler(coproduct(EvaluateIntp, NormalizeIntp)):
    ...     evaluate(term)
    10
    """
    assert inspect.isgenerator(comprehension), "Input must be a generator expression"

    watermark = set(_OPAQUE_VALUES)
    try:
        tree = disassemble(comprehension).body
        assert isinstance(tree, ast.GeneratorExp)
        namespace = _namespace(comprehension)
    finally:
        for name in set(_OPAQUE_VALUES) - watermark:
            del _OPAQUE_VALUES[name]

    streams: dict[Operation[[], Any], Iterable[Any]] = {}
    conditions: list[Any] = []
    bound: set[str] = set()

    for generator in tree.generators:
        if generator.is_async:
            raise NotImplementedError("Asynchronous comprehensions are not supported")

        stream = _materialize(_evaluate(_prepare(generator.iter, bound, stream=True), namespace))
        bound.update(_bind_target(generator.target, stream, namespace, streams))

        conditions.extend(
            _evaluate(_prepare(condition, bound), namespace)
            for condition in generator.ifs
        )

    body = _evaluate(_prepare(tree.elt, bound), namespace)
    if conditions:
        body = monoid.mask(body, _as_condition(_conjoin(*conditions)))

    return monoid.reduce(body, streams)


def _prepare(
    node: ast.expr, bound: collections.abc.Set[str], *, stream: bool = False
) -> ast.expr:
    """Rewrite a subexpression of the comprehension for evaluation.

    Loop targets become calls to the operations standing for them, and Python's
    syntax becomes calls that symbolic operands can survive.
    """
    prepared = BindTargets(bound).visit(copy.deepcopy(node))
    return ast.fix_missing_locations(LiftOperators(stream).visit(prepared))


def _evaluate(node: ast.expr, namespace: dict[str, Any]) -> Any:
    return eval(compile(ast.Expression(body=node), "<comprehension>", "eval"), namespace)


def _namespace(comprehension: Any) -> dict[str, Any]:
    """Build the namespace the comprehension's free names resolve in.

    A generator's frame exposes its closure variables alongside the outermost
    iterable, so globals and frame locals together cover every name the
    comprehension can refer to.
    """
    frame = comprehension.gi_frame
    namespace = dict(frame.f_globals)
    namespace.update({k: v for k, v in frame.f_locals.items() if k != ".0"})
    namespace.update(_OPAQUE_VALUES)
    namespace.update(
        {
            _CONJUNCTION: _conjoin,
            _DISJUNCTION: _disjoin,
            _NEGATION: _negate,
            _CONDITIONAL: ite,
            _CALL: _dispatch_call,
            _STREAM_CALL: _dispatch_stream_call,
        }
    )
    return namespace


def _materialize(stream: Any) -> Any:
    """Give a one-shot stream a form that can be read more than once.

    A reduce reads each stream repeatedly -- once to unroll the nest, again in
    any rewrite that inspects it -- and inferring an element type reads one
    element more. An iterator such as ``zip(...)`` survives none of that, so it
    is drained here. Anything already re-iterable, symbolic or otherwise, is
    left as it is.
    """
    if isinstance(stream, Term) or isinstance(stream, collections.abc.Collection):
        return stream
    if inspect.isgenerator(stream) and isinstance(
        stream.gi_frame.f_locals.get(".0"), Term
    ):
        # Draining this would not terminate: a symbolic iterable yields terms
        # forever rather than stopping. Only a stream handed straight to the
        # monoid may be symbolic.
        raise NotImplementedError(
            "A comprehension used as a stream cannot itself range over a "
            "symbolic iterable"
        )
    if isinstance(stream, collections.abc.Iterator):
        return tuple(stream)
    return stream


def _target_names(target: ast.expr) -> list[str]:
    match target:
        case ast.Name(id=name):
            return [name]
        case ast.Tuple(elts=elts) | ast.List(elts=elts):
            return [name for elt in elts for name in _target_names(elt)]
        case _:
            raise NotImplementedError(f"Unsupported loop target: {ast.dump(target)}")


def _bind_target(
    target: ast.expr,
    stream: Iterable[Any],
    namespace: dict[str, Any],
    streams: dict[Operation[[], Any], Iterable[Any]],
) -> list[str]:
    """Mint the operations standing for a loop target and bind them.

    A single name becomes one operation, which is also the key of the stream. A
    tuple target still gets exactly one operation -- the stream has one element
    per iteration, whatever its shape -- and each name is bound to a projection
    out of it.
    """
    names = _target_names(target)
    elem_type = element_type(stream)

    if isinstance(target, ast.Name):
        operation = defop(elem_type, name=target.id)
        namespace[target.id] = operation
        streams[operation] = stream
        return names

    operation = defop(elem_type, name="_".join(names))
    streams[operation] = stream
    _bind_projections(target, operation, (), elem_type, namespace)
    return names


def _bind_projections(
    target: ast.expr,
    operation: Operation[[], Any],
    steps: tuple[tuple[int, Any], ...],
    tp: Any,
    namespace: dict[str, Any],
) -> None:
    """Bind each name in a tuple target to its projection out of one element."""
    match target:
        case ast.Name(id=name):
            # `BindTargets` rewrites each name into a call, so a thunk stands
            # in for the operation an untupled target would have had.
            namespace[name] = _projector(operation, steps)
        case ast.Tuple(elts=elts) | ast.List(elts=elts):
            for index, (element, component) in enumerate(
                zip(elts, _component_types(tp, len(elts)), strict=True)
            ):
                _bind_projections(
                    element, operation, (*steps, (index, component)), component, namespace
                )
        case _:
            raise NotImplementedError(f"Unsupported loop target: {ast.dump(target)}")


def _projector(
    operation: Operation[[], Any], steps: tuple[tuple[int, Any], ...]
) -> Callable[[], Any]:
    def project() -> Any:
        value: Any = operation()
        for index, component in steps:
            value = _project(value, index, component)
        return value

    project.__name__ = operation.__name__ + "".join(f"[{i}]" for i, _ in steps)
    return project


def _project(element: Any, index: int, component: Any) -> Any:
    """Index into a stream element for a tuple loop target.

    ``Sequence.__getitem__`` is generic in one element type, so it can type
    ``tuple[int, int]`` but not ``tuple[int, str]``. A heterogeneous element
    falls back to an operation minted for the component's own type, so that the
    projection is still typed and the body built from it still dispatches.
    """
    try:
        return operator.getitem(element, index)
    except TypeError:
        return _heterogeneous_projection(component)(element, index)


@functools.cache
def _heterogeneous_projection(component: Any) -> Operation:
    def getitem_(sequence, index):
        if isinstance(sequence, Term) or isinstance(index, Term):
            raise NotHandled
        return sequence[index]

    getitem_.__annotations__ = {"sequence": Any, "index": int, "return": component}
    return Operation.define(getitem_)
