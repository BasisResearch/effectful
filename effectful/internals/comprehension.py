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
outright when the condition is concrete and a term when it is not. ``in`` asks
its container the same question, and is rewritten into a disjunction over the
container's elements. Ordering and equality are left alone: numeric terms
already implement ``==`` and ``<`` symbolically.

:func:`reduce_to_comprehension` goes the other way, rewriting a reduction as a
comprehension that desugars back to it. The two are inverse in meaning rather
than in syntax.

.. warning::

   Some of what a comprehension may contain answers about the term rather than
   about the element it stands for, and so quietly computes something else:
   equality on a non-numeric term, ``is``, ``isinstance``, a lookup keyed by an
   element, and any formatting of one. Numeric streams, which are what a monoid
   reduces, are unaffected by the first.
"""

import ast
import builtins
import collections.abc
import contextlib
import copy
import functools
import inspect
import itertools
import keyword
import operator
import types
import typing
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from effectful.internals.disassembly import CompExp, disassemble, ensure_ast
from effectful.internals.unification import Box, unify
from effectful.ops.monoid import (
    And,
    CartesianProduct,
    Max,
    Min,
    Monoid,
    Or,
    Sum,
    Union,
    complement,
)
from effectful.ops.semantics import apply, evaluate, typeof
from effectful.ops.syntax import as_dict, defop, ite, iter_, range_
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

PASSTHROUGH: Sequence[Any] = (list, tuple)
"""Stream constructors that, in stream position, are the stream they are given.

Draining a symbolic iterable would not finish, and a reduction does not need it
drained: what these produce holds the same elements in the same order.
"""

ADAPTORS: Sequence[Any] = (filter, map)
"""Stream adaptors that, in stream position, are part of the loop nest.

Over a symbolic iterable there is nothing for one of these to run over, but
what it does to a stream is something the nest can say for itself: a ``filter``
is a filter clause, and a ``map`` is what the loop target stands for.
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
    if typeof(cond) is bool or _is_boolean_sum(cond):
        return cond
    return ite(cond, True, False)


def _is_boolean_sum(cond: Any) -> bool:
    """True if ``cond`` is a sum over :data:`And` or :data:`Or`."""
    owner = getattr(getattr(cond, "op", None), "__self__", None)
    return owner in (And, Or) and cond.op is owner.plus


def _all(*conditions: Any) -> Any:
    """``and`` between conditions, as :class:`MaskFusion` writes a mask."""
    return _boolean_sum(And, False, conditions)


def _any(*conditions: Any) -> Any:
    """``or`` between conditions, as :class:`MaskFusion` writes a mask."""
    return _boolean_sum(Or, True, conditions)


def _boolean_sum(monoid: Monoid, decisive: bool, conditions: Iterable[Any]) -> Any:
    """Add up conditions over ``monoid``, settling the concrete ones here.

    ``decisive`` is the condition that fixes the answer on its own.
    """
    undecided = []
    for condition in conditions:
        condition = _as_condition(condition)
        if condition is decisive:
            return decisive
        if not isinstance(condition, bool):
            undecided.append(condition)

    if not undecided:
        return not decisive
    return undecided[0] if len(undecided) == 1 else monoid.plus(*undecided)


def _negate(cond: Any) -> Any:
    """``not cond``, as the complementary comparison or as a conditional.

    ``not (a == b)`` is ``a != b``, which the rules that read a mask apart are
    written to match; anything without a registered complement becomes a
    conditional a term can survive.
    """
    if isinstance(cond, Term) and not cond.kwargs:
        opposite = complement.of(cond.op)
        if opposite is not None:
            return opposite(*cond.args)
    return ite(cond, False, True)


def _subscript(value: Any, index: Any) -> Any:
    """``value[index]``, projecting out of a heterogeneous element.

    ``Sequence.__getitem__`` is generic in one element type, so a term standing
    for a ``tuple[int, str]`` needs a projection minted for the component.
    """
    if isinstance(value, Term) and isinstance(index, int):
        components = typing.get_args(annotation_of(value))
        if -len(components) <= index < len(components):
            return _project(value, index, components[index])
    return operator.getitem(value, index)


def _membership(element: Any, container: Any, join: Callable = _disjoin) -> Any:
    """``element in container``, as a disjunction over the container's elements.

    ``__contains__`` has to answer with a :class:`bool`, which a term cannot
    supply; comparing against each element yields one a term can survive.
    """
    if not isinstance(element, Term) and not isinstance(container, Term):
        return element in container
    if not isinstance(container, collections.abc.Collection):
        raise NotImplementedError(
            "A symbolic element can only be looked for in a concrete container"
        )
    return join(*(element == item for item in container))


def _contained(element: Any, container: Any) -> Any:
    """``element in container`` in condition position, as :data:`Or` adds."""
    return _membership(element, container, join=_any)


# Names bound in the evaluation namespace for the rewritten syntax. These are
# deliberately not valid identifiers, so they cannot collide with anything the
# comprehension itself refers to.
_CONJUNCTION = ".conjunction"
_DISJUNCTION = ".disjunction"
_NEGATION = ".negation"
_CONDITIONAL = ".conditional"
_SUBSCRIPT = ".subscript"
_ALL = ".all"
_ANY = ".any"
_MEMBERSHIP = ".membership"
_CONTAINED = ".contained"
_CALL = ".call"
_STREAM_CALL = ".stream_call"

_RESERVED = frozenset(
    {
        _CONJUNCTION,
        _DISJUNCTION,
        _SUBSCRIPT,
        _ALL,
        _ANY,
        _NEGATION,
        _CONDITIONAL,
        _MEMBERSHIP,
        _CONTAINED,
        _CALL,
        _STREAM_CALL,
    }
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

    ``in`` is rewritten too. Ordering and equality already answer symbolically
    on a numeric term, but ``__contains__`` has to return a concrete
    :class:`bool`, so a container is asked about each of its elements instead.

    Which dispatcher depends on where the call sits. A stream may be symbolic,
    so a constructor in that position is replaced by a counterpart that accepts
    terms. Anywhere else -- a body, a filter, or inside a nested comprehension
    or lambda, all of which Python evaluates eagerly -- it is left alone: a
    symbolic iterable has no end, so eagerly consuming one would never finish,
    whereas the builtin rejects a term outright.
    """

    def __init__(self, substituting: bool = False, condition: bool = False):
        self.substituting = substituting
        self.condition = condition

    def _values(self) -> "LiftOperators":
        return self if not self.condition else LiftOperators(self.substituting)

    def _conditions(self) -> "LiftOperators":
        return self if self.condition else LiftOperators(self.substituting, True)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.Call:
        inner = self._conditions()
        values = [inner.visit(value) for value in node.values]
        if self.condition:
            name = _ALL if isinstance(node.op, ast.And) else _ANY
        else:
            name = _CONJUNCTION if isinstance(node.op, ast.And) else _DISJUNCTION
        return _reserved_call(name, values)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        if not isinstance(node.op, ast.Not):
            self._values().generic_visit(node)
            return node
        return _reserved_call(_NEGATION, [self._conditions().visit(node.operand)])

    def visit_IfExp(self, node: ast.IfExp) -> ast.Call:
        # `a if not c else b` is `b if c else a`, and the disassembler renders a
        # conditional from jump-if-false bytecode in the negated form.
        test, body, orelse = node.test, node.body, node.orelse
        while isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            test, body, orelse = test.operand, orelse, body

        values = self._values()
        return _reserved_call(
            _CONDITIONAL,
            [
                self._conditions().visit(test),
                values.visit(body),
                values.visit(orelse),
            ],
        )

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        # A slice is syntax rather than a value, so it is left as it is.
        self._values().generic_visit(node)
        if isinstance(node.ctx, ast.Load) and not isinstance(node.slice, ast.Slice):
            return _reserved_call(_SUBSCRIPT, [node.value, node.slice])
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        # A chain containing ``in`` becomes the conjunction of its links.
        self._values().generic_visit(node)
        if not any(isinstance(op, ast.In | ast.NotIn) for op in node.ops):
            return node

        operands = [node.left, *node.comparators]
        contains = _CONTAINED if self.condition else _MEMBERSHIP
        links: list[ast.expr] = []
        for index, op in enumerate(node.ops):
            left, right = operands[index], operands[index + 1]
            match op:
                case ast.In():
                    links.append(_reserved_call(contains, [left, right]))
                case ast.NotIn():
                    links.append(
                        _reserved_call(
                            _NEGATION, [_reserved_call(contains, [left, right])]
                        )
                    )
                case _:
                    links.append(ast.Compare(left=left, ops=[op], comparators=[right]))

        return links[0] if len(links) == 1 else _reserved_call(_CONJUNCTION, links)

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self._values().generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in _RESERVED:
            return node
        dispatcher = _STREAM_CALL if self.substituting else _CALL
        return ast.Call(
            func=ast.Name(id=dispatcher, ctx=ast.Load()),
            args=[node.func, *node.args],
            keywords=node.keywords,
        )

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        # A lambda body is evaluated where the lambda is called, which for a
        # lambda written in stream position is that stream position.
        return LiftOperators(self.substituting).generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        # A generator expression is lazy, so every one of its iterables is a
        # stream position and every one of its filters is a condition.
        inner = LiftOperators()
        stream = LiftOperators(True)
        filters = LiftOperators(condition=True)
        generators = [
            ast.comprehension(
                target=generator.target,
                iter=stream.visit(generator.iter),
                ifs=[filters.visit(condition) for condition in generator.ifs],
                is_async=generator.is_async,
            )
            for generator in node.generators
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
    return ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[]
    )


def _dispatch_stream_call(callee: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Interpret a call that builds a stream, where a term is admissible."""
    if not kwargs and any(map(_is_endless, args)):
        if callee in PASSTHROUGH and len(args) == 1:
            return args[0]
        if callee in ADAPTORS:
            # Building one of these reads the iterable but does not run it;
            # `_unwrap_adaptor` takes it apart once the stream is in hand.
            return callee(*args)

    try:
        callee = SUBSTITUTIONS.get(callee, callee)
    except TypeError:  # an unhashable callee matches nothing
        pass
    return _dispatch_call(callee, *args, **kwargs)


def _unwrap_adaptor(stream: Any) -> tuple[Any, list[Any], Any]:
    """An adaptor over a symbolic stream, as the nest it stands for.

    ``map(f, filter(p, E))`` is ``(f(e) for e in E if p(e))``, so adaptors are
    taken from the outside in and each predicate is composed with the mappings
    between it and the stream underneath. Every predicate that comes back
    therefore reads an element of that stream, whatever it read where it was
    written.

    :returns: The stream underneath, the predicates each of its elements has to
        satisfy, and what the loop target stands for.
    """
    predicates: list[Any] = []
    mapping: Any = None
    while _is_endless(stream):
        parts = _reduced(stream)
        if len(parts) != 2:
            break
        if isinstance(stream, filter):
            predicates.append(_as_condition if parts[0] is None else parts[0])
        elif isinstance(stream, map):
            predicates = [_after(keep, parts[0]) for keep in predicates]
            mapping = parts[0] if mapping is None else _after(mapping, parts[0])
        else:
            break
        stream = _iterated(parts[1])
    return stream, predicates, mapping


def _after(outer: Any, inner: Any) -> Callable[[Any], Any]:
    """``outer`` read one mapping further in."""

    def composed(value: Any) -> Any:
        return outer(inner(value))

    return composed


def _reduced(stream: Any) -> tuple[Any, ...]:
    """The parts an iterator adaptor pickles as, which say what it wraps.

    `itertools` warns that pickling an adaptor is on its way out; the
    disassembler reads one the same way.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            reduced = stream.__reduce__()
        except TypeError:
            return ()
    return () if isinstance(reduced, str) or len(reduced) < 2 else tuple(reduced[1])


def _iterated(stream: Any) -> Any:
    """The iterable an adaptor holds, which it holds an iterator over."""
    if isinstance(stream, Term) and stream.op is iter_ and len(stream.args) == 1:
        return stream.args[0]
    return stream


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

    if monoid is not None and not kwargs:
        folded = _fold_symbolically(monoid, callee, args)
        if folded is not None:
            return folded

    # An operation may be handed a symbolic stream; anything else would read
    # one to an end it does not have.
    if not isinstance(callee, Operation) and any(map(_is_endless, args)):
        raise NotImplementedError(
            f"{getattr(callee, '__name__', callee)} cannot consume a "
            "comprehension over a symbolic iterable"
        )

    return callee(*args, **kwargs)


def _fold_symbolically(monoid: Monoid, callee: Any, args: Sequence[Any]) -> Any:
    """``max(a, b)`` and friends over terms, as the ``plus`` of their monoid.

    Over a symbolic iterable there are no elements to add up, so the call
    becomes the reduction it means. ``sum(values, start)`` is not one of these
    spellings and is left alone.
    """
    if len(args) > 1 and callee in (max, min):
        values: Sequence[Any] = args
    elif len(args) == 1 and isinstance(args[0], Term):
        if not isinstance(args[0], collections.abc.Iterable):
            return None
        element = defop(element_type(args[0]), name="element")
        return monoid.reduce(element(), {element: args[0]})
    elif len(args) == 1 and isinstance(args[0], collections.abc.Collection):
        values = tuple(args[0])
    else:
        return None

    if not any(isinstance(value, Term) for value in values):
        return None
    if monoid in (And, Or):
        values = [_as_condition(value) for value in values]
    return monoid.plus(*values)


def _is_endless(stream: Any, depth: int = 0) -> bool:
    """True if draining ``stream`` would never finish.

    A symbolic iterable yields terms forever, and so does anything built on
    one. An iterator adaptor is read through the parts it pickles as.
    """
    if isinstance(stream, Term):
        return isinstance(stream, collections.abc.Iterable)
    if depth > _ADAPTOR_DEPTH or not isinstance(stream, collections.abc.Iterator):
        return False
    if inspect.isgenerator(stream):
        frame = stream.gi_frame
        return frame is not None and _is_endless(frame.f_locals.get(".0"), depth + 1)
    return any(_is_endless(part, depth + 1) for part in _reduced(stream))


_ADAPTOR_DEPTH = 8
"""How far to look through nested iterator adaptors for a symbolic source."""


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


_BY_REFERENCE = 0


@contextlib.contextmanager
def _by_reference() -> collections.abc.Iterator[None]:
    """Let :func:`ensure_ast` name a value it has no way to write out.

    A desugaring evaluates the tree it built in a namespace of its own, so a
    captured monoid, operation or array can be passed through by name.
    """
    global _BY_REFERENCE
    _BY_REFERENCE += 1
    try:
        yield
    finally:
        _BY_REFERENCE -= 1


_ensure_ast_unrepresentable = ensure_ast.dispatch(object)


@ensure_ast.register(object)
def _ensure_ast_opaque(value: Any) -> ast.expr:
    if not _BY_REFERENCE:
        return _ensure_ast_unrepresentable(value)
    return opaque(value)


_ensure_ast_class = ensure_ast.dispatch(type)


@ensure_ast.register(type)
def _ensure_ast_type(value: type) -> ast.expr:
    """Refer to a builtin class by name and name any other."""
    try:
        return _ensure_ast_class(value)
    except AssertionError:
        if not _BY_REFERENCE:
            raise
        return opaque(value)


_ensure_ast_lambda = ensure_ast.dispatch(types.FunctionType)


@ensure_ast.register(types.FunctionType)
def _ensure_ast_function(value: types.FunctionType) -> ast.expr:
    """Write a lambda out as its source and name any other function."""
    if value.__name__.endswith("<lambda>"):
        return _ensure_ast_lambda(value)
    return opaque(value)


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
    """Infer the type of the elements of ``stream``, as a loop target's type."""
    return _as_annotation(_element_type(stream))


def _as_annotation(tp: Any) -> Any:
    """``object`` where the inferred type is not one an operation can have."""
    return object if isinstance(tp, typing.TypeVar) or tp is Any else tp


def _element_type(stream: Any) -> Any:
    """Infer the type of the elements of ``stream``.

    A symbolic stream carries its element type in its own type, recovered by
    unifying that type against ``Iterable[T]``. A concrete stream usually does
    not: Python erases the element type of a list at runtime. Where the elements
    are available without consuming the stream they are inspected directly;
    otherwise this falls back to :class:`object`, which is honest but leaves the
    loop target untyped.
    """
    if _is_row_stream(stream):
        return collections.abc.Mapping
    if isinstance(stream, Term):
        owner = getattr(stream.op, "__self__", None)
        if isinstance(owner, Monoid) and stream.op is owner.weighted:
            # Weighting pairs each element with a weight; the elements are the
            # stream's own.
            return element_type(stream.args[0])
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
    """The type of a stream element, keeping what it is made of.

    A tuple is described by each of its components, since a target may unpack
    one; a list or a mapping by its first entry.
    """
    if isinstance(value, Term):
        return annotation_of(value)
    if isinstance(value, tuple):
        return tuple[*(_value_type(element) for element in value)]  # type: ignore[misc]
    if isinstance(value, list):
        return list[_first_type(value)] if value else list  # type: ignore[misc]
    if isinstance(value, collections.abc.Mapping):
        if not value:
            return type(value)
        key = next(iter(value))
        return dict[_value_type(key), _value_type(value[key])]  # type: ignore[misc]
    return type(value)


def _first_type(values: Sequence[Any]) -> Any:
    """What a sequence holds, judged by what it holds first."""
    return _value_type(values[0])


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
    comprehension: collections.abc.Generator[Any, Any, Any],
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
        with _by_reference():
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

        stream, predicates, mapping = _unwrap_adaptor(
            _evaluate(_prepare(generator.iter, bound, stream=True), namespace)
        )
        names, element = _bind_target(
            generator.target, _materialize(stream), namespace, streams, monoid, mapping
        )
        bound.update(names)
        conditions.extend(keep(element()) for keep in predicates)

        conditions.extend(
            _evaluate(_prepare(condition, bound, condition=True), namespace)
            for condition in generator.ifs
        )

    body = _lift_body(monoid, _evaluate(_prepare(tree.elt, bound), namespace), streams)
    if conditions:
        body = monoid.mask(body, _all(*conditions))

    return monoid.reduce(body, streams)


def _lift_body[W](monoid: Monoid[W], body: Any, streams: Mapping) -> Any:
    """Lift a comprehension body into the monoid's element representation.

    Most monoids reduce a body of their own element type directly, so there is
    nothing to do. :data:`~effectful.ops.monoid.CartesianProduct` is the
    exception: its elements are *rows* -- mappings from an index tuple to a
    value -- because ``plus`` merges rows across the loop nest. A body written
    as a comprehension yields plain values, so it has to be tagged with the
    loop targets it was produced under before sibling rows can be merged.

    ``CartesianProduct(range(K) for t in range(T))`` therefore reduces not
    ``range(K)`` but the row stream ``[{(t,): v} for v in range(K)]``, whose
    elements are the assignments ``t -> value`` that the comprehension means.

    A body that is already a stream of rows -- a nested comprehension over
    :data:`~effectful.ops.monoid.Union` or over ``CartesianProduct`` itself --
    is left alone: its rows already say which targets they assign.
    """
    if monoid in (And, Or):
        # The monoids of `all` and `any`, which take the truth of an element.
        return _as_condition(body)

    if monoid is not CartesianProduct or _is_rows(body):
        return body

    index = tuple(target() for target in streams)
    if not _is_stream(body):
        # One value rather than a choice, so the row assigns it outright.
        return [as_dict((index, body))]

    element = defop(element_type(body), name="row_value")
    return Union.reduce([as_dict((index, element()))], {element: body})


def _is_stream(body: Any) -> bool:
    """True if ``body`` offers several values rather than standing for one."""
    if isinstance(body, Term):
        return issubclass(typeof(body), collections.abc.Iterable)
    return isinstance(body, collections.abc.Iterable)


def _prepare(
    node: ast.expr,
    bound: collections.abc.Set[str],
    *,
    stream: bool = False,
    condition: bool = False,
) -> ast.expr:
    """Rewrite a subexpression of the comprehension for evaluation.

    Loop targets become calls to the operations standing for them, and Python's
    syntax becomes calls that symbolic operands can survive.
    """
    prepared = BindTargets(bound).visit(copy.deepcopy(node))
    lift = LiftOperators(stream, condition)
    return ast.fix_missing_locations(lift.visit(prepared))


def _evaluate(node: ast.expr, namespace: dict[str, Any]) -> Any:
    return eval(
        compile(ast.Expression(body=node), "<comprehension>", "eval"), namespace
    )


def _namespace(comprehension: Any) -> dict[str, Any]:
    """Build the namespace the comprehension's free names resolve in.

    A generator's frame exposes its closure variables alongside the outermost
    iterable, so globals and frame locals together cover every name the
    comprehension can refer to.
    """
    frame = comprehension.gi_frame
    assert frame is not None, "Generator must not be exhausted"
    namespace = dict(frame.f_globals)
    namespace.update({k: v for k, v in frame.f_locals.items() if k != ".0"})
    namespace.update(_OPAQUE_VALUES)
    namespace.update(
        {
            _CONJUNCTION: _conjoin,
            _DISJUNCTION: _disjoin,
            _ALL: _all,
            _ANY: _any,
            _NEGATION: _negate,
            _CONDITIONAL: ite,
            _MEMBERSHIP: _membership,
            _CONTAINED: _contained,
            _SUBSCRIPT: _subscript,
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
    if _is_endless(stream):
        # Draining this would not terminate: a symbolic iterable yields terms
        # forever rather than stopping. Only a stream handed straight to the
        # monoid may be symbolic.
        raise NotImplementedError(
            "A stream that has to be read more than once cannot itself range "
            "over a symbolic iterable"
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
    monoid: Monoid,
    mapping: Any = None,
) -> tuple[list[str], Callable[[], Any]]:
    """Mint the operation standing for a loop target and bind its names.

    A tuple target still gets exactly one operation -- the stream has one
    element per iteration, whatever its shape -- and each name is bound to a
    projection out of it. ``mapping`` is what a ``map`` applies to an element.

    :returns: The names bound, and an element of ``stream`` -- which is what
        the target stands for only when there is no ``mapping``.
    """
    names = _target_names(target)
    elem_type = (
        _vacuous_type(monoid, names) if _is_empty(stream) else element_type(stream)
    )
    operation = defop(
        elem_type, name=target.id if isinstance(target, ast.Name) else "_".join(names)
    )
    streams[operation] = stream

    bound: Callable[[], Any] = operation
    if mapping is not None:
        bound = _mapped(operation, mapping)
        elem_type = _value_type(bound())
    elif _is_row_stream(stream):
        bound = _row_view(operation)

    if isinstance(target, ast.Name):
        namespace[target.id] = bound
    else:
        _bind_projections(target, bound, (), elem_type, namespace)
    return names, operation


def _mapped(operation: Operation[[], Any], function: Any) -> Callable[[], Any]:
    def mapped() -> Any:
        return function(operation())

    mapped.__name__ = operation.__name__
    return mapped


def _is_empty(stream: Any) -> bool:
    """True if ``stream`` is a concrete stream with nothing in it."""
    return (
        not isinstance(stream, Term)
        and isinstance(stream, collections.abc.Collection)
        and len(stream) == 0
    )


def _vacuous_type(monoid: Monoid, names: Sequence[str]) -> Any:
    """A type for the target of a stream that has no elements.

    No element can contradict it, and the reduction is the identity whatever
    the body computes, so the type ``monoid`` adds up is as good as any.
    """
    component = type(monoid.identity)
    if len(names) == 1:
        return component
    return tuple[*([component] * len(names))]  # type: ignore[misc]


def _bind_projections(
    target: ast.expr,
    element: Callable[[], Any],
    steps: tuple[tuple[int, Any], ...],
    tp: Any,
    namespace: dict[str, Any],
) -> None:
    """Bind each name in a tuple target to its projection out of one element."""
    match target:
        case ast.Name(id=name):
            # `BindTargets` rewrites each name into a call, so a thunk stands
            # in for the operation an untupled target would have had.
            namespace[name] = _projector(element, steps)
        case ast.Tuple(elts=elts) | ast.List(elts=elts):
            for index, (part, component) in enumerate(
                zip(elts, _component_types(tp, len(elts)), strict=True)
            ):
                _bind_projections(
                    part,
                    element,
                    (*steps, (index, component)),
                    component,
                    namespace,
                )
        case _:
            raise NotImplementedError(f"Unsupported loop target: {ast.dump(target)}")


def _is_row_stream(stream: Any) -> bool:
    """True if ``stream`` is a :data:`CartesianProduct` stream of rows."""
    return isinstance(stream, Term) and stream.op is CartesianProduct.reduce


_ROW_OPERATIONS = (
    CartesianProduct.reduce,
    CartesianProduct.plus,
    Union.reduce,
    Union.plus,
)


def _is_rows(body: Any) -> bool:
    """True if ``body`` is already a sequence of plate assignments."""
    if isinstance(body, Term):
        return body.op in _ROW_OPERATIONS
    return isinstance(body, Sequence) and all(_is_row(row) for row in body)


def _is_row(row: Any) -> bool:
    """True if ``row`` is one plate assignment, symbolic or not."""
    if isinstance(row, Term):
        return issubclass(typeof(row), collections.abc.Mapping)
    return isinstance(row, Mapping)


class _RowView:
    """A cartesian-product element, subscripted the way it was written.

    A row is keyed by the tuple of plate indices it assigns, but Python hands
    ``row[t]`` a bare ``t`` and only ``row[i, j]`` a tuple. Normalizing the key
    here keeps one internal representation -- every row is tuple-keyed, as the
    rules that consume them expect -- while letting a comprehension subscript a
    single-plate assignment as ``ixs[t]`` rather than ``ixs[t,]``.
    """

    def __init__(self, row: Any):
        self._row = row

    def __getitem__(self, key: Any) -> Any:
        return self._row[key if isinstance(key, tuple) else (key,)]


def _row_view(operation: Operation[[], Any]) -> Callable[[], _RowView]:
    def view() -> _RowView:
        return _RowView(operation())

    view.__name__ = operation.__name__
    return view


def _projector(
    element: Callable[[], Any], steps: tuple[tuple[int, Any], ...]
) -> Callable[[], Any]:
    def project() -> Any:
        value: Any = element()
        for index, component in steps:
            value = _project(value, index, component)
        return value

    project.__name__ = element.__name__ + "".join(f"[{i}]" for i, _ in steps)
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


# ============================================================================
# INVERTING A REDUCTION
# ============================================================================


def reduce_to_comprehension(reduction: Any) -> tuple[ast.expr, dict[str, Any]]:
    """Rewrite a reduction as a comprehension that desugars back to it.

    ``Sum.reduce(Sum.mask(f(x()), p(x())), {x: xs()})`` becomes
    ``Sum(f(x) for x in xs() if p(x))``. A reduction whose streams are not a
    dict literal keyed by names has no comprehension that means it and is left
    alone; where this does apply, the two agree up to renaming and to the order
    of a loop nest.

    :param reduction: The :class:`ast.expr` of an expression containing
        :meth:`Monoid.reduce` calls, or an :class:`~effectful.ops.types.Expr`
        to be rendered as one.
    :returns: The comprehension, and the namespace its generated names mean
        something in. A syntactic input contributes no names.

    **Example usage**:

    >>> import ast
    >>> node = ast.parse("Sum.reduce(x() * 2, {x: (1, 2, 3)})", mode="eval").body
    >>> ast.unparse(reduce_to_comprehension(node)[0])
    'Sum((x * 2 for x in (1, 2, 3)))'
    """
    namespace: dict[str, Any] = {}
    if isinstance(reduction, ast.AST):
        # Rewriting is destructive and the caller still holds the input.
        node = copy.deepcopy(typing.cast(ast.expr, reduction))
    else:
        renderer = _Render()
        node = renderer.expression(reduction)
        namespace = renderer.namespace

    return ast.fix_missing_locations(_Uncomprehend().visit(node)), namespace


class _Render:
    """Write an expression out as source, naming what has no source form.

    An :class:`Operation`, and any value that is not a literal, is bound to a
    name in :attr:`namespace`, keyed by identity. An operation belonging to a
    monoid is written as an attribute of it.
    """

    _MONOID_METHODS = ("reduce", "plus", "mask", "weighted", "delta", "inverse")

    def __init__(self) -> None:
        self.namespace: dict[str, Any] = {}
        self._names: dict[int, str] = {}
        self._keep: list[Any] = []

    def name(self, value: Any, hint: str) -> ast.Name:
        # Identity, not equality: two operations may share a name and mean
        # different loops. Only the second of them is renamed.
        if id(value) not in self._names:
            base = _identifier(hint)
            name = next(
                candidate
                for candidate in itertools.chain(
                    (base,), (f"{base}_{n}" for n in itertools.count(1))
                )
                if candidate not in self.namespace
            )
            self._names[id(value)] = name
            self.namespace[name] = value
            self._keep.append(value)  # ids are only unique while the value lives
        return ast.Name(id=self._names[id(value)], ctx=ast.Load())

    def expression(self, value: Any) -> ast.expr:
        match value:
            case Term():
                return ast.Call(
                    func=self.callee(value.op),
                    args=[self.expression(argument) for argument in value.args],
                    keywords=[
                        ast.keyword(arg=keyword, value=self.expression(argument))
                        for keyword, argument in value.kwargs.items()
                    ],
                )
            case Operation():
                return self.name(value, value.__name__)
            case bool() | int() | float() | str() | bytes() | None:
                return ast.Constant(value=value)
            case tuple():
                return ast.Tuple(
                    elts=[self.expression(element) for element in value],
                    ctx=ast.Load(),
                )
            case list():
                return ast.List(
                    elts=[self.expression(element) for element in value],
                    ctx=ast.Load(),
                )
            case collections.abc.Mapping():
                return ast.Dict(
                    keys=[self.expression(key) for key in value],
                    values=[self.expression(element) for element in value.values()],
                )
            case _:
                return self.name(value, type(value).__name__)

    def callee(self, operation: Operation) -> ast.expr:
        owner = getattr(operation, "__self__", None)
        if isinstance(owner, Monoid):
            for method in self._MONOID_METHODS:
                if operation is getattr(owner, method, None):
                    return ast.Attribute(
                        value=self.name(owner, owner.__name__),
                        attr=method,
                        ctx=ast.Load(),
                    )
        return self.name(operation, operation.__name__)


def _identifier(hint: str) -> str:
    """A name a loop target may be spelled with, however the value was named."""
    name = "".join(character if character.isalnum() else "_" for character in hint)
    if not name.isidentifier() or keyword.iskeyword(name) or hasattr(builtins, name):
        return f"_{name}"
    return name


class _Uncomprehend(ast.NodeTransformer):
    """Rewrite every invertible :meth:`Monoid.reduce` call into a comprehension.

    Nested reductions are rewritten first, so an inner one becomes a monoid
    applied to a comprehension in the body of the outer one.
    """

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)

        match node:
            case ast.Call(
                func=ast.Attribute(value=monoid, attr="reduce"),
                args=[body, ast.Dict() as streams],
                keywords=[],
            ):
                pairs = _stream_pairs(streams)
            case _:
                return node

        if pairs is None:
            return node

        ordered = _order_streams(pairs)
        if ordered is None:
            return node

        body, conditions = _peel_masks(body, monoid)

        generators: list[ast.comprehension] = []
        bound: set[str] = set()
        for name, stream in ordered:
            generators.append(
                ast.comprehension(
                    target=ast.Name(id=name, ctx=ast.Store()),
                    iter=_UnbindTargets(bound).visit(stream),
                    ifs=[],
                    is_async=0,
                )
            )
            bound.add(name)

        if not generators:
            # Reducing over a one-element stream the body ignores is the body.
            generators.append(
                ast.comprehension(
                    target=ast.Name(id=_unused_target(node), ctx=ast.Store()),
                    iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
                    ifs=[],
                    is_async=0,
                )
            )

        unbind = _UnbindTargets(bound)
        generators[-1].ifs = [unbind.visit(condition) for condition in conditions]

        return ast.Call(
            func=monoid,
            args=[ast.GeneratorExp(elt=unbind.visit(body), generators=generators)],
            keywords=[],
        )


def _stream_pairs(streams: ast.Dict) -> list[tuple[str, ast.expr]] | None:
    """The loop nest of a reduction, as names and the expressions they range over."""
    pairs = []
    for key, value in zip(streams.keys, streams.values, strict=True):
        if not isinstance(key, ast.Name):
            return None
        pairs.append((key.id, value))
    if len({name for name, _ in pairs}) != len(pairs):
        return None
    return pairs


def _order_streams(
    pairs: Sequence[tuple[str, ast.expr]],
) -> list[tuple[str, ast.expr]] | None:
    """Order a loop nest so that each stream follows the targets it uses."""
    names = {name for name, _ in pairs}
    remaining = list(pairs)
    ordered: list[tuple[str, ast.expr]] = []
    placed: set[str] = set()
    while remaining:
        for index, (name, stream) in enumerate(remaining):
            if not (_free_names(stream) & names) - placed - {name}:
                ordered.append(remaining.pop(index))
                placed.add(name)
                break
        else:
            return None  # a cyclic nest is not a comprehension
    return ordered


def _free_names(node: ast.expr) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _peel_masks(body: ast.expr, monoid: ast.expr) -> tuple[ast.expr, list[ast.expr]]:
    """Separate a masked body into the body and the filters that mask it."""
    conditions: list[ast.expr] = []
    while True:
        match body:
            case ast.Call(
                func=ast.Attribute(value=masked, attr="mask"),
                args=[value, condition],
                keywords=[],
            ) if ast.unparse(masked) == ast.unparse(monoid):
                conditions.append(condition)
                body = value
            case _:
                return body, conditions[::-1]


def _unused_target(node: ast.AST) -> str:
    """A loop target for an empty nest, named so that it captures nothing."""
    taken = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    return next(
        name for name in (f"_unused{i}" for i in itertools.count()) if name not in taken
    )


class _UnbindTargets(ast.NodeTransformer):
    """Rewrite each application of a loop operation back into the loop target.

    The dual of :class:`BindTargets`, and scoped the same way.
    """

    def __init__(self, bound: collections.abc.Set[str]):
        self.bound = frozenset(bound)

    def _without(self, names: collections.abc.Set[str]) -> "_UnbindTargets":
        return _UnbindTargets(self.bound - set(names))

    def visit_Call(self, node: ast.Call) -> ast.expr:
        match node:
            case ast.Call(func=ast.Name(id=name), args=[], keywords=[]) if (
                name in self.bound
            ):
                return ast.Name(id=name, ctx=ast.Load())
        self.generic_visit(node)
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
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
