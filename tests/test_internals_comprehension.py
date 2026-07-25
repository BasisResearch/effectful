"""Tests for desugaring generator expressions into :meth:`Monoid.reduce`.

Comprehensions are spelled as thunks rather than as generator objects: a
generator survives being desugared, but not being reduced twice, and a
parametrized case is built once and shared by every monoid it runs under.
"""

import collections.abc
import functools
import operator
import typing
from collections.abc import Iterable, Mapping

import pytest

from effectful.internals.comprehension import annotation_of, element_type
from effectful.ops.monoid import (
    And,
    EvaluateIntp,
    Factor,
    Max,
    Min,
    Monoid,
    NormalizeIntp,
    Or,
    Product,
    ReduceFusion,
    Sum,
    Union,
    distributes_over,
)
from effectful.ops.semantics import coproduct, evaluate, fvsof, handler, typeof
from effectful.ops.syntax import defop, ite, range_, syntactic_eq
from effectful.ops.types import NotHandled, Operation, Term
from tests._monoid_helpers import syntactic_eq_alpha

# ``EvaluateIntp`` unrolls a concrete loop nest; ``NormalizeIntp`` supplies the
# rewrites that discharge a mask once its condition is concrete.
CONCRETE = coproduct(EvaluateIntp, NormalizeIntp)


@defop
def f(i: int) -> int:
    raise NotHandled


@defop
def h(i: int) -> int:
    raise NotHandled


@defop
def g(i: int, j: int) -> int:
    raise NotHandled


@defop
def p(i: int) -> bool:
    raise NotHandled


@defop
def xs() -> Iterable[int]:
    raise NotHandled


@defop
def zs() -> Iterable[int]:
    raise NotHandled


@defop
def ys(i: int) -> Iterable[int]:
    raise NotHandled


# An empty stream has no element to inspect, so its target stays untyped. The
# annotation is for the reader and the type checker; at runtime it is just `[]`.
EMPTY: list[int] = []


class Box[T](collections.abc.Iterable[T]):
    """A stream that records its element type, as a builtin container does not."""

    def __init__(self, *items: T):
        self._items = items

    def __iter__(self) -> typing.Iterator[T]:
        return iter(self._items)


def streams_of(term: Term) -> Mapping[Operation, typing.Any]:
    """The loop nest of a ``reduce`` term, keyed by the operations it binds."""
    return typing.cast(Mapping[Operation, typing.Any], term.args[1])


def targets_of(term: Term) -> list[Operation]:
    return list(streams_of(term))


def body_of(term: Term) -> typing.Any:
    return term.args[0]


def reduce_concretely(term):
    with handler(CONCRETE):
        return evaluate(term)


FOLDS = {
    Sum: operator.add,
    Product: operator.mul,
    Min: min,
    Max: max,
    And: lambda a, b: a and b,
    Or: lambda a, b: a or b,
}


def fold(monoid, values):
    """What plain Python computes for this monoid.

    ``functools.reduce`` with an explicit initial value rather than
    ``sum``/``min``/``max``, so that an empty stream yields the identity
    instead of raising, as reducing over an empty nest does.
    """
    return functools.reduce(FOLDS[monoid], values, monoid.identity)


ALL_MONOIDS = [
    pytest.param(Sum, id="Sum"),
    pytest.param(Product, id="Product"),
    pytest.param(Min, id="Min"),
    pytest.param(Max, id="Max"),
]

BOOLEAN_MONOIDS = [pytest.param(And, id="And"), pytest.param(Or, id="Or")]

MONOID_PAIRS = [
    pytest.param(outer.values[0], inner.values[0], id=f"{outer.id}-{inner.id}")
    for outer in ALL_MONOIDS
    for inner in ALL_MONOIDS
    if distributes_over(
        typing.cast(Monoid, inner.values[0]), typing.cast(Monoid, outer.values[0])
    )
]


# ============================================================================
# WHAT A COMPREHENSION DESUGARS TO
# ============================================================================

# Each case pairs a comprehension with the reduce it means, written as a
# function of the monoid and of the operations the desugaring minted for the
# loop targets. Comparison is up to renaming, since those operations are fresh.


w = defop(int, name="w")


def _inner_monoid_call(monoid, x):
    y = defop(int, name="y")
    return monoid.reduce(Sum.reduce(g(x(), y()), {y: ys(x())}), {x: (1, 2)})


def _inner_other_monoid(monoid, x):
    y = defop(int, name="y")
    return monoid.reduce(Max.reduce(g(x(), y()), {y: ys(x())}), {x: (1, 2)})


def _explicit_reduce(monoid, x):
    return monoid.reduce(Sum.reduce(g(x(), w()), {w: (1, 2)}), {x: (1, 2)})


def _inner_sum(monoid, x):
    y = defop(int, name="y")
    return monoid.reduce(Sum.reduce(g(x(), y()), {y: ys(x())}), {x: (1, 2)})


DESUGARINGS = [
    pytest.param(
        lambda: (x * 2 for x in (1, 2, 3)),
        lambda M, x: M.reduce(x() * 2, {x: (1, 2, 3)}),
        id="target-in-the-body",
    ),
    pytest.param(
        lambda: (f(x) * g(x, x) for x in (1, 2)),
        lambda M, x: M.reduce(f(x()) * g(x(), x()), {x: (1, 2)}),
        id="arithmetic-stays-arithmetic",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if x == 1),
        lambda M, x: M.reduce(M.mask(f(x()), x() == 1), {x: (1, 2)}),
        id="comparisons-stay-comparisons",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if p(x)),
        lambda M, x: M.reduce(M.mask(f(x()), p(x())), {x: (1, 2)}),
        id="filter-becomes-a-mask",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if p(x) if p(x + 1)),
        lambda M, x: M.reduce(
            M.mask(f(x()), ite(p(x()), p(x() + 1), False)), {x: (1, 2)}
        ),
        id="filter-clauses-conjoin",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if p(x) and p(x + 1)),
        lambda M, x: M.reduce(
            M.mask(f(x()), ite(p(x()), p(x() + 1), False)), {x: (1, 2)}
        ),
        id="and-becomes-a-conditional",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if p(x) or p(x + 1)),
        lambda M, x: M.reduce(
            M.mask(f(x()), ite(p(x()), True, p(x() + 1))), {x: (1, 2)}
        ),
        id="or-becomes-a-conditional",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if not p(x)),
        lambda M, x: M.reduce(
            M.mask(f(x()), ite(p(x()), False, True)), {x: (1, 2)}
        ),
        id="not-becomes-a-conditional",
    ),
    pytest.param(
        lambda: (f(x) if p(x) else g(x, x) for x in (1, 2)),
        lambda M, x: M.reduce(
            ite(ite(p(x()), False, True), g(x(), x()), f(x())), {x: (1, 2)}
        ),
        id="conditional-expression-becomes-ite",
    ),
    pytest.param(
        lambda: (g(x, y) for x in (1, 2) for y in (3, 4)),
        lambda M, x, y: M.reduce(g(x(), y()), {x: (1, 2), y: (3, 4)}),
        id="one-stream-per-loop",
    ),
    pytest.param(
        lambda: (g(x, y) for x in (1, 2) for y in ys(x)),
        lambda M, x, y: M.reduce(g(x(), y()), {x: (1, 2), y: ys(x())}),
        id="dependent-stream",
    ),
    pytest.param(
        lambda: (g(x, y) for x in (1, 2) if p(x) for y in ys(x) if p(y)),
        lambda M, x, y: M.reduce(
            M.mask(g(x(), y()), ite(p(x()), p(y()), False)), {x: (1, 2), y: ys(x())}
        ),
        id="filters-across-the-nest-conjoin",
    ),
    pytest.param(
        lambda: (f(x) for x in xs()),
        lambda M, x: M.reduce(f(x()), {x: xs()}),
        id="symbolic-stream",
    ),
    pytest.param(
        lambda: (f(y) for x in (1, 2) for y in range(x)),
        lambda M, x, y: M.reduce(f(y()), {x: (1, 2), y: range_(x())}),
        id="dependent-range-goes-symbolic",
    ),
    pytest.param(
        lambda: (g(a, b) for a, b in ((1, 2), (3, 4))),
        lambda M, e: M.reduce(g(e()[0], e()[1]), {e: ((1, 2), (3, 4))}),
        id="tuple-target-projects",
    ),
    pytest.param(
        lambda: (sum(g(x, y) for y in ys(x)) for x in (1, 2)),
        _inner_sum,
        id="inner-sum-becomes-a-nested-reduce",
    ),
    pytest.param(
        lambda: (Sum(g(x, y) for y in ys(x)) for x in (1, 2)),
        _inner_monoid_call,
        id="an-inner-monoid-call-becomes-a-nested-reduce",
    ),
    pytest.param(
        lambda: (Max(g(x, y) for y in ys(x)) for x in (1, 2)),
        _inner_other_monoid,
        id="an-inner-monoid-may-differ-from-the-outer",
    ),
    pytest.param(
        # `reduce` binds its stream keys over values, not over names, so a
        # reduce written in the body cannot capture a loop target -- and its
        # own bound operation is left alone.
        lambda: (Sum.reduce(g(x, w()), {w: (1, 2)}) for x in (1, 2)),
        _explicit_reduce,
        id="an-explicit-reduce-keeps-its-own-binding",
    ),
    pytest.param(
        lambda: (f(x) * g(x, y) for x in xs() for y in ys(x)),
        lambda M, x, y: M.reduce(f(x()) * g(x(), y()), {x: xs(), y: ys(x())}),
        id="the-motivating-example",
    ),
]


@pytest.mark.parametrize("comprehension,expected", DESUGARINGS)
@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_desugars_to(comprehension, expected, monoid):
    term = monoid(comprehension())
    assert syntactic_eq_alpha(term, expected(monoid, *targets_of(term)))


# ============================================================================
# WHAT A COMPREHENSION REDUCES TO
# ============================================================================

# The shapes the disassembler covers, each checked against folding the same
# comprehension in plain Python.

STREAMS = [
    pytest.param(lambda: (x for x in range(5)), id="range"),
    pytest.param(lambda: (x for x in range(0)), id="empty-range"),
    pytest.param(lambda: (x for x in range(1)), id="single-range"),
    pytest.param(lambda: (x for x in range(-5, 5)), id="negative-range"),
    pytest.param(lambda: (x for x in range(0, 10, 2)), id="step-range"),
    pytest.param(lambda: (x for x in range(10, 0, -1)), id="reverse-range"),
    pytest.param(lambda: (x for x in [1, 2, 3]), id="list"),
    pytest.param(lambda: (x for x in (1, 2, 3)), id="tuple"),
    pytest.param(lambda: (x for x in {1, 2, 3}), id="set"),
    pytest.param(lambda: (x for x in frozenset({1, 2})), id="frozenset"),
    pytest.param(lambda: (k for k in {1: "a", 2: "b"}), id="dict-keys"),
    pytest.param(lambda: (v for v in {1: 10, 2: 20}.values()), id="dict-values"),
    pytest.param(lambda: (b for b in b"abc"), id="bytes"),
    pytest.param(lambda: (b for b in bytearray(b"abc")), id="bytearray"),
    pytest.param(lambda: (x for x in iter([1, 2, 3])), id="iterator"),
    pytest.param(lambda: (x for x in reversed([1, 2, 3])), id="reversed"),
    pytest.param(lambda: (x for x in sorted([3, 1, 2])), id="sorted"),
    pytest.param(lambda: (x for x in map(abs, [-1, 2])), id="map"),
    pytest.param(lambda: (x for x in filter(None, [0, 1, 2])), id="filter"),
    pytest.param(lambda: (x for x in (y for y in range(5) if y % 2)), id="generator"),
    pytest.param(lambda: (x for x in [y for y in range(5) if y % 2]), id="list-comp"),
    pytest.param(lambda: (x for x in {y for y in range(5) if y % 2}), id="set-comp"),
    pytest.param(lambda: (x for x in {y: y for y in range(3)}), id="dict-comp"),
]

ARITHMETIC = [
    pytest.param(lambda: (x + 1 for x in range(5)), id="add"),
    pytest.param(lambda: (x - 1 for x in range(5)), id="sub"),
    pytest.param(lambda: (x * 3 for x in range(5)), id="mul"),
    pytest.param(lambda: (x // 2 for x in range(1, 6)), id="floordiv"),
    pytest.param(lambda: (x % 3 for x in range(6)), id="mod"),
    pytest.param(lambda: (x**2 for x in range(4)), id="pow"),
    pytest.param(lambda: (-x for x in range(4)), id="neg"),
    pytest.param(lambda: (abs(x - 2) for x in range(5)), id="abs"),
    pytest.param(lambda: (x & 3 for x in range(8)), id="bitand"),
    pytest.param(lambda: (x | 1 for x in range(6)), id="bitor"),
    pytest.param(lambda: (x ^ 2 for x in range(6)), id="bitxor"),
    pytest.param(lambda: (x << 1 for x in range(5)), id="lshift"),
    pytest.param(lambda: (x >> 1 for x in range(1, 6)), id="rshift"),
    pytest.param(lambda: (~x for x in range(4)), id="invert"),
    pytest.param(lambda: ((x + 1) * (x - 1) for x in range(5)), id="compound"),
]

FILTERS = [
    pytest.param(lambda: (x for x in range(8) if x % 2 == 0), id="equality"),
    pytest.param(lambda: (x for x in range(8) if x % 3 != 0), id="disequality"),
    pytest.param(lambda: (x for x in range(8) if x > 4), id="greater"),
    pytest.param(lambda: (x for x in range(8) if x <= 4), id="less-equal"),
    pytest.param(lambda: (x for x in range(8) if x), id="truthy"),
    pytest.param(lambda: (x for x in range(8) if not x % 3), id="not"),
    pytest.param(lambda: (x for x in range(8) if x > 2 if x < 6), id="two-clauses"),
    pytest.param(lambda: (x for x in range(8) if x > 2 and x < 6), id="and"),
    pytest.param(lambda: (x for x in range(8) if x < 2 or x > 6), id="or"),
    pytest.param(lambda: (x for x in range(8) if x < 100), id="always-true"),
    pytest.param(lambda: (x for x in range(8) if x > 100), id="always-false"),
    pytest.param(lambda: (x for x in range(10) if 2 < x < 7), id="chained"),
    pytest.param(lambda: (x for x in range(10) if 0 <= x <= 3), id="chained-inclusive"),
    pytest.param(
        lambda: (x for x in range(30) if 5 < x < 15 or 20 < x < 25), id="chained-or"
    ),
    pytest.param(
        lambda: (x for x in range(30) if 5 < x < 15 and (x % 2 == 0 or x % 3 == 0)),
        id="chained-and",
    ),
    pytest.param(
        lambda: (x for x in range(20) if (x > 2 or x < 1) and (x < 10 or x > 15)),
        id="and-of-or",
    ),
    pytest.param(
        lambda: (x for x in range(30) if not (x % 2 == 0 or x % 3 == 0)), id="not-of-or"
    ),
    pytest.param(
        lambda: (x for x in range(30) if not (not (x > 5) or not (x < 20))),
        id="double-not",
    ),
    pytest.param(
        lambda: (
            x
            for x in range(40)
            if (x < 5 and x % 2 == 0) or (10 < x < 15) or (x > 35 and x % 3 == 0)
        ),
        id="three-way-disjunction",
    ),
]

NESTED_LOOPS = [
    pytest.param(lambda: (x * y for x in range(3) for y in range(4)), id="two"),
    pytest.param(
        lambda: (x * y * z for x in range(2) for y in range(3) for z in range(2)),
        id="three",
    ),
    pytest.param(lambda: (x * y for x in range(4) for y in range(x)), id="dependent"),
    pytest.param(
        lambda: (x * y for x in range(4) for y in range(x) if x != y),
        id="dependent-filter",
    ),
    pytest.param(
        lambda: (x + y for x in range(6) if x < 2 or x > 4 for y in range(6) if y > 4),
        id="filters-on-both",
    ),
    pytest.param(
        lambda: (x * y for x in range(4) if x % 2 == 0 for y in range(x)),
        id="filter-then-dependent",
    ),
    pytest.param(
        lambda: (x * y for x in range(3) for y in [z for z in range(2)]),
        id="comprehension-as-inner-stream",
    ),
    pytest.param(
        lambda: (x * y for x in range(3) for y in (z for z in range(x))),
        id="generator-as-inner-stream",
        marks=pytest.mark.xfail(raises=NotImplementedError, strict=True),
    ),
]

CONDITIONALS = [
    pytest.param(lambda: (x if x > 3 else -x for x in range(6)), id="simple"),
    pytest.param(
        lambda: (x if x > 3 else (x if x > 1 else 0) for x in range(6)), id="nested"
    ),
    pytest.param(
        lambda: ((x if (x > 2 or x < 1) else -x) for x in range(10) if x % 2 == 0),
        id="lazy-arms-and-filter",
    ),
    pytest.param(
        lambda: (
            (x if x > 5 or x < 2 else (0 if x % 2 == 0 or x == 3 else 1))
            for x in range(12)
        ),
        id="lazy-nested",
    ),
    pytest.param(lambda: ((x if 5 < x < 15 else 0) for x in range(20)), id="chained"),
    pytest.param(
        lambda: (y for x in range(4) for y in (range(x) if x % 2 == 0 else range(1))),
        id="as-the-iterable",
    ),
]

UNPACKING = [
    pytest.param(lambda: (a * b for a, b in [(1, 2), (3, 4)]), id="pair"),
    pytest.param(lambda: (a + b + c for a, b, c in [(1, 2, 3), (4, 5, 6)]), id="triple"),
    pytest.param(
        lambda: (a * b + c for (a, b), c in [((1, 2), 3), ((4, 5), 6)]), id="nested"
    ),
    pytest.param(lambda: (a - b for a, b in zip(range(4), range(4, 8))), id="zip"),
    pytest.param(lambda: (i * v for i, v in enumerate([10, 20, 30])), id="enumerate"),
    pytest.param(lambda: (k * v for k, v in {1: 10, 2: 20}.items()), id="dict-items"),
    pytest.param(
        lambda: (a + b for a, b in ((x, x * 2) for x in range(3))), id="from-generator"
    ),
]

# An eagerly built comprehension is evaluated once, symbolically, while the
# body is being built, so it can only range over something concrete. A lazy
# generator handed to a reduction is reduced instead of run, so it may depend
# on a loop target; the dependent eager forms are in ``REJECTED``.
INNER_COMPREHENSIONS = [
    pytest.param(lambda: (len([y for y in range(3)]) for x in range(4)), id="list"),
    pytest.param(lambda: (len({y for y in range(3)}) for x in range(4)), id="set"),
    pytest.param(lambda: (len({y: y for y in range(3)}) for x in range(4)), id="dict"),
    pytest.param(
        lambda: (len(list(y for y in range(3))) for x in range(4)), id="generator"
    ),
    pytest.param(
        lambda: (len([{y: y} for y in range(3)]) for x in range(4)), id="dict-in-list"
    ),
    pytest.param(
        lambda: (len([y for y in {z for z in range(3)}]) for x in range(4)),
        id="set-in-list",
    ),
    pytest.param(
        lambda: (x * len([y for y in range(3)]) for x in range(4)), id="with-the-target"
    ),
    pytest.param(
        lambda: (x for x in range(5) if [y for y in range(3)]), id="list-as-filter"
    ),
    pytest.param(
        lambda: (x for x in range(5) if len([y for y in range(3) if y % 2]) > x),
        id="list-in-filter",
    ),
    pytest.param(
        lambda: (x for x in range(6) if len([y for y in range(3)]) > 2 or x == 0),
        id="list-in-lazy-filter",
    ),
    pytest.param(
        lambda: (x for x in range(5) if any(y > 2 for y in range(x))),
        id="dependent-any-as-filter",
    ),
    pytest.param(
        lambda: (x for x in range(5) if all(y < 3 for y in range(x))),
        id="dependent-all-as-filter",
    ),
    pytest.param(
        lambda: (
            len([y for y in range(3) if y or y == 0])
            for x in range(5)
            if x > 1 or x == 0
        ),
        id="lazy-in-both-positions",
    ),
]

INNER_REDUCTIONS = [
    pytest.param(lambda: (sum(x * y for y in range(3)) for x in range(4)), id="sum"),
    pytest.param(lambda: (max(x * y for y in range(1, 3)) for x in range(4)), id="max"),
    pytest.param(lambda: (min(x - y for y in range(1, 3)) for x in range(4)), id="min"),
    pytest.param(
        lambda: (sum(x * y for y in range(x)) for x in range(5)), id="dependent"
    ),
    pytest.param(
        lambda: (sum(y for y in range(4) if y != x) for x in range(3)), id="filtered"
    ),
    pytest.param(
        lambda: (
            sum(sum(x * y * z for z in range(2)) for y in range(2)) for x in range(3)
        ),
        id="doubly-nested",
    ),
    pytest.param(
        lambda: (
            sum(y for y in range(x)) + max(z for z in range(1, x + 2))
            for x in range(4)
        ),
        id="two-reductions",
    ),
]

LAMBDAS = [
    pytest.param(lambda: ((lambda v: v * 2)(x) for x in range(4)), id="immediate"),
    pytest.param(
        lambda: ((lambda v, w=3: v + w)(x) for x in range(4)),  # type: ignore[assignment]
        id="default-argument",
    ),
    pytest.param(
        lambda: ((lambda *vs: sum(vs))(x, x + 1) for x in range(4)), id="variadic"
    ),
    pytest.param(
        lambda: ((lambda v: v if v > 1 else -v)(x) for x in range(4)), id="ternary-body"
    ),
    pytest.param(
        lambda: ((lambda v: v + len([w for w in range(3)]))(x) for x in range(4)),
        id="comprehension-body",
    ),
]

WALRUS = [
    pytest.param(lambda: ((y := x * 2) + y for x in range(4)), id="in-the-body"),
    pytest.param(lambda: (x for x in range(8) if (y := x % 3) == 0), id="in-a-filter"),
    pytest.param(
        lambda: (y for x in range(5) if (y := x * 2) > 2), id="bound-by-a-filter"
    ),
]

# These must stay on one line: Python 3.12's `dis` mis-reports jumps for
# multiline comprehensions, which the disassembler suite covers directly.
STRESS = [
    pytest.param(lambda: (x + y for x in range(10) if x % 2 == 0 if x > 2 for y in range(10) if y % 3 == 0 if y < x), id="many-filters"),  # fmt: skip
    pytest.param(lambda: (len([y if y > 1 else -y for y in range(3)]) for x in range(4) if (x if x % 2 == 1 else x % 2 == 0)), id="nested-ternary"),  # fmt: skip
    pytest.param(lambda: (sum(y * z for z in range(y)) for x in range(4) for y in range(x) if y % 2 == 0 or y == 1), id="reduction-in-a-nest"),  # fmt: skip
]

MULTILINE = [
    pytest.param(
        lambda: (
            x * y
            for x in range(4)
            if x % 2 == 0
            for y in range(x)
            if y > 0
        ),
        id="split-generators",
    ),
    pytest.param(
        lambda: (
            x
            if x > 2
            else -x
            for x in range(6)
        ),
        id="split-ternary",
    ),
]


def _shadowed_reduction():
    def total(values):
        return 100

    sum = total  # noqa: A001 - deliberately shadowing the builtin
    return (sum(y for y in [1, 2]) for x in [1, 2])


def _shadowed_stream_constructor():
    def bounds(stop):
        return [stop, stop + 1]

    range = bounds  # noqa: A001 - deliberately shadowing the builtin
    return (y for x in [3] for y in range(x))


def _aliased_reduction():
    total = sum
    return (total(y for y in range(x)) for x in range(4))


def _aliased_stream_constructor():
    interval = range
    return (y for x in [1, 2] for y in interval(x))


def _closure_variable():
    scale = 7
    return (x * scale for x in [1, 2, 3])


DISPATCH = [
    pytest.param(_shadowed_reduction, id="shadowed-reduction"),
    pytest.param(_shadowed_stream_constructor, id="shadowed-stream-constructor"),
    pytest.param(_aliased_reduction, id="aliased-reduction"),
    pytest.param(_aliased_stream_constructor, id="aliased-stream-constructor"),
    pytest.param(_closure_variable, id="closure-variable"),
    pytest.param(lambda: (sum([1, 2, 3], x) for x in [10, 20]), id="reduction-with-start"),
]

ALL_SHAPES = [
    *STREAMS,
    *ARITHMETIC,
    *FILTERS,
    *NESTED_LOOPS,
    *CONDITIONALS,
    *UNPACKING,
    *INNER_COMPREHENSIONS,
    *INNER_REDUCTIONS,
    *LAMBDAS,
    *WALRUS,
    *STRESS,
    *MULTILINE,
    *DISPATCH,
]


@pytest.mark.parametrize("comprehension", ALL_SHAPES)
def test_reduces_like_python(comprehension):
    assert reduce_concretely(Sum(comprehension())) == fold(Sum, comprehension())


@pytest.mark.parametrize(
    "comprehension",
    [*STREAMS, *ARITHMETIC, *FILTERS, *NESTED_LOOPS, *UNPACKING, *INNER_REDUCTIONS],
)
@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduces_like_python_over_every_monoid(comprehension, monoid):
    assert reduce_concretely(monoid(comprehension())) == fold(monoid, comprehension())


PREDICATES = [
    pytest.param(lambda: (x > 2 for x in range(6)), id="mixed"),
    pytest.param(lambda: (x >= 0 for x in range(6)), id="all-true"),
    pytest.param(lambda: (x < 0 for x in range(6)), id="all-false"),
    pytest.param(lambda: (x % 2 == 0 for x in range(6) if x > 3), id="filtered"),
]


@pytest.mark.parametrize("comprehension", PREDICATES)
@pytest.mark.parametrize("monoid", BOOLEAN_MONOIDS)
def test_reduces_predicates_like_python(comprehension, monoid):
    assert reduce_concretely(monoid(comprehension())) == fold(monoid, comprehension())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_an_empty_nest_reduces_to_the_identity(monoid):
    assert reduce_concretely(monoid(x for x in [])) == monoid.identity


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_a_wholly_masked_nest_reduces_to_the_identity(monoid):
    assert reduce_concretely(monoid(x for x in range(5) if x > 100)) == monoid.identity


NON_SCALAR_BODIES = [
    pytest.param(Sum, lambda: ((x, x * 2) for x in range(4)), (6, 12), id="tuple"),
    pytest.param(
        Sum, lambda: ({"a": x, "b": x * 2} for x in range(4)), {"a": 6, "b": 12},
        id="mapping",
    ),
    pytest.param(
        Union, lambda: ([{"x": v}] for v in range(3)), [{"x": 0}, {"x": 1}, {"x": 2}],
        id="union-of-rows",
    ),
]


@pytest.mark.parametrize("monoid,comprehension,expected", NON_SCALAR_BODIES)
def test_a_non_scalar_body_reduces_pointwise(monoid, comprehension, expected):
    """``MonoidOverSequence`` and ``MonoidOverMapping`` see the desugared body."""
    assert reduce_concretely(monoid(comprehension())) == expected


# ============================================================================
# EQUATIONS BETWEEN SPELLINGS
# ============================================================================

# Pairs of comprehensions that describe the same reduction. Each takes the
# monoid, so a spelling may refer to its identity.

PAIRS = ((1, 2), (3, 4), (5, 6))

EQUIVALENCES = [
    pytest.param(
        lambda M: (x for x in range(8) if x % 2 == 0),
        lambda M: (v for v in [w for w in range(8) if w % 2 == 0]),
        id="filter-or-prefiltered-stream",
    ),
    pytest.param(
        lambda M: (x for x in range(8) if x > 2 if x < 6),
        lambda M: (x for x in range(8) if x > 2 and x < 6),
        id="two-clauses-or-conjunction",
    ),
    pytest.param(
        lambda M: (x for x in range(8) if x < 2 or x > 6),
        lambda M: (x for x in range(8) if not (not (x < 2) and not (x > 6))),
        id="de-morgan",
    ),
    pytest.param(
        lambda M: (x for x in range(8) if not x % 3),
        lambda M: (x for x in range(8) if x % 3 == 0),
        id="negation-or-equality",
    ),
    pytest.param(
        lambda M: (x + 0.0 for x in range(6) if x > 2),
        lambda M: (x + 0.0 if x > 2 else M.identity + 0.0 for x in range(6)),
        id="filter-or-identity-arm",
    ),
    pytest.param(
        lambda M: (x * y for x in range(3) for y in range(4)),
        lambda M: (x * y for y in range(4) for x in range(3)),
        id="loop-order",
    ),
    pytest.param(
        lambda M: (x for x in range(5)),
        lambda M: (x for x in tuple(range(5))),
        id="stream-form",
    ),
    pytest.param(
        lambda M: (a * b for a, b in PAIRS),
        lambda M: (t[0] * t[1] for t in PAIRS),
        id="unpacking-or-indexing",
    ),
    pytest.param(
        lambda M: (x * y for x in range(4) for y in range(x)),
        lambda M: (a * b for a, b in [(i, j) for i in range(4) for j in range(i)]),
        id="nest-or-flattened-pairs",
    ),
    pytest.param(
        lambda M: (sum(x * y for y in range(x)) for x in range(5)),
        lambda M: (Sum(x * y for y in range(x)) for x in range(5)),
        id="builtin-or-monoid-call",
    ),
    pytest.param(
        lambda M: (max(x * y for y in range(1, 3)) for x in range(4)),
        lambda M: (Max(x * y for y in range(1, 3)) for x in range(4)),
        id="builtin-or-monoid-call-max",
    ),
    pytest.param(
        lambda M: (sum(x * 2 for x in range(3)) for x in range(4)),
        lambda M: (Sum(x * 2 for x in range(3)) for x in range(4)),
        id="monoid-call-with-a-shadowed-target",
    ),
    pytest.param(
        lambda M: (x * 2 for x in range(5) if x % 2 == 0),
        lambda M: (y * 2 for y in range(0, 5, 2)),
        id="filter-or-step",
    ),
]


@pytest.mark.parametrize("left,right", EQUIVALENCES)
@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_equivalent_spellings_reduce_alike(left, right, monoid):
    assert reduce_concretely(monoid(left(monoid))) == reduce_concretely(
        monoid(right(monoid))
    )


@pytest.mark.parametrize(
    "monoid,builtin", [(Sum, sum), (Max, max), (Min, min)], ids=["Sum", "Max", "Min"]
)
def test_an_inner_reduction_equals_a_flattened_nest(monoid, builtin):
    """``M(M(body, inner), outer) == M(body, inner + outer)``."""
    nested = monoid(builtin(x * y for y in range(1, x + 2)) for x in range(4))
    flat = monoid(x * y for x in range(4) for y in range(1, x + 2))
    assert reduce_concretely(nested) == reduce_concretely(flat)


@pytest.mark.parametrize(
    "monoid,builtin", [(Sum, sum), (Max, max), (Min, min)], ids=["Sum", "Max", "Min"]
)
def test_an_inner_reduction_fuses_into_one_nest(monoid, builtin):
    """``ReduceFusion`` recognizes the nest a nested reduction produced."""
    nested = monoid(builtin(g(x, y) for y in ys(x)) for x in xs())
    with handler(ReduceFusion()):
        fused = evaluate(nested)
    assert isinstance(fused, Term) and fused.op is monoid.reduce
    assert len(streams_of(fused)) == 2
    assert not isinstance(body_of(fused), Term) or body_of(fused).op is not monoid.reduce


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_a_desugared_nest_factors_over_independent_streams(outer, inner):
    """``Factor`` splits a desugared nest exactly as it splits a written one."""
    lhs = outer(inner.plus(f(x), h(y)) for x in xs() for y in zs())
    a, b = defop(int, name="x"), defop(int, name="y")
    rhs = inner.plus(
        outer.reduce(inner.plus(f(a())), {a: xs()}),
        outer.reduce(inner.plus(h(b())), {b: zs()}),
    )
    with handler(Factor()):
        assert syntactic_eq_alpha(evaluate(lhs), rhs)


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_a_desugared_nest_normalizes_like_a_written_one(monoid):
    x = defop(int, name="x")
    written = monoid.reduce(monoid.mask(f(x()), p(x())), {x: (1, 2, 3)})
    with handler(NormalizeIntp):
        assert syntactic_eq_alpha(
            evaluate(monoid(f(v) for v in (1, 2, 3) if p(v))), evaluate(written)
        )


# ============================================================================
# NAME COLLISIONS
# ============================================================================

# A name bound by a nested comprehension, a lambda or a later generator means
# that binding, not the loop target of the comprehension being desugared.

SHADOWING = [
    pytest.param(
        lambda: (sum(x for x in range(x)) for x in range(4)),
        id="generator-target-shadows-and-depends",
    ),
    pytest.param(
        lambda: (sum(x * 2 for x in range(3)) for x in range(4)),
        id="generator-target-shadows",
    ),
    pytest.param(
        lambda: (len([x for x in range(3)]) for x in range(4)), id="list-target-shadows"
    ),
    pytest.param(
        lambda: (len({x for x in range(3)}) for x in range(4)), id="set-target-shadows"
    ),
    pytest.param(
        lambda: (len({x: x for x in range(3)}) for x in range(4)),
        id="dict-target-shadows",
    ),
    pytest.param(
        lambda: (x for x in range(5) if len([x for x in range(2)]) > 1),
        id="target-shadowed-inside-a-filter",
    ),
    pytest.param(
        lambda: (sum(sum(x for x in range(2)) for x in range(2)) for x in range(3)),
        id="target-shadowed-twice-over",
    ),
    pytest.param(
        lambda: ((lambda x: x * 3)(x) for x in range(4)), id="lambda-parameter-shadows"
    ),
    pytest.param(
        lambda: ((lambda x: x + 1)(2) for x in range(4)),
        id="lambda-parameter-shadows-unused",
    ),
    pytest.param(
        lambda: ((lambda y=x: y * 2)() for x in range(4)),  # type: ignore[assignment]
        id="lambda-default-sees-the-target",
    ),
    pytest.param(
        lambda: (x + 1 for x in range(2) for x in range(3)), id="two-loops-one-name"
    ),
    pytest.param(lambda: (range * 2 for range in [1, 2, 3]), id="target-named-range"),
    pytest.param(lambda: (sum * 2 for sum in [1, 2, 3]), id="target-named-sum"),
    pytest.param(lambda: (f * 2 for f in [1, 2, 3]), id="target-named-like-an-operation"),
]


@pytest.mark.parametrize("comprehension", SHADOWING)
@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_shadowed_names_keep_their_own_binding(comprehension, monoid):
    assert reduce_concretely(monoid(comprehension())) == fold(monoid, comprehension())


def test_a_shadowing_target_is_not_a_stream_of_the_outer_reduce():
    """The inner binding gets its own reduce, not a second stream of the outer."""
    term = Sum(sum(x for x in ys(x)) for x in xs())
    (outer,) = targets_of(term)
    (inner,) = targets_of(body_of(term))
    assert outer is not inner
    assert syntactic_eq_alpha(
        term, Sum.reduce(Sum.reduce(inner(), {inner: ys(outer())}), {outer: xs()})
    )

# ============================================================================
# ELEMENT TYPE INFERENCE
# ============================================================================

ELEMENT_TYPES = [
    pytest.param([1, 2, 3], int, id="list-int"),
    pytest.param((1.5, 2.5), float, id="tuple-float"),
    pytest.param({"a", "b"}, str, id="set-str"),
    pytest.param(range(10), int, id="range"),
    pytest.param("abc", str, id="str"),
    pytest.param(b"abc", int, id="bytes"),
    pytest.param(bytearray(b"abc"), int, id="bytearray"),
    pytest.param({1: "a"}, int, id="dict-keys"),
    pytest.param({1: "a"}.values(), str, id="dict-values"),
    pytest.param([[1], [2]], list, id="nested-list"),
    pytest.param([None], type(None), id="none"),
    pytest.param([], object, id="empty"),
    pytest.param(Box[int](1, 2), int, id="parameterized-generic"),
    pytest.param(Box(1, 2), object, id="unparameterized-generic"),
    pytest.param(iter([1, 2, 3]), object, id="unrepeatable"),
]


@pytest.mark.parametrize("stream,expected", ELEMENT_TYPES)
def test_element_type_of_a_concrete_stream(stream, expected):
    assert element_type(stream) is expected


def test_element_type_of_a_tuple_stream_keeps_its_shape():
    assert element_type([(1, "a"), (2, "b")]) == tuple[int, str]


SYMBOLIC_ELEMENT_TYPES = [
    pytest.param(lambda: xs(), int, id="stream"),
    pytest.param(lambda: ys(defop(int, name="x")()), int, id="dependent-stream"),
    pytest.param(lambda: range_(defop(int, name="x")()), int, id="symbolic-range"),
]


@pytest.mark.parametrize("stream,expected", SYMBOLIC_ELEMENT_TYPES)
def test_element_type_of_a_symbolic_stream(stream, expected):
    """A term carries its element type in its own type."""
    assert element_type(stream()) is expected


def test_an_annotation_keeps_what_typeof_erases():
    assert typeof(xs()) is Iterable
    assert annotation_of(xs()) == Iterable[int]


TARGET_TYPES = [
    pytest.param(lambda: (f(1) for i in [1, 2, 3]), int, id="list"),
    pytest.param(lambda: (f(1) for c in "abc"), str, id="str"),
    pytest.param(lambda: (f(1) for v in [1.5]), float, id="float"),
    pytest.param(lambda: (f(1) for i in range(3)), int, id="range"),
    pytest.param(lambda: (f(1) for i in xs()), int, id="symbolic"),
    pytest.param(lambda: (f(1) for i in EMPTY), object, id="unknown"),
    pytest.param(lambda: (f(1) for i in Box[int](1, 2)), int, id="parameterized"),
    pytest.param(lambda: (f(1) for a, b in [(1, 2)]), tuple, id="tuple-target"),
]


@pytest.mark.parametrize("comprehension,expected", TARGET_TYPES)
def test_a_target_operation_has_the_inferred_element_type(comprehension, expected):
    (target,) = targets_of(Sum(comprehension()))
    assert typeof(target()) is expected


def test_a_dependent_target_is_typed_from_its_dependent_stream():
    """``y``'s type is only knowable once ``x`` exists to be applied."""
    _, y = targets_of(Sum(g(x, y) for x in [1, 2] for y in ys(x)))
    assert typeof(y()) is int


def test_a_heterogeneous_tuple_target_types_each_component_separately():
    """``Sequence.__getitem__`` is generic in one element type, so a mixed
    tuple needs a projection minted for each component's own type."""
    pairs = [(1, "x"), (2, "y")]
    assert element_type(pairs) == tuple[int, str]
    assert reduce_concretely(Sum(a for a, b in pairs)) == 3


# ============================================================================
# SCOPE AND SYMBOLIC STREAMS
# ============================================================================


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_targets_are_bound_by_the_reduce(monoid):
    term = monoid(g(x, y) for x in xs() for y in ys(x))
    assert not set(targets_of(term)) & fvsof(term)
    assert {g, xs, ys} <= fvsof(term)


def test_targets_are_fresh_per_desugaring():
    (first,) = targets_of(Sum(x for x in [1, 2, 3]))
    (second,) = targets_of(Sum(x for x in [1, 2, 3]))
    assert first is not second


def test_a_symbolic_stream_is_unwrapped_from_its_iterator():
    """Creating the generator applied ``iter``; the stream is the iterable."""
    (stream,) = streams_of(Sum(f(x) for x in xs())).values()
    assert syntactic_eq(stream, xs())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_a_reduce_over_a_symbolic_stream_stays_symbolic(monoid):
    assert isinstance(reduce_concretely(monoid(f(x) for x in xs())), Term)


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_a_symbolic_stream_reduces_once_it_is_bound(monoid):
    term = monoid(f(x) for x in xs())
    with handler(CONCRETE), handler({xs: lambda: [1, 2, 3], f: lambda i: i * 3}):
        assert evaluate(term) == fold(monoid, [3, 6, 9])


# ============================================================================
# UNSUPPORTED INPUT
# ============================================================================


def _started_generator():
    comprehension = (x for x in [1, 2, 3])
    next(comprehension)
    return comprehension


def _async_comprehension():
    import asyncio

    async def counter():
        yield 1

    async def build():
        return (x async for x in counter())

    return asyncio.run(build())


REJECTED = [
    pytest.param(lambda: [1, 2, 3], AssertionError, "must be a generator", id="a-list"),
    pytest.param(
        lambda: [x for x in range(3)],
        AssertionError,
        "must be a generator",
        id="a-list-comprehension",
    ),
    pytest.param(_started_generator, AssertionError, None, id="a-started-generator"),
    pytest.param(
        _async_comprehension, AssertionError, None, id="an-async-comprehension"
    ),
    pytest.param(
        lambda: (len(rest) for first, *rest in [(1, 2, 3)]),
        NotImplementedError,
        "Unsupported loop target",
        id="a-starred-target",
    ),
    pytest.param(
        lambda: (x * y for x in range(3) for y in [z for z in range(x)]),
        TypeError,
        "__index__ returned non-int",
        id="an-eager-comprehension-over-a-symbolic-stream",
    ),
    pytest.param(
        lambda: (len([y for y in range(x)]) for x in range(4)),
        TypeError,
        "__index__ returned non-int",
        id="an-eager-comprehension-depending-on-a-target",
    ),
    pytest.param(
        lambda: (len(list(y for y in range(x))) for x in range(4)),
        NotImplementedError,
        "cannot consume a comprehension over a symbolic iterable",
        id="an-eager-consumer-of-a-dependent-generator",
    ),
    pytest.param(
        lambda: (x * y for x in range(3) for y in (z for z in range(x))),
        NotImplementedError,
        "cannot itself range over a symbolic iterable",
        id="a-generator-stream-over-a-symbolic-stream",
    ),
]


@pytest.mark.parametrize("argument,exception,match", REJECTED)
@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_rejects(argument, exception, match, monoid):
    with pytest.raises(exception, match=match):
        monoid(argument())


def test_equality_on_a_non_numeric_element_is_not_symbolic():
    """A hazard inherited from the term classes, not introduced here.

    ``__eq__`` on a non-numeric term answers ``False`` rather than building a
    term, so a filter comparing such elements keeps nothing. Numeric streams --
    what a monoid reduces -- are unaffected.
    """
    c = defop(str, name="c")
    assert (c() == "a") is False
    assert reduce_concretely(Sum(1 for c in "abc" if c == "a")) == 0


def test_streams_passed_by_reference_do_not_accumulate():
    from effectful.internals.comprehension import _OPAQUE_VALUES

    before = len(_OPAQUE_VALUES)
    for _ in range(5):
        Sum(f(x) for x in xs())
    assert len(_OPAQUE_VALUES) == before


def test_a_monoid_with_a_zero_desugars_the_same_way():
    assert isinstance(Product, Monoid)
    assert reduce_concretely(Product(x for x in range(1, 5))) == 24
