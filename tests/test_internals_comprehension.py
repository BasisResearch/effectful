"""Tests for desugaring generator expressions into :meth:`Monoid.reduce`.

Comprehensions are spelled as thunks rather than as generator objects: a
generator survives being desugared, but not being reduced twice, and a
parametrized case is built once and shared by every monoid it runs under.
"""

import ast
import builtins
import collections.abc
import enum
import functools
import itertools
import keyword
import math
import operator
import typing
from collections.abc import Iterable, Mapping

import pytest

from effectful.internals.comprehension import (
    annotation_of,
    element_type,
    reduce_to_comprehension,
)
from effectful.ops.monoid import (  # noqa: F401 -- named by the reduction sources
    And,
    ArgMax,
    ArgMin,
    CartesianProduct,
    EvaluateIntp,
    Factor,
    LogSumExp,
    Max,
    Min,
    Monoid,
    NormalizeIntp,
    Or,
    Product,
    ReduceDisequalityMask,
    ReduceEqualityMaskRange,
    ReduceFusion,
    ReduceMaskHoist,
    ReduceSplit,
    Sum,
    Union,
    distributes_over,
    is_commutative,
)
from effectful.ops.semantics import coproduct, evaluate, fvsof, handler, typeof
from effectful.ops.syntax import (  # noqa: F401 -- `as_dict` names a row
    as_dict,
    defdata,
    defop,
    ite,
    range_,
    syntactic_eq,
)
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
def A(i: int, j: int) -> int:
    raise NotHandled


@defop
def B(i: int, j: int) -> int:
    raise NotHandled


@defop
def scaled(i: int, scale: int = 1) -> int:
    raise NotHandled


@defop
def q(i: int) -> bool:
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
EMPTY_PAIRS: list[tuple[int, int]] = []

# Structured streams, the shape a comprehension usually reads real data in.
RECORDS = [{"k": 1, "v": 10}, {"k": 2, "v": 20}, {"k": 3, "v": 30}]
LOOKUP = {0: 1, 1: 2, 2: 3}
GRID = [[1, 2, 3], [4, 5, 6]]


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


def _heterogeneous_pair(monoid, t):
    from effectful.internals.comprehension import _project

    return monoid.reduce(
        _project(t(), 0, int) * _project(t(), 1, str), {t: ((1, "a"), (2, "b"))}
    )


def _summed_symbolic_stream(monoid, x):
    element = defop(int, name="element")
    return monoid.reduce(Sum.reduce(element(), {element: xs()}) * f(x()), {x: (1, 2)})


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
            M.mask(f(x()), And.plus(p(x()), p(x() + 1))), {x: (1, 2)}
        ),
        id="filter-clauses-conjoin",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if p(x) and p(x + 1)),
        lambda M, x: M.reduce(
            M.mask(f(x()), And.plus(p(x()), p(x() + 1))), {x: (1, 2)}
        ),
        id="and-becomes-a-conjunction",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if p(x) or p(x + 1)),
        lambda M, x: M.reduce(M.mask(f(x()), Or.plus(p(x()), p(x() + 1))), {x: (1, 2)}),
        id="or-becomes-a-disjunction",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if not p(x)),
        lambda M, x: M.reduce(M.mask(f(x()), ite(p(x()), False, True)), {x: (1, 2)}),
        id="not-becomes-a-conditional",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if not x == w()),
        lambda M, x: M.reduce(M.mask(f(x()), x() != w()), {x: (1, 2)}),
        id="not-becomes-the-complementary-comparison",
    ),
    pytest.param(
        lambda: (f(x) if p(x) else g(x, x) for x in (1, 2)),
        lambda M, x: M.reduce(ite(p(x()), f(x()), g(x(), x())), {x: (1, 2)}),
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
            M.mask(g(x(), y()), And.plus(p(x()), p(y()))), {x: (1, 2), y: ys(x())}
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
        lambda: (f(x) if p(x) else 0 for x in (1, 2)),
        lambda M, x: M.reduce(ite(p(x()), f(x()), 0), {x: (1, 2)}),
        id="a-conditional-stands-in-for-a-mask",
    ),
    pytest.param(
        lambda: (f(x) + g(x, x) for x in (1, 2) if p(x) or x > 2),
        lambda M, x: M.reduce(
            M.mask(f(x()) + g(x(), x()), Or.plus(p(x()), x() > 2)), {x: (1, 2)}
        ),
        id="a-symbolic-filter-beside-a-comparison",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2, 3) if x in (1, 3)),
        lambda M, x: M.reduce(
            M.mask(f(x()), Or.plus(x() == 1, x() == 3)), {x: (1, 2, 3)}
        ),
        id="membership-becomes-a-disjunction",
    ),
    pytest.param(
        lambda: (max(f(x), 2) for x in (1, 2)),
        lambda M, x: M.reduce(Max.plus(f(x()), 2), {x: (1, 2)}),
        id="max-becomes-its-monoid-addition",
    ),
    pytest.param(
        lambda: (min([f(x), h(x)]) for x in (1, 2)),
        lambda M, x: M.reduce(Min.plus(f(x()), h(x())), {x: (1, 2)}),
        id="min-of-a-list-becomes-its-monoid-addition",
    ),
    pytest.param(
        lambda: (f(x) for x in (1, 2) if all([p(x), x > 0])),
        lambda M, x: M.reduce(M.mask(f(x()), And.plus(p(x()), x() > 0)), {x: (1, 2)}),
        id="all-of-a-list-becomes-its-monoid-addition",
    ),
    pytest.param(
        lambda: (sum(xs()) * f(x) for x in (1, 2)),
        _summed_symbolic_stream,
        id="a-builtin-over-a-symbolic-stream-becomes-a-reduce",
    ),
    pytest.param(
        lambda: (x * 2 for x in EMPTY),
        lambda M, x: M.reduce(x() * 2, {x: []}),
        id="an-empty-stream-still-binds-a-target",
    ),
    pytest.param(
        lambda: (t[0] * t[1] for t in ((1, "a"), (2, "b"))),
        _heterogeneous_pair,
        id="a-heterogeneous-element-is-projected-per-component",
    ),
    pytest.param(
        lambda: (f(y) for x in xs() for y in list(ys(x))),
        lambda M, x, y: M.reduce(f(y()), {x: xs(), y: ys(x())}),
        id="a-listed-stream-is-the-stream",
    ),
    pytest.param(
        lambda: (f(y) for x in xs() for y in map(abs, ys(x))),
        lambda M, x, y: M.reduce(f(abs(y())), {x: xs(), y: ys(x())}),
        id="a-mapped-stream-is-what-the-target-stands-for",
    ),
    pytest.param(
        lambda: (f(y) for x in xs() for y in filter(p, ys(x))),
        lambda M, x, y: M.reduce(M.mask(f(y()), p(y())), {x: xs(), y: ys(x())}),
        id="a-filtered-stream-is-a-filter-clause",
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
    pytest.param(
        lambda: (y for x in range(3) for y in list(range(x))), id="a-listed-stream"
    ),
    pytest.param(
        lambda: (y for x in range(3) for y in tuple(range(x))), id="a-tupled-stream"
    ),
    pytest.param(
        lambda: (y for x in range(4) for y in filter(lambda v: v % 2, range(x))),
        id="a-filtered-dependent-stream",
    ),
    pytest.param(
        lambda: (y for x in range(4) for y in filter(None, range(x))),
        id="a-truthiness-filtered-dependent-stream",
    ),
    pytest.param(
        lambda: (y for x in range(4) for y in map(abs, range(x))),
        id="a-mapped-dependent-stream",
    ),
    pytest.param(
        lambda: (y for x in range(4) for y in list(filter(lambda v: v % 2, range(x)))),
        id="a-listed-filtered-dependent-stream",
    ),
    pytest.param(
        lambda: (
            y for x in range(5) for y in filter(lambda v: v > 1, map(abs, range(x)))
        ),
        id="a-filtered-mapped-dependent-stream",
    ),
    pytest.param(
        lambda: (
            y
            for x in range(6)
            for y in filter(lambda v: v != 2, filter(lambda v: v % 2, range(x)))
        ),
        id="two-filters-on-a-dependent-stream",
    ),
    pytest.param(
        lambda: (
            a * b for x in range(3) for a, b in map(lambda v: (v, v + 1), range(x))
        ),
        id="a-mapped-stream-unpacked",
    ),
    pytest.param(
        lambda: (
            y for x in range(4) for y in filter(lambda v: v % 2, range(x)) if y > 1
        ),
        id="a-filtered-dependent-stream-with-a-filter-clause",
    ),
    pytest.param(lambda: (x for x in filter(None, [0, 1, 2])), id="filter"),
    pytest.param(lambda: (x for x in (y for y in range(5) if y % 2)), id="generator"),
    pytest.param(lambda: (x for x in [y for y in range(5) if y % 2]), id="list-comp"),
    pytest.param(lambda: (x for x in {y for y in range(5) if y % 2}), id="set-comp"),
    pytest.param(lambda: (x for x in {y: y for y in range(3)}), id="dict-comp"),
    pytest.param(lambda: (x * 2 for x in EMPTY), id="empty-with-arithmetic"),
    pytest.param(lambda: (x for x in EMPTY if x > 1), id="empty-with-a-filter"),
    pytest.param(
        lambda: (x * y for x in EMPTY for y in range(3)), id="empty-in-a-nest"
    ),
    pytest.param(
        lambda: (a * b for a, b in itertools.product(range(3), range(2))),
        id="itertools-product",
    ),
    pytest.param(
        lambda: (x for x in itertools.chain(range(3), [10, 20])), id="itertools-chain"
    ),
    pytest.param(
        lambda: (x for x in itertools.islice(range(10), 2, 6)), id="itertools-islice"
    ),
    pytest.param(
        lambda: (x for x in itertools.accumulate(range(5))), id="itertools-accumulate"
    ),
    pytest.param(
        lambda: (a * b for a, b in itertools.combinations(range(4), 2)),
        id="itertools-combinations",
    ),
    pytest.param(lambda: (x for x in [3, 1, 2].copy()), id="from-a-method-call"),
    pytest.param(lambda: (v for v in dict(a=1, b=2).values()), id="from-a-constructor"),
    pytest.param(
        lambda: (x for x in list(y * 2 for y in range(4))), id="a-listed-generator"
    ),
    pytest.param(
        lambda: (x for x in tuple(filter(lambda v: v % 2, range(8)))),
        id="a-tupled-filter",
    ),
    pytest.param(
        lambda: (x for x in sorted([5, 1, 4], reverse=True)), id="sorted-with-a-keyword"
    ),
    pytest.param(lambda: (x for x in list(range(10))[2:8:2]), id="a-slice"),
    pytest.param(lambda: (x for x in [[1, 2], [3, 4]][1]), id="an-indexed-list"),
    pytest.param(lambda: (b - 96 for b in b"abc"), id="bytes-shifted"),
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
            for x in range(16)
            if (x < 5 and x % 2 == 0) or (9 < x < 12) or (x > 13 and x % 3 == 0)
        ),
        id="three-way-disjunction",
    ),
    pytest.param(
        lambda: (
            x
            for x in range(30)
            if (x > 2 and x < 25) and not (x % 3 == 0 or (x % 5 == 0 and x > 10))
        ),
        id="deep-boolean-structure",
    ),
    pytest.param(
        lambda: (x for x in range(8) if not any(y > 4 for y in range(x))),
        id="a-negated-reduction",
    ),
    pytest.param(
        lambda: (x for x in range(8) if 0 <= sum(y for y in range(x)) <= 6),
        id="a-chained-comparison-of-reductions",
    ),
    pytest.param(
        lambda: (
            x
            for x in range(40)
            if ((x > 2 and (x < 30 or x == 35)) and not (x % 7 == 0 and x > 10))
            or x == 0
        ),
        id="deeply-nested-booleans",
    ),
    pytest.param(
        lambda: (
            x
            for x in range(12)
            if (sum(y for y in range(x)) > 3 and x % 2 == 0) or not (x > 8)
        ),
        id="reductions-mixed-with-comparisons",
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
    pytest.param(
        lambda: (
            a + b + c + d
            for a in range(3)
            for b in range(a)
            for c in range(b + 1)
            for d in range(c + 1)
        ),
        id="four-deep-and-dependent",
    ),
    pytest.param(
        lambda: (
            a * b * c
            for a in range(5)
            if a % 2
            for b in range(a)
            if b != 1
            for c in range(b + 1)
            if c < 2
        ),
        id="a-filter-at-every-level",
    ),
    pytest.param(
        lambda: (
            a * b
            for a in range(4)
            if sum(y for y in range(a)) > 1
            for b in range(a)
            if max(z for z in range(b + 1)) == b
        ),
        id="a-reduction-in-a-filter-at-every-level",
    ),
    pytest.param(
        lambda: (x + 1 for x in range(3) if x > 0 for x in range(4) if x % 2),
        id="two-loops-one-name-with-filters",
    ),
    pytest.param(
        lambda: (a * b for a in range(1, 4) for b in range(a, a + 2)),
        id="a-target-at-both-ends-of-a-range",
    ),
    pytest.param(
        lambda: (b for a in range(1, 4) for b in range(a, a * 2, 1)),
        id="a-three-argument-dependent-range",
    ),
    pytest.param(
        lambda: (a * b for a in range(4) for b in range(4) if a < b),
        id="two-targets-compared",
    ),
    pytest.param(
        lambda: (a * b for a in range(4) for b in range(4) if a == b),
        id="two-targets-equated",
    ),
    pytest.param(
        lambda: (a * b for a, b in [(1, 2), (3, 4), (5, 6)] if a > 2),
        id="a-filter-on-a-tuple-component",
    ),
    pytest.param(
        lambda: (
            a + b + c for a, b, c in [(1, 2, 3), (4, 5, 6), (7, 8, 9)] if a % 2 if c > 3
        ),
        id="a-triple-target-with-filters",
    ),
    pytest.param(
        lambda: (k * (a + b) for k, (a, b) in {1: (2, 3), 4: (5, 6)}.items()),
        id="nested-unpacking-of-dict-items",
    ),
    pytest.param(
        lambda: (
            a * b for a, b in zip((y for y in range(3)), [z + 1 for z in range(3)])
        ),
        id="a-zip-of-a-generator-and-a-list",
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
    pytest.param(
        lambda: (
            y
            for x in range(4)
            for y in (range(x) if x % 2 == 0 else (range(1) if x > 1 else range(2)))
        ),
        id="nested-as-the-iterable",
    ),
    pytest.param(
        lambda: (y for x in range(3) for y in (range(x) if x else [9])),
        id="as-the-iterable-with-differently-typed-arms",
    ),
    pytest.param(
        lambda: (
            (x if x > 2 else -x) if x % 2 else (x * 2 if x else 7) for x in range(8)
        ),
        id="in-both-arms",
    ),
    pytest.param(
        lambda: ((x if 2 < x < 8 else 0) for x in range(12) if 1 <= x <= 10),
        id="a-comparison-chain-in-both-places",
    ),
    pytest.param(
        lambda: (
            y
            for x in range(4)
            for y in (range(x) if x % 2 else (range(2) if x else range(1)))
        ),
        id="nested-in-the-iterable",
    ),
]

UNPACKING = [
    pytest.param(lambda: (a * b for a, b in [(1, 2), (3, 4)]), id="pair"),
    pytest.param(
        lambda: (a + b + c for a, b, c in [(1, 2, 3), (4, 5, 6)]), id="triple"
    ),
    pytest.param(
        lambda: (a * b + c for (a, b), c in [((1, 2), 3), ((4, 5), 6)]), id="nested"
    ),
    pytest.param(lambda: (a - b for a, b in zip(range(4), range(4, 8))), id="zip"),
    pytest.param(lambda: (i * v for i, v in enumerate([10, 20, 30])), id="enumerate"),
    pytest.param(lambda: (k * v for k, v in {1: 10, 2: 20}.items()), id="dict-items"),
    pytest.param(
        lambda: (a + b for a, b in ((x, x * 2) for x in range(3))), id="from-generator"
    ),
    pytest.param(
        lambda: (
            a * b + c * d for (a, b), (c, d) in [((1, 2), (3, 4)), ((5, 6), (7, 8))]
        ),
        id="nested-on-both-sides",
    ),
    pytest.param(
        lambda: (a * b * c for a, b in [(1, 2), (3, 4)] for c in range(a)),
        id="with-a-dependent-stream",
    ),
    pytest.param(
        lambda: (
            a - b for a, b in zip([y for y in range(4)], [z * 2 for z in range(4)])
        ),
        id="a-zip-of-comprehensions",
    ),
    pytest.param(
        lambda: (i * v for i, v in enumerate([y * y for y in range(4)])),
        id="an-enumerate-of-a-comprehension",
    ),
    pytest.param(
        lambda: (i * (a + b) for i, (a, b) in enumerate([(1, 2), (3, 4)])),
        id="a-nested-enumerate",
    ),
    pytest.param(
        lambda: (k * v for k, v in {1: 10, 2: 20, 3: 30}.items() if k != 2),
        id="filtered-dict-items",
    ),
    pytest.param(
        lambda: (i for i, c in enumerate("abcd")), id="an-enumerate-of-a-string"
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
            sum(y for y in range(x)) + max(z for z in range(1, x + 2)) for x in range(4)
        ),
        id="two-reductions",
    ),
    pytest.param(
        lambda: (sum(y * z for y in range(x) for z in range(2)) for x in range(4)),
        id="over-a-nest",
    ),
    pytest.param(
        lambda: (
            sum(y for y in range(x)) * max(z for z in range(1, x + 2))
            - min(w for w in range(1, x + 2))
            for x in range(4)
        ),
        id="three-combined-arithmetically",
    ),
    pytest.param(
        lambda: (sum(y for y in {z % 3 for z in range(7)}) * x for x in range(3)),
        id="over-a-set-comprehension",
    ),
    pytest.param(
        lambda: (sum(k * v for k, v in {1: 2, 3: 4}.items()) * x for x in range(3)),
        id="over-dict-items",
    ),
    pytest.param(
        lambda: (
            sum(y for y in [z for w in range(2) for z in range(w + 1)]) * x
            for x in range(3)
        ),
        id="over-a-nested-list-comprehension",
    ),
    pytest.param(
        lambda: (
            (sum(y for y in range(x)), max(z for z in range(1, x + 2)))[0]
            for x in range(4)
        ),
        id="in-a-tuple-that-is-indexed",
    ),
    pytest.param(
        lambda: ({"s": sum(y for y in range(x))}["s"] for x in range(4)),
        id="in-a-dict-that-is-indexed",
    ),
    pytest.param(
        lambda: (
            sum(
                sum(sum(a * b * c for c in range(2)) for b in range(2))
                for a in range(x)
            )
            for x in range(4)
        ),
        id="four-deep",
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
    pytest.param(
        # The binding is deliberately unused: what matters is that a walrus in
        # a filter is reconstructed at all.
        lambda: (x for x in range(8) if (y := x % 3) == 0),  # noqa: F841
        id="in-a-filter",
    ),
    pytest.param(
        lambda: (y for x in range(5) if (y := x * 2) > 2), id="bound-by-a-filter"
    ),
    pytest.param(
        lambda: ((y := a * 2) + b + y for a in range(3) for b in range(a)),
        id="in-a-nested-loop",
    ),
    pytest.param(
        lambda: (y for x in range(6) if (y := x * 2) > 2 if y < 9),
        id="read-by-a-later-filter",
    ),
    pytest.param(
        lambda: (len([(z := w * 2) + z for w in range(3)]) for x in range(3)),
        id="in-an-inner-comprehension",
    ),
]

# These must stay on one line: Python 3.12's `dis` mis-reports jumps for
# multiline comprehensions, which the disassembler suite covers directly.
STRESS = [
    pytest.param(
        lambda: (
            x + y
            for x in range(10)
            if x % 2 == 0
            if x > 2
            for y in range(10)
            if y % 3 == 0
            if y < x
        ),
        id="many-filters",
    ),  # fmt: skip
    pytest.param(
        lambda: (
            len([y if y > 1 else -y for y in range(3)])
            for x in range(4)
            if (x if x % 2 == 1 else x % 2 == 0)
        ),
        id="nested-ternary",
    ),  # fmt: skip
    pytest.param(
        lambda: (
            sum(y * z for z in range(y))
            for x in range(4)
            for y in range(x)
            if y % 2 == 0 or y == 1
        ),
        id="reduction-in-a-nest",
    ),  # fmt: skip
]

PROGRAMS = [
    pytest.param(
        lambda: (
            (a * b + c if c % 2 else a - b)
            for a in range(4)
            if a % 2 == 0
            for b in range(a)
            if b != 1
            for c in range(b + 2)
            if c < 3
        ),
        id="three-deep-with-filters-and-conditionals",
    ),
    pytest.param(
        lambda: (
            (lambda v: v + (t := v * 2) + t)(x) + len([y for y in range(3)])
            for x in range(4)
        ),
        id="a-walrus-in-a-lambda-beside-a-comprehension",
    ),
    pytest.param(
        lambda: (
            x
            for x in range(60)
            if x > 1
            if x % 2
            if x % 3
            if x % 5
            if x < 55
            if x != 49
        ),
        id="a-long-filter-chain",
    ),
    pytest.param(
        lambda: (i * sum(y for y in range(v)) for i, v in enumerate([1, 2, 3])),
        id="unpacking-around-a-reduction",
    ),
    pytest.param(
        lambda: (y for x in range(3) for y in range(sum(z for z in range(x)))),
        id="a-range-of-a-reduction",
    ),
    pytest.param(
        lambda: (a * b + c for (a, b) in [(1, 2), (3, 4)] for c in range(a + b)),
        id="unpacking-with-a-stream-built-from-both-parts",
    ),
    pytest.param(
        lambda: (
            (lambda v: v * 2)(a) + b + (c := a * b) + c
            for a in range(1, 4)
            if a in (1, 3)
            for b in range(a, a + 2)
            if b % 2 == 0 or b > 3
        ),
        id="every-clause-at-once",
    ),
]

MULTILINE = [
    pytest.param(
        lambda: (x * y for x in range(4) if x % 2 == 0 for y in range(x) if y > 0),
        id="split-generators",
    ),
    pytest.param(
        lambda: (x if x > 2 else -x for x in range(6)),
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


class Color(enum.Enum):
    """An enumeration is iterated by a generator that closes over the class."""

    RED = 1
    BLUE = 2


@defop
def rank(c: Color) -> int:
    raise NotHandled


PLATE = CartesianProduct(range(2) for t in range(2))


class Holder:
    """An object a comprehension may be written inside a method of."""

    scale = 3

    def __init__(self, n: int):
        self.n = n

    def over_an_attribute(self):
        return (x * self.n for x in range(4))

    def over_a_class_attribute(self):
        return (x * self.scale for x in range(4))

    @staticmethod
    def over_nothing():
        return (x * 2 for x in range(4))


HOLDER = Holder(5)

GLOBAL_SCALE = 7

DOUBLE = functools.partial(lambda a, b: a * b, 2)


def _two_levels_of_closure():
    def outer(n):
        def inner(m):
            return (x * n * m for x in range(3))

        return inner(2)

    return outer(4)


def _a_default_argument_stream(values=(1, 2, 3)):
    return (v * 2 for v in values)


def _a_module_global():
    return (x * GLOBAL_SCALE for x in range(4))


def _a_captured_lambda_stream():
    build = lambda n: list(range(n))  # noqa: E731
    return (y for x in range(3) for y in build(x))


# Where a comprehension is written decides what its names mean.
SCOPES = [
    pytest.param(HOLDER.over_an_attribute, id="an-attribute-of-self"),
    pytest.param(HOLDER.over_a_class_attribute, id="a-class-attribute"),
    pytest.param(Holder.over_nothing, id="a-static-method"),
    pytest.param(_two_levels_of_closure, id="two-levels-of-closure"),
    pytest.param(_a_default_argument_stream, id="a-defaulted-stream"),
    pytest.param(_a_module_global, id="a-module-global"),
    pytest.param(lambda: (DOUBLE(x) for x in range(4)), id="a-partial"),
    pytest.param(
        lambda: ((lambda v, u=[y for y in range(3)]: v + len(u))(x) for x in range(4)),  # type: ignore[assignment]
        id="a-comprehension-in-a-default",
    ),
    pytest.param(
        lambda: (abs(min(max(x, 1), 3)) for x in range(6)), id="a-chain-of-calls"
    ),
    pytest.param(
        lambda: (y for x in range(3) for y in (lambda n: list(range(n)))(x)),
        id="a-lambda-building-a-stream",
    ),
    pytest.param(_a_captured_lambda_stream, id="a-captured-lambda-building-a-stream"),
    pytest.param(
        lambda: (
            y
            for x in range(4)
            for y in (lambda n: filter(lambda v: v % 2, range(n)))(x)
        ),
        id="a-lambda-building-a-filtered-stream",
    ),
]

DISPATCH = [
    pytest.param(_shadowed_reduction, id="shadowed-reduction"),
    pytest.param(_shadowed_stream_constructor, id="shadowed-stream-constructor"),
    pytest.param(_aliased_reduction, id="aliased-reduction"),
    pytest.param(_aliased_stream_constructor, id="aliased-stream-constructor"),
    pytest.param(_closure_variable, id="closure-variable"),
    pytest.param(
        lambda: (sum([1, 2, 3], x) for x in [10, 20]), id="reduction-with-start"
    ),
]

# Streams whose elements are themselves containers.
STRUCTURED = [
    pytest.param(lambda: (r["v"] * 2 for r in RECORDS if r["k"] != 2), id="a-record"),
    pytest.param(
        lambda: (sum(y for y in range(r["k"])) for r in RECORDS),
        id="a-record-in-a-reduction",
    ),
    pytest.param(
        lambda: (a * b for row in GRID for a in row for b in row if a != b),
        id="a-row-looped-over-twice",
    ),
    pytest.param(
        lambda: (r["k"] * r["v"] for r in RECORDS if r["v"] in (10, 30)),
        id="a-record-tested-for-membership",
    ),
    pytest.param(
        lambda: (max(row) - min(row) for row in GRID), id="a-row-reduced-by-builtins"
    ),
]


# ``max`` and ``min`` reduce as well, without a comprehension to reduce over.
BUILTIN_FOLDS = [
    pytest.param(lambda: (max(x, 2) for x in range(6)), id="two-argument-max"),
    pytest.param(lambda: (min(x, 2) for x in range(6)), id="two-argument-min"),
    pytest.param(lambda: (max(x, 2, x * 3) for x in range(6)), id="three-argument-max"),
    pytest.param(lambda: (max([x, 2, 5]) for x in range(6)), id="max-of-a-list"),
    pytest.param(lambda: (min((x, 2, 5)) for x in range(6)), id="min-of-a-tuple"),
    pytest.param(lambda: (sum([x, 2, 5]) for x in range(6)), id="sum-of-a-list"),
    pytest.param(
        lambda: (x for x in range(6) if any([x > 3, x == 0])), id="any-of-a-list"
    ),
    pytest.param(
        lambda: (x for x in range(6) if all([x >= 0, x != 3])), id="all-of-a-list"
    ),
]


# ``in`` is the one comparison a term does not already answer symbolically, so
# it is asked of each element of the container instead.
MEMBERSHIP = [
    pytest.param(lambda: (x for x in range(6) if x in (1, 3, 5)), id="a-tuple"),
    pytest.param(lambda: (x for x in range(6) if x not in (1, 3, 5)), id="not-in"),
    pytest.param(lambda: (x for x in range(6) if x in [0, 2, 4]), id="a-list"),
    pytest.param(lambda: (x for x in range(6) if x in {2, 3}), id="a-set"),
    pytest.param(lambda: (x for x in range(6) if x in {1: "a", 4: "b"}), id="a-dict"),
    pytest.param(lambda: (x for x in range(10) if x in range(3, 7)), id="a-range"),
    pytest.param(
        lambda: (x for x in range(8) if x in [y * 2 for y in range(3)]),
        id="a-comprehension",
    ),
    pytest.param(
        lambda: (x for x in range(10) if 0 < x in (1, 3, 5)), id="chained-with-an-order"
    ),
    pytest.param(
        lambda: (x for x in range(10) if x in (1, 3, 5, 7) and x > 2),
        id="in-a-conjunction",
    ),
    pytest.param(
        lambda: ((x * 2 if x in (1, 3) else x) for x in range(6)),
        id="in-a-conditional",
    ),
    pytest.param(lambda: (x for x in range(6) if not x in (1, 3)), id="negated"),  # noqa: E713
    pytest.param(
        lambda: (x for x in range(6) if x % 3 in (0, 2)), id="of-a-computed-value"
    ),
    pytest.param(
        lambda: (a * b for a in range(4) for b in range(4) if a in (b, b + 1)),
        id="between-two-targets",
    ),
    pytest.param(lambda: (x for x in range(6) if 2 in (1, 2, 3)), id="wholly-concrete"),
]


# What a body may be built out of besides arithmetic: calls with every shape of
# argument list, container displays, subscripts, attributes and methods.
CALLS_AND_DISPLAYS = [
    pytest.param(
        lambda: (sorted([3, 1, 2], reverse=True)[0] * x for x in range(4)),
        id="a-builtin-with-a-keyword",
    ),
    pytest.param(
        lambda: (max(*[1, 5, 3]) * x for x in range(4)), id="starred-constants"
    ),
    pytest.param(
        lambda: (len(set(y % 3 for y in range(5))) * x for x in range(4)),
        id="an-eager-consumer-of-a-concrete-generator",
    ),
    pytest.param(lambda: ((x, x * 2)[1] for x in range(4)), id="an-indexed-tuple"),
    pytest.param(
        lambda: ({"a": x, "b": x * 2}["b"] for x in range(4)), id="an-indexed-dict"
    ),
    pytest.param(lambda: (len([x, x + 1, x + 2]) * x for x in range(4)), id="a-list"),
    pytest.param(lambda: (x for x in range(6) if len({1, 2, 3}) > 2), id="a-set"),
    pytest.param(
        lambda: (sum([1, 2, 3, 4][1:3]) * x for x in range(3)), id="a-sliced-list"
    ),
    pytest.param(lambda: ((3 + 0j).real * x for x in range(4)), id="an-attribute"),
    pytest.param(lambda: (len("abc".upper()) * x for x in range(4)), id="a-method"),
    pytest.param(
        lambda: (math.gcd(12, 8) * x for x in range(4)), id="a-module-function"
    ),
    pytest.param(
        lambda: (len({x: x * 2}) for x in range(4)), id="a-dict-keyed-by-a-target"
    ),
    pytest.param(lambda: (len({x, x + 1}) for x in range(4)), id="a-set-of-targets"),
    pytest.param(lambda: (--x + ~-x for x in range(5)), id="chained-unary-operators"),
    pytest.param(
        lambda: ((x + 1) * (x - 1) // 2 % 5**2 & 7 | 1 ^ 2 for x in range(1, 8)),
        id="every-arithmetic-operator",
    ),
    pytest.param(
        lambda: (sum(g for g in (y * 2 for y in range(3))) * x for x in range(3)),
        id="a-reduction-over-a-generator",
    ),
    pytest.param(
        lambda: (y for x in range(3) for y in (lambda: [1, 2])()),
        id="a-stream-from-a-lambda",
    ),
]


# Comprehensions built out of other comprehensions, in every position one may
# appear in: as a stream, as a body, and inside a filter.
COMPREHENSION_POSITIONS = [
    pytest.param(lambda: (x for x in [y * 2 for y in range(4)]), id="list-as-stream"),
    pytest.param(lambda: (x for x in {y * 2 for y in range(4)}), id="set-as-stream"),
    pytest.param(
        lambda: (x for x in {y: y * 2 for y in range(4)}.values()),
        id="dict-values-as-stream",
    ),
    pytest.param(
        lambda: (x for x in sorted({y % 3 for y in range(7)})),
        id="sorted-set-as-stream",
    ),
    pytest.param(
        lambda: (x * len([y for y in range(3) if y]) for x in range(4)),
        id="filtered-list-in-the-body",
    ),
    pytest.param(
        lambda: (len([y for y in [z for z in range(4)] if y > 1]) for x in range(3)),
        id="list-of-a-list-in-the-body",
    ),
    pytest.param(
        lambda: (len({y: [z for z in range(y)] for y in range(3)}) for x in range(3)),
        id="list-inside-a-dict-in-the-body",
    ),
    pytest.param(
        lambda: (x for x in range(5) if len({y for y in range(3)}) > 2),
        id="set-in-a-filter",
    ),
    pytest.param(
        lambda: (sum(y for y in [z for z in range(3)]) for x in range(3)),
        id="list-as-an-inner-reduction-stream",
    ),
    pytest.param(
        lambda: (sum(y * z for y in range(2) for z in range(2)) for x in range(3)),
        id="inner-reduction-over-a-nest",
    ),
    pytest.param(
        lambda: (sum(sum(z for z in range(y)) for y in range(x)) for x in range(4)),
        id="inner-reduction-in-an-inner-reduction",
    ),
    pytest.param(
        lambda: (any(y > 1 for y in range(x)) for x in range(5)),
        id="boolean-inner-reduction",
    ),
    pytest.param(
        lambda: (all(y > 1 for y in range(1, x)) for x in range(5)),
        id="boolean-inner-reduction-all",
    ),
    pytest.param(
        lambda: (max(y for y in range(1, x + 2)) for x in range(4)), id="inner-max"
    ),
    pytest.param(
        lambda: (min(y for y in range(1, x + 2)) for x in range(4)), id="inner-min"
    ),
]

# A lambda is a binder that a comprehension may contain anywhere; each of these
# puts one somewhere different.
LAMBDA_POSITIONS = [
    pytest.param(
        lambda: ((lambda a, b: a * b)(x, x + 1) for x in range(4)), id="two-arguments"
    ),
    pytest.param(
        lambda: ((lambda a: lambda b: a + b)(x)(2) for x in range(4)), id="curried"
    ),
    pytest.param(
        lambda: (x for x in range(6) if (lambda v: v % 2 == 0)(x)), id="in-a-filter"
    ),
    pytest.param(
        lambda: (sum((lambda v: v * 2)(y) for y in range(x)) for x in range(4)),
        id="in-an-inner-reduction",
    ),
    pytest.param(
        lambda: ((lambda v: v if v > 1 else 0)(x) + 1 for x in range(4)),
        id="conditional-body",
    ),
    pytest.param(
        lambda: ((lambda *vs, **kw: sum(vs) + kw["e"])(x, x, e=1) for x in range(4)),
        id="variadic-and-keyword",
    ),
    pytest.param(
        lambda: ((lambda a, *, b=2: a * b)(x) for x in range(4)),  # type: ignore[assignment]
        id="keyword-only-default",
    ),
    pytest.param(
        lambda: ((lambda a, /, b=1: a + b)(x) for x in range(4)),  # type: ignore[assignment]
        id="positional-only",
    ),
    pytest.param(
        lambda: (sum((lambda v: v + y)(x) for y in range(2)) for x in range(3)),
        id="closing-over-an-inner-target",
    ),
    pytest.param(
        lambda: ((lambda a: lambda b: a * b)(x)(x + 1) for x in range(4)),
        id="returning-a-lambda-over-the-target",
    ),
    pytest.param(
        lambda: ((lambda v: sum(w for w in range(3)) + v)(x) for x in range(4)),
        id="with-a-reduction-in-its-body",
    ),
    pytest.param(
        lambda: ((lambda v: v * 2)(sum(y for y in range(x))) for x in range(4)),
        id="applied-to-a-reduction",
    ),
    pytest.param(
        lambda: ((lambda v, u=sum(w for w in range(3)): v + u)(x) for x in range(4)),  # type: ignore[assignment,misc]
        id="defaulted-from-a-reduction",
    ),
    pytest.param(
        lambda: ((lambda x: (lambda x: x + 1)(x * 2))(x) for x in range(4)),
        id="shadowing-inside-a-lambda",
    ),
    pytest.param(
        lambda: ((lambda a, b=2, *, c=3: a * b + c)(x, c=x) for x in range(4)),  # type: ignore[assignment]
        id="called-with-a-keyword",
    ),
    pytest.param(
        lambda: ((lambda **kw: kw["a"] * kw["b"])(a=x, b=x + 1) for x in range(4)),
        id="called-with-only-keywords",
    ),
    pytest.param(
        lambda: ((lambda *vs: sum(vs))(*[x, x + 1]) for x in range(4)),  # type: ignore[arg-type]
        id="called-with-a-starred-list",
    ),
    pytest.param(
        lambda: (sum((lambda x: x * 2)(y) for y in range(x)) for x in range(4)),
        id="shadowing-inside-an-inner-reduction",
    ),
]

# What a term cannot do, reached through a comprehension. Each of these fails
# the same way written out as a reduction, so the gap is in the term classes.
BEYOND_A_TERM = [
    pytest.param(
        # The disassembler mis-reads a conditional jump whose target needs an
        # `EXTENDED_ARG`, which a chain of about a dozen filters reaches.
        lambda: (
            f(a)
            for a in xs()
            if p(a + 0)
            if p(a + 1)
            if p(a + 2)
            if p(a + 3)
            if p(a + 4)
            if p(a + 5)
            if p(a + 6)
            if p(a + 7)
            if p(a + 8)
            if p(a + 9)
            if p(a + 10)
            if p(a + 11)
        ),
        id="a-dozen-filter-clauses",
        marks=pytest.mark.xfail(raises=AssertionError, strict=True),
    ),
    pytest.param(
        lambda: (divmod(x, 3)[0] for x in range(9)),
        id="divmod-of-an-element",
        marks=pytest.mark.xfail(raises=TypeError, strict=True),
    ),
    pytest.param(
        lambda: (round(x) for x in range(5)),
        id="rounding-an-element",
        marks=pytest.mark.xfail(raises=TypeError, strict=True),
    ),
    pytest.param(
        # The disassembler cannot merge the branches of a conditional in the
        # body with those of one in a later iterable.
        lambda: (
            (y * 2 if y % 2 else y)
            for x in range(4)
            for y in (range(x) if x % 2 else range(1))
        ),
        id="conditionals-in-the-body-and-in-a-later-stream",
        marks=pytest.mark.xfail(raises=ValueError, strict=True),
    ),
    pytest.param(
        lambda: (pow(x, 2, 7) for x in range(6)),
        id="three-argument-pow",
        marks=pytest.mark.xfail(raises=TypeError, strict=True),
    ),
    pytest.param(
        lambda: (len(c) for c in ["a", "bb", "ccc"]),
        id="the-length-of-an-element",
        marks=pytest.mark.xfail(raises=NotImplementedError, strict=True),
    ),
]


# A comprehension may name a monoid where a builtin reduction would do. Python
# evaluates such a call by desugaring it, so these describe a reduction.
# Streams whose elements only an operation can read, so the body is a term.
SYMBOLIC_BODIES = [
    pytest.param(lambda: (rank(c) for c in Color), id="an-enumeration"),
]


MONOID_CALLS = [
    pytest.param(
        lambda: (Sum(a * b for b in range(3) if a != b) for a in range(3)),
        id="a-contraction-with-a-mask",
    ),
    pytest.param(
        lambda: (
            Sum(y for y in range(x) if y % 2 == 0)  # type: ignore[operator]
            + Max(z for z in range(1, x + 2) if z != 2)
            for x in range(4)
        ),
        id="two-filtered-monoid-calls",
    ),
    pytest.param(
        lambda: ((Sum if x % 2 else Max)(y for y in range(1, x + 2)) for x in range(4)),
        id="a-conditionally-chosen-monoid",
    ),
    pytest.param(
        lambda: (
            sum(Sum(z for z in range(y + 1)) for y in range(x))  # type: ignore[misc]
            for x in range(4)
        ),
        id="a-monoid-call-under-a-builtin-reduction",
    ),
    pytest.param(
        lambda: (
            Sum(A(i, j) * B(j, k) for j in range(2)) for i in range(2) for k in range(2)
        ),
        id="a-matrix-product",
    ),
    pytest.param(
        lambda: (scaled(x, scale=2) for x in range(4)), id="a-keyword-argument"
    ),
    pytest.param(lambda: (scaled(x) for x in range(4)), id="a-defaulted-argument"),
    pytest.param(
        lambda: (
            Sum(
                Max(Min(a * b * c for c in range(1, 3)) for b in range(1, 3))
                for a in range(1, 3)
            )
            for x in range(2)
        ),
        id="monoid-calls-nested-three-deep",
    ),
    pytest.param(
        lambda: (
            sum(Max(y * z for z in range(1, 3)) for y in range(x))  # type: ignore[misc]
            for x in range(4)
        ),
        id="a-monoid-call-inside-a-builtin-reduction",
    ),
    pytest.param(
        lambda: (Sum(a * b for a in range(x) for b in range(a + 1)) for x in range(4)),
        id="a-monoid-call-over-a-dependent-nest",
    ),
    pytest.param(
        lambda: (
            Sum(y for y in range(x)) if x % 2 else Max(y for y in range(1, x + 2))
            for x in range(5)
        ),
        id="a-monoid-call-in-a-conditional-arm",
    ),
    pytest.param(
        lambda: (
            Sum(y for y in range(x)) * 2  # type: ignore[operator]
            - Min(z for z in range(1, x + 2))
            for x in range(4)
        ),
        id="arithmetic-on-monoid-calls",
    ),
]

PLAIN_SHAPES = [
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
    *COMPREHENSION_POSITIONS,
    *LAMBDA_POSITIONS,
    *MEMBERSHIP,
    *STRUCTURED,
    *BUILTIN_FOLDS,
    *PROGRAMS,
    *SCOPES,
    *CALLS_AND_DISPLAYS,
    *BEYOND_A_TERM,
]
"""Comprehensions plain Python evaluates to a number, a bool or a string."""

ALL_SHAPES = [*PLAIN_SHAPES, *MONOID_CALLS, *SYMBOLIC_BODIES]


@pytest.mark.parametrize("comprehension", PLAIN_SHAPES)
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
        Sum,
        lambda: ({"a": x, "b": x * 2} for x in range(4)),
        {"a": 6, "b": 12},
        id="mapping",
    ),
    pytest.param(
        Union,
        lambda: ([{"x": v}] for v in range(3)),
        [{"x": 0}, {"x": 1}, {"x": 2}],
        id="union-of-rows",
    ),
]


@pytest.mark.parametrize("monoid,comprehension,expected", NON_SCALAR_BODIES)
def test_a_non_scalar_body_reduces_pointwise(monoid, comprehension, expected):
    """``MonoidOverSequence`` and ``MonoidOverMapping`` see the desugared body."""
    assert reduce_concretely(monoid(comprehension())) == expected


PLATES = [
    pytest.param(
        lambda: CartesianProduct(range(2) for t in range(2)),
        [
            {(0,): 0, (1,): 0},
            {(0,): 0, (1,): 1},
            {(0,): 1, (1,): 0},
            {(0,): 1, (1,): 1},
        ],
        id="a-stream-of-values-per-plate",
    ),
    pytest.param(
        lambda: CartesianProduct(t for t in range(2)),
        [{(0,): 0, (1,): 1}],
        id="one-value-per-plate",
    ),
    pytest.param(
        lambda: Union([as_dict(((t,), t))] for t in range(3)),
        [{(0,): 0}, {(1,): 1}, {(2,): 2}],
        id="a-row-per-plate",
    ),
]


def test_a_plate_can_be_looped_over_and_subscripted():
    """A row is keyed by the tuple of plate indices it assigns, and a
    comprehension over one subscripts it by a bare index."""
    plate = CartesianProduct(range(2) for t in range(2))
    row = defop(Mapping, name="row")
    assert syntactic_eq_alpha(
        reduce_concretely(Sum(f(ixs[0]) * h(ixs[1]) for ixs in plate)),
        reduce_concretely(Sum.reduce(f(row()[(0,)]) * h(row()[(1,)]), {row: plate})),
    )


@pytest.mark.parametrize("comprehension,expected", PLATES)
def test_a_body_under_a_row_monoid_is_tagged_with_its_plates(comprehension, expected):
    """``CartesianProduct`` reduces rows, so a body of plain values is tagged
    with the targets it was produced under."""
    assert reduce_concretely(comprehension()) == expected


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
    assert (
        not isinstance(body_of(fused), Term) or body_of(fused).op is not monoid.reduce
    )


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


# ============================================================================
# WHAT THE RULES MAKE OF A DESUGARED NEST
# ============================================================================

# A desugared comprehension has to be a term the rules recognize, or the
# spelling costs the optimizer. Each case pairs a comprehension with the
# reduction it means and applies one rule to both.

c = defop(int, name="c")


def _written(rule, build, *names):
    return evaluate_under(rule, build(*(defop(int, name=name) for name in names)))


def evaluate_under(rule, term):
    with handler(rule):
        return evaluate(term)


RULES = [
    pytest.param(
        ReduceEqualityMaskRange(),
        lambda: Sum(f(x) for x in range(3) if x == c()),
        lambda a: Sum.reduce(Sum.mask(f(a()), a() == c()), {a: range(3)}),
        ["a"],
        id="ReduceEqualityMaskRange",
    ),
    pytest.param(
        ReduceEqualityMaskRange(),
        lambda: Sum(f(x) for x in range(4) if x == c() and c() < 3),
        lambda a: Sum.reduce(
            Sum.mask(f(a()), And.plus(a() == c(), c() < 3)), {a: range(4)}
        ),
        ["a"],
        id="ReduceEqualityMaskRange-with-a-residual-conjunct",
    ),
    pytest.param(
        ReduceMaskHoist(),
        lambda: Sum(f(x) for x in xs() if p(c())),
        lambda a: Sum.reduce(Sum.mask(f(a()), p(c())), {a: xs()}),
        ["a"],
        id="ReduceMaskHoist",
    ),
    pytest.param(
        ReduceFusion(),
        lambda: Sum(sum(g(x, y) for y in ys(x)) for x in xs()),
        lambda a, b: Sum.reduce(Sum.reduce(g(a(), b()), {b: ys(a())}), {a: xs()}),
        ["a", "b"],
        id="ReduceFusion",
    ),
    pytest.param(
        Factor(),
        lambda: Sum(Product.plus(f(x), h(y)) for x in xs() for y in zs()),
        lambda a, b: Sum.reduce(Product.plus(f(a()), h(b())), {a: xs(), b: zs()}),
        ["a", "b"],
        id="Factor",
    ),
    pytest.param(
        ReduceDisequalityMask(),
        lambda: Sum(f(x) for x in range(4) if not x == w()),
        lambda a: Sum.reduce(Sum.mask(f(a()), a() != w()), {a: range(4)}),
        ["a"],
        id="ReduceDisequalityMask",
    ),
    pytest.param(
        ReduceSplit(),
        lambda: Sum(Sum.plus(f(x), c()) for x in xs()),
        lambda a: Sum.reduce(Sum.plus(f(a()), c()), {a: xs()}),
        ["a"],
        id="ReduceSplit",
    ),
]


@pytest.mark.parametrize("rule,comprehension,written,names", RULES)
def test_a_rule_rewrites_a_desugared_nest_as_it_rewrites_a_written_one(
    rule, comprehension, written, names
):
    assert syntactic_eq_alpha(
        evaluate_under(rule, comprehension()), _written(rule, written, *names)
    )


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
    pytest.param(
        lambda: (f * 2 for f in [1, 2, 3]), id="target-named-like-an-operation"
    ),
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
    pytest.param([[1], [2]], list[int], id="nested-list"),
    pytest.param([{"a": 1}], dict[str, int], id="record"),
    pytest.param([[]], list, id="empty-nested-list"),
    pytest.param([{}], dict, id="empty-record"),
    pytest.param([None], type(None), id="none"),
    pytest.param([], object, id="empty"),
    pytest.param(Box[int](1, 2), int, id="parameterized-generic"),
    pytest.param(Box(1, 2), object, id="unparameterized-generic"),
    pytest.param(iter([1, 2, 3]), object, id="unrepeatable"),
]


@pytest.mark.parametrize("stream,expected", ELEMENT_TYPES)
def test_element_type_of_a_concrete_stream(stream, expected):
    assert element_type(stream) == expected


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
    pytest.param(lambda: (f(1) for i in EMPTY), int, id="an-empty-stream"),
    pytest.param(lambda: (f(1) for i in Box[int](1, 2)), int, id="parameterized"),
    pytest.param(lambda: (f(1) for a, b in [(1, 2)]), tuple, id="tuple-target"),
]


@pytest.mark.parametrize("comprehension,expected", TARGET_TYPES)
def test_a_target_operation_has_the_inferred_element_type(comprehension, expected):
    """A stream with no elements has no element type to read off, so the target
    takes the type the monoid adds up: nothing can contradict it."""
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
    pytest.param(
        _started_generator,
        ValueError,
        "has not been started",
        id="a-started-generator",
    ),
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
    pytest.param(
        lambda: (sum(a * b for a, b in zip(range(x), range(x))) for x in range(5)),
        NotImplementedError,
        "cannot consume a comprehension over a symbolic iterable",
        id="a-zip-of-dependent-ranges",
    ),
    pytest.param(
        lambda: (len(set(y % 3 for y in range(x))) for x in range(5)),
        NotImplementedError,
        "cannot consume a comprehension over a symbolic iterable",
        id="an-eager-set-over-a-dependent-generator",
    ),
    pytest.param(
        # `set` drops the repetitions a reduction over a non-idempotent monoid
        # counts, so it is not the stream it is given.
        lambda: (y for x in range(3) for y in set(range(x))),
        NotImplementedError,
        "set cannot consume",
        id="a-dependent-stream-made-a-set",
    ),
    pytest.param(
        lambda: (i * v for x in range(3) for i, v in enumerate(range(x))),
        NotImplementedError,
        "enumerate cannot consume",
        id="an-enumerated-dependent-stream",
    ),
    pytest.param(
        lambda: (a * b for x in range(3) for a, b in zip(range(x), range(x))),
        NotImplementedError,
        "zip cannot consume",
        id="a-zip-of-two-dependent-streams",
    ),
    pytest.param(
        lambda: (i * v for x in range(3) for i, v in enumerate(range(x))),
        NotImplementedError,
        "enumerate cannot consume",
        id="an-enumerated-dependent-range",
    ),
    pytest.param(
        lambda: (max((y for y in range(1, x + 2)), default=0) for x in range(3)),
        NotImplementedError,
        "max cannot consume",
        id="a-reduction-with-a-default",
    ),
    pytest.param(
        lambda: (len(s) for s in ["ab", "cde"]),
        NotImplementedError,
        "len cannot consume",
        id="the-length-of-a-string-element",
    ),
    pytest.param(
        lambda: (ord(c) for c in "abc"),
        NotImplementedError,
        "ord cannot consume",
        id="the-code-point-of-a-character",
    ),
    pytest.param(
        lambda: (float(x) for x in range(3)),
        ValueError,
        "Cannot convert term to float",
        id="a-target-converted-to-a-float",
    ),
    pytest.param(
        lambda: (int(x > 1) for x in range(4)),
        ValueError,
        "Cannot convert term to int",
        id="a-comparison-counted-as-an-int",
    ),
    pytest.param(
        lambda: (sorted([b, a])[0] for a, b in [(2, 1), (5, 4)]),
        ValueError,
        "Cannot convert term to bool",
        id="a-sort-of-two-targets",
    ),
    pytest.param(
        lambda: (x.bit_length() for x in range(8)),
        AttributeError,
        "has no attribute",
        id="a-method-of-a-target",
    ),
    pytest.param(
        lambda: (x.real for x in range(4)),
        NotHandled,
        None,
        id="an-attribute-of-a-target",
    ),
    pytest.param(
        lambda: (len([*range(x), 1]) for x in range(3)),
        AssertionError,
        None,
        id="a-starred-item-in-a-display",
    ),
    pytest.param(
        lambda: (len({**{"a": 1}, "b": x}) for x in range(3)),
        AssertionError,
        None,
        id="a-double-starred-item-in-a-display",
    ),
    pytest.param(
        lambda: (y for x in range(3) for y in {x, x + 1}),
        TypeError,
        "Terms should not appear",
        id="a-set-display-as-a-stream",
    ),
    pytest.param(
        lambda: (len("abcde"[:x]) for x in range(4)),
        TypeError,
        "__index__ returned non-int",
        id="a-slice-by-a-target",
    ),
    pytest.param(
        lambda: (len("ab" * x) for x in range(4)),
        TypeError,
        "has no len",
        id="a-string-repeated-by-a-target",
    ),
    pytest.param(
        lambda: (len("%d" % x) for x in range(12)),  # noqa: UP031 - the point
        ValueError,
        "Cannot convert term to int",
        id="a-percent-formatted-target",
    ),
    pytest.param(
        lambda: (len(f"{x:>3}") for x in range(5)),
        TypeError,
        "unsupported format string",
        id="a-format-specification",
    ),
    pytest.param(
        lambda: ({0: 10, 1: 20, 2: 30}[x] for x in range(3)),
        KeyError,
        None,
        id="a-concrete-dict-indexed-by-a-target",
    ),
    pytest.param(
        lambda: ([10, 20, 30][x] for x in range(3)),
        TypeError,
        "__index__ returned non-int",
        id="a-concrete-list-indexed-by-a-target",
    ),
    pytest.param(
        lambda: (y for x in range(3) for y in {0: [1], 1: [2, 3], 2: []}[x]),
        AssertionError,
        "must not contain temporary nodes",
        id="a-concrete-dict-indexed-by-a-target-as-a-stream",
    ),
]


@pytest.mark.parametrize("argument,exception,match", REJECTED)
@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_rejects(argument, exception, match, monoid):
    with pytest.raises(exception, match=match):
        monoid(argument())


# What a comprehension may ask that a term answers about *itself* rather than
# about the element it stands for, with the answer it gives and the answer it
# should have given. A reduction written out does the same.

SILENT_HAZARDS = [
    pytest.param(
        lambda: (LOOKUP.get(x, 0) for x in range(4)),
        0,
        6,
        id="a-lookup-misses-every-key",
    ),
    pytest.param(
        lambda: (isinstance(x, int) for x in [1, 2, 3]), 0, 3, id="no-term-is-an-int"
    ),
    pytest.param(
        lambda: (x.__class__ is int for x in range(3)),
        0,
        3,
        id="a-term-is-not-of-its-own-class",
    ),
    pytest.param(
        lambda: (1 for c in "abc" if c in "ab"),
        0,
        2,
        id="membership-in-a-string-is-not-a-substring",
    ),
    pytest.param(lambda: (len(str(x)) for x in range(12)), 36, 14, id="text"),
    pytest.param(lambda: (len(repr(x)) for x in range(12)), 552, 14, id="repr"),
    pytest.param(lambda: (len(f"{x}") for x in range(12)), 36, 14, id="an-f-string"),
    pytest.param(
        lambda: (x for x in range(4) if (x % 2 == 0) is True), 0, 2, id="identity"
    ),
]


@pytest.mark.parametrize("comprehension,answered,meant", SILENT_HAZARDS)
def test_what_a_term_answers_about_itself(comprehension, answered, meant):
    """A hazard inherited from the term classes, not introduced here."""
    assert reduce_concretely(Sum(comprehension())) == answered != meant
    assert reduce_in_python(Sum, comprehension()) == meant


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


# ============================================================================
# COMPREHENSIONS AND REDUCTIONS AS TWO SPELLINGS OF ONE THING
# ============================================================================

# Neither direction is a syntactic inverse of the other -- loop targets come
# back as fresh operations, a loop nest has no inherent order, and a filter and
# a mask are two spellings of one thing -- so what has to hold is in the
# semantic domain: each direction preserves meaning, and so does either round
# trip. A comprehension is written as a thunk, so that it can be both desugared
# and run; a reduction is written as source, so that it can be transformed
# before it is evaluated.


def _run(node: ast.expr, namespace: Mapping[str, typing.Any]) -> typing.Any:
    return eval(
        compile(ast.Expression(body=node), "<equations>", "eval"), dict(namespace)
    )


def as_reduction(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


class _IterateInPython:
    """A monoid that runs the loop nest itself rather than reducing over it.

    Every other attribute is the monoid's own, so a body that uses ``mask`` or
    ``delta`` still means what it says.
    """

    def __init__(self, monoid: Monoid):
        self._monoid = monoid

    def __call__(self, comprehension) -> typing.Any:
        return self.plus(*comprehension)

    def plus(self, *values) -> typing.Any:
        # Added up right away rather than at the end: a sum may be the
        # condition of a filter, and Python needs a decision there.
        return reduce_concretely(self._monoid.plus(*values))

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._monoid, name)


def reduce_in_python(monoid, values):
    """Reduce what Python yields for a comprehension, without desugaring it."""
    return reduce_concretely(monoid.plus(*values))


def run_as_written(node: ast.expr, namespace=()):
    """Evaluate an expression with every monoid meaning its own reduction."""
    return reduce_concretely(_run(node, {**globals(), **dict(namespace)}))


def run_in_python(node: ast.expr, namespace=()):
    """Evaluate an expression with every monoid running its loops in Python."""
    bindings = {**globals(), **dict(namespace)}
    return reduce_concretely(
        _run(
            node,
            {
                name: _IterateInPython(value) if isinstance(value, Monoid) else value
                for name, value in bindings.items()
            },
        )
    )


def means_the_same(left, right) -> bool:
    """Equality up to the renaming of bound operations and the order of a sum.

    ``PlusOrder`` orders the arguments of a commutative ``plus`` by their
    hashes, which two alpha-equivalent expressions need not agree on.
    """
    return syntactic_eq_alpha(_ordered(left), _ordered(right))


def _ordered(expr):
    """Reorder the arguments of every commutative sum in ``expr``."""
    if isinstance(expr, Term):
        args = tuple(_ordered(argument) for argument in expr.args)
        owner = getattr(expr.op, "__self__", None)
        if (
            isinstance(owner, Monoid)
            and expr.op is owner.plus
            and is_commutative(owner)
        ):
            args = tuple(sorted(args, key=repr))
        return defdata(
            expr.op, *args, **{k: _ordered(v) for k, v in expr.kwargs.items()}
        )
    if isinstance(expr, tuple):
        return tuple(_ordered(element) for element in expr)
    if isinstance(expr, list):
        return [_ordered(element) for element in expr]
    if isinstance(expr, Mapping):
        return {key: _ordered(value) for key, value in expr.items()}
    return expr


# Operations standing for the loop targets of a written-out reduction, and for
# the per-element weight of a weighted stream.

i = defop(int, name="i")
j = defop(int, name="j")
k = defop(int, name="k")
pair = defop(tuple[int, int], name="pair")
u = defop(float, name="u")
v = defop(float, name="v")


@defop
def wt(v: int) -> float:
    raise NotHandled


# ---------------------------------------------------------------------------
# Reductions over loop nests Python can iterate
# ---------------------------------------------------------------------------

BASIC_REDUCTIONS = [
    pytest.param("Sum.reduce(i(), {i: (1, 2, 3)})", id="bare-target"),
    pytest.param("Sum.reduce(i() * 2, {i: range(4)})", id="arithmetic"),
    pytest.param("Sum.reduce((i() + 1) * (i() - 1), {i: range(5)})", id="compound"),
    pytest.param("Sum.reduce(-i(), {i: range(4)})", id="unary"),
    pytest.param("Sum.reduce(i() ** 2, {i: range(4)})", id="power"),
    pytest.param("Sum.reduce(i() // 2 + i() % 3, {i: range(1, 7)})", id="integer"),
    pytest.param("Sum.reduce(i() & 3 | 1 ^ 2, {i: range(8)})", id="bitwise"),
    pytest.param("Sum.reduce(abs(i() - 2), {i: range(5)})", id="builtin-call"),
    pytest.param("Sum.reduce(7, {i: range(4)})", id="constant-body"),
    pytest.param("Sum.reduce(5, {})", id="no-streams"),
    pytest.param("Sum.reduce(i(), {i: []})", id="empty-stream"),
    pytest.param("Sum.reduce(i(), {i: (7,)})", id="singleton-stream"),
    pytest.param("Sum.reduce(i() * 2, {i: []})", id="an-empty-stream-with-arithmetic"),
    pytest.param(
        "Sum.reduce(i() * j(), {i: (), j: range(3)})", id="an-empty-stream-in-a-nest"
    ),
    pytest.param("And.reduce(i() > 0, {i: []})", id="an-empty-stream-under-and"),
    pytest.param("Sum.reduce(i(), {i: [1, 2, 3]})", id="list-stream"),
    pytest.param("Sum.reduce(i(), {i: {1, 2, 3}})", id="set-stream"),
    pytest.param("Sum.reduce(i(), {i: frozenset({1, 2})})", id="frozenset-stream"),
    pytest.param("Sum.reduce(i(), {i: {1: 'a', 2: 'b'}})", id="dict-keys-stream"),
    pytest.param("Sum.reduce(i(), {i: tuple({1: 10, 2: 20}.values())})", id="values"),
    pytest.param("Sum.reduce(i(), {i: range(10, 0, -2)})", id="stepped-range"),
    pytest.param("Sum.reduce(i(), {i: {1: 'a', 2: 'b'}})", id="a-dict-stream"),
    pytest.param(
        "Sum.reduce((i() + 1) * (i() - 1) // 2 % 5**2 & 7 | 1 ^ 2, {i: range(1, 8)})",
        id="every-arithmetic-operator",
    ),
    pytest.param("Sum.reduce(i(), {i: tuple(sorted([3, 1, 2]))})", id="sorted-stream"),
    pytest.param("Product.reduce(i(), {i: range(1, 5)})", id="product"),
    pytest.param("Min.reduce(i(), {i: (3, 1, 2)})", id="min"),
    pytest.param("Max.reduce(i(), {i: (3, 1, 2)})", id="max"),
    pytest.param("And.reduce(i() > 0, {i: range(1, 4)})", id="and"),
    pytest.param("Or.reduce(i() > 2, {i: range(4)})", id="or"),
    pytest.param("Sum.reduce(f(i()), {i: (1, 2)})", id="operation-body"),
    pytest.param("Sum.reduce(f(i()) * h(i()), {i: (1, 2)})", id="two-operations"),
    pytest.param(
        "Sum.reduce(pair()[0] * pair()[1], {pair: ((1, 2), (3, 4))})", id="pairs"
    ),
    pytest.param("Sum.reduce(f(i()), {i: (w(),)})", id="symbolic-element"),
    pytest.param("Sum.reduce(i(), {i: [w(), w() + 1]})", id="symbolic-elements"),
    pytest.param(
        "Sum.reduce(pair()[0] + pair()[1], {pair: ((w(), 1), (2, w()))})",
        id="symbolic-pairs",
    ),
    pytest.param("Sum.reduce(i() + w(), {i: range(3)})", id="a-free-operation"),
    pytest.param(
        "Sum.reduce(i() * j(), {i: (3,), j: range(4)})", id="a-singleton-beside"
    ),
    pytest.param("Sum.reduce(i() * j(), {i: range(3), j: []})", id="an-empty-beside"),
    pytest.param(
        "And.reduce(And.mask(i() > 0, i() != 2), {i: range(1, 5)})", id="masked-and"
    ),
    pytest.param(
        "Or.reduce(i() > j(), {i: range(3), j: range(3)})", id="or-over-a-nest"
    ),
]

NESTED_REDUCTIONS = [
    pytest.param("Sum.reduce(i() + j(), {i: range(3), j: range(4)})", id="two-streams"),
    pytest.param("Sum.reduce(i() * j(), {j: range(4), i: range(3)})", id="unordered"),
    pytest.param(
        "Sum.reduce(i() * j() * k(), {i: range(2), j: range(3), k: range(2)})",
        id="three-streams",
    ),
    pytest.param(
        "Sum.reduce(i() * j(), {i: range(4), j: range_(i())})", id="dependent-stream"
    ),
    pytest.param(
        "Sum.reduce(i() * j(), {j: range_(i()), i: range(4)})",
        id="dependent-stream-written-first",
    ),
    pytest.param(
        "Sum.reduce(i() + j() + k(), {k: range_(j()), j: range_(i()), i: range(4)})",
        id="chain-of-dependent-streams",
    ),
    pytest.param(
        "Sum.reduce(Sum.reduce(i() * j(), {j: range(3)}), {i: range(4)})",
        id="reduce-in-a-reduce",
    ),
    pytest.param(
        "Sum.reduce(Max.reduce(i() * j(), {j: range(1, 3)}), {i: range(4)})",
        id="a-different-inner-monoid",
    ),
    pytest.param(
        "Sum.reduce(Sum.reduce(Sum.reduce(i() * j() * k(), {k: range(2)}),"
        " {j: range(2)}), {i: range(3)})",
        id="three-deep",
    ),
    pytest.param(
        "Sum.reduce(Sum.reduce(i() * j(), {j: range_(i())}), {i: range(4)})",
        id="a-dependent-inner-nest",
    ),
    pytest.param(
        "Sum.reduce(i() + Sum.reduce(j(), {j: range(3)}), {i: range(4)})",
        id="a-reduce-beside-the-body",
    ),
    pytest.param(
        "Sum.reduce(Sum.reduce(j(), {j: range(3)}) * Max.reduce(k(), {k: range(3)}),"
        " {i: range(2)})",
        id="two-inner-reductions",
    ),
    pytest.param(
        "Sum.reduce(i() + j() + k() + pair()[0],"
        " {i: range(2), j: range(2), k: range(2), pair: ((1, 2),)})",
        id="four-streams",
    ),
    pytest.param(
        "Sum.reduce(Sum.plus(Sum.reduce(j(), {j: range_(i())}),"
        " Max.reduce(k(), {k: range(1, 3)})), {i: range(4)})",
        id="a-plus-of-two-reductions",
    ),
    pytest.param(
        "Sum.reduce(Product.plus(Sum.reduce(Product.plus(i(), j()), {j: range(1, 3)}),"
        " i()), {i: range(1, 4)})",
        id="a-product-of-sums-of-products",
    ),
    pytest.param(
        "Sum.reduce(i(), {i: range_(Sum.reduce(j(), {j: range(3)}))})",
        id="a-reduction-in-a-stream",
    ),
    pytest.param(
        "Sum.reduce(Max.reduce(Min.reduce(i() * j() * k(), {k: range(1, 3)}),"
        " {j: range(1, 3)}), {i: range(1, 3)})",
        id="three-monoids-deep",
    ),
    pytest.param(
        "Sum.reduce(Product.reduce(i() + j(), {j: range(1, 3)}), {i: range(1, 4)})",
        id="a-product-inside-a-sum",
    ),
    pytest.param(
        "Min.reduce(Max.reduce(i() * j(), {j: range(1, 3)}), {i: range(1, 4)})",
        id="a-max-inside-a-min",
    ),
    pytest.param(
        "Sum.reduce(i() + j() + k() + u() + v(),"
        " {i: range(2), j: range(2), k: range(2), u: (0.5,), v: (1.5,)})",
        id="five-streams",
    ),
    pytest.param(
        "Sum.reduce(j(), {i: range(1, 4), j: range_(i(), i() * 2, 1)})",
        id="a-three-argument-dependent-range",
    ),
    pytest.param(
        "Sum.reduce(Sum.reduce(i() * j(), {j: range(2)}) + j(),"
        " {i: range(2), j: range(3)})",
        id="a-key-bound-inside-and-outside",
    ),
    pytest.param(
        "Product.reduce(Sum.reduce(Sum.mask(j(), j() != 1), {j: range_(i() + 1)}),"
        " {i: range(1, 4)})",
        id="a-product-of-masked-sums",
    ),
    pytest.param(
        "And.reduce(And.reduce(i() <= j(), {j: range_(i(), i() + 2)}), {i: range(3)})",
        id="a-nested-conjunction",
    ),
]

MASKED_REDUCTIONS = [
    pytest.param(
        "Sum.reduce(Sum.mask(i(), i() != 1), {i: range(4)})", id="disequality"
    ),
    pytest.param("Sum.reduce(Sum.mask(i(), i() % 2 == 0), {i: range(8)})", id="modulo"),
    pytest.param(
        "Sum.reduce(Sum.mask(Sum.mask(i(), i() > 1), i() < 4), {i: range(6)})",
        id="two-masks",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(i() * j(), i() != j()), {i: range(3), j: range(3)})",
        id="over-a-nest",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(i(), i() < 100), {i: range(5)})", id="always-true"
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(i(), i() > 100), {i: range(5)})", id="always-false"
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(i(), ite(i() > 1, i() < 4, False)), {i: range(6)})",
        id="a-conditional-condition",
    ),
    pytest.param(
        "Sum.reduce(Sum.reduce(Sum.mask(i() * j(), i() != j()), {j: range(3)}),"
        " {i: range(3)})",
        id="inside-an-inner-reduce",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(Sum.reduce(j(), {j: range_(i())}), i() % 2 == 0),"
        " {i: range(5)})",
        id="over-an-inner-reduce",
    ),
    pytest.param(
        "Max.reduce(Max.mask(i(), i() % 2 == 0), {i: range(10)})", id="under-max"
    ),
    pytest.param("Min.reduce(Min.mask(i(), i() > 3), {i: range(10)})", id="under-min"),
    pytest.param(
        "Product.reduce(Product.mask(i(), i() != 3), {i: range(1, 5)})",
        id="under-product",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(Sum.reduce(Sum.mask(i() * j(), j() != 1),"
        " {j: range_(i())}), i() % 2 == 0), {i: range(5)})",
        id="at-every-level",
    ),
    pytest.param(
        "Sum.reduce(Sum.plus(Sum.mask(i(), i() > 1), Sum.mask(i() * 2, i() < 3)),"
        " {i: range(5)})",
        id="under-a-plus",
    ),
    pytest.param(
        "Sum.reduce(ite(i() > 2, Sum.mask(i(), i() != 4), Sum.mask(i() * 2, i() != 0)),"
        " {i: range(6)})",
        id="under-a-conditional",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(ite(i() % 2 == 0, i(), -i()), i() > 1), {i: range(6)})",
        id="over-a-conditional",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(i() * j() * k(), k() != j()),"
        " {i: range(4), j: range_(i()), k: range_(j())})",
        id="over-a-dependent-chain",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(i(), Sum.reduce(j(), {j: range_(i())}) > 2), {i: range(6)})",
        id="conditioned-on-a-reduction",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(1, i() > 0), {i: []})", id="over-an-empty-stream"
    ),
]

CONDITIONAL_REDUCTIONS = [
    pytest.param("Sum.reduce(ite(i() > 2, i(), -i()), {i: range(6)})", id="simple"),
    pytest.param(
        "Sum.reduce(ite(i() > 3, i(), ite(i() > 1, 0, 1)), {i: range(6)})", id="nested"
    ),
    pytest.param(
        "Sum.reduce(ite(i() > 2, i(), Sum.identity), {i: range(6)})",
        id="an-identity-arm",
    ),
    pytest.param(
        "Sum.reduce(ite(p(i()), f(i()), h(i())), {i: (1, 2)})",
        id="a-symbolic-condition",
    ),
    pytest.param(
        "Sum.reduce(i(), {i: ite(True, range(3), range(4))})", id="in-the-stream"
    ),
]

MONOID_API_REDUCTIONS = [
    pytest.param(
        "Sum.reduce(Sum.plus(i(), j()), {i: range(3), j: range(3)})", id="plus"
    ),
    pytest.param(
        "Sum.reduce(Product.plus(i(), j()), {i: range(1, 3), j: range(1, 3)})",
        id="a-different-plus",
    ),
    pytest.param("Sum.reduce(Sum.plus(i()), {i: range(4)})", id="unary-plus"),
    pytest.param(
        "Sum.reduce(Sum.plus(i(), j(), k(), 1),"
        " {i: range(2), j: range(2), k: range(2)})",
        id="a-plus-of-many",
    ),
    pytest.param(
        "Product.reduce(ite(i() == 0, Product.identity, i()), {i: range(4)})",
        id="an-identity-arm",
    ),
    pytest.param("Sum.reduce(Sum.inverse(i()), {i: range(4)})", id="inverse"),
    pytest.param(
        "Sum.reduce(Sum.plus(i(), Sum.inverse(j())), {i: range(3), j: range(3)})",
        id="plus-and-inverse",
    ),
    pytest.param("Sum.reduce(Sum.delta((i(),), f(i())), {i: range(3)})", id="delta"),
    pytest.param(
        "Sum.reduce(Sum.delta((i(), j()), i() * j()), {i: range(2), j: range(3)})",
        id="a-two-index-delta",
    ),
    pytest.param("Sum.reduce(Product.identity * i(), {i: range(4)})", id="identity"),
    pytest.param("Product.reduce(i() + Product.zero, {i: range(1, 4)})", id="zero"),
    pytest.param(
        "Sum.reduce(Sum.mask(Sum.delta((i(),), f(i())), i() != 1), {i: range(3)})",
        id="a-masked-delta",
    ),
    pytest.param(
        "Sum.reduce(Sum.delta((i(), j()), f(i()) * h(j())), {i: range(2), j: range(2)})",
        id="a-delta-over-a-nest",
    ),
    pytest.param(
        "Sum.reduce(Sum.delta((i(),), Sum.reduce(j(), {j: range_(i())})), {i: range(3)})",
        id="a-delta-of-a-reduction",
    ),
    pytest.param(
        "Sum.reduce(Sum.inverse(Sum.reduce(j(), {j: range_(i())})), {i: range(4)})",
        id="an-inverted-reduction",
    ),
    pytest.param(
        "Sum.reduce(Sum.plus(Sum.inverse(i()), Sum.inverse(j())),"
        " {i: range(3), j: range(3)})",
        id="a-plus-of-inverses",
    ),
]

SYNTAX_REDUCTIONS = [
    pytest.param(
        "Sum.reduce(i(), {i: [y * 2 for y in range(4)]})", id="a-list-comprehension"
    ),
    pytest.param(
        "Sum.reduce(i(), {i: {y % 3 for y in range(7)}})", id="a-set-comprehension"
    ),
    pytest.param(
        "Sum.reduce(i(), {i: tuple({y: y * 2 for y in range(4)}.values())})",
        id="a-dict-comprehension",
    ),
    pytest.param(
        "Sum.reduce(i(), {i: tuple(y for y in range(4) if y % 2)})",
        id="a-generator-expression",
    ),
    pytest.param(
        "Sum.reduce((lambda v: v * 2)(i()), {i: range(4)})", id="a-lambda-body"
    ),
    pytest.param(
        "Sum.reduce((lambda v, u=3: v + u)(i()), {i: range(4)})", id="a-lambda-default"
    ),
    pytest.param(
        "Sum.reduce((lambda *vs: sum(vs))(i(), i() + 1), {i: range(4)})",
        id="a-variadic-lambda",
    ),
    pytest.param(
        "Sum.reduce(i() * len([y for y in range(3)]), {i: range(4)})",
        id="a-comprehension-in-the-body",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(i(), len([y for y in range(3)]) > i()), {i: range(5)})",
        id="a-comprehension-in-a-mask",
    ),
    pytest.param(
        "Sum.reduce((lambda v: Sum.reduce(v * j(), {j: range(3)}))(i()), {i: range(4)})",
        id="a-reduce-inside-a-lambda",
    ),
    pytest.param(
        "Sum.reduce(i(), {i: tuple(y * 2 for y in range(4))})",
        id="a-materialised-generator",
    ),
    pytest.param(
        "Sum.reduce(i(), {i: tuple({y: y * 2 for y in range(3)}.items())[0]})",
        id="an-item-of-a-dict-comprehension",
    ),
]

CONCRETE_REDUCTIONS = [
    *BASIC_REDUCTIONS,
    *NESTED_REDUCTIONS,
    *MASKED_REDUCTIONS,
    *CONDITIONAL_REDUCTIONS,
    *MONOID_API_REDUCTIONS,
    *SYNTAX_REDUCTIONS,
]

# ---------------------------------------------------------------------------
# Reductions over loop nests only a monoid can walk
# ---------------------------------------------------------------------------

# A symbolic stream has no end, a weighted one is a term, and a symbolic filter
# has no truth value, so Python cannot run these comprehensions at all.

SYMBOLIC_REDUCTIONS = [
    pytest.param(
        "Sum.reduce(Sum.mask(f(i()), p(i())), {i: (1, 2, 3)})",
        id="a-symbolic-filter",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(Sum.mask(f(i()), p(i())), q(i())), {i: xs()})",
        id="two-symbolic-filters",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(Sum.mask(Sum.mask(f(i()), p(i())), q(i())), p(i() + 1)),"
        " {i: xs()})",
        id="three-symbolic-filters",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(Sum.mask(g(i(), j()), p(i())), p(j())),"
        " {i: xs(), j: zs()})",
        id="a-symbolic-filter-per-target",
    ),
    pytest.param(
        "Sum.reduce(Sum.reduce(Sum.mask(Sum.mask(g(i(), j()), p(i())), p(j())),"
        " {j: ys(i())}), {i: xs()})",
        id="symbolic-filters-in-an-inner-reduce",
    ),
    pytest.param(
        "CartesianProduct.reduce([as_dict(((i(),), i()))], {i: range(2)})",
        id="rows-written-as-a-literal",
    ),
    pytest.param(
        "Sum.reduce(f(i()), {i: Product.weighted((1, 2, 3), abs)})",
        id="a-stream-weighted-by-a-builtin",
    ),
    pytest.param(
        "Sum.reduce(pair()[0], {pair: ((1, 'a'), (2, 'b'))})",
        id="a-heterogeneous-element-indexed",
    ),
    pytest.param("Sum.reduce(f(i()), {i: xs()})", id="a-symbolic-stream"),
    pytest.param(
        "Sum.reduce(g(i(), j()), {i: xs(), j: ys(i())})",
        id="a-dependent-symbolic-stream",
    ),
    pytest.param(
        "Sum.reduce(f(i()) * h(j()), {i: xs(), j: zs()})", id="two-symbolic-streams"
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(f(i()), p(i())), {i: xs()})", id="a-masked-symbolic-stream"
    ),
    pytest.param(
        "Sum.reduce(Sum.reduce(g(i(), j()), {j: ys(i())}), {i: xs()})",
        id="a-nested-symbolic-stream",
    ),
    pytest.param(
        "Sum.reduce(ite(p(i()), f(i()), h(i())), {i: xs()})",
        id="a-conditional-over-a-symbolic-stream",
    ),
    pytest.param(
        "Sum.reduce(f(i()), {i: Product.weighted((1, 2, 3, 4), wt)})",
        id="a-weighted-stream",
    ),
    pytest.param(
        "Sum.reduce(f(i()), {i: Product.weighted(xs(), wt)})",
        id="a-weighted-symbolic-stream",
    ),
    pytest.param(
        "Sum.reduce(Product.plus(f(i()), h(j())),"
        " {i: Product.weighted(xs(), wt), j: zs()})",
        id="a-weighted-stream-beside-another",
    ),
    pytest.param("Sum.reduce((f(i()), h(i())), {i: (1, 2)})", id="a-sequence-body"),
    pytest.param(
        "Sum.reduce({0: f(i()), 1: h(i())}, {i: (1, 2)})", id="a-mapping-body"
    ),
    pytest.param(
        "Union.reduce([as_dict(((i(),), i()))], {i: range(3)})", id="a-union-of-rows"
    ),
    pytest.param(
        "CartesianProduct.reduce("
        "Union.reduce([as_dict(((i(),), j()))], {j: range(2)}), {i: range(2)})",
        id="a-cartesian-product-of-rows",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(f(i()), p(i())), {i: Product.weighted((1, 2, 3), wt)})",
        id="a-masked-weighted-stream",
    ),
    pytest.param(
        "Sum.reduce(Product.plus(f(i()), h(j())),"
        " {i: Product.weighted((1, 2), wt), j: Product.weighted((3, 4), wt)})",
        id="two-weighted-streams",
    ),
    pytest.param(
        "Sum.reduce(g(i(), j()), {i: xs(), j: Product.weighted(ys(i()), wt)})",
        id="a-weighted-dependent-stream",
    ),
    pytest.param(
        "Sum.reduce(Sum.reduce(g(i(), j()), {j: Product.weighted(ys(i()), wt)}),"
        " {i: xs()})",
        id="a-nested-weighted-reduce",
    ),
    pytest.param(
        "Union.reduce([as_dict(((i(), j()), i() * j()))], {i: range(2), j: range(2)})",
        id="rows-over-a-nest",
    ),
    pytest.param(
        "Union.reduce([as_dict(((i(),), Sum.mask(i(), i() != 1)))], {i: range(3)})",
        id="masked-rows",
    ),
    pytest.param(
        "CartesianProduct.reduce("
        "Union.reduce([as_dict(((i(), j()), k()))], {k: range(2)}),"
        " {i: range(2), j: range(2)})",
        id="a-cartesian-product-of-two-plates",
    ),
    pytest.param(
        "CartesianProduct.reduce(CartesianProduct.mask("
        "Union.reduce([as_dict(((i(),), j()))], {j: range(2)}), i() != 1), {i: range(3)})",
        id="a-masked-cartesian-product",
    ),
    pytest.param("LogSumExp.reduce(u(), {u: (0.0, 1.0, 2.0)})", id="log-sum-exp"),
    pytest.param(
        "LogSumExp.reduce(LogSumExp.mask(u(), u() != 1.0), {u: (0.0, 1.0, 2.0)})",
        id="a-masked-log-sum-exp",
    ),
    pytest.param(
        "LogSumExp.reduce(LogSumExp.reduce(u() + v(), {v: (0.5, 1.5)}), {u: (0.0, 1.0)})",
        id="a-nested-log-sum-exp",
    ),
    pytest.param("ArgMax.reduce((u(), 1), {u: (0.0, 2.0, 1.0)})", id="arg-max"),
    pytest.param("ArgMin.reduce((u(), 1), {u: (0.0, 2.0, 1.0)})", id="arg-min"),
    pytest.param(
        "Sum.reduce((Sum.reduce(j(), {j: range_(i())}), i()), {i: range(4)})",
        id="a-sequence-of-reductions",
    ),
    pytest.param(
        "Sum.reduce({0: Sum.mask(i(), i() != 1), 1: i() * 2}, {i: range(4)})",
        id="a-mapping-of-masked-values",
    ),
    pytest.param(
        "Sum.reduce({0: {1: f(i())}}, {i: (1, 2)})", id="a-nested-mapping-body"
    ),
]

ALL_REDUCTIONS = [*CONCRETE_REDUCTIONS, *SYMBOLIC_REDUCTIONS]


# ---------------------------------------------------------------------------
# The four equations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("comprehension", ALL_SHAPES)
@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_desugaring_a_comprehension_preserves_its_meaning(comprehension, monoid):
    """``eval(c2r(c)) == eval(c)``."""
    assert means_the_same(
        reduce_concretely(monoid(comprehension())),
        reduce_in_python(monoid, comprehension()),
    )


# What a reduction is written as, once the loop nest is a comprehension again.

REWRITINGS = [
    pytest.param(
        "Sum.reduce(i() * 2, {i: (1, 2, 3)})",
        "Sum((i * 2 for i in (1, 2, 3)))",
        id="a-stream-becomes-a-generator",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(f(i()), p(i())), {i: xs()})",
        "Sum((f(i) for i in xs() if p(i)))",
        id="a-mask-becomes-a-filter",
    ),
    pytest.param(
        "Sum.reduce(Sum.mask(Sum.mask(f(i()), p(i())), q(i())), {i: xs()})",
        "Sum((f(i) for i in xs() if p(i) if q(i)))",
        id="stacked-masks-become-filter-clauses",
    ),
    pytest.param(
        "Sum.reduce(g(i(), j()), {j: ys(i()), i: xs()})",
        "Sum((g(i, j) for i in xs() for j in ys(i)))",
        id="a-nest-is-ordered-by-what-it-depends-on",
    ),
    pytest.param(
        "Sum.reduce(Max.reduce(g(i(), j()), {j: ys(i())}), {i: xs()})",
        "Sum((Max((g(i, j) for j in ys(i))) for i in xs()))",
        id="an-inner-reduce-becomes-an-inner-comprehension",
    ),
    pytest.param(
        "Sum.reduce(f(w()), {})",
        "Sum((f(w()) for _unused0 in (0,)))",
        id="an-empty-nest-gets-a-loop-that-binds-nothing",
    ),
    pytest.param(
        "Sum.reduce(f(i()), {i: xs(), j: zs()})",
        "Sum((f(i) for i in xs() for j in zs()))",
        id="a-stream-the-body-ignores-is-kept",
    ),
    pytest.param(
        "Sum.reduce(f(i()), streams)",
        "Sum.reduce(f(i()), streams)",
        id="a-nest-that-is-not-written-out-is-left-alone",
    ),
    pytest.param(
        "Sum.reduce(f(i()), {**others})",
        "Sum.reduce(f(i()), {**others})",
        id="a-splatted-nest-is-left-alone",
    ),
]


@pytest.mark.parametrize("reduction,expected", REWRITINGS)
def test_a_reduction_is_rewritten_as(reduction, expected):
    comprehension, _ = reduce_to_comprehension(as_reduction(reduction))
    assert ast.unparse(comprehension) == expected


@pytest.mark.parametrize("name", ["class", "None", "lambda", "range", "sum"])
def test_a_rewritten_reduction_is_spelled_with_names_it_may_use(name):
    """A loop target is a name, so an operation named like a keyword or a
    builtin is renamed rather than written out where it cannot be read."""
    target = defop(int, name=name)
    comprehension, namespace = reduce_to_comprehension(
        Sum.reduce(target(), {target: range(3)})
    )
    assert compile(ast.Expression(body=comprehension), "<names>", "eval")
    assert not {n for n in namespace if hasattr(builtins, n) or keyword.iskeyword(n)}


@pytest.mark.parametrize("reduction", CONCRETE_REDUCTIONS)
def test_rewriting_a_reduction_preserves_its_meaning(reduction):
    """``eval(r2c(r)) == eval(r)``."""
    node = as_reduction(reduction)
    comprehension, namespace = reduce_to_comprehension(node)
    assert means_the_same(run_in_python(comprehension, namespace), run_as_written(node))


@pytest.mark.parametrize("comprehension", ALL_SHAPES)
@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_a_comprehension_survives_a_round_trip(comprehension, monoid):
    """``eval(r2c(c2r(c))) == eval(c)``."""
    back, namespace = reduce_to_comprehension(monoid(comprehension()))
    assert means_the_same(
        run_in_python(back, namespace),
        reduce_in_python(monoid, comprehension()),
    )


@pytest.mark.xfail(strict=True, reason="a callable body keeps its own binding")
def test_a_reduction_over_a_callable_body_rewrites_to_one_too():
    """``MonoidOverCallable`` makes the reduction of a function a function.

    Two functions are never equal, so they are compared by what they answer --
    and they do not agree. A Python lambda closes over the operation the stream
    is keyed by rather than being bound by it, so ``MonoidOverCallable`` leaves
    that operation free in what it answers, and the two spellings leave a
    different one free. Recorded here because a comprehension is not what makes
    it so: the written-out reduction does the same.
    """
    node = as_reduction("Sum.reduce(lambda a: a * i(), {i: range(3)})")
    comprehension, namespace = reduce_to_comprehension(node)
    written = _run(node, globals())
    rewritten = _run(comprehension, {**globals(), **namespace})
    assert means_the_same(
        reduce_concretely(rewritten(2)), reduce_concretely(written(2))
    )


@pytest.mark.parametrize("reduction", ALL_REDUCTIONS)
def test_a_reduction_survives_a_round_trip(reduction):
    """``eval(c2r(r2c(r))) == eval(r)``."""
    node = as_reduction(reduction)
    comprehension, namespace = reduce_to_comprehension(node)
    assert means_the_same(
        run_as_written(comprehension, namespace), run_as_written(node)
    )
