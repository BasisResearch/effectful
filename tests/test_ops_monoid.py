import typing

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from effectful.ops.monoid import (
    CartesianProduct,
    EvaluateIntp,
    Max,
    Min,
    Monoid,
    NormalizeIntp,
    Product,
    Sum,
    distributes_over,
    is_commutative,
)
from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.types import NotHandled, Operation
from tests._monoid_helpers import define_vars, random_interpretation, syntactic_eq_alpha


@pytest.fixture(autouse=True)
def _install_normalize_intp():
    """Install :data:`NormalizeIntp` for every test in this module.

    :data:`NormalizeIntp` is a superset of :data:`EvaluateIntp` — direct
    monoid calls evaluate, rewrites also fire. Will be replaced by a global
    interpretation once that lands.
    """
    with handler(NormalizeIntp):
        yield


_INT = st.integers(min_value=-100, max_value=100)

ALL_MONOIDS = [
    pytest.param(Sum, id="Sum"),
    pytest.param(Product, id="Product"),
    pytest.param(Min, id="Min"),
    pytest.param(Max, id="Max"),
]

COMMUTATIVE = [
    pytest.param(Sum, id="Sum"),
    pytest.param(Product, id="Product"),
    pytest.param(Min, id="Min"),
    pytest.param(Max, id="Max"),
]

IDEMPOTENT = [
    pytest.param(Min, id="Min"),
    pytest.param(Max, id="Max"),
]

WITH_ZERO = [
    pytest.param(Product, id="Product"),
]

# Pairs (outer, inner) such that inner distributes over outer — i.e. the lifting
# identity ``outer(inner(body, A), CartesianProduct...) == inner(outer(body, D), ...)``
# is valid for that semiring pair.
MONOID_PAIRS = [
    pytest.param(o.values[0], i.values[0], id=f"{o.id}-{i.id}")
    for o in ALL_MONOIDS
    for i in ALL_MONOIDS
    if distributes_over(
        typing.cast(Monoid, i.values[0]), typing.cast(Monoid, o.values[0])
    )
]


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
@given(a=_INT, b=_INT, c=_INT)
@settings(max_examples=50, deadline=None)
def test_associativity(monoid, a, b, c):
    left = monoid.plus(monoid.plus(a, b), c)
    right = monoid.plus(a, monoid.plus(b, c))
    assert left == right


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
@given(a=_INT)
@settings(max_examples=50, deadline=None)
def test_identity(monoid, a):
    assert monoid.plus(monoid.identity, a) == a
    assert monoid.plus(a, monoid.identity) == a


@pytest.mark.parametrize("monoid", COMMUTATIVE)
@given(a=_INT, b=_INT)
@settings(max_examples=50, deadline=None)
def test_commutativity(monoid, a, b):
    assert monoid.plus(a, b) == monoid.plus(b, a)


@pytest.mark.parametrize("monoid", IDEMPOTENT)
@given(a=_INT)
@settings(max_examples=50, deadline=None)
def test_idempotence(monoid, a):
    assert monoid.plus(a, a) == a


@pytest.mark.parametrize("monoid", WITH_ZERO)
@given(a=_INT)
@settings(max_examples=50, deadline=None)
def test_zero_absorbs(monoid, a):
    assert monoid.plus(monoid.zero, a) == monoid.zero
    assert monoid.plus(a, monoid.zero) == monoid.zero


def _check_pair(lhs, rhs, *, free_vars=[], max_examples: int = 25) -> None:
    """Run structural + semantic checks on a TermPair."""
    with handler(NormalizeIntp):
        norm = evaluate(lhs)

    assert syntactic_eq_alpha(norm, rhs)

    @given(intp=random_interpretation(free_vars))
    @settings(max_examples=max_examples, deadline=None)
    def _check_semantics(intp):
        with handler(intp):
            lhs_val = evaluate(lhs)
            rhs_val = evaluate(rhs)
        assert lhs_val == rhs_val

    _check_semantics()


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_empty(monoid):
    _check_pair(lhs=monoid.plus(), rhs=monoid.identity)


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_single(monoid):
    x = define_vars("x", typ=type(monoid.identity))
    _check_pair(lhs=monoid.plus(x()), rhs=x(), free_vars=[x])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_identity_right(monoid):
    x = define_vars("x", typ=type(monoid.identity))
    _check_pair(lhs=monoid.plus(x(), monoid.identity), rhs=x(), free_vars=[x])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_identity_left(monoid):
    x = define_vars("x", typ=type(monoid.identity))
    _check_pair(lhs=monoid.plus(monoid.identity, x()), rhs=x(), free_vars=[x])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_assoc_right(monoid):
    x, y, z = define_vars("x", "y", "z", typ=type(monoid.identity))
    _check_pair(
        lhs=monoid.plus(x(), monoid.plus(y(), z())),
        rhs=monoid.plus(x(), y(), z()),
        free_vars=[x, y, z],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_assoc_left(monoid):
    x, y, z = define_vars("x", "y", "z", typ=type(monoid.identity))
    _check_pair(
        lhs=monoid.plus(monoid.plus(x(), y()), z()),
        rhs=monoid.plus(x(), y(), z()),
        free_vars=[x, y, z],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_sequence(monoid):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=type(monoid.identity))
    _check_pair(
        lhs=monoid.plus((a(), b()), (c(), d())),
        rhs=(monoid.plus(a(), c()), monoid.plus(b(), d())),
        free_vars=[a, b, c, d],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_mapping(monoid):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=type(monoid.identity))
    _check_pair(
        lhs=monoid.plus({"x": a(), "y": b()}, {"x": c(), "z": d()}),
        rhs={"x": monoid.plus(a(), c()), "y": b(), "z": d()},
        free_vars=[a, b, c, d],
    )


def test_plus_distributes():
    a, b, c, d = define_vars("a", "b", "c", "d")
    lhs = Product.plus(Sum.plus(a(), b()), Sum.plus(c(), d()))
    rhs = Sum.plus(
        Product.plus(a(), c()),
        Product.plus(a(), d()),
        Product.plus(b(), c()),
        Product.plus(b(), d()),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[a, b, c, d])


def test_plus_distributes_constant():
    a, b, c, d = define_vars("a", "b", "c", "d")
    lhs = Product.plus(Sum.plus(a(), b()), Sum.plus(c(), d()), 5)
    rhs = Product.plus(
        5,
        Sum.plus(
            Product.plus(a(), c()),
            Product.plus(a(), d()),
            Product.plus(b(), c()),
            Product.plus(b(), d()),
        ),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[a, b, c, d])


def test_plus_distributes_multiple():
    a, b, c, d = define_vars("a", "b", "c", "d")
    lhs = Sum.plus(
        Min.plus(a(), b()),
        Min.plus(c(), d()),
        Max.plus(a(), b()),
        Max.plus(c(), d()),
    )
    rhs = Sum.plus(
        Min.plus(
            Sum.plus(a(), c()),
            Sum.plus(a(), d()),
            Sum.plus(b(), c()),
            Sum.plus(b(), d()),
        ),
        Max.plus(
            Sum.plus(a(), c()),
            Sum.plus(a(), d()),
            Sum.plus(b(), c()),
            Sum.plus(b(), d()),
        ),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[a, b, c, d])


@pytest.mark.parametrize("monoid", IDEMPOTENT)
def test_plus_idempotent_consecutive(monoid):
    """``a, a, b → a, b`` — only consecutive duplicates collapse."""
    a, b = define_vars("a", "b")
    lhs = monoid.plus(a(), a(), b())
    return _check_pair(lhs=lhs, rhs=monoid.plus(a(), b()), free_vars=[a, b])


@pytest.mark.parametrize("monoid", IDEMPOTENT)
def test_plus_idempotent_non_consecutive(monoid):
    """``a, b, a`` — Semilattice (Min/Max) collapses via commutative
    PlusDups; plain IdempotentMonoid leaves it as-is (consecutive-only)."""
    a, b = define_vars("a", "b")
    lhs = monoid.plus(a(), b(), a())
    if is_commutative(monoid):
        rhs = monoid.plus(a(), b())
    else:
        rhs = monoid.plus(a(), b(), a())
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[a, b])


def test_plus_commutative_idempotent_long():
    """Long alternation collapses via commutative dedup (Min/Max only)."""
    a, b = define_vars("a", "b")
    lhs = Min.plus(a(), b(), a(), b(), b(), a(), a())
    _check_pair(lhs=lhs, rhs=Min.plus(a(), b()), free_vars=[a, b])


@pytest.mark.parametrize("monoid", WITH_ZERO)
def test_plus_zero(monoid):
    a = define_vars("a")
    lhs_right = monoid.plus(a(), monoid.zero)
    lhs_left = monoid.plus(monoid.zero, a())
    _check_pair(lhs=lhs_right, rhs=monoid.zero, free_vars=[a])
    _check_pair(lhs=lhs_left, rhs=monoid.zero, free_vars=[a])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_1(monoid):
    x, y = define_vars("x", "y")

    lhs = monoid.reduce(x(), {x: []})
    rhs = monoid.identity

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[x, y])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_2(monoid):
    x, y = define_vars("x", "y")
    Y = define_vars("Y", typ=list[int])

    lhs = monoid.reduce(x(), {y: Y(), x: []})
    rhs = monoid.identity

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[x, y, Y])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_3(monoid):
    x, y, a, b = define_vars("x", "y", "a", "b")
    Y = define_vars("Y", typ=list[int])

    lhs = monoid.reduce(x(), {y: Y(), x: [a(), b()]})
    rhs = monoid.plus(monoid.reduce(a(), {y: Y()}), monoid.reduce(b(), {y: Y()}))

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[x, y, a, b, Y])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_4(monoid):
    x, y, a, b = define_vars("x", "y", "a", "b")

    @Operation.define
    def f(_x: int) -> list[int]:
        raise NotHandled

    lhs = monoid.reduce(x(), {y: f(x()), x: [a(), b()]})
    rhs = monoid.plus(monoid.reduce(a(), {y: f(a())}), monoid.reduce(b(), {y: f(b())}))

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[x, y, a, b, f])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_sequence(monoid):
    x = Operation.define(int, name="x")
    X = Operation.define(list[int], name="X")

    @Operation.define
    def f(_x: int) -> int:
        raise NotHandled

    g = Operation.define(f, name="g")

    lhs = monoid.reduce((f(x()), g(x())), {x: X()})
    rhs = (monoid.reduce(f(x()), {x: X()}), monoid.reduce(g(x()), {x: X()}))

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[X, f, g])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_sequence_2(monoid):
    x, y = define_vars("x", "y")
    X, Y = define_vars("X", "Y", typ=list[int])

    @Operation.define
    def f(_x: int) -> int:
        raise NotHandled

    g = Operation.define(f, name="g")

    lhs = monoid.reduce((f(x()), g(y())), {x: X(), y: Y()})
    rhs = (
        monoid.reduce(f(x()), {x: X(), y: Y()}),
        monoid.reduce(g(y()), {x: X(), y: Y()}),
    )

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[X, Y, f, g])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_mapping(monoid):
    x = Operation.define(int, name="x")
    X = Operation.define(list[int], name="X")

    @Operation.define
    def f(_x: int) -> int:
        raise NotHandled

    g = Operation.define(f, name="g")

    lhs = monoid.reduce({"a": f(x()), "b": g(x())}, {x: X()})
    rhs = {
        "a": monoid.reduce(f(x()), {x: X()}),
        "b": monoid.reduce(g(x()), {x: X()}),
    }
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[X, f, g])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_no_streams(monoid):
    a = define_vars("a")
    lhs = monoid.reduce(a(), {})
    rhs = monoid.identity

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[a])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_reduce(monoid):
    a, b = define_vars("a", "b")
    A, B = define_vars("A", "B", typ=list[int])

    @Operation.define
    def f(_x: int, _y: int) -> int:
        raise NotHandled

    lhs = monoid.reduce(monoid.reduce(f(a(), b()), {a: A()}), {b: B()})
    rhs = monoid.reduce(f(a(), b()), {a: A(), b: B()})

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[A, B, f])


@pytest.mark.parametrize("monoid", COMMUTATIVE)
def test_reduce_plus(monoid):
    a, b = define_vars("a", "b")
    A, B = define_vars("A", "B", typ=list[int])
    lhs = monoid.reduce(monoid.plus(a(), b()), {a: A(), b: B()})
    rhs = monoid.plus(
        monoid.reduce(a(), {a: A(), b: B()}),
        monoid.reduce(b(), {a: A(), b: B()}),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[A, B])


def test_reduce_independent_1():
    a, b = define_vars("a", "b")
    A, B = define_vars("A", "B", typ=list[int])
    lhs = Sum.reduce(Product.plus(a(), b()), {a: A(), b: B()})
    rhs = Product.plus(Sum.reduce(a(), {a: A()}), Sum.reduce(b(), {b: B()}))
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[A, B])


def test_reduce_independent_2():
    a, b, c = define_vars("a", "b", "c")
    A, B, C = define_vars("A", "B", "C", typ=list[int])

    @Operation.define
    def f(_x: int, _y: int) -> int:
        raise NotHandled

    lhs = Sum.reduce(Product.plus(a(), b(), f(b(), c())), {a: A(), b: B(), c: C()})
    rhs = Product.plus(
        Sum.reduce(a(), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: B(), c: C()}),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[A, B, C, f])


def test_reduce_independent_3_negative():
    """Stream `b` depends on `a` (b: g(a())), so the proposed factorization
    is unsound — the normalizer must NOT apply it."""
    a, b, c = define_vars("a", "b", "c")
    A, C = define_vars("A", "C", typ=list[int])

    @Operation.define
    def f(_x: int, _y: int) -> int:
        raise NotHandled

    @Operation.define
    def g(_x: int) -> list[int]:
        raise NotHandled

    with handler(NormalizeIntp):
        lhs = Sum.reduce(
            Product.plus(a(), b(), f(b(), c())), {a: A(), b: g(a()), c: C()}
        )
    bogus_rhs = Product.plus(
        Sum.reduce(a(), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: g(a()), c: C()}),
    )
    assert fvsof(bogus_rhs) != fvsof(lhs)
    assert not syntactic_eq_alpha(lhs, bogus_rhs)


def test_reduce_independent_4():
    a, b, c = define_vars("a", "b", "c")
    A, B, C = define_vars("A", "B", "C", typ=list[int])

    @Operation.define
    def f(_x: int, _y: int) -> int:
        raise NotHandled

    lhs = Sum.reduce(Product.plus(a(), b(), f(b(), c()), 7), {a: A(), b: B(), c: C()})
    rhs = Product.plus(
        7,
        Sum.reduce(a(), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: B(), c: C()}),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[A, B, C, f])


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lifted_1(outer, inner):
    a, i = define_vars("a", "i")
    A, N, A_domain = define_vars("A", "N", "A_domain", typ=list[int])

    @Operation.define
    def f(_: int) -> float:
        raise NotHandled

    term1 = outer.reduce(
        inner.reduce(f(a()), {a: A()}),
        {A: CartesianProduct.reduce(A_domain(), {i: N()})},
    )
    term2 = inner.reduce(outer.reduce(f(a()), {a: A_domain()}), {i: N()})
    _check_pair(lhs=term1, rhs=term2, free_vars=[N, A_domain, f])


def test_reduce_cartesian_1():
    a, i = define_vars("a", "i")
    A = define_vars("A", typ=list[int])

    term1 = Sum.reduce(
        Product.reduce(a(), {a: []}),
        {A: CartesianProduct.reduce([], {i: []})},
    )
    term2 = Product.reduce(Sum.reduce(a(), {a: []}), {i: []})
    assert term1 == term2


def test_reduce_cartesian_2():
    a, i = define_vars("a", "i")
    A = define_vars("A", typ=list[int])

    term1 = Sum.reduce(
        Product.reduce(a(), {a: A()}),
        {A: CartesianProduct.reduce([(0,)], {i: [0]})},
    )
    term2 = Product.reduce(Sum.reduce(a(), {a: [0]}), {i: [0]})
    assert term1 == term2


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lifted_multi_index(outer, inner):
    a, i, j = define_vars("a", "i", "j")
    A, N, M, A_domain = define_vars("A", "N", "M", "A_domain", typ=list[int])

    @Operation.define
    def f(_: int) -> float:
        raise NotHandled

    term1 = outer.reduce(
        inner.reduce(f(a()), {a: A()}),
        {A: CartesianProduct.reduce(A_domain(), {i: N(), j: M()})},
    )
    term2 = inner.reduce(
        outer.reduce(f(a()), {a: A_domain()}),
        {i: N(), j: M()},
    )
    _check_pair(lhs=term1, rhs=term2, free_vars=[N, M, A_domain, f])


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lifted_2(outer, inner):
    """The worked example on page 396 of 'Lifted Variable Elimination:
    Decoupling the Operators from the Constraint Language'.

    """
    a, i, s, t = define_vars("a", "i", "s", "t")
    A, N, T = define_vars("A", "N", "T", typ=list[int])

    @Operation.define
    def A_domain(_i: int) -> list[int]:
        raise NotHandled

    @Operation.define
    def f1(_a: int, _s: int) -> float:
        raise NotHandled

    @Operation.define
    def f2(_t: int, _a: int) -> float:
        raise NotHandled

    term1 = outer.reduce(
        inner.reduce(inner.plus(f1(a(), s()), f2(t(), a())), {a: A()}),
        {A: CartesianProduct.reduce(A_domain(i()), {i: N()}), t: T()},
    )

    term2 = outer.reduce(
        inner.reduce(
            outer.reduce(inner.plus(f1(a(), s()), f2(t(), a())), {a: A_domain(i())}),
            {i: N()},
        ),
        {t: T()},
    )

    _check_pair(lhs=term1, rhs=term2, free_vars=[a, i, s, t, A, N, T, A_domain, f1, f2])
