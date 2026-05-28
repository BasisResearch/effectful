import math
import typing
from collections.abc import Iterable

import jax
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import effectful.handlers.jax.monoid  # noqa: F401
import effectful.handlers.jax.numpy as jnp
from effectful.ops.monoid import (
    CartesianProduct,
    Max,
    Min,
    Monoid,
    MonoidOverMapping,
    MonoidOverSequence,
    NormalizeIntp,
    PlusAssoc,
    PlusConsecutiveDups,
    PlusDistr,
    PlusDups,
    PlusEmpty,
    PlusIdentity,
    PlusSingle,
    PlusZero,
    Product,
    ReduceCartesianWeightedStream,
    ReduceDistributeCartesianProduct,
    ReduceFactorization,
    ReduceFusion,
    ReduceNoStreams,
    ReduceSplit,
    ReduceWeightedStream,
    Sum,
    distributes_over,
)
from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.syntax import deffn
from effectful.ops.types import NotHandled, Operation, Term
from tests._monoid_helpers import (
    INT_BACKEND,
    JAX_BACKEND,
    Backend,
    check_rewrite,
    define_vars,
    syntactic_eq_alpha,
)


@pytest.fixture(params=[INT_BACKEND, JAX_BACKEND], ids=["int", "jax"])
def backend(request) -> Backend:
    return request.param


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
@given(data=st.data())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_associativity(monoid, backend, data):
    a = data.draw(backend.scalar_strategy)
    b = data.draw(backend.scalar_strategy)
    c = data.draw(backend.scalar_strategy)
    with handler(NormalizeIntp):
        left = monoid.plus(monoid.plus(a, b), c)
        right = monoid.plus(a, monoid.plus(b, c))
        assert backend.eq(left, right)


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
@given(data=st.data())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_identity(monoid, backend, data):
    a = data.draw(backend.scalar_strategy)
    with handler(NormalizeIntp):
        assert backend.eq(monoid.plus(monoid.identity, a), a)
        assert backend.eq(monoid.plus(a, monoid.identity), a)


@pytest.mark.parametrize("monoid", COMMUTATIVE)
@given(data=st.data())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_commutativity(monoid, backend, data):
    a = data.draw(backend.scalar_strategy)
    b = data.draw(backend.scalar_strategy)
    with handler(NormalizeIntp):
        assert backend.eq(monoid.plus(a, b), monoid.plus(b, a))


@pytest.mark.parametrize("monoid", IDEMPOTENT)
@given(data=st.data())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_idempotence(monoid, backend, data):
    a = data.draw(backend.scalar_strategy)
    with handler(NormalizeIntp):
        assert backend.eq(monoid.plus(a, a), a)


@pytest.mark.parametrize("monoid", WITH_ZERO)
@given(data=st.data())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_zero_absorbs(monoid, backend, data):
    a = data.draw(backend.scalar_strategy)
    with handler(NormalizeIntp):
        assert backend.eq(monoid.plus(monoid.zero, a), monoid.zero)
        assert backend.eq(monoid.plus(a, monoid.zero), monoid.zero)


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_empty(monoid, backend):
    check_rewrite(
        lhs=monoid.plus(), rhs=monoid.identity, rule=PlusEmpty(), backend=backend
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_single(monoid, backend):
    x = define_vars("x", typ=backend.scalar_typ)
    check_rewrite(
        lhs=monoid.plus(x()), rhs=x(), rule=PlusSingle(), backend=backend, free_vars=[x]
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_identity_right(monoid, backend):
    x = define_vars("x", typ=backend.scalar_typ)

    lhs = monoid.plus(x(), monoid.identity)
    rhs = monoid.plus(x())

    check_rewrite(lhs=lhs, rhs=rhs, rule=PlusIdentity(), backend=backend, free_vars=[x])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_identity_left(monoid, backend):
    x = define_vars("x", typ=backend.scalar_typ)

    lhs = monoid.plus(monoid.identity, x())
    rhs = monoid.plus(x())

    check_rewrite(lhs=lhs, rhs=rhs, rule=PlusIdentity(), backend=backend, free_vars=[x])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_assoc_right(monoid, backend):
    x, y, z = define_vars("x", "y", "z", typ=backend.scalar_typ)
    check_rewrite(
        lhs=monoid.plus(x(), monoid.plus(y(), z())),
        rhs=monoid.plus(x(), y(), z()),
        rule=PlusAssoc(),
        backend=backend,
        free_vars=[x, y, z],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_assoc_left(monoid, backend):
    x, y, z = define_vars("x", "y", "z", typ=backend.scalar_typ)
    check_rewrite(
        lhs=monoid.plus(monoid.plus(x(), y()), z()),
        rhs=monoid.plus(x(), y(), z()),
        rule=PlusAssoc(),
        backend=backend,
        free_vars=[x, y, z],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_sequence(monoid, backend):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=backend.scalar_typ)
    check_rewrite(
        lhs=monoid.plus((a(), b()), (c(), d())),
        rhs=(monoid.plus(a(), c()), monoid.plus(b(), d())),
        rule=MonoidOverSequence(),
        backend=backend,
        free_vars=[a, b, c, d],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_mapping(monoid, backend):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=backend.scalar_typ)

    lhs = monoid.plus({0: a(), 1: b()}, {0: c(), 2: d()})
    rhs = {0: monoid.plus(a(), c()), 1: monoid.plus(b()), 2: monoid.plus(d())}

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=MonoidOverMapping(),
        backend=backend,
        free_vars=[a, b, c, d],
    )


def test_plus_distributes(backend):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=backend.scalar_typ)
    lhs = Product.plus(Sum.plus(a(), b()), Sum.plus(c(), d()))
    rhs = Product.plus(
        Sum.plus(
            Product.plus(a(), c()),
            Product.plus(a(), d()),
            Product.plus(b(), c()),
            Product.plus(b(), d()),
        )
    )
    check_rewrite(
        lhs=lhs, rhs=rhs, rule=PlusDistr(), backend=backend, free_vars=[a, b, c, d]
    )


def test_plus_distributes_constant(backend):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=backend.scalar_typ)
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
    check_rewrite(
        lhs=lhs, rhs=rhs, rule=PlusDistr(), backend=backend, free_vars=[a, b, c, d]
    )


def test_plus_distributes_multiple(backend):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=backend.scalar_typ)
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
    check_rewrite(
        lhs=lhs, rhs=rhs, rule=PlusDistr(), backend=backend, free_vars=[a, b, c, d]
    )


@pytest.mark.parametrize("monoid", IDEMPOTENT)
def test_plus_idempotent_consecutive(monoid, backend):
    """``a, a, b → a, b`` — only consecutive duplicates collapse."""
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    lhs = monoid.plus(a(), a(), b())
    return check_rewrite(
        lhs=lhs,
        rhs=monoid.plus(a(), b()),
        rule=PlusConsecutiveDups(),
        backend=backend,
        free_vars=[a, b],
    )


@pytest.mark.parametrize("monoid", IDEMPOTENT)
def test_plus_idempotent_non_consecutive(monoid, backend):
    """``a, b, a`` — Semilattice (Min/Max) collapses via commutative
    PlusDups."""
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    lhs = monoid.plus(a(), b(), a())
    rhs = monoid.plus(a(), b())
    check_rewrite(lhs=lhs, rhs=rhs, rule=PlusDups(), backend=backend, free_vars=[a, b])


@pytest.mark.parametrize("monoid", [Min, Max])
def test_plus_commutative_idempotent_long(monoid, backend):
    """Long alternation collapses via commutative dedup (Min/Max only)."""
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    lhs = monoid.plus(a(), b(), a(), b(), b(), a(), a())
    rhs = monoid.plus(a(), b())
    check_rewrite(lhs=lhs, rhs=rhs, rule=PlusDups(), backend=backend, free_vars=[a, b])


@pytest.mark.parametrize("monoid", WITH_ZERO)
def test_plus_zero(monoid, backend):
    a = define_vars("a", typ=backend.scalar_typ)
    lhs_right = monoid.plus(a(), monoid.zero)
    lhs_left = monoid.plus(monoid.zero, a())
    rhs = monoid.zero
    check_rewrite(
        lhs=lhs_right, rhs=rhs, rule=PlusZero(), backend=backend, free_vars=[a]
    )
    check_rewrite(
        lhs=lhs_left, rhs=rhs, rule=PlusZero(), backend=backend, free_vars=[a]
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_1(monoid, backend):
    x, y = define_vars("x", "y", typ=backend.scalar_typ)
    lhs = monoid.reduce(x(), {x: []})
    rhs = monoid.identity
    check_rewrite(lhs=lhs, rhs=rhs, rule={}, backend=backend, free_vars=[x, y])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_2(monoid, backend):
    x, y = define_vars("x", "y", typ=backend.scalar_typ)
    Y = define_vars("Y", typ=backend.stream_typ)

    lhs = monoid.reduce(x(), {y: Y(), x: []})
    rhs = monoid.identity

    check_rewrite(lhs=lhs, rhs=rhs, rule={}, backend=backend, free_vars=[x, y, Y])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_3(monoid, backend):
    x, y, a, b = define_vars("x", "y", "a", "b", typ=backend.scalar_typ)
    Y = define_vars("Y", typ=backend.stream_typ)

    lhs = monoid.reduce(x(), {y: Y(), x: [a(), b()]})
    rhs = monoid.plus(monoid.reduce(a(), {y: Y()}), monoid.reduce(b(), {y: Y()}))

    check_rewrite(lhs=lhs, rhs=rhs, rule={}, backend=backend, free_vars=[x, y, a, b, Y])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_4(monoid, backend):
    x, y, a, b = define_vars("x", "y", "a", "b", typ=backend.scalar_typ)
    f = backend.fresh_op("f", n_args=1, ret="stream")

    lhs = monoid.reduce(x(), {y: f(x()), x: [a(), b()]})
    rhs = monoid.plus(monoid.reduce(a(), {y: f(a())}), monoid.reduce(b(), {y: f(b())}))

    check_rewrite(lhs=lhs, rhs=rhs, rule={}, backend=backend, free_vars=[x, y, a, b, f])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_sequence(monoid, backend):
    x = Operation.define(backend.scalar_typ, name="x")
    X = Operation.define(backend.stream_typ, name="X")
    f = backend.fresh_op("f", n_args=1, ret="scalar")
    g = Operation.define(f, name="g")

    lhs = monoid.reduce((f(x()), g(x())), {x: X()})
    rhs = (monoid.reduce(f(x()), {x: X()}), monoid.reduce(g(x()), {x: X()}))

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=MonoidOverSequence(),
        backend=backend,
        free_vars=[X, f, g],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_sequence_2(monoid, backend):
    x, y = define_vars("x", "y", typ=backend.scalar_typ)
    X, Y = define_vars("X", "Y", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=1, ret="scalar")
    g = Operation.define(f, name="g")

    lhs = monoid.reduce((f(x()), g(y())), {x: X(), y: Y()})
    rhs = (
        monoid.reduce(f(x()), {x: X(), y: Y()}),
        monoid.reduce(g(y()), {x: X(), y: Y()}),
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=MonoidOverSequence(),
        backend=backend,
        free_vars=[X, Y, f, g],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_mapping(monoid, backend):
    x = Operation.define(backend.scalar_typ, name="x")
    X = Operation.define(backend.stream_typ, name="X")
    f = backend.fresh_op("f", n_args=1, ret="scalar")
    g = Operation.define(f, name="g")

    lhs = monoid.reduce({0: f(x()), 1: g(x())}, {x: X()})
    rhs = {
        0: monoid.reduce(f(x()), {x: X()}),
        1: monoid.reduce(g(x()), {x: X()}),
    }
    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=MonoidOverMapping(),
        backend=backend,
        free_vars=[X, f, g],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_no_streams(monoid, backend):
    a = define_vars("a", typ=backend.scalar_typ)
    lhs = monoid.reduce(a(), {})
    rhs = monoid.identity

    check_rewrite(
        lhs=lhs, rhs=rhs, rule=ReduceNoStreams(), backend=backend, free_vars=[a]
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_reduce(monoid, backend):
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    A, B = define_vars("A", "B", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=2, ret="scalar")

    lhs = monoid.reduce(monoid.reduce(f(a(), b()), {a: A()}), {b: B()})
    rhs = monoid.reduce(f(a(), b()), {a: A(), b: B()})

    check_rewrite(
        lhs=lhs, rhs=rhs, rule=ReduceFusion(), backend=backend, free_vars=[A, B, f]
    )


@pytest.mark.parametrize("monoid", COMMUTATIVE)
def test_reduce_plus(monoid, backend):
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    A, B = define_vars("A", "B", typ=backend.stream_typ)
    lhs = monoid.reduce(monoid.plus(a(), b()), {a: A(), b: B()})
    rhs = monoid.plus(
        monoid.reduce(a(), {a: A(), b: B()}),
        monoid.reduce(b(), {a: A(), b: B()}),
    )
    check_rewrite(
        lhs=lhs, rhs=rhs, rule=ReduceSplit(), backend=backend, free_vars=[A, B]
    )


def test_reduce_independent_1(backend):
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    A, B = define_vars("A", "B", typ=backend.stream_typ)
    lhs = Sum.reduce(Product.plus(a(), b()), {a: A(), b: B()})
    rhs = Product.plus(
        Sum.reduce(Product.plus(a()), {a: A()}), Sum.reduce(Product.plus(b()), {b: B()})
    )
    check_rewrite(
        lhs=lhs, rhs=rhs, rule=ReduceFactorization(), backend=backend, free_vars=[A, B]
    )


def test_reduce_independent_2(backend):
    a, b, c = define_vars("a", "b", "c", typ=backend.scalar_typ)
    A, B, C = define_vars("A", "B", "C", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=2, ret="scalar")

    lhs = Sum.reduce(Product.plus(a(), b(), f(b(), c())), {a: A(), b: B(), c: C()})
    rhs = Product.plus(
        Sum.reduce(Product.plus(a()), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: B(), c: C()}),
    )
    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ReduceFactorization(),
        backend=backend,
        free_vars=[A, B, C, f],
    )


def test_reduce_independent_3_negative(backend):
    """Stream `b` depends on `a` (b: g(a())), so the proposed factorization
    is unsound — the normalizer must NOT apply it."""
    a, b, c = define_vars("a", "b", "c", typ=backend.scalar_typ)
    A, C = define_vars("A", "C", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=2, ret="scalar")
    g = backend.fresh_op("g", n_args=1, ret="stream")

    with handler(ReduceFactorization()):  # ty:ignore[invalid-argument-type]
        lhs = Sum.reduce(
            Product.plus(a(), b(), f(b(), c())), {a: A(), b: g(a()), c: C()}
        )
    bogus_rhs = Product.plus(
        Sum.reduce(a(), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: g(a()), c: C()}),
    )
    assert fvsof(bogus_rhs) != fvsof(lhs)
    assert not syntactic_eq_alpha(lhs, bogus_rhs)


def test_reduce_independent_4(backend):
    a, b, c = define_vars("a", "b", "c", typ=backend.scalar_typ)
    A, B, C = define_vars("A", "B", "C", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=2, ret="scalar")

    lhs = Sum.reduce(Product.plus(a(), b(), f(b(), c()), 7), {a: A(), b: B(), c: C()})
    rhs = Product.plus(
        7,
        Sum.reduce(Product.plus(a()), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: B(), c: C()}),
    )
    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ReduceFactorization(),
        backend=backend,
        free_vars=[A, B, C, f],
    )


def test_reduce_cartesian_3():
    i = define_vars("i", typ=jax.Array)

    with handler(NormalizeIntp):
        value = CartesianProduct.reduce(jnp.zeros(2), {i: jnp.arange(3)})
    assert value.shape == (2**3, 3)

    with handler(NormalizeIntp):
        value = CartesianProduct.reduce(jnp.zeros(2), {i: jnp.arange(1)})
    assert value.shape == (2**1, 1)

    with handler(NormalizeIntp):
        value = CartesianProduct.reduce(jnp.zeros(1), {i: jnp.arange(3)})
    assert value.shape == (1**3, 3)


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lifted_1(outer, inner, backend):
    a, i = define_vars("a", "i", typ=backend.scalar_typ)
    A, N, A_domain = define_vars("A", "N", "A_domain", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=1, ret="scalar")

    term1 = outer.reduce(
        inner.reduce(f(a()), {a: A()}),
        {A: CartesianProduct.reduce(A_domain(), {i: N()})},
    )
    term2 = inner.reduce(outer.reduce(inner.plus(f(a())), {a: A_domain()}), {i: N()})

    check_rewrite(
        lhs=term1,
        rhs=term2,
        rule=ReduceDistributeCartesianProduct(),
        backend=backend,
        free_vars=[N, A_domain, f],
    )


def test_reduce_cartesian_1():
    a, i = define_vars("a", "i", typ=int)
    A = define_vars("A", typ=tuple[int])

    with handler(NormalizeIntp):
        term1 = Sum.reduce(
            Product.reduce(a(), {a: []}),
            {A: CartesianProduct.reduce([], {i: []})},
        )
        term2 = Product.reduce(Sum.reduce(a(), {a: []}), {i: []})
    assert term1 == term2


def test_reduce_cartesian_2():
    a, i = define_vars("a", "i", typ=int)
    A = define_vars("A", typ=tuple[int])

    with handler(NormalizeIntp):
        term1 = Sum.reduce(
            Product.reduce(a(), {a: A()}),
            {A: CartesianProduct.reduce([(0,)], {i: [0]})},
        )
        term2 = Product.reduce(Sum.reduce(a(), {a: [0]}), {i: [0]})
    assert term1 == term2


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lifted_multi_index(outer, inner, backend):
    a, i, j = define_vars("a", "i", "j", typ=backend.scalar_typ)
    A, N, M, A_domain = define_vars("A", "N", "M", "A_domain", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=1, ret="scalar")

    term1 = outer.reduce(
        inner.reduce(f(a()), {a: A()}),
        {A: CartesianProduct.reduce(A_domain(), {i: N(), j: M()})},
    )
    term2 = inner.reduce(
        outer.reduce(inner.plus(f(a())), {a: A_domain()}),
        {i: N(), j: M()},
    )
    check_rewrite(
        lhs=term1,
        rhs=term2,
        rule=ReduceDistributeCartesianProduct(),
        backend=backend,
        free_vars=[N, M, A_domain, f],
    )


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lifted_2(outer, inner, backend):
    """The worked example on page 396 of 'Lifted Variable Elimination:
    Decoupling the Operators from the Constraint Language'.

    """
    a, i, s, t = define_vars("a", "i", "s", "t", typ=backend.scalar_typ)
    A, N, T = define_vars("A", "N", "T", typ=backend.stream_typ)
    A_domain = backend.fresh_op("A_domain", n_args=1, ret="stream")
    f1 = backend.fresh_op("f1", n_args=2, ret="scalar")
    f2 = backend.fresh_op("f2", n_args=2, ret="scalar")

    term1 = outer.reduce(
        inner.reduce(inner.plus(f1(a(), s()), f2(t(), a())), {a: A()}),
        {A: CartesianProduct.reduce(A_domain(i()), {i: N()}), t: T()},
    )

    term2 = outer.reduce(
        inner.reduce(
            outer.reduce(
                inner.plus(inner.plus(f1(a(), s()), f2(t(), a()))), {a: A_domain(i())}
            ),
            {i: N()},
        ),
        {t: T()},
    )

    check_rewrite(
        lhs=term1,
        rhs=term2,
        rule=ReduceDistributeCartesianProduct(),
        backend=backend,
        free_vars=[a, i, s, t, A, N, T, A_domain, f1, f2],
    )


# ---------------------------------------------------------------------------
# Weighted streams
# ---------------------------------------------------------------------------


def test_reduce_single_weighted_stream(backend):
    """Single weighted stream desugars:
    Sum.reduce(body, {a: WS(A, w, Product)})
      = Sum.reduce(Product.plus(w(a), body), {a: A})
    """
    a = define_vars("a", typ=backend.scalar_typ)
    A = define_vars("A", typ=backend.stream_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")
    w = backend.fresh_op("w", n_args=1, ret="scalar")

    lhs = Sum.reduce(body(a()), {a: Product.weighted(A(), w)})
    rhs = Sum.reduce(Product.plus(w(a()), body(a())), {a: A()})

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ReduceWeightedStream(),
        backend=backend,
        free_vars=[A, body, w],
    )


def test_reduce_weighted_factorization(backend):
    """Two independent weighted streams under Sum with Product weights factor:
        Sum.reduce(f(a)*g(b), {a: Product.weighted(A, a, w_a), b: Product.weighted(B, b, w_b)})
          = (Sum.reduce(w_a(a)*f(a), {a: A})) * (Sum.reduce(w_b(b)*g(b), {b: B}))

    Exercises chaining of ``ReduceWeightedStream`` with ``ReduceFactorization``
    inside ``NormalizeIntp``.
    """
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    A, B = define_vars("A", "B", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=1, ret="scalar")
    g = backend.fresh_op("g", n_args=1, ret="scalar")
    w_a = backend.fresh_op("w_a", n_args=1, ret="scalar")
    w_b = backend.fresh_op("w_b", n_args=1, ret="scalar")

    lhs = Sum.reduce(
        Product.plus(f(a()), g(b())),
        {a: Product.weighted(A(), w_a), b: Product.weighted(B(), w_b)},
    )
    rhs = Product.plus(
        Sum.reduce(Product.plus(w_a(a()), Product.plus(f(a()))), {a: A()}),
        Sum.reduce(Product.plus(w_b(b()), Product.plus(g(b()))), {b: B()}),
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=coproduct(ReduceWeightedStream(), ReduceFactorization()),
        backend=backend,
        free_vars=[A, B, f, g, w_a, w_b],
    )


def test_reduce_cartesian_weighted_stream(backend):
    """``CartesianProduct.reduce`` over a ``WeightedStream`` body whose weight
    is independent of the plate var rewrites to a single joint
    ``WeightedStream``:

        CartesianProduct.reduce(M.weighted(s, e, w(e)), {p: P})
          = M.weighted(CartesianProduct.reduce(s, {p: P}), row, M.reduce(w(e), {e: row()}))
    """
    p, e_var = define_vars("p", "e_var", typ=backend.scalar_typ)
    S, P = define_vars("S", "P", typ=backend.stream_typ)
    w = backend.fresh_op("w", n_args=1, ret="scalar")

    lhs = CartesianProduct.reduce(Product.weighted(S(), w), {p: P()})

    row_var = Operation.define(Iterable[backend.scalar_typ], name="row")
    rhs = Product.weighted(
        CartesianProduct.reduce(S(), {p: P()}),
        deffn(Product.reduce(w(e_var()), {e_var: row_var()}), row_var),
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ReduceCartesianWeightedStream(),
        backend=backend,
        free_vars=[S, P, w],
    )


def test_lift_weighted_cartesian(backend):
    """Compose ``ReduceCartesianWeightedStream`` + ``ReduceWeightedStream`` +
    ``ReduceDistributeCartesianProduct`` on a Sum-of-Product-of-weighted shape:

        Sum.reduce(
            Product.reduce(body(a()), {a: A()}),
            {A: CartesianProduct.reduce(Product.weighted(S, e, w(e)), {p: P})},
        )

    The inner ``weighted`` becomes a joint ``weighted`` (rule 1), lifts its
    per-element weight into the outer Sum body (rule 2), and the lifted form
    matches the inversion pattern (rule 3), yielding::

        Product.reduce(
            Sum.reduce(Product.plus(w(a()), body(a())), {a: S}),
            {p: P},
        )
    """
    a = define_vars("a", typ=backend.scalar_typ)
    p = define_vars("p", typ=backend.scalar_typ)
    A, S, P = define_vars("A", "S", "P", typ=backend.stream_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")
    w = backend.fresh_op("w", n_args=1, ret="scalar")

    lhs = Sum.reduce(
        Product.reduce(body(a()), {a: A()}),
        {A: CartesianProduct.reduce(Product.weighted(S(), w), {p: P()})},
    )
    rhs = Product.reduce(
        Sum.reduce(Product.plus(w(a()), body(a())), {a: S()}), {p: P()}
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=coproduct(
            coproduct(ReduceWeightedStream(), ReduceCartesianWeightedStream()),
            ReduceDistributeCartesianProduct(),
        ),
        backend=backend,
        free_vars=[S, P, body, w],
    )


def test_weighted_expectation_demo():
    """Demo: compute E[f(X)] = Σ_x w(x)·f(x) via a weighted reduce.

    X ranges over [1, 2, 3, 4] with weights w(x) = x/10 (a valid distribution
    since the weights sum to 1) and f(x) = x*x. Expected value:
        0.1·1 + 0.2·4 + 0.3·9 + 0.4·16 = 10.0
    """
    weights = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}

    def _w(v: int) -> float:
        if isinstance(v, Term):
            raise NotHandled
        return weights[v]

    def _f(v: int) -> float:
        if isinstance(v, Term):
            raise NotHandled
        return float(v * v)

    a = define_vars("a", typ=int)
    w = Operation.define(_w, name="w")
    f = Operation.define(_f, name="f")

    with handler(NormalizeIntp):
        result = evaluate(Sum.reduce(f(a()), {a: Product.weighted([1, 2, 3, 4], w)}))

    assert math.isclose(result, 10.0)
