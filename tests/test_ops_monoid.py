import typing

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effectful.handlers.jax.monoid import JaxEvaluateIntp
from effectful.ops.monoid import (
    CartesianProduct,
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
from effectful.ops.types import Operation
from tests._monoid_helpers import (
    INT_BACKEND,
    JAX_BACKEND,
    Backend,
    define_vars,
    random_interpretation,
    syntactic_eq_alpha,
)


@pytest.fixture(params=[INT_BACKEND, JAX_BACKEND], ids=["int", "jax"])
def backend(request) -> Backend:
    return request.param


@pytest.fixture(autouse=True)
def _install_normalize_intp(backend):
    """Install :data:`NormalizeIntp` (plus JAX kernels when the backend is
    jax) for every test in this module.
    """
    intp = NormalizeIntp
    if backend.scalar_typ is not int:
        intp = coproduct(intp, JaxEvaluateIntp)
    with handler(intp):
        yield


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
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_associativity(monoid, backend, data):
    a = data.draw(backend.scalar_strategy)
    b = data.draw(backend.scalar_strategy)
    c = data.draw(backend.scalar_strategy)
    left = monoid.plus(monoid.plus(a, b), c)
    right = monoid.plus(a, monoid.plus(b, c))
    assert backend.eq(left, right)


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
@given(data=st.data())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_identity(monoid, backend, data):
    a = data.draw(backend.scalar_strategy)
    ident = backend.lift(monoid.identity)
    assert backend.eq(monoid.plus(ident, a), a)
    assert backend.eq(monoid.plus(a, ident), a)


@pytest.mark.parametrize("monoid", COMMUTATIVE)
@given(data=st.data())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_commutativity(monoid, backend, data):
    a = data.draw(backend.scalar_strategy)
    b = data.draw(backend.scalar_strategy)
    assert backend.eq(monoid.plus(a, b), monoid.plus(b, a))


@pytest.mark.parametrize("monoid", IDEMPOTENT)
@given(data=st.data())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_idempotence(monoid, backend, data):
    a = data.draw(backend.scalar_strategy)
    assert backend.eq(monoid.plus(a, a), a)


@pytest.mark.parametrize("monoid", WITH_ZERO)
@given(data=st.data())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_zero_absorbs(monoid, backend, data):
    a = data.draw(backend.scalar_strategy)
    zero = backend.lift(monoid.zero)
    assert backend.eq(monoid.plus(zero, a), monoid.zero)
    assert backend.eq(monoid.plus(a, zero), monoid.zero)


def _check_pair(
    lhs, rhs, *, backend: Backend, free_vars=[], max_examples: int = 25
) -> None:
    """Run structural + semantic checks on a TermPair."""
    with handler(NormalizeIntp):
        norm = evaluate(lhs)

    assert syntactic_eq_alpha(norm, rhs)

    @given(intp=random_interpretation(free_vars))
    @settings(
        max_examples=max_examples,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def _check_semantics(intp):
        with handler(intp):
            lhs_val = evaluate(lhs)
            rhs_val = evaluate(rhs)
        assert backend.eq(lhs_val, rhs_val)

    _check_semantics()


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_empty(monoid, backend):
    _check_pair(lhs=monoid.plus(), rhs=monoid.identity, backend=backend)


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_single(monoid, backend):
    x = define_vars("x", typ=backend.scalar_typ)
    _check_pair(lhs=monoid.plus(x()), rhs=x(), backend=backend, free_vars=[x])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_identity_right(monoid, backend):
    x = define_vars("x", typ=backend.scalar_typ)
    _check_pair(
        lhs=monoid.plus(x(), monoid.identity),
        rhs=x(),
        backend=backend,
        free_vars=[x],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_identity_left(monoid, backend):
    x = define_vars("x", typ=backend.scalar_typ)
    _check_pair(
        lhs=monoid.plus(monoid.identity, x()),
        rhs=x(),
        backend=backend,
        free_vars=[x],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_assoc_right(monoid, backend):
    x, y, z = define_vars("x", "y", "z", typ=backend.scalar_typ)
    _check_pair(
        lhs=monoid.plus(x(), monoid.plus(y(), z())),
        rhs=monoid.plus(x(), y(), z()),
        backend=backend,
        free_vars=[x, y, z],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_assoc_left(monoid, backend):
    x, y, z = define_vars("x", "y", "z", typ=backend.scalar_typ)
    _check_pair(
        lhs=monoid.plus(monoid.plus(x(), y()), z()),
        rhs=monoid.plus(x(), y(), z()),
        backend=backend,
        free_vars=[x, y, z],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_sequence(monoid, backend):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=backend.scalar_typ)
    _check_pair(
        lhs=monoid.plus((a(), b()), (c(), d())),
        rhs=(monoid.plus(a(), c()), monoid.plus(b(), d())),
        backend=backend,
        free_vars=[a, b, c, d],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_mapping(monoid, backend):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=backend.scalar_typ)
    _check_pair(
        lhs=monoid.plus({0: a(), 1: b()}, {0: c(), 2: d()}),
        rhs={0: monoid.plus(a(), c()), 1: b(), 2: d()},
        backend=backend,
        free_vars=[a, b, c, d],
    )


def test_plus_distributes(backend):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=backend.scalar_typ)
    lhs = Product.plus(Sum.plus(a(), b()), Sum.plus(c(), d()))
    rhs = Sum.plus(
        Product.plus(a(), c()),
        Product.plus(a(), d()),
        Product.plus(b(), c()),
        Product.plus(b(), d()),
    )
    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[a, b, c, d])


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
    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[a, b, c, d])


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
    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[a, b, c, d])


@pytest.mark.parametrize("monoid", IDEMPOTENT)
def test_plus_idempotent_consecutive(monoid, backend):
    """``a, a, b → a, b`` — only consecutive duplicates collapse."""
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    lhs = monoid.plus(a(), a(), b())
    return _check_pair(
        lhs=lhs, rhs=monoid.plus(a(), b()), backend=backend, free_vars=[a, b]
    )


@pytest.mark.parametrize("monoid", IDEMPOTENT)
def test_plus_idempotent_non_consecutive(monoid, backend):
    """``a, b, a`` — Semilattice (Min/Max) collapses via commutative
    PlusDups; plain IdempotentMonoid leaves it as-is (consecutive-only)."""
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    lhs = monoid.plus(a(), b(), a())
    if is_commutative(monoid):
        rhs = monoid.plus(a(), b())
    else:
        rhs = monoid.plus(a(), b(), a())
    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[a, b])


def test_plus_commutative_idempotent_long(backend):
    """Long alternation collapses via commutative dedup (Min/Max only)."""
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    lhs = Min.plus(a(), b(), a(), b(), b(), a(), a())
    _check_pair(
        lhs=lhs, rhs=Min.plus(a(), b()), backend=backend, free_vars=[a, b]
    )


@pytest.mark.parametrize("monoid", WITH_ZERO)
def test_plus_zero(monoid, backend):
    a = define_vars("a", typ=backend.scalar_typ)
    lhs_right = monoid.plus(a(), monoid.zero)
    lhs_left = monoid.plus(monoid.zero, a())
    _check_pair(lhs=lhs_right, rhs=monoid.zero, backend=backend, free_vars=[a])
    _check_pair(lhs=lhs_left, rhs=monoid.zero, backend=backend, free_vars=[a])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_1(monoid, backend):
    x, y = define_vars("x", "y", typ=backend.scalar_typ)

    lhs = monoid.reduce(x(), {x: []})
    rhs = monoid.identity

    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[x, y])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_2(monoid, backend):
    x, y = define_vars("x", "y", typ=backend.scalar_typ)
    Y = define_vars("Y", typ=backend.stream_typ)

    lhs = monoid.reduce(x(), {y: Y(), x: []})
    rhs = monoid.identity

    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[x, y, Y])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_3(monoid, backend):
    x, y, a, b = define_vars("x", "y", "a", "b", typ=backend.scalar_typ)
    Y = define_vars("Y", typ=backend.stream_typ)

    lhs = monoid.reduce(x(), {y: Y(), x: [a(), b()]})
    rhs = monoid.plus(monoid.reduce(a(), {y: Y()}), monoid.reduce(b(), {y: Y()}))

    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[x, y, a, b, Y])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_4(monoid, backend):
    x, y, a, b = define_vars("x", "y", "a", "b", typ=backend.scalar_typ)
    f = backend.fresh_op("f", n_args=1, ret="stream")

    lhs = monoid.reduce(x(), {y: f(x()), x: [a(), b()]})
    rhs = monoid.plus(monoid.reduce(a(), {y: f(a())}), monoid.reduce(b(), {y: f(b())}))

    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[x, y, a, b, f])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_sequence(monoid, backend):
    x = Operation.define(backend.scalar_typ, name="x")
    X = Operation.define(backend.stream_typ, name="X")
    f = backend.fresh_op("f", n_args=1, ret="scalar")
    g = Operation.define(f, name="g")

    lhs = monoid.reduce((f(x()), g(x())), {x: X()})
    rhs = (monoid.reduce(f(x()), {x: X()}), monoid.reduce(g(x()), {x: X()}))

    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[X, f, g])


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

    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[X, Y, f, g])


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
    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[X, f, g])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_no_streams(monoid, backend):
    a = define_vars("a", typ=backend.scalar_typ)
    lhs = monoid.reduce(a(), {})
    rhs = monoid.identity

    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[a])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_reduce(monoid, backend):
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    A, B = define_vars("A", "B", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=2, ret="scalar")

    lhs = monoid.reduce(monoid.reduce(f(a(), b()), {a: A()}), {b: B()})
    rhs = monoid.reduce(f(a(), b()), {a: A(), b: B()})

    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[A, B, f])


@pytest.mark.parametrize("monoid", COMMUTATIVE)
def test_reduce_plus(monoid, backend):
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    A, B = define_vars("A", "B", typ=backend.stream_typ)
    lhs = monoid.reduce(monoid.plus(a(), b()), {a: A(), b: B()})
    rhs = monoid.plus(
        monoid.reduce(a(), {a: A(), b: B()}),
        monoid.reduce(b(), {a: A(), b: B()}),
    )
    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[A, B])


def test_reduce_independent_1(backend):
    a, b = define_vars("a", "b", typ=backend.scalar_typ)
    A, B = define_vars("A", "B", typ=backend.stream_typ)
    lhs = Sum.reduce(Product.plus(a(), b()), {a: A(), b: B()})
    rhs = Product.plus(Sum.reduce(a(), {a: A()}), Sum.reduce(b(), {b: B()}))
    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[A, B])


def test_reduce_independent_2(backend):
    a, b, c = define_vars("a", "b", "c", typ=backend.scalar_typ)
    A, B, C = define_vars("A", "B", "C", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=2, ret="scalar")

    lhs = Sum.reduce(Product.plus(a(), b(), f(b(), c())), {a: A(), b: B(), c: C()})
    rhs = Product.plus(
        Sum.reduce(a(), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: B(), c: C()}),
    )
    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[A, B, C, f])


def test_reduce_independent_3_negative(backend):
    """Stream `b` depends on `a` (b: g(a())), so the proposed factorization
    is unsound — the normalizer must NOT apply it."""
    a, b, c = define_vars("a", "b", "c", typ=backend.scalar_typ)
    A, C = define_vars("A", "C", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=2, ret="scalar")
    g = backend.fresh_op("g", n_args=1, ret="stream")

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


def test_reduce_independent_4(backend):
    a, b, c = define_vars("a", "b", "c", typ=backend.scalar_typ)
    A, B, C = define_vars("A", "B", "C", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=2, ret="scalar")

    lhs = Sum.reduce(Product.plus(a(), b(), f(b(), c()), 7), {a: A(), b: B(), c: C()})
    rhs = Product.plus(
        7,
        Sum.reduce(a(), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: B(), c: C()}),
    )
    _check_pair(lhs=lhs, rhs=rhs, backend=backend, free_vars=[A, B, C, f])


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lifted_1(outer, inner, backend):
    a, i = define_vars("a", "i", typ=backend.scalar_typ)
    A, N, A_domain = define_vars("A", "N", "A_domain", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=1, ret="scalar")

    term1 = outer.reduce(
        inner.reduce(f(a()), {a: A()}),
        {A: CartesianProduct.reduce(A_domain(), {i: N()})},
    )
    term2 = inner.reduce(outer.reduce(f(a()), {a: A_domain()}), {i: N()})
    _check_pair(lhs=term1, rhs=term2, backend=backend, free_vars=[N, A_domain, f])


def test_reduce_cartesian_1(backend):
    a, i = define_vars("a", "i", typ=backend.scalar_typ)
    A = define_vars("A", typ=backend.stream_typ)

    term1 = Sum.reduce(
        Product.reduce(a(), {a: []}),
        {A: CartesianProduct.reduce([], {i: []})},
    )
    term2 = Product.reduce(Sum.reduce(a(), {a: []}), {i: []})
    assert term1 == term2


def test_reduce_cartesian_2(backend):
    a, i = define_vars("a", "i", typ=backend.scalar_typ)
    A = define_vars("A", typ=backend.stream_typ)

    term1 = Sum.reduce(
        Product.reduce(a(), {a: A()}),
        {A: CartesianProduct.reduce([(0,)], {i: [0]})},
    )
    term2 = Product.reduce(Sum.reduce(a(), {a: [0]}), {i: [0]})
    assert term1 == term2


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lifted_multi_index(outer, inner, backend):
    a, i, j = define_vars("a", "i", "j", typ=backend.scalar_typ)
    A, N, M, A_domain = define_vars(
        "A", "N", "M", "A_domain", typ=backend.stream_typ
    )
    f = backend.fresh_op("f", n_args=1, ret="scalar")

    term1 = outer.reduce(
        inner.reduce(f(a()), {a: A()}),
        {A: CartesianProduct.reduce(A_domain(), {i: N(), j: M()})},
    )
    term2 = inner.reduce(
        outer.reduce(f(a()), {a: A_domain()}),
        {i: N(), j: M()},
    )
    _check_pair(
        lhs=term1, rhs=term2, backend=backend, free_vars=[N, M, A_domain, f]
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
            outer.reduce(inner.plus(f1(a(), s()), f2(t(), a())), {a: A_domain(i())}),
            {i: N()},
        ),
        {t: T()},
    )

    _check_pair(
        lhs=term1,
        rhs=term2,
        backend=backend,
        free_vars=[a, i, s, t, A, N, T, A_domain, f1, f2],
    )
