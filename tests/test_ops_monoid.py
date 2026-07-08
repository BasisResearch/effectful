import math
import typing
from collections.abc import Iterable

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import effectful.handlers.jax.monoid  # noqa: F401
import effectful.handlers.jax.numpy as jnp
from effectful.ops.monoid import (
    And,
    CartesianProduct,
    CartesianProductPlus,
    DeltaEmpty,
    EliminateSingletonStreams,
    Factor,
    Max,
    Min,
    Monoid,
    MonoidOverMapping,
    MonoidOverSequence,
    NormalizeIntp,
    Or,
    PlusAssoc,
    PlusConsecutiveDups,
    PlusDistr,
    PlusEmpty,
    PlusOrder,
    PlusSingle,
    Product,
    ReduceCartesianWeightedStream,
    ReduceDistributeCartesianProduct,
    ReduceEqualityMaskRange,
    ReduceFusion,
    ReduceMaskHoist,
    ReducePartial,
    ReduceSplit,
    ReduceUnion,
    ReduceWeightedStream,
    SplitDisjointProduct,
    Sum,
    Union,
    distributes_over,
)
from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.syntax import Array, deffn
from effectful.ops.types import NotHandled, Operation, Term
from tests._monoid_helpers import Backend, IntBackend, JaxBackend, syntactic_eq_alpha


@pytest.fixture(params=[IntBackend, JaxBackend], ids=["int", "jax"])
def backend(request) -> Backend:
    return request.param()


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
def test_associativity(monoid, backend: Backend, data):
    a = data.draw(backend.strategy(ret="scalar"))()
    b = data.draw(backend.strategy(ret="scalar"))()
    c = data.draw(backend.strategy(ret="scalar"))()
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
def test_identity(monoid, backend: Backend, data):
    a = data.draw(backend.strategy(ret="scalar"))()
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
def test_commutativity(monoid, backend: Backend, data):
    a = data.draw(backend.strategy(ret="scalar"))()
    b = data.draw(backend.strategy(ret="scalar"))()
    with handler(NormalizeIntp):
        assert backend.eq(monoid.plus(a, b), monoid.plus(b, a))


@pytest.mark.parametrize("monoid", IDEMPOTENT)
@given(data=st.data())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_idempotence(monoid, backend: Backend, data):
    a = data.draw(backend.strategy(ret="scalar"))()
    with handler(NormalizeIntp):
        assert backend.eq(monoid.plus(a, a), a)


@pytest.mark.parametrize("monoid", WITH_ZERO)
@given(data=st.data())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_zero_absorbs(monoid, backend: Backend, data):
    a = data.draw(backend.strategy(ret="scalar"))()
    with handler(NormalizeIntp):
        assert backend.eq(monoid.plus(monoid.zero, a), monoid.zero)
        assert backend.eq(monoid.plus(a, monoid.zero), monoid.zero)


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_empty(monoid, backend: Backend):
    backend.check_rewrite(lhs=monoid.plus(), rhs=monoid.identity, rule=PlusEmpty())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_single(monoid, backend: Backend):
    x = backend.define_vars("x", ret="scalar")
    backend.check_rewrite(lhs=monoid.plus(x()), rhs=x(), rule=PlusSingle())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_identity_right(monoid, backend: Backend):
    x = backend.define_vars("x", ret="scalar")

    lhs = monoid.plus(x(), monoid.identity)
    rhs = monoid.plus(x())

    backend.check_rewrite(lhs=lhs, rhs=rhs, rule={})


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_identity_left(monoid, backend: Backend):
    x = backend.define_vars("x", ret="scalar")

    lhs = monoid.plus(monoid.identity, x())
    rhs = monoid.plus(x())

    backend.check_rewrite(lhs=lhs, rhs=rhs, rule={})


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_assoc_right(monoid, backend: Backend):
    x, y, z = backend.define_vars("x", "y", "z", ret="scalar")
    backend.check_rewrite(
        lhs=monoid.plus(x(), monoid.plus(y(), z())),
        rhs=monoid.plus(x(), y(), z()),
        rule=PlusAssoc(),
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_assoc_left(monoid, backend: Backend):
    x, y, z = backend.define_vars("x", "y", "z", ret="scalar")
    backend.check_rewrite(
        lhs=monoid.plus(monoid.plus(x(), y()), z()),
        rhs=monoid.plus(x(), y(), z()),
        rule=PlusAssoc(),
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_sequence(monoid, backend: Backend):
    a, b, c, d = backend.define_vars("a", "b", "c", "d", ret="scalar")
    backend.check_rewrite(
        lhs=monoid.plus((a(), b()), (c(), d())),
        rhs=(monoid.plus(a(), c()), monoid.plus(b(), d())),
        rule=MonoidOverSequence(),
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_mapping(monoid, backend: Backend):
    a, b, c, d = backend.define_vars("a", "b", "c", "d", ret="scalar")

    lhs = monoid.plus({0: a(), 1: b()}, {0: c(), 2: d()})
    rhs = {0: monoid.plus(a(), c()), 1: monoid.plus(b()), 2: monoid.plus(d())}

    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=MonoidOverMapping())


def test_plus_distributes_1(backend: Backend):
    a, b, c = backend.define_vars("a", "b", "c", ret="scalar")
    lhs = Product.plus(c(), Sum.plus(a(), b()))
    rhs = Sum.plus(Product.plus(c(), a()), Product.plus(c(), b()))
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=coproduct(PlusDistr(), coproduct(PlusSingle(), PlusAssoc())),
    )


def test_plus_distributes_2(backend: Backend):
    a, b, c = backend.define_vars("a", "b", "c", ret="scalar")
    lhs = Product.plus(Sum.plus(a(), b()), c())
    rhs = Sum.plus(Product.plus(a(), c()), Product.plus(b(), c()))
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=coproduct(PlusDistr(), coproduct(PlusSingle(), PlusAssoc())),
    )


def test_plus_distributes_3(backend: Backend):
    a, b, c, d = backend.define_vars("a", "b", "c", "d", ret="scalar")
    lhs = Product.plus(Sum.plus(a(), b()), Sum.plus(c(), d()))
    rhs = Sum.plus(
        Product.plus(a(), c()),
        Product.plus(a(), d()),
        Product.plus(b(), c()),
        Product.plus(b(), d()),
    )
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=coproduct(PlusDistr(), coproduct(PlusSingle(), PlusAssoc())),
    )


def test_plus_distributes_4(backend: Backend):
    a, b, c, d = backend.define_vars("a", "b", "c", "d", ret="scalar")
    lhs = Product.plus(Sum.plus(a(), b()), c(), d())
    rhs = Sum.plus(Product.plus(a(), c(), d()), Product.plus(b(), c(), d()))
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=coproduct(PlusDistr(), coproduct(PlusSingle(), PlusAssoc())),
    )


def test_plus_distributes_constant(backend: Backend):
    a, b, c, d, e = backend.define_vars("a", "b", "c", "d", "e", ret="scalar")
    lhs = Product.plus(Sum.plus(a(), b()), Sum.plus(c(), d()), e())
    rhs = Sum.plus(
        Product.plus(a(), c(), e()),
        Product.plus(a(), d(), e()),
        Product.plus(b(), c(), e()),
        Product.plus(b(), d(), e()),
    )
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=coproduct(PlusDistr(), coproduct(PlusSingle(), PlusAssoc())),
    )


def test_plus_distributes_multiple(backend: Backend):
    a, b, c, d = backend.define_vars("a", "b", "c", "d", ret="scalar")
    lhs = Sum.plus(Min.plus(a(), b()), Max.plus(c(), d()))
    rhs = Min.plus(
        Max.plus(Sum.plus(a(), c()), Sum.plus(a(), d())),
        Max.plus(Sum.plus(b(), c()), Sum.plus(b(), d())),
    )
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=coproduct(PlusDistr(), coproduct(PlusSingle(), PlusAssoc())),
    )


@pytest.mark.parametrize("monoid", IDEMPOTENT)
def test_plus_idempotent_consecutive(monoid, backend: Backend):
    """``a, a, b → a, b`` — only consecutive duplicates collapse."""
    a, b = backend.define_vars("a", "b", ret="scalar")
    lhs = monoid.plus(a(), a(), b())
    return backend.check_rewrite(
        lhs=lhs, rhs=monoid.plus(a(), b()), rule=PlusConsecutiveDups()
    )


@pytest.mark.parametrize("monoid", IDEMPOTENT)
def test_plus_idempotent_non_consecutive(monoid, backend: Backend):
    a, b = backend.define_vars("a", "b", ret="scalar")
    lhs = monoid.plus(a(), b(), a())
    rhs = monoid.plus(a(), b(), a())
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=PlusConsecutiveDups())


@pytest.mark.parametrize("monoid", [Min, Max])
def test_plus_commutative_idempotent_long(monoid, backend: Backend):
    """Long alternation collapses via commutative dedup (Min/Max only)."""
    lhs = monoid.plus(0, 1, 0, 1, 1, 0, 0)
    rhs = monoid.plus(0, 1)
    backend.check_rewrite(
        lhs=lhs, rhs=rhs, rule=coproduct(PlusOrder(), PlusConsecutiveDups())
    )


@pytest.mark.parametrize("monoid", WITH_ZERO)
def test_plus_zero(monoid, backend: Backend):
    a = backend.define_vars("a", ret="scalar")
    lhs_right = monoid.plus(a(), monoid.zero)
    lhs_left = monoid.plus(monoid.zero, a())
    rhs = monoid.zero
    backend.check_rewrite(lhs=lhs_right, rhs=rhs, rule={})
    backend.check_rewrite(lhs=lhs_left, rhs=rhs, rule={})


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_1(monoid, backend: Backend):
    x = backend.define_vars("x", ret="scalar")
    lhs = monoid.reduce(x(), {x: []})
    rhs = monoid.plus()
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReducePartial())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_2(monoid, backend: Backend):
    x, y = backend.define_vars("x", "y", ret="scalar")
    Y = backend.define_vars("Y", ret="stream")

    lhs = monoid.reduce(x(), {y: Y(), x: []})
    rhs = monoid.plus()
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReducePartial())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_3(monoid, backend: Backend):
    x, y, a, b = backend.define_vars("x", "y", "a", "b", ret="scalar")
    Y = backend.define_vars("Y", ret="stream")

    lhs = monoid.reduce(x(), {y: Y(), x: [a(), b()]})
    rhs = monoid.plus(monoid.reduce(a(), {y: Y()}), monoid.reduce(b(), {y: Y()}))
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReducePartial())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_partial_4(monoid, backend: Backend):
    x, y, a, b = backend.define_vars("x", "y", "a", "b", ret="scalar")
    f = backend.define_vars("f", arg_types=(backend.scalar_typ,), ret="stream")

    lhs = monoid.reduce(x(), {y: f(x()), x: [a(), b()]})
    rhs = monoid.plus(monoid.reduce(a(), {y: f(a())}), monoid.reduce(b(), {y: f(b())}))
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReducePartial())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_eliminate_singleton_into_sibling(monoid, backend: Backend):
    """A length-1 stream substitutes its element into the body *and* into a
    sibling stream's definition, then drops out of the nest."""
    x, y, a = backend.define_vars("x", "y", "a", ret="scalar")
    f = backend.define_vars("f", arg_types=(backend.scalar_typ,), ret="stream")
    g = backend.define_vars(
        "g", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = monoid.reduce(g(x(), y()), {x: (a(),), y: f(x())})
    rhs = monoid.reduce(g(a(), y()), {y: f(a())})
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=EliminateSingletonStreams())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_eliminate_singleton_only_stream(monoid, backend: Backend):
    """When the length-1 stream is the only stream, reducing over the now-empty
    nest yields the substituted body itself (not the monoid identity)."""
    x, a = backend.define_vars("x", "a", ret="scalar")
    f = backend.define_vars("f", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = monoid.reduce(f(x()), {x: (a(),)})
    rhs = f(a())
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=EliminateSingletonStreams())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_sequence(monoid, backend: Backend):
    x = backend.define_vars("x", ret="scalar")
    X = backend.define_vars("X", ret="stream")
    f, g = backend.define_vars("f", "g", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = monoid.reduce((f(x()), g(x())), {x: X()})
    rhs = (monoid.reduce(f(x()), {x: X()}), monoid.reduce(g(x()), {x: X()}))
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=MonoidOverSequence())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_sequence_2(monoid, backend: Backend):
    x, y = backend.define_vars("x", "y", ret="scalar")
    X, Y = backend.define_vars("X", "Y", ret="stream")
    f, g = backend.define_vars("f", "g", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = monoid.reduce((f(x()), g(y())), {x: X(), y: Y()})
    rhs = (
        monoid.reduce(f(x()), {x: X(), y: Y()}),
        monoid.reduce(g(y()), {x: X(), y: Y()}),
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=MonoidOverSequence())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_mapping(monoid, backend: Backend):
    x = backend.define_vars("x", ret="scalar")
    X = backend.define_vars("X", ret="stream")
    f, g = backend.define_vars("f", "g", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = monoid.reduce({0: f(x()), 1: g(x())}, {x: X()})
    rhs = {
        0: monoid.reduce(f(x()), {x: X()}),
        1: monoid.reduce(g(x()), {x: X()}),
    }
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=MonoidOverMapping())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_no_streams(monoid, backend: Backend):
    a = backend.define_vars("a", ret="scalar")

    lhs = monoid.reduce(a(), {})
    rhs = monoid.identity
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReducePartial())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_reduce(monoid, backend: Backend):
    a, b = backend.define_vars("a", "b", ret="scalar")
    A, B = backend.define_vars("A", "B", ret="stream")
    f = backend.define_vars(
        "f", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = monoid.reduce(monoid.reduce(f(a(), b()), {a: A()}), {b: B()})
    rhs = monoid.reduce(f(a(), b()), {a: A(), b: B()})
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceFusion())


@pytest.mark.parametrize("monoid", COMMUTATIVE)
def test_reduce_split_subset(monoid, backend: Backend):
    """ReduceSplit confines a stream to the summands that use it: ``a`` is used
    only by the first summand, so it is pushed into a reduce over just that
    summand (the constant summand keeps its -- now innermost -- ``a`` reduce).
    """
    a = backend.define_vars("a", ret="scalar")
    A = backend.define_vars("A", ret="stream")
    c = backend.define_vars("c", ret="scalar")
    f = backend.define_vars("f", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = monoid.reduce(monoid.plus(f(a()), c()), {a: A()})
    rhs = monoid.plus(
        monoid.reduce(f(a()), {a: A()}),
        monoid.reduce(c(), {a: A()}),
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceSplit())


@pytest.mark.parametrize("monoid", COMMUTATIVE)
def test_reduce_split_shared_noop(monoid, backend: Backend):
    """ReduceSplit leaves a reduce fused when every summand uses the stream --
    the form ``ReduceDistributeCartesianProduct`` relies on.
    """
    a = backend.define_vars("a", ret="scalar")
    A = backend.define_vars("A", ret="stream")
    f, g = backend.define_vars("f", "g", arg_types=(backend.scalar_typ,), ret="scalar")

    term = monoid.reduce(monoid.plus(f(a()), g(a())), {a: A()})
    backend.check_rewrite(lhs=term, rhs=term, rule=ReduceSplit())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_mask_hoist(monoid):
    """A reduce-stream-independent mask condition lifts out of the reduce:
    ``M.reduce(M.mask(v, c), {a: A}) == M.mask(M.reduce(v, {a: A}), c)``.

    Pinned to ``IntBackend``: ``Monoid.mask`` evaluates symbolically (via
    ``MaskConcrete``) under the ops stack used by ``check_rewrite``; jax-array
    masks are exercised separately in ``test_handlers_jax_monoid``.
    """
    backend = IntBackend()
    a, c, d = backend.define_vars("a", "c", "d", ret="scalar")
    A = backend.define_vars("A", ret="stream")
    f = backend.define_vars("f", arg_types=(backend.scalar_typ,), ret="scalar")

    cond = c() == d()  # independent of the reduced stream `a`
    lhs = monoid.reduce(monoid.mask(f(a()), cond), {a: A()})
    rhs = monoid.mask(monoid.reduce(f(a()), {a: A()}), cond)
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceMaskHoist())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_mask_hoist_dependent_noop(monoid):
    """The mask does NOT hoist when its condition depends on the reduced
    stream (that case is gather territory for ``ReduceEqualityMaskRange``).
    """
    backend = IntBackend()
    a, c = backend.define_vars("a", "c", ret="scalar")
    A = backend.define_vars("A", ret="stream")
    f = backend.define_vars("f", arg_types=(backend.scalar_typ,), ret="scalar")

    term = monoid.reduce(monoid.mask(f(a()), a() == c()), {a: A()})
    backend.check_rewrite(lhs=term, rhs=term, rule=ReduceMaskHoist())


@pytest.mark.parametrize("mask_monoid, plus_monoid", MONOID_PAIRS)
def test_mask_push_plus(mask_monoid, plus_monoid):
    """A conjunctive mask on a product pushes each conjunct down onto the
    factors that mention its variables::

        M.mask(WM.plus(f(a), g(b)), And.plus(a == c, b == d))
          == WM.plus(M.mask(f(a), a == c), M.mask(g(b), b == d))

    when ``distributes_over(WM, M)``. Sound because ``M.identity`` is the
    annihilator of ``WM``: a false conjunct collapses the factor it lands on to
    ``M.identity == WM.zero``, so the whole product is ``WM.zero == M.identity``.
    """
    backend = IntBackend()
    a, b, c, d = backend.define_vars("a", "b", "c", "d", ret="scalar")
    f, g = backend.define_vars("f", "g", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = mask_monoid.mask(
        plus_monoid.plus(f(a()), g(b())),
        And.plus(a() == c(), b() == d()),
    )
    rhs = plus_monoid.plus(
        mask_monoid.mask(f(a()), a() == c()),
        mask_monoid.mask(g(b()), b() == d()),
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=Factor())


@pytest.mark.parametrize("mask_monoid, plus_monoid", MONOID_PAIRS)
def test_mask_push_plus_orphan_residual(mask_monoid, plus_monoid):
    """A conjunct that no factor mentions stays as a residual outer mask while
    the pushable conjuncts move down onto their factors."""
    backend = IntBackend()
    a, c, e, h, g = backend.define_vars("a", "c", "e", "h", "g", ret="scalar")
    f = backend.define_vars("f", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = mask_monoid.mask(
        plus_monoid.plus(f(a()), g()),
        And.plus(a() == c(), e() == h()),  # e == h mentions no factor
    )
    rhs = mask_monoid.mask(
        plus_monoid.plus(mask_monoid.mask(f(a()), a() == c()), g()),
        e() == h(),
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=Factor())


@pytest.mark.parametrize("mask_monoid, plus_monoid", MONOID_PAIRS)
def test_mask_push_plus_all_orphan_noop(mask_monoid, plus_monoid):
    """A mask whose condition mentions no factor is left in place: there is
    nowhere to fuse it, so nothing is pushed."""
    backend = IntBackend()
    x, y, c, d = backend.define_vars("x", "y", "c", "d", ret="scalar")

    term = mask_monoid.mask(plus_monoid.plus(x(), y()), c() == d())
    backend.check_rewrite(lhs=term, rhs=term, rule=Factor())


@pytest.mark.parametrize("mask_monoid, plus_monoid", MONOID_PAIRS)
def test_mask_push_plus_nondistributing_noop(mask_monoid, plus_monoid):
    """The mask does NOT push into a plus of the *same* (non-distributing)
    monoid: ``distributes_over(M, M)`` is false, so ``M.mask(M.plus(x, y), c)``
    is left untouched."""
    backend = IntBackend()
    a, c = backend.define_vars("a", "c", ret="scalar")
    f, g = backend.define_vars("f", "g", arg_types=(backend.scalar_typ,), ret="scalar")

    term = mask_monoid.mask(mask_monoid.plus(f(a()), g(a())), a() == c())
    backend.check_rewrite(lhs=term, rhs=term, rule=MaskPushPlus())


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_equality_mask_plus(backend: Backend, monoid):
    """ReduceEqualityMaskRange distributes over a plus body, discharging an
    equality on the reduced stream in one summand via a singleton-stream gather
    while leaving the other summand as an ordinary masked reduce.
    """
    a, c = backend.define_vars("a", "c", ret="scalar")
    f, g = backend.define_vars("f", "g", arg_types=(backend.scalar_typ,), ret="scalar")

    body = monoid.plus(
        monoid.mask(f(a()), a() == c()),  # eliminable: a == c over range
        monoid.mask(g(a()), c() == 0),  # not eliminable (no reduced-stream eq)
    )
    lhs = monoid.reduce(body, {a: range(3)})
    rhs = monoid.plus(
        monoid.mask(
            monoid.reduce(monoid.mask(f(a()), And.plus()), {a: (c(),)}),
            And.plus(0 <= c(), c() < 3),
        ),
        monoid.reduce(monoid.mask(g(a()), c() == 0), {a: range(3)}),
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceEqualityMaskRange())


def test_reduce_independent_1(backend: Backend):
    a, b = backend.define_vars("a", "b", ret="scalar")
    A, B = backend.define_vars("A", "B", ret="stream")

    lhs = Sum.reduce(Product.plus(a(), b()), {a: A(), b: B()})
    rhs = Product.plus(
        Sum.reduce(Product.plus(a()), {a: A()}), Sum.reduce(Product.plus(b()), {b: B()})
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=Factor())


def test_reduce_independent_2(backend: Backend):
    a, b, c = backend.define_vars("a", "b", "c", ret="scalar")
    A, B, C = backend.define_vars("A", "B", "C", ret="stream")
    f = backend.define_vars(
        "f", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = Sum.reduce(Product.plus(a(), b(), f(b(), c())), {a: A(), b: B(), c: C()})
    rhs = Product.plus(
        Sum.reduce(Product.plus(a()), {a: A()}),
        Sum.reduce(
            Product.plus(b(), Sum.reduce(Product.plus(f(b(), c())), {c: C()})),
            {b: B()},
        ),
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=Factor())


def test_reduce_independent_3_negative(backend: Backend):
    """Stream `b` depends on `a` (b: g(a())), so the proposed factorization
    is unsound — the normalizer must NOT apply it."""
    a, b, c = backend.define_vars("a", "b", "c", ret="scalar")
    A, C = backend.define_vars("A", "C", ret="stream")
    f = backend.define_vars(
        "f", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )
    g = backend.define_vars("g", arg_types=(backend.scalar_typ,), ret="stream")

    with handler(Factor()):  # ty:ignore[invalid-argument-type]
        lhs = Sum.reduce(
            Product.plus(a(), b(), f(b(), c())), {a: A(), b: g(a()), c: C()}
        )
    bogus_rhs = Product.plus(
        Sum.reduce(a(), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: g(a()), c: C()}),
    )
    assert fvsof(bogus_rhs) != fvsof(lhs)
    assert not syntactic_eq_alpha(lhs, bogus_rhs)


def test_reduce_independent_4(backend: Backend):
    a, b, c, d = backend.define_vars("a", "b", "c", "d", ret="scalar")
    A, B, C = backend.define_vars("A", "B", "C", ret="stream")
    f = backend.define_vars(
        "f", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = Sum.reduce(Product.plus(a(), b(), f(b(), c()), d()), {a: A(), b: B(), c: C()})
    rhs = Product.plus(
        d(),
        Sum.reduce(Product.plus(a()), {a: A()}),
        Sum.reduce(
            Product.plus(b(), Sum.reduce(Product.plus(f(b(), c())), {c: C()})),
            {b: B()},
        ),
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=Factor())


def test_reduce_chain(backend: Backend):
    x, y = backend.define_vars("x", "y", ret="scalar")
    X, Y = backend.define_vars("X", "Y", ret="stream")
    f, h = backend.define_vars("f", "h", arg_types=(backend.scalar_typ,), ret="scalar")
    g = backend.define_vars(
        "g", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = Sum.reduce(Product.plus(f(x()), g(x(), y()), h(y())), {x: X(), y: Y()})
    rhs = Sum.reduce(
        Product.plus(h(y()), Sum.reduce(Product.plus(f(x()), g(x(), y())), {x: X()})),
        {y: Y()},
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=Factor())


def test_reduce_factorization_mask(backend: Backend):
    """A reduce-monoid mask wrapping the plus is preserved across factorization.

    The mask condition depends only on the outer stream ``a``, so ``b`` factors
    into an inner reduce while the mask stays at the outer level (``a`` is
    treated as universal because the mask gates every factor)."""
    a, b, k = backend.define_vars("a", "b", "k", ret="scalar")
    A, B = backend.define_vars("A", "B", ret="stream")
    f, g = backend.define_vars("f", "g", arg_types=(backend.scalar_typ,), ret="scalar")

    cond = a() == k()  # depends on the outer stream `a`, not on `b`
    lhs = Sum.reduce(Sum.mask(Product.plus(f(a()), g(b())), cond), {a: A(), b: B()})
    rhs = Sum.reduce(
        Sum.mask(
            Product.plus(f(a()), Sum.reduce(Product.plus(g(b())), {b: B()})),
            cond,
        ),
        {a: A()},
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=Factor())


def test_reduce_factorization_mask_universal_noop(backend: Backend):
    """When the mask condition depends on every stream, the mask makes each
    stream universal, so no stream is innermost-eligible and the rule is a
    no-op."""
    a, b, k = backend.define_vars("a", "b", "k", ret="scalar")
    A, B = backend.define_vars("A", "B", ret="stream")
    f, g = backend.define_vars("f", "g", arg_types=(backend.scalar_typ,), ret="scalar")

    cond = And.plus(a() == k(), b() == k())
    term = Sum.reduce(Sum.mask(Product.plus(f(a()), g(b())), cond), {a: A(), b: B()})
    backend.check_rewrite(lhs=term, rhs=term, rule=Factor())


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lift_shared(outer, inner, backend: Backend):
    """A stream free in every factor is hoisted into an outer reduce:
    Sum.reduce(f(a, c) * g(b, c), {a: A, b: B, c: C})
      = Sum.reduce(Sum.reduce(f(a, c), {a: A}) * Sum.reduce(g(b, c), {b: B}), {c: C})
    """
    a, b, c = backend.define_vars("a", "b", "c", ret="scalar")
    A, B, C = backend.define_vars("A", "B", "C", ret="stream")
    f, g = backend.define_vars(
        "f", "g", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = outer.reduce(inner.plus(f(a(), c()), g(b(), c())), {a: A(), b: B(), c: C()})
    rhs = outer.reduce(
        inner.plus(
            outer.reduce(inner.plus(f(a(), c())), {a: A()}),
            outer.reduce(inner.plus(g(b(), c())), {b: B()}),
        ),
        {c: C()},
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=Factor())


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lift_shared_deps(outer, inner, backend: Backend):
    """A shared stream is lifted together with its dependencies: both ``c``
    and ``d = h(c)`` appear in every factor, so both are hoisted."""
    a, b, c, d = backend.define_vars("a", "b", "c", "d", ret="scalar")
    A, B, C = backend.define_vars("A", "B", "C", ret="stream")
    h = backend.define_vars("h", arg_types=(backend.scalar_typ,), ret="stream")
    f, g = backend.define_vars(
        "f",
        "g",
        arg_types=(backend.scalar_typ, backend.scalar_typ, backend.scalar_typ),
        ret="scalar",
    )

    lhs = outer.reduce(
        inner.plus(f(a(), c(), d()), g(b(), c(), d())),
        {a: A(), b: B(), c: C(), d: h(c())},
    )
    rhs = outer.reduce(
        inner.plus(
            outer.reduce(inner.plus(f(a(), c(), d())), {a: A()}),
            outer.reduce(inner.plus(g(b(), c(), d())), {b: B()}),
        ),
        {c: C(), d: h(c())},
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=Factor())


def test_reduce_cartesian_3():
    backend = JaxBackend()
    i = backend.define_vars("i", ret="scalar")

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
def test_reduce_lifted_1(outer, inner, backend: Backend):
    a, i = backend.define_vars("a", "i", ret="scalar")
    N, A_domain = backend.define_vars("N", "A_domain", ret="stream")
    f = backend.define_vars("f", arg_types=(backend.scalar_typ,), ret="scalar")
    A = Operation.define(Array)

    lhs = outer.reduce(
        inner.reduce(f(A()[i()]), {i: range(3)}),
        {
            A: CartesianProduct.reduce(
                Union.reduce([Union.delta((i(),), a())], {a: A_domain()}), {i: range(3)}
            )
        },
    )
    rhs = inner.reduce(outer.reduce(f(a()), {a: A_domain()}), {i: range(3)})
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=coproduct(
            ReduceDistributeCartesianProduct(),
            coproduct(
                ReduceUnion(), coproduct(DeltaEmpty(), EliminateSingletonStreams())
            ),
        ),
    )


def test_reduce_cartesian_1():
    backend = IntBackend()
    a, i = backend.define_vars("a", "i", ret="scalar")
    A = backend.define_vars("A", ret="stream")

    with handler(NormalizeIntp):
        term1 = Sum.reduce(
            Product.reduce(a(), {a: []}),
            {A: CartesianProduct.reduce([], {i: []})},
        )
        term2 = Product.reduce(Sum.reduce(a(), {a: []}), {i: []})
    assert term1 == term2


def test_reduce_cartesian_2():
    backend = IntBackend()
    a, i = backend.define_vars("a", "i", ret="scalar")
    A = backend.define_vars("A", ret="stream")

    with handler(NormalizeIntp):
        term1 = Sum.reduce(
            Product.reduce(a(), {a: A()}),
            {A: CartesianProduct.reduce([(0,)], {i: [0]})},
        )
        term2 = Product.reduce(Sum.reduce(a(), {a: [0]}), {i: [0]})
    assert term1 == term2


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lifted_multi_index(outer, inner, backend: Backend):
    a, i, j = backend.define_vars("a", "i", "j", ret="scalar")
    A, N, M, A_domain = backend.define_vars("A", "N", "M", "A_domain", ret="stream")
    f = backend.define_vars("f", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = outer.reduce(
        inner.reduce(f(a()), {a: A()}),
        {A: CartesianProduct.reduce(A_domain(), {i: N(), j: M()})},
    )
    rhs = inner.reduce(
        outer.reduce(inner.plus(f(a())), {a: A_domain()}), {i: N(), j: M()}
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDistributeCartesianProduct())


@pytest.mark.parametrize("outer,inner", MONOID_PAIRS)
def test_reduce_lifted_2(outer, inner, backend: Backend):
    """The worked example on page 396 of 'Lifted Variable Elimination:
    Decoupling the Operators from the Constraint Language'.

    """
    a, i, s, t = backend.define_vars("a", "i", "s", "t", ret="scalar")
    A, N, T = backend.define_vars("A", "N", "T", ret="stream")
    A_domain = backend.define_vars(
        "A_domain", arg_types=(backend.scalar_typ,), ret="stream"
    )
    f1, f2 = backend.define_vars(
        "f1", "f2", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = outer.reduce(
        inner.reduce(inner.plus(f1(a(), s()), f2(t(), a())), {a: A()}),
        {A: CartesianProduct.reduce(A_domain(i()), {i: N()}), t: T()},
    )
    rhs = outer.reduce(
        inner.reduce(
            outer.reduce(
                inner.plus(inner.plus(f1(a(), s()), f2(t(), a()))), {a: A_domain(i())}
            ),
            {i: N()},
        ),
        {t: T()},
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDistributeCartesianProduct())


# ---------------------------------------------------------------------------
# Weighted streams
# ---------------------------------------------------------------------------


def test_reduce_single_weighted_stream(backend: Backend):
    """Single weighted stream desugars:
    Sum.reduce(body, {a: WS(A, w, Product)})
      = Sum.reduce(Product.plus(w(a), body), {a: A})
    """
    a = backend.define_vars("a", ret="scalar")
    A = backend.define_vars("A", ret="stream")
    body, w = backend.define_vars(
        "body", "w", arg_types=(backend.scalar_typ,), ret="scalar"
    )

    lhs = Sum.reduce(body(a()), {a: Product.weighted(A(), w)})
    rhs = Sum.reduce(Product.plus(w(a()), body(a())), {a: A()})
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceWeightedStream())


def test_reduce_weighted_factorization(backend: Backend):
    """Two independent weighted streams under Sum with Product weights factor:
        Sum.reduce(f(a)*g(b), {a: Product.weighted(A, a, w_a), b: Product.weighted(B, b, w_b)})
          = (Sum.reduce(w_a(a)*f(a), {a: A})) * (Sum.reduce(w_b(b)*g(b), {b: B}))

    Exercises chaining of ``ReduceWeightedStream`` with ``Factor``
    inside ``NormalizeIntp``.
    """
    a, b = backend.define_vars("a", "b", ret="scalar")
    A, B = backend.define_vars("A", "B", ret="stream")
    f, g, w_a, w_b = backend.define_vars(
        "f", "g", "w_a", "w_b", arg_types=(backend.scalar_typ,), ret="scalar"
    )

    lhs = Sum.reduce(
        Product.plus(f(a()), g(b())),
        {a: Product.weighted(A(), w_a), b: Product.weighted(B(), w_b)},
    )
    rhs = Product.plus(
        Sum.reduce(Product.plus(w_a(a()), Product.plus(f(a()))), {a: A()}),
        Sum.reduce(Product.plus(w_b(b()), Product.plus(g(b()))), {b: B()}),
    )
    backend.check_rewrite(
        lhs=lhs, rhs=rhs, rule=coproduct(ReduceWeightedStream(), Factor())
    )


def test_reduce_cartesian_weighted_stream(backend: Backend):
    """``CartesianProduct.reduce`` over a ``WeightedStream`` body whose weight
    is independent of the plate var rewrites to a single joint
    ``WeightedStream``:

        CartesianProduct.reduce(M.weighted(s, e, w(e)), {p: P})
          = M.weighted(CartesianProduct.reduce(s, {p: P}), row, M.reduce(w(e), {e: row()}))
    """
    p, e_var = backend.define_vars("p", "e_var", ret="scalar")
    S, P = backend.define_vars("S", "P", ret="stream")
    w = backend.define_vars("w", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = CartesianProduct.reduce(Product.weighted(S(), w), {p: P()})
    row_var = Operation.define(Iterable[backend.scalar_typ], name="row")  # type: ignore[name-defined]
    rhs = Product.weighted(
        CartesianProduct.reduce(S(), {p: P()}),
        deffn(Product.reduce(w(e_var()), {e_var: row_var()}), row_var),
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceCartesianWeightedStream())


def test_lift_weighted_cartesian(backend: Backend):
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
    a, p = backend.define_vars("a", "p", ret="scalar")
    A, S, P = backend.define_vars("A", "S", "P", ret="stream")
    body, w = backend.define_vars(
        "body", "w", arg_types=(backend.scalar_typ,), ret="scalar"
    )

    lhs = Sum.reduce(
        Product.reduce(body(a()), {a: A()}),
        {A: CartesianProduct.reduce(Product.weighted(S(), w), {p: P()})},
    )
    rhs = Product.reduce(
        Sum.reduce(Product.plus(w(a()), body(a())), {a: S()}), {p: P()}
    )
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=coproduct(
            coproduct(ReduceWeightedStream(), ReduceCartesianWeightedStream()),
            ReduceDistributeCartesianProduct(),
        ),
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

    a = Operation.define(int, name="a")
    w = Operation.define(_w, name="w")
    f = Operation.define(_f, name="f")

    with handler(NormalizeIntp):
        result = evaluate(Sum.reduce(f(a()), {a: Product.weighted([1, 2, 3, 4], w)}))

    assert math.isclose(result, 10.0)


# ---------------------------------------------------------------------------
# CartesianProduct.plus (pure-Python ``CartesianProductPlus`` implementation)
# ---------------------------------------------------------------------------
#
# A ``CartesianProduct`` value is a list of "rows", each row a ``dict`` mapping
# index variables to values. ``plus`` takes the cartesian product of its
# argument lists, disjoint-merging the dicts of each combination into a single
# row. The identity is ``[{}]`` (one empty row) and the zero is ``[]`` (no
# rows).


@pytest.fixture
def cprod():
    """A handler scope in which ``CartesianProduct.plus`` is concrete."""
    with handler(CartesianProductPlus()):
        yield


def test_cprod_plus_two_singletons(cprod):
    """Two single-row lists merge into one row with the union of their keys."""
    assert CartesianProduct.plus([{"a": 1}], [{"b": 2}]) == [{"a": 1, "b": 2}]


def test_cprod_plus_multi_key_rows(cprod):
    """Every key of a multi-key row survives the merge (regression: the merge
    used to keep only the last key of each dict)."""
    assert CartesianProduct.plus([{"a": 1, "b": 2}], [{"c": 3, "d": 4}]) == [
        {"a": 1, "b": 2, "c": 3, "d": 4}
    ]


def test_cprod_plus_cartesian_expansion(cprod):
    """The result enumerates the full cartesian product of the input rows."""
    result = CartesianProduct.plus([{"a": 1}, {"a": 2}, {"a": 3}], [{"b": 4}, {"b": 5}])
    assert result == [
        {"a": 1, "b": 4},
        {"a": 1, "b": 5},
        {"a": 2, "b": 4},
        {"a": 2, "b": 5},
        {"a": 3, "b": 4},
        {"a": 3, "b": 5},
    ]


def test_cprod_plus_cardinality(cprod):
    """|plus(A, B, C)| == |A| * |B| * |C|."""
    a = [{"a": i} for i in range(2)]
    b = [{"b": i} for i in range(3)]
    c = [{"c": i} for i in range(4)]
    assert len(CartesianProduct.plus(a, b, c)) == 2 * 3 * 4


def test_cprod_plus_three_args(cprod):
    """``plus`` is variadic and merges across all arguments at once."""
    assert CartesianProduct.plus([{"a": 1}], [{"b": 2}], [{"c": 3}]) == [
        {"a": 1, "b": 2, "c": 3}
    ]


def test_cprod_plus_single_arg(cprod):
    """A single argument is returned row-for-row (product of one factor)."""
    assert CartesianProduct.plus([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]


def test_cprod_plus_identity_right(cprod):
    """The identity ``[{}]`` is a right unit: merging an empty row changes
    nothing."""
    assert CartesianProduct.plus([{"a": 1}, {"a": 2}], CartesianProduct.identity) == [
        {"a": 1},
        {"a": 2},
    ]


def test_cprod_plus_identity_left(cprod):
    """The identity ``[{}]`` is a left unit."""
    assert CartesianProduct.plus(CartesianProduct.identity, [{"a": 1}, {"a": 2}]) == [
        {"a": 1},
        {"a": 2},
    ]


def test_cprod_plus_identity_with_identity(cprod):
    """Identity ⊕ identity == identity."""
    assert (
        CartesianProduct.plus(CartesianProduct.identity, CartesianProduct.identity)
        == CartesianProduct.identity
    )


def test_cprod_plus_zero_right(cprod):
    """The zero ``[]`` absorbs on the right (empty cartesian product)."""
    assert CartesianProduct.plus([{"a": 1}], CartesianProduct.zero) == []


def test_cprod_plus_zero_left(cprod):
    """The zero ``[]`` absorbs on the left."""
    assert CartesianProduct.plus(CartesianProduct.zero, [{"a": 1}]) == []


def test_cprod_plus_zero_among_many(cprod):
    """A single zero factor anywhere collapses the whole product to ``[]``."""
    assert CartesianProduct.plus([{"a": 1}], [], [{"c": 3}]) == []


def test_cprod_plus_empty_rows_preserved(cprod):
    """Rows that are themselves empty dicts merge cleanly."""
    assert CartesianProduct.plus([{}], [{"a": 1}]) == [{"a": 1}]
    assert CartesianProduct.plus([{}], [{}]) == [{}]


def test_cprod_plus_associative(cprod):
    """plus(plus(A, B), C) == plus(A, plus(B, C)) == plus(A, B, C)."""
    a = [{"a": 1}, {"a": 2}]
    b = [{"b": 3}]
    c = [{"c": 4}, {"c": 5}]
    flat = CartesianProduct.plus(a, b, c)
    assert CartesianProduct.plus(CartesianProduct.plus(a, b), c) == flat
    assert CartesianProduct.plus(a, CartesianProduct.plus(b, c)) == flat


def test_cprod_plus_duplicate_key_raises(cprod):
    """Merging rows that share a key is ill-defined and rejected."""
    with pytest.raises(ValueError, match="Duplicate key found: 'a'"):
        CartesianProduct.plus([{"a": 1}], [{"a": 2}])


def test_cprod_plus_duplicate_key_in_multi_key_row_raises(cprod):
    """The duplicate-key check sees every key of a multi-key row, not just the
    last (regression for the same merge bug)."""
    with pytest.raises(ValueError, match="Duplicate key found: 'a'"):
        CartesianProduct.plus([{"a": 1, "x": 9}], [{"a": 2}])


def test_cprod_plus_does_not_mutate_inputs(cprod):
    """The input rows are left untouched; merges build fresh dicts."""
    left = [{"a": 1}]
    right = [{"b": 2}]
    CartesianProduct.plus(left, right)
    assert left == [{"a": 1}]
    assert right == [{"b": 2}]


def test_cprod_plus_forwards_on_term():
    """A symbolic (``Term``) argument cannot be enumerated, so ``plus`` forwards
    and the call builds an unevaluated ``Term`` instead of a value."""
    x = Operation.define(Iterable, name="x")
    with handler(CartesianProductPlus()):
        result = CartesianProduct.plus(x(), [{"a": 1}])
    assert isinstance(result, Term)
    assert result.op is CartesianProduct.plus


def test_split_disjoint_product_simple():
    backend = IntBackend()
    i, j, i_out, j_out = backend.define_vars("i", "j", "i_out", "j_out", ret="scalar")
    f, g = backend.define_vars("f", "g", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = Product.reduce(
        Sum.plus(
            Sum.mask(f(i()), And.plus(i() == i_out(), j() == j_out())),
            Sum.mask(g(i()), i() != i_out()),
        ),
        {i: range(3)},
    )

    rhs = Product.reduce(
        Product.plus(
            Product.mask(
                Sum.mask(f(i()), And.plus(j() == j_out())), And.plus(i() == i_out())
            ),
            Product.mask(Sum.mask(g(i()), True), Or.plus(i() != i_out())),
        ),
        {i: range(3)},
    )

    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=SplitDisjointProduct())
