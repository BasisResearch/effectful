import jax
import pytest

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.monoid import (
    ArrayReduce,
    LogSumExp,
    ReduceDeltaEmpty,
    ReduceDeltaIndependent,
    ReduceDependentRangeMask,
    delta,
)
from effectful.handlers.jax.monoid import range as Range
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import Max, Min, Product, Sum
from tests._monoid_helpers import JAX_BACKEND, Backend, check_rewrite, define_vars

MONOIDS = [
    pytest.param(Sum, jnp.sum, id="Sum"),
    pytest.param(Product, jnp.prod, id="Product"),
    pytest.param(Min, jnp.min, id="Min"),
    pytest.param(Max, jnp.max, id="Max"),
    pytest.param(LogSumExp, logsumexp, id="LogSumExp"),
]


@pytest.fixture
def backend() -> Backend:
    return JAX_BACKEND


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_1(monoid, reductor, backend: Backend):
    (x, k) = define_vars("x", "k", typ=jax.Array)
    X = define_vars("X", typ=backend.stream_typ)

    lhs = monoid.reduce(x(), {x: X()})
    rhs = reductor(bind_dims(unbind_dims(X(), k), k), axis=0)

    check_rewrite(
        lhs=lhs, rhs=rhs, rule=ArrayReduce(), backend=backend, free_vars=[x, X, k]
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_2(monoid, reductor, backend: Backend):
    (x, y, k1, k2) = define_vars("x", "y", "k1", "k2", typ=backend.scalar_typ)
    (X, Y) = define_vars("X", "Y", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=2, ret="scalar")

    lhs = monoid.reduce(f(x(), y()), {x: X(), y: Y()})
    rhs = reductor(
        bind_dims(
            reductor(
                bind_dims(f(unbind_dims(X(), k1), unbind_dims(Y(), k2)), k2),
                axis=0,
            ),
            k1,
        ),
        axis=0,
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ArrayReduce(),
        backend=backend,
        free_vars=[x, y, k1, k2, X, Y, f],
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_3(monoid, reductor, backend: Backend):
    """Stream `y` is `g(x())` — depends on the bound element of X. The reducer
    must inline ``g`` along the same named dim used to unbind `x`."""
    (x, y, k1, k2) = define_vars("x", "y", "k1", "k2", typ=backend.scalar_typ)
    X = define_vars("X", typ=backend.stream_typ)

    f = backend.fresh_op("f", n_args=2, ret="scalar")
    g = backend.fresh_op("g", n_args=1, ret="stream")

    lhs = monoid.reduce(f(x(), y()), {x: X(), y: g(x())})
    rhs = reductor(
        bind_dims(
            reductor(
                bind_dims(
                    f(unbind_dims(X(), k1), unbind_dims(g(unbind_dims(X(), k1)), k2)),
                    k2,
                ),
                axis=0,
            ),
            k1,
        ),
        axis=0,
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ArrayReduce(),
        backend=backend,
        free_vars=[x, y, k1, k2, X, f, g],
    )


# ---------------------------------------------------------------------------
# Delta rules. All tests use the operation form ``delta(idx, body)`` rather
# than the ``Delta`` dataclass; the delta op is the user-facing surface.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_delta_empty(monoid, reductor, backend: Backend):
    """An empty-index delta unwraps to its body.

    reduce(M, streams, delta((), body)) ≡ reduce(M, streams, body)
    """
    x = define_vars("x", typ=backend.scalar_typ)
    X = define_vars("X", typ=backend.stream_typ)

    lhs = monoid.reduce(delta((), x()), {x: X()})
    rhs = monoid.reduce(x(), {x: X()})

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ReduceDeltaEmpty(),
        backend=backend,
        free_vars=[x, X],
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_delta_independent_one(monoid, reductor, backend: Backend):
    """One R1 step: peel the final preserved index off a delta.

    reduce(M, {y: Y()}, delta((y(),), f(y())))
    ≡ reduce(M, {}, delta((), bind_dims(f(unbind_dims(Y(), k)), k)))
    """
    (y, k) = define_vars("y", "k", typ=backend.scalar_typ)
    Y = define_vars("Y", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=1, ret="scalar")

    lhs = monoid.reduce(delta((y(),), f(y())), {y: Y()})
    rhs = monoid.reduce(
        delta((), bind_dims(f(unbind_dims(Y(), k)), k)),
        {},
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ReduceDeltaIndependent(),
        backend=backend,
        free_vars=[y, k, Y, f],
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_delta_independent_preserves_others(monoid, reductor, backend: Backend):
    """R1 peels only the final index. Streams not matching the peeled index op
    stay untouched, as do earlier entries in the index tuple.

    reduce(M, {x: X(), y: Y()}, delta((x(), y()), f(x(), y())))
    ≡ reduce(M, {x: X()}, delta((x(),), bind_dims(f(x(), unbind_dims(Y(), k)), k)))
    """
    (x, y, k) = define_vars("x", "y", "k", typ=backend.scalar_typ)
    (X, Y) = define_vars("X", "Y", typ=backend.stream_typ)
    f = backend.fresh_op("f", n_args=2, ret="scalar")

    lhs = monoid.reduce(delta((x(), y()), f(x(), y())), {x: X(), y: Y()})
    rhs = monoid.reduce(
        delta((x(),), bind_dims(f(x(), unbind_dims(Y(), k)), k)),
        {x: X()},
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ReduceDeltaIndependent(),
        backend=backend,
        free_vars=[x, y, k, X, Y, f],
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_dependent_range_mask(monoid, reductor, backend: Backend):
    """A dependent range stream gets rewritten to the referent's bbox stream,
    with the original constraint folded into the body as a where-guard.

    reduce(M, {u: range(0, N, 1), v: range(0, u(), 1)}, body)
    ≡ reduce(M, {u: range(0, N, 1), v: range(0, N, 1)}, where(v() < u(), body, M.identity))
    """
    (u, v) = define_vars("u", "v", typ=backend.scalar_typ)
    N = 5
    f = backend.fresh_op("f", n_args=2, ret="scalar")

    body = f(u(), v())

    lhs = monoid.reduce(body, {u: Range(0, N, 1), v: Range(0, u(), 1)})
    rhs = monoid.reduce(
        jnp.where(v() < u(), body, monoid.identity),
        {u: Range(0, N, 1), v: Range(0, N, 1)},
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ReduceDependentRangeMask(),
        backend=backend,
        free_vars=[u, v, f],
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_dependent_range_mask_delta_body(monoid, reductor, backend: Backend):
    """When the body is a delta term, R4 folds the constraint into the delta's
    weight while leaving its index tuple untouched.

    reduce(M, {u: range(N), v: range(u())}, delta((u(), v()), w))
    ≡ reduce(M, {u: range(N), v: range(N)},
             delta((u(), v()), where(v() < u(), w, M.identity)))
    """
    (u, v) = define_vars("u", "v", typ=backend.scalar_typ)
    N = 5
    f = backend.fresh_op("f", n_args=2, ret="scalar")

    weight = f(u(), v())
    idx = (u(), v())

    lhs = monoid.reduce(delta(idx, weight), {u: Range(0, N, 1), v: Range(0, u(), 1)})
    rhs = monoid.reduce(
        delta(idx, jnp.where(v() < u(), weight, monoid.identity)),
        {u: Range(0, N, 1), v: Range(0, N, 1)},
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=ReduceDependentRangeMask(),
        backend=backend,
        free_vars=[u, v, f],
    )
