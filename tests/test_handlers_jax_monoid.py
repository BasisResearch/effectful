import functools

import jax
import pytest

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.monoid import (
    ArrayReduce,
    LogSumExp,
    ProductPlusJax,
    SumPlusJax,
)
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import (
    Max,
    Min,
    NormalizeIntp,
    Product,
    ReduceWeightedStream,
    Sum,
    WeightedStream,
)
from effectful.ops.semantics import coproduct
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


def test_jax_weighted_reduce(backend: Backend):
    """Sum over a single ``WeightedStream`` with ``Product`` weights lowers to
    ``jnp.sum(w(X) * body(X))`` under ``NormalizeIntp`` ∘ ``ArrayReduce``.

    Verifies that the desugaring rule composes cleanly with the JAX lowering
    so existing handlers need no changes to support weighted streams.
    """
    (x, k) = define_vars("x", "k", typ=jax.Array)
    X = define_vars("X", typ=backend.stream_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")
    w = backend.fresh_op("w", n_args=1, ret="scalar")

    ws = WeightedStream(stream=X(), weight=w, monoid=Product)
    lhs = Sum.reduce(body(x()), {x: ws})
    rhs = jnp.sum(
        bind_dims(w(unbind_dims(X(), k)) * body(unbind_dims(X(), k)), k), axis=0
    )

    check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=functools.reduce(
            coproduct, [ReduceWeightedStream(), ArrayReduce(), ProductPlusJax()]
        ),
        backend=backend,
        free_vars=[x, k, X, body, w],
    )
