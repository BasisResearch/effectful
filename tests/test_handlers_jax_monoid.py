import jax
import pytest

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.monoid import LogSumExp, Max, Min, Product, Sum
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.types import NotHandled, Operation
from tests._monoid_helpers import define_vars, syntactic_eq_alpha

MONOIDS = [
    pytest.param(Sum, jnp.sum, id="Sum"),
    pytest.param(Product, jnp.prod, id="Product"),
    pytest.param(Min, jnp.min, id="Min"),
    pytest.param(Max, jnp.max, id="Max"),
    pytest.param(LogSumExp, logsumexp, id="LogSumExp"),
]


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_1(monoid, reductor):
    (x, X, k) = define_vars("x", "X", "k", typ=jax.Array)

    lhs = monoid.reduce(x(), {x: X()})
    rhs = reductor(bind_dims(unbind_dims(X(), k), k), axis=0)

    assert syntactic_eq_alpha(lhs, rhs)


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_2(monoid, reductor):
    (x, y, X, Y, k1, k2) = define_vars("x", "y", "X", "Y", "k1", "k2", typ=jax.Array)

    @Operation.define
    def f(_a: jax.Array, _b: jax.Array) -> jax.Array:
        raise NotHandled

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

    assert syntactic_eq_alpha(lhs, rhs)


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_3(monoid, reductor):
    """Stream `y` is `g(x())` — depends on the bound element of X. The reducer
    must inline ``g`` along the same named dim used to unbind `x`."""
    (x, y, X, k1, k2) = define_vars("x", "y", "X", "k1", "k2", typ=jax.Array)

    @Operation.define
    def f(_a: jax.Array, _b: jax.Array) -> jax.Array:
        raise NotHandled

    @Operation.define
    def g(_a: jax.Array) -> jax.Array:
        raise NotHandled

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

    assert syntactic_eq_alpha(lhs, rhs)
