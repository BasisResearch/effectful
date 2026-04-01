import jax
from jax.numpy import allclose
from weighted.handlers.jax import DenseTensorReduce
from weighted.handlers.optimization.quadrature import GaussHermiteQuadrature
from weighted.ops.jax import reals
from weighted.ops.sugar import Sum

import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numpyro as dist
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import defop


def test_quadrature():
    x = defop(jax.Array, name="x")
    polynomial = 2 * x() + x() ** 2
    mu, sigma = 2.5, 5.0
    d = dist.Normal(mu, sigma)

    body = jnp.exp(d.log_prob(x())) * polynomial
    expr = Sum({x: reals()}, body)
    # 3 points are sufficient as polynomial is quadratic
    with handler(DenseTensorReduce()), handler(GaussHermiteQuadrature(3)):
        expr = evaluate(expr)

    expected = 2 * mu + mu**2 + sigma**2
    assert allclose(expr, expected)


def test_bivariate_quadrature():
    x = defop(jax.Array, name="x")
    y = defop(jax.Array, name="y")
    mu_x, sigma_x = 5.0, 3.0
    mu_y, sigma_y = 1.0, 2.0
    d_x = dist.Normal(mu_x, sigma_x)
    d_y = dist.Normal(mu_y, sigma_y)
    prob_x = jnp.exp(d_x.log_prob(x()))
    prob_y = jnp.exp(d_y.log_prob(y()))

    polynomial = 2 * x() + y() ** 2
    body = prob_x * prob_y * polynomial
    expr = Sum({x: reals(), y: reals()}, body)

    # 3 points are sufficient as polynomial is quadratic
    with handler(DenseTensorReduce()), handler(GaussHermiteQuadrature(3)):
        expr = evaluate(expr)

    expected = 2 * mu_x + mu_y**2 + sigma_y**2
    assert allclose(expr, expected)
