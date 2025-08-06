from functools import reduce

import jax
from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import Term
from numpyro import distributions as dist
from scipy.special import roots_hermite

from weighted.handlers.optimization.utils import parse_terms
from weighted.ops.distribution import log_prob, reals
from weighted.ops.fold import fold
from weighted.ops.monoid import SumMonoid


def _parse_gaussian_prob(term: Term):
    if not (isinstance(term, Term) and term.op is jnp.exp and len(term.args) == 1):
        return None
    term = term.args[0]
    if not (isinstance(term, Term) and term.op is log_prob and len(term.args) == 2):
        return None
    d, x = term.args
    if not isinstance(d, dist.Normal):
        return None
    return d.loc, d.scale, x.op


class GaussHermiteQuadrature(ObjectInterpretation):
    """
    Transforms a fold over a normal distribution into
    a quadrature of Hermite polynomials.
        Sum({x: reals()}, normal_distribution.prob(x()) * f(x()))
        => Sum({i: range(nb_points)}, weight[i()] * f(points[i()]))
         where the points and weight arrays are the roots and weights
         of the n^th order Hermite polynomial.

    Given n/2 quadrature points, the error of this quadrature
    is low when the n^th derivative of the function f is low.
    More precisely, the error is bounded by (c.f. [1])
        max_{x ∈ domain} [ f^(n)(x) (n/2)! / (n)! ].
    This is exact for polynomials of degree at most n-1
    (up to floating point errors).

    [1] Davis, Tom P. "A general expression for Hermite
    expansions with applications." (2024)
    """

    def __init__(self, nb_points: int):
        self.nb_points = nb_points
        # pre-compute Hermite roots
        points, weights = roots_hermite(nb_points)
        # need to normalize as scipy uses the physicist's convention
        self.points = jnp.array(points) * jnp.sqrt(2)
        self.weights = jnp.array(weights) / jnp.sqrt(jax.numpy.pi)

    @implements(fold)
    def fold(self, monoid, streams, body):
        if monoid is not SumMonoid:
            return fwd()

        # Try to parse the body into a gaussian distribution
        # multiplied with some remaining term
        mul, terms = parse_terms(body, monoid)
        for i, term in enumerate(terms):
            gaussian = _parse_gaussian_prob(term)
            if gaussian is not None:
                mu, sigma, x = gaussian
                remaining_terms = (t for j, t in enumerate(terms) if i != j)
                remaining_body = reduce(mul, remaining_terms)
                break
        else:
            return fwd()

        # Only integration over ℝ supported for now
        if streams[x].op is not reals:
            return fwd()

        fresh_index = defop(jax.Array, name="fresh_i")
        points = jnp.expand_dims(sigma * self.points + mu, -1)
        new_streams = {
            fresh_index: jnp.arange(self.nb_points),
            x: jax_getitem(points, (fresh_index(),)),
        }
        new_body = jax_getitem(self.weights, (fresh_index(),)) * remaining_body
        return fold(monoid, streams | new_streams, new_body)
