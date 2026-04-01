import functools

import jax
from scipy.special import roots_hermite
from weighted.handlers.optimization.utils import parse_terms
from weighted.ops.jax import reals
from weighted.ops.monoid import SumMonoid
from weighted.ops.reduce import reduce

import effectful.handlers.numpyro as dist
from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import Term


class GaussHermiteQuadrature(ObjectInterpretation):
    """
    Transforms a reduce over a normal distribution into
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

    @implements(reduce)
    def reduce(self, monoid, streams, body):
        if monoid != SumMonoid:
            return fwd()

        # Try to parse the body into a gaussian distribution
        # multiplied with some remaining term
        mul, terms = parse_terms(body, monoid)
        for i, term in enumerate(terms):
            match term:
                case Term(
                    jnp.exp,
                    (
                        Term(
                            dist._DistributionTerm.log_prob,
                            (Term(dist.Normal, (loc, scale)), x),
                        ),
                    ),
                ):
                    mu, sigma, x = loc, scale, x.op
                    remaining_terms = (t for j, t in enumerate(terms) if i != j)
                    break
        else:
            return fwd()

        # Only integration over ℝ supported for now
        if streams[x].op != reals:
            return fwd()

        fresh_index = defop(jax.Array, name="fresh_i")
        points = jnp.expand_dims(sigma * self.points + mu, -1)
        new_streams = {
            fresh_index: jnp.arange(self.nb_points),
            x: jax_getitem(points, (fresh_index(),)),
        }
        remaining_body = functools.reduce(mul, remaining_terms)
        new_body = jax_getitem(self.weights, (fresh_index(),)) * remaining_body
        return reduce(monoid, streams | new_streams, new_body)
