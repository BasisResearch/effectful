from functools import reduce

import effectful.handlers.numpyro as dist
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import coproduct, fwd
from effectful.ops.syntax import ObjectInterpretation, implements, syntactic_eq
from effectful.ops.types import Term

from weighted.handlers.jax import sample

"""
Elementary transforms and simplification on Normal
distributions and samples on them.

Note: these transforms preserve the distribution,
    but may not give the exact same samples under 
    the same RNG.
"""


def _is_sample_term(x):
    return isinstance(x, Term) and x.op is sample


def _is_normal_term(x):
    return isinstance(x, Term) and x.op is dist.Normal


class NormalVerticalFusion(ObjectInterpretation):
    """
    Implements the identity
        Normal(sample(Normal(μ, σ₁²)), σ₂²)
        => Normal(μ, σ₁² + σ₂²)
    """

    @implements(dist.Normal)
    def Normal(self, loc, scale):
        if not _is_sample_term(loc):
            return fwd()

        body = loc.args[1]
        if not _is_normal_term(body):
            return fwd()
        loc2, scale2 = body.args
        return dist.Normal(loc2, jnp.sqrt(scale**2 + scale2**2))


class SampleMulConstantFusion(ObjectInterpretation):
    """
    Implements the identity
        c⋅sample(Normal(μ, σ²)) => sample(Normal(c⋅μ, c²⋅σ²))
        where c is a constant.
    """

    @implements(jnp.multiply)  # type: ignore
    def multiply(self, a, b):
        if _is_sample_term(a):
            a, b = b, a

        if not _is_sample_term(b) or isinstance(a, Term):
            return fwd()

        key, d, sample_shape = b.args
        if not _is_normal_term(d):
            return fwd()

        mu, sigma = d.args
        new_d = dist.Normal(a * mu, jnp.abs(a) * sigma)
        return sample(key, new_d, sample_shape)


class SampleAddNormalFusion(ObjectInterpretation):
    """
    Implements the identity
        sample(Normal(μ₁,σ₁²)) + sample(Normal(μ₂,σ₂²))
        => sample(Normal(μ₁ + μ₂, σ₁² + σ₂²))
    """

    @implements(jnp.add)  # type: ignore
    def add(self, a, b):
        if not (_is_sample_term(a) and _is_sample_term(b)):
            return fwd()

        k1, d1, shape1 = a.args
        k2, d2, shape2 = b.args

        if not syntactic_eq(shape1, shape2):
            return fwd()
        if not (_is_normal_term(d1) and _is_normal_term(d2)):
            return fwd()

        mu1, sigma1 = d1.args
        mu2, sigma2 = d2.args
        new_d = dist.Normal(mu1 + mu2, jnp.sqrt(sigma1**2 + sigma2**2))
        return sample(k1, new_d, shape1)


interpretation = reduce(
    coproduct,  # type: ignore
    [
        NormalVerticalFusion(),
        SampleAddNormalFusion(),
        SampleMulConstantFusion(),
    ],
)
