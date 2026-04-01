import functools

from weighted.ops.distribution import kl_divergence

import effectful.handlers.numpyro as dist
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import coproduct, fwd
from effectful.ops.syntax import ObjectInterpretation, implements, syntactic_eq
from effectful.ops.types import Term

"""
Elementary transforms and simplification on Normal
distributions and samples on them.

Note: these transforms preserve the distribution,
    but may not give the exact same samples under 
    the same RNG.
"""


class NormalVerticalFusion(ObjectInterpretation):
    """
    Implements the identity
        Normal(sample(Normal(μ, σ₁²)), σ₂²)
        => Normal(μ, σ₁² + σ₂²)
    """

    @implements(dist.Normal)
    def Normal(self, loc, scale):
        match loc:
            case Term(
                dist._DistributionTerm.sample,
                (Term(dist.Normal, (loc2, scale2)), _, ()),
            ):
                return dist.Normal(loc2, jnp.sqrt(scale**2 + scale2**2))
        return fwd()


class SampleMulConstantFusion(ObjectInterpretation):
    """
    Implements the identity
        c⋅sample(Normal(μ, σ²)) => sample(Normal(c⋅μ, c²⋅σ²))
        where c is a constant.
    """

    @implements(jnp.multiply)  # type: ignore
    def multiply(self, c, body):
        match body:
            case Term(
                dist._DistributionTerm.sample,
                (Term(dist.Normal, (loc, scale)), key, shape),
            ):
                new_d = dist.Normal(c * loc, jnp.abs(c) * scale)
                return new_d.sample(key, shape)
        return fwd()


class SampleAddNormalFusion(ObjectInterpretation):
    """
    Implements the identity
        sample(Normal(μ₁,σ₁²)) + sample(Normal(μ₂,σ₂²))
        => sample(Normal(μ₁ + μ₂, σ₁² + σ₂²))
    """

    @implements(jnp.add)  # type: ignore
    def add(self, a, b):
        match (a, b):
            case (
                Term(
                    dist._DistributionTerm.sample,
                    (Term(dist.Normal, (loc1, scale1)), key1, shape1),
                ),
                Term(
                    dist._DistributionTerm.sample,
                    (Term(dist.Normal, (loc2, scale2)), key2, shape2),
                ),
            ) if syntactic_eq(shape1, shape2) and syntactic_eq(key1, key2):
                new_d = dist.Normal(loc1 + loc2, jnp.sqrt(scale1**2 + scale2**2))
                return new_d.sample(key1, shape1)
        return fwd()


class NormalDivergence(ObjectInterpretation):
    """
    Exact computation of KL divergence on normal distributions.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    @implements(kl_divergence)
    def kl_divergence(self, p, q):
        match (p, q):
            case (Term(dist.Normal, (mu1, s1)), Term(dist.Normal, (mu2, s2))):
                mu_norm = (mu1 - mu2) ** 2
                log_s2 = jnp.log(s2 + self.eps)
                log_s1 = jnp.log(s1 + self.eps)
                return log_s2 - log_s1 + (s1**2 + mu_norm) / (2 * s2**2) - 0.5
        return fwd()


interpretation = functools.reduce(
    coproduct,  # type: ignore
    [
        NormalDivergence(),
        NormalVerticalFusion(),
        SampleAddNormalFusion(),
        SampleMulConstantFusion(),
    ],
)
