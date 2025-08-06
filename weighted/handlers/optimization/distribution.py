from functools import reduce

import effectful.handlers.numpyro as dist
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import coproduct, fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Term

import weighted.handlers.jax as handler

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
            case Term(handler.sample, (_, Term(dist.Normal, (loc2, scale2)), _)):
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
            case Term(handler.sample, (key, Term(dist.Normal, (loc, scale)), shape)):
                new_d = dist.Normal(c * loc, jnp.abs(c) * scale)
                return handler.sample(key, new_d, shape)
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
                Term(handler.sample, (key1, Term(dist.Normal, (loc1, scale1)), shape1)),
                Term(handler.sample, (key2, Term(dist.Normal, (loc2, scale2)), shape2)),
            ) if handler.syntactic_eq_jax(shape1, shape2) and handler.syntactic_eq_jax(
                key1, key2
            ):
                new_d = dist.Normal(loc1 + loc2, jnp.sqrt(scale1**2 + scale2**2))
                return handler.sample(key1, new_d, shape1)
        return fwd()


interpretation = reduce(
    coproduct,  # type: ignore
    [
        NormalVerticalFusion(),
        SampleAddNormalFusion(),
        SampleMulConstantFusion(),
    ],
)
