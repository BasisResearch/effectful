import jax
from effectful.handlers.jax import numpy as jnp
from effectful.handlers.jax._handlers import _register_jax_op, is_eager_array
from effectful.ops.syntax import defop
from effectful.ops.types import Term
from numpyro.distributions import Distribution

from weighted.ops.fold import fold
from weighted.ops.jax import reals
from weighted.ops.monoid import SumMonoid


@defop
def D(*args: tuple[tuple[int, ...], jax.Array]) -> jax.Array:
    if not all(isinstance(kv, tuple) and len(kv) == 2 for kv in args):
        raise ValueError("Expected a sequence of key-value pairs")
    raise NotImplementedError


@defop
def sample(key: jax.Array, d: Distribution, sample_shape: tuple[int]) -> jax.Array:
    # TODO: remove when DistributionTerm gets refactored
    ground_shape = not isinstance(sample_shape, Term)
    if is_eager_array(key) and ground_shape and is_eager_distribution(d):
        return d.sample(key, sample_shape=sample_shape)
    raise NotImplementedError


@defop
def rsample(key, d: Distribution, sample_shape: tuple[int]) -> jax.Array:
    # TODO: remove when DistributionTerm gets refactored
    if not (isinstance(d, Distribution) and isinstance(sample_shape, tuple)):
        raise NotImplementedError
    return d.rsample(key, sample_shape=sample_shape)


def is_eager_distribution(d: Distribution) -> bool:
    # TODO: remove when DistributionTerm gets refactored
    return isinstance(d, Distribution) and (
        not isinstance(d, Term) or all(is_eager_array(a) for a in d.args)
    )


@defop
def log_prob(d: Distribution, value: jax.Array) -> jax.Array:  # todo
    # TODO: remove when DistributionTerm gets refactored
    if not is_eager_distribution(d) or not is_eager_array(value):
        raise NotImplementedError
    return _register_jax_op(d.log_prob)(value)


@defop
def kl_divergence(p: Distribution, q: Distribution) -> jax.Array:
    # KL(p, q) = ∫ p(x) (log p(x) - log q(x)) dx
    x = defop(jax.Array, name="x")
    log_p = log_prob(p, x())  # type: ignore
    log_q = log_prob(q, x())  # type: ignore
    return fold(SumMonoid, {x: reals()}, jnp.exp(log_p) * (log_p - log_q))  # type: ignore
