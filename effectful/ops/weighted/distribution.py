import jax
from numpyro.distributions import Distribution

from effectful.handlers.jax import numpy as jnp
from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Operation
from effectful.ops.weighted.jax import reals
from effectful.ops.weighted.sugar import Sum


@defop
def D(*args: tuple[tuple[int, ...], jax.Array]) -> jax.Array:
    if not all(isinstance(kv, tuple) and len(kv) == 2 for kv in args):
        raise ValueError("Expected a sequence of key-value pairs")
    raise NotHandled


@defop
def kl_divergence(p: Distribution, q: Distribution) -> jax.Array:
    # KL(p, q) = ∫ p(x) (log p(x) - log q(x)) dx
    x: Operation[[], jax.Array] = defop(jax.Array, name="x")  # type: ignore
    log_p = p.log_prob(x())
    log_q = q.log_prob(x())
    return Sum({x: reals()}, jnp.exp(log_p) * (log_p - log_q))
