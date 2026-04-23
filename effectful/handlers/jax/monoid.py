import jax

import effectful.handlers.jax.numpy as jnp
from effectful.ops.types import Operation
from effectful.ops.weighted.monoid import (
    CommutativeMonoid,
    CommutativeMonoidWithZero,
    Monoid,
    Semilattice,
    distributes_over,
)


class _SumMonoid(CommutativeMonoid[jax.Array]):
    def scalar_mul(self, v: jax.Array, x: int) -> jax.Array:
        return v * x


Sum = _SumMonoid.from_binary(jnp.add, 0)


class _ProductMonoid(CommutativeMonoidWithZero[jax.Array]):
    def scalar_mul(self, v: jax.Array, x: int) -> jax.Array:
        return v**x


Product = _ProductMonoid.from_binary(jnp.multiply, 1, 0)

Min = Semilattice.from_binary(jnp.minimum, float("-inf"))
Max = Semilattice.from_binary(jnp.maximum, float("inf"))
LogSumExp = CommutativeMonoid.from_binary(jnp.logaddexp, float("-inf"))


@Operation.define
def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = jnp.repeat(x, y.shape[0], axis=0), jnp.tile(y, (x.shape[0], 1))
    return jnp.hstack([x, y])


CartesianProd = Monoid.from_binary(cartesian_prod, jnp.array([]))

distributes_over.register(jnp.maximum, jnp.minimum)
distributes_over.register(jnp.minimum, jnp.maximum)
distributes_over.register(jnp.add, jnp.minimum)
distributes_over.register(jnp.add, jnp.maximum)
distributes_over.register(jnp.multiply, jnp.add)
distributes_over.register(jnp.add, jnp.logaddexp)
