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


Sum = _SumMonoid(kernel=jnp.add, identity=jnp.asarray(0))


class _ProductMonoid(CommutativeMonoidWithZero[jax.Array]):
    def scalar_mul(self, v: jax.Array, x: int) -> jax.Array:
        return v**x


Product = _ProductMonoid(
    kernel=jnp.multiply, identity=jnp.asarray(1), zero=jnp.asarray(0)
)

Min = Semilattice(kernel=jnp.minimum, identity=jnp.asarray(float("-inf")))
Max = Semilattice(kernel=jnp.maximum, identity=jnp.asarray(float("inf")))
LogSumExp = CommutativeMonoid(kernel=jnp.logaddexp, identity=jnp.asarray(float("-inf")))


@Operation.define
def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = jnp.repeat(x, y.shape[0], axis=0), jnp.tile(y, (x.shape[0], 1))
    return jnp.hstack([x, y])


CartesianProd = Monoid(kernel=cartesian_prod, identity=jnp.array([]))

distributes_over.register(Max.plus, Min.plus)
distributes_over.register(Min.plus, Max.plus)
distributes_over.register(Sum.plus, Min.plus)
distributes_over.register(Sum.plus, Max.plus)
distributes_over.register(Product.plus, Sum.plus)
distributes_over.register(Sum.plus, LogSumExp.plus)
