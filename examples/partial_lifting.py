import jax
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import deffn, defop
from jax import random

from weighted.handlers.jax import DenseTensorFold
from weighted.handlers.optimization.cartesian_product import (
    FoldDistributeCartesianProduct,
    SplitCartesianProductFold,
)
from weighted.ops.sugar import CartesianProd, Prod, Sum

"""
Simplest plated factor graph that is not liftable
in general as it is non-hierarchical. However,
we can still lift one of the plates (here j)
and ground the other one (here i).

+-----------------+
| i      +---------|--------+
|        |         |        |
| ( X )--|--[ F ]--|--( Y ) |
|        |         |        |
|--------|---------+      j |
         +------------------+
"""


def main():
    key = random.PRNGKey(0)

    # define the plate indices
    i = defop(jax.Array, name="i")()
    j = defop(jax.Array, name="j")()
    i_size, j_size = 2, 3
    i_stream = {i.op: jnp.arange(i_size)}
    j_stream = {j.op: jnp.arange(j_size)}

    # define the variables
    x = defop(jax.Array, name="x")()
    y = defop(jax.Array, name="y")()
    x_size, y_size = 2, 3
    x_stream = {x.op: CartesianProd(i_stream, jnp.arange(x_size))}
    y_stream = {y.op: CartesianProd(j_stream, jnp.arange(y_size))}

    # define the factors
    f = defop(jax.Array, name="f")()
    F_arr = random.uniform(key, shape=(x_size, y_size))

    # try lifting
    fold_opt = coproduct(SplitCartesianProductFold(), FoldDistributeCartesianProduct())
    with handler(fold_opt):
        model = Sum(x_stream | y_stream, Prod(j_stream, Prod(i_stream, f[x[i], y[j]])))

    # Now, we can just instantiate the arrays and compute the result.
    factor_intp = {f.op: deffn(F_arr)}
    with handler(factor_intp), handler(DenseTensorFold()):
        contraction = evaluate(model)
    print("result:", contraction)

    # Compare with unrolled computation
    expected = jnp.einsum("ac,bc,ad,bd,ae,be->", *([F_arr] * 6))
    print("expected", expected)
    assert jnp.allclose(expected, contraction)


if __name__ == "__main__":
    main()
