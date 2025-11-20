import jax
import pyro
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import deffn, defop
from jax import random
from torch import tensor

from weighted.handlers.jax import DenseTensorReduce
from weighted.handlers.optimization import ReduceDistributeTerm
from weighted.handlers.optimization.cartesian_product import (
    ReduceDistributeCartesianProduct,
)
from weighted.handlers.optimization.reorder import ReduceNoStreams
from weighted.ops.sugar import CartesianProd, Prod, Sum

"""
Example for a plated factor graph (taken from Example 3.1. in [1]).
This is liftable as J is fully contained within I (c.f. [2]).

                    +---------------------------------------+
                    |     +---------+                       |
[ F ] --- ( X ) --- | --- |  [ H ]  | --- ( Y ) --- [ G ]   |
                    |     |       j |                       |
                    |     +---------+                     i |
                    +---------------------------------------+


[1] Obermeyer, F., et al. "Tensor variable elimination for 
     plated factor graphs." ICML. 2019.
[2] Taghipour, Nima, et al. "Completeness results for 
    lifted variable elimination." AISTATS. 2013.

A plated variable in a factor graph is implemented by
taking a cartesian product over its stream. Computing this
naively is highly inefficient, and the ReduceDistributeCartesianProduct
reduces this to polynomial complexity.
"""


def main():
    keys = random.split(random.PRNGKey(0), 5)

    # define the plate indices
    i = defop(jax.Array, name="i")
    j = defop(jax.Array, name="j")
    i_size = 5
    j_size = 6
    i_stream = {i: jnp.arange(i_size)}
    j_stream = {j: jnp.arange(j_size)}

    # define the variables
    x = defop(jax.Array, name="x")
    y = defop(jax.Array, name="y")
    x_size = 2
    y_size = 3
    x_stream = {x: jnp.arange(x_size)}
    y_stream = {y: CartesianProd(i_stream, jnp.arange(y_size))}

    # define the factors
    f = defop(jax.Array, name="f")
    g = defop(jax.Array, name="g")
    h = defop(jax.Array, name="h")

    F_arr = random.uniform(keys[0], shape=(x_size,))
    G_arr = random.uniform(keys[1], shape=(y_size, i_size))
    H_arr = random.uniform(keys[2], shape=(x_size, y_size, i_size, j_size))

    # construct model with lifting optimizations
    reduce_opt = coproduct(ReduceDistributeTerm(), ReduceDistributeCartesianProduct())
    with handler(reduce_opt), handler(ReduceNoStreams()):
        F = f()[x()]
        G = g()[y()[i()], i()]
        H = h()[x(), y()[i()], i(), j()]
        body = F * Prod(i_stream, G * Prod(j_stream, H))
        model = Sum(x_stream | y_stream, body)

    # Now, we can just instantiate the arrays and compute the result.
    factor_intp = {f: deffn(F_arr), g: deffn(G_arr), h: deffn(H_arr)}
    with handler(factor_intp), handler(DenseTensorReduce()):
        contraction = evaluate(model)
    print("result:", contraction)

    # Compare with pyro result
    torch_arrs = tensor(F_arr), tensor(G_arr), tensor(H_arr)
    expected = pyro.ops.contract.einsum("x,yi,xyij->", *torch_arrs, plates="ij")
    expected = expected[0].item()
    print("expected", expected)
    assert jnp.allclose(expected, contraction)


if __name__ == "__main__":
    main()
