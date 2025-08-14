import jax
import pyro
from effectful.handlers.jax import jax_getitem, unbind_dims
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import deffn, defop
from jax import random
from torch import tensor

from weighted.handlers.jax import DenseTensorFold
from weighted.handlers.optimization.plates import PlateUnrolling, plated
from weighted.ops.sugar import Sum

"""
Consider the plated factor graph from Example 3.1. in [1],
which is liftable as J contained within I (c.f. [2]).

                    +---------------------------------------+
                    |     +---------+                       |
[ F ] --- ( X ) --- | --- |  [ H ]  | --- ( Y ) --- [ G ]   |
                    |     |       j |                       |
                    |     +---------+                     i |
                    +---------------------------------------+
                    
                    
[1]: Obermeyer, F., et al. "Tensor variable elimination for 
     plated factor graphs." ICML. 2019.
[2] Taghipour, Nima, et al. "Completeness results for 
    lifted variable elimination." AISTATS. 2013.

Question 1: What is the naive representation in Weighted.
A naive einsum can just be thought of multiplying tensors 
together along the right axes and then summing everything.
"""


keys = random.split(random.PRNGKey(42), 5)

# define the plate indices
i = defop(jax.Array, name="i")
j = defop(jax.Array, name="j")
i_size = 2
j_size = 3
plate_streams = {i: jnp.arange(i_size), j: jnp.arange(j_size)}

# define the variables
x = defop(jax.Array, name="x")
y = defop(jax.Array, name="y")
x_size = 2
y_size = 3
var_streams = {x: jnp.arange(x_size), y: jnp.arange(y_size)}

# define the factors
f = defop(jax.Array, name="f")
g = defop(jax.Array, name="g")
h = defop(jax.Array, name="h")

F_arr = random.uniform(keys[0], shape=(x_size,))
H_arr = random.uniform(keys[1], shape=(x_size, y_size, i_size, j_size))
G_arr = random.uniform(keys[2], shape=(y_size, i_size))

yi = unbind_dims(y(), i)  # type: ignore
F = jax_getitem(f(), (x(),))  # type: ignore
H = jax_getitem(h(), (x(), yi, i(), j()))  # type: ignore
G = jax_getitem(g(), (yi, i()))  # type: ignore

factor_intp = {f: deffn(F_arr), h: deffn(H_arr), g: deffn(G_arr)}

# the Plated op marks indices as first-order variables
model = plated(plate_streams, Sum(var_streams, F * G * H))

# LiftR handles the Plated op, reducing it to nested folds.
with handler(PlateUnrolling()):
    model = evaluate(model)

# Now, we can just evaluate the folds as usual to get the final result.
with handler(factor_intp), handler(DenseTensorFold()):
    contraction = evaluate(model)
print("result:", contraction)

expected = pyro.ops.contract.einsum(
    "x,xyij,yi->", tensor(F_arr), tensor(H_arr), tensor(G_arr), plates="ij"
)
print("expected", expected[0].item())
