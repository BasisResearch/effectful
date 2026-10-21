"""Control-flow effects whose defaults delegate directly to JAX."""

import jax

from effectful.ops.syntax import defop

cond = defop(jax.lax.cond)
switch = defop(jax.lax.switch)
while_loop = defop(jax.lax.while_loop)
scan = defop(jax.lax.scan)
