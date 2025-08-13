from typing import TYPE_CHECKING

import jax.numpy.linalg

from effectful.handlers.jax._handlers import _register_jax_op

for name, op in jax.numpy.linalg.__dict__.items():
    if callable(op):
        globals()[name] = _register_jax_op(op)

# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.numpy.linalg import *  # noqa: F403
