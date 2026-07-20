from typing import TYPE_CHECKING

import jax.lax

from effectful.handlers.jax._handlers import _register_jax_op

for name, op in jax.lax.__dict__.items():
    wrapped_value = None
    if callable(op):
        wrapped_value = _register_jax_op(op)
    else:
        continue

    globals()[name] = wrapped_value


# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.lax import *  # noqa: F403
