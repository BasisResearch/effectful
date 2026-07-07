from typing import TYPE_CHECKING

import jax.lax

from effectful.handlers.jax._handlers import JaxOperation

for name, op in jax.lax.__dict__.items():
    wrapped_value = None
    if callable(op):
        wrapped_value = JaxOperation.define(op)
    else:
        continue

    globals()[name] = wrapped_value


# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.lax import *  # type: ignore[assignment] # noqa: F403
