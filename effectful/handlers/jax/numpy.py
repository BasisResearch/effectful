from typing import TYPE_CHECKING

import jax.numpy

from ._handlers import _register_jax_op

_no_overload = ["array", "asarray"]

for name, op in jax.numpy.__dict__.items():
    if callable(op):
        globals()[name] = _register_jax_op(op)

for name in _no_overload:
    globals()[name] = jax.numpy.__dict__[name]

# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.numpy import *  # noqa: F403
