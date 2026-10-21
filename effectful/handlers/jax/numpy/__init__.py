import types
from typing import TYPE_CHECKING

import jax.numpy

from .._handlers import _register_jax_op, _register_jax_op_no_partial_eval

_no_overload = ["array", "asarray"]

for name, op in jax.numpy.__dict__.items():
    if isinstance(op, types.ModuleType):
        continue

    # Dtype classes are static configuration, not numerical operations.
    if isinstance(op, type):
        globals()[name] = op
        continue

    # copy constants
    if isinstance(op, float | types.NoneType):
        globals()[name] = op

    if callable(op):
        if name == "__getattr__":
            continue

        register = (
            _register_jax_op_no_partial_eval
            if name in _no_overload
            else _register_jax_op
        )
        globals()[name] = register(op)

# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.numpy import *  # noqa: F403
else:
    from . import linalg as linalg
