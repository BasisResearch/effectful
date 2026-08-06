import types
from typing import TYPE_CHECKING

import jax.numpy

from .._handlers import _register_jax_op, _register_jax_op_no_partial_eval

_no_overload = ["array", "asarray"]

for name, op in jax.numpy.__dict__.items():
    if isinstance(op, types.ModuleType):
        continue
    elif isinstance(op, float | types.NoneType):
        globals()[name] = op
    elif callable(op):
        if name == "__getattr__":
            continue
        elif name in _no_overload:
            globals()[name] = _register_jax_op_no_partial_eval(op)
        else:
            globals()[name] = _register_jax_op(op)

# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.numpy import *  # noqa: F403
