from typing import TYPE_CHECKING

import jax.numpy

from effectful.handlers.jax._handlers import (
    _reduce_named,
    _register_jax_op,
    _register_jax_op_no_partial_eval,
)
from effectful.ops.semantics import handler

_NO_OVERLOAD = ["array", "asarray"]
_REDUCTION = ["sum", "prod", "min", "max", "any", "all", "mean", "argmax"]

__all__ = []
for name, op in jax.numpy.__dict__.items():
    wrapped_value = None
    if not callable(op):
        wrapped_value = op
    elif name in _NO_OVERLOAD:
        wrapped_value = _register_jax_op_no_partial_eval(op)
    else:
        wrapped_value = _register_jax_op(op)

    globals()[name] = wrapped_value
    __all__.append(name)

for name in _REDUCTION:
    op = globals()[name]
    globals()[name] = handler({op: _reduce_named})(op)

# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.numpy import *  # noqa: F403
