from types import NoneType
from typing import TYPE_CHECKING

import jax.numpy

from effectful.handlers.jax._handlers import (
    _einsum_named,
    _reduce_named,
    _register_jax_op,
    _register_jax_op_no_partial_eval,
)
from effectful.ops.semantics import handler
from effectful.ops.types import Operation

_NO_OVERLOAD = ["array", "asarray"]
_REDUCTION = ["sum", "prod", "min", "max", "any", "all", "mean", "argmax"]

for name, op in jax.numpy.__dict__.items():
    wrapped_value = None
    if type(op) in (float, NoneType):
        wrapped_value = op
    elif name in _NO_OVERLOAD:
        wrapped_value = _register_jax_op_no_partial_eval(op)
    elif callable(op):
        wrapped_value = _register_jax_op(op)
    else:
        continue

    globals()[name] = wrapped_value

for name in _REDUCTION:
    op = globals()[name]
    globals()[name] = handler({op: _reduce_named})(op)


einsum = Operation.define(_einsum_named)

# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.numpy import *  # type: ignore[assignment] # noqa: F403
