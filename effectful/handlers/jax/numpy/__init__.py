from typing import TYPE_CHECKING

import jax.numpy

from .._handlers import _register_jax_op, _register_jax_op_no_partial_eval

_no_overload = ["array", "asarray"]

for name, op in jax.numpy.__dict__.items():
    if not callable(op):
        continue

    jax_op = (
        _register_jax_op_no_partial_eval(op)
        if name in _no_overload
        else _register_jax_op(op)
    )
    globals()[name] = jax_op


# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.numpy import *  # noqa: F403
