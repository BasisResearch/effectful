import types
import typing
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
    if isinstance(op, types.ModuleType):
        continue

    # copy constants
    if isinstance(op, float | types.NoneType):
        globals()[name] = op

    if callable(op):
        if name == "__getattr__":
            continue

        elif name in _NO_OVERLOAD:
            globals()[name] = _register_jax_op_no_partial_eval(op)

        else:
            globals()[name] = _register_jax_op(op)
        jax_op = (
            _register_jax_op_no_partial_eval(op)
            if name in _NO_OVERLOAD
            else _register_jax_op(op)
        )
        globals()[name] = jax_op

for name in _REDUCTION:
    op = globals()[name]
    globals()[name] = handler({op: _reduce_named})(op)


einsum = Operation.define(_einsum_named)


@Operation.define
def asarray(a, **kwargs) -> jax.Array:
    import jax.core

    from effectful.ops.semantics import typeof
    from effectful.ops.types import NotHandled, Term

    if isinstance(a, Term):
        if issubclass(typeof(a), jax.Array | jax.core.Tracer) and not kwargs:
            return typing.cast(jax.Array, a)
        else:
            raise NotHandled
    return jax.numpy.asarray(a, **kwargs)


# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.numpy import *  # type: ignore[assignment] # noqa: F403
