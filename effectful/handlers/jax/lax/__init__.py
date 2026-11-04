"""Mirrored lax operations and representation-aware control flow."""

import inspect
from typing import TYPE_CHECKING

import jax.lax

from .._handlers import _register_jax_op

for _name, _fn in vars(jax.lax).items():
    if _name.startswith("_"):
        continue
    if inspect.isclass(_fn) or not callable(_fn):
        globals()[_name] = _fn
    else:
        globals()[_name] = _register_jax_op(_fn)

if TYPE_CHECKING:
    from jax.lax import *  # noqa: F403
else:
    from ._control_flow import cond as cond
    from ._control_flow import scan as scan
    from ._control_flow import switch as switch
    from ._control_flow import while_loop as while_loop
