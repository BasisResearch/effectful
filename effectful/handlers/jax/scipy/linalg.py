"""Mirrored SciPy linear algebra operations."""

from typing import TYPE_CHECKING

import jax.scipy.linalg

from .._handlers import _register_jax_op

for _name, _fn in vars(jax.scipy.linalg).items():
    if not _name.startswith("_") and callable(_fn) and not isinstance(_fn, type):
        globals()[_name] = _register_jax_op(_fn)

if TYPE_CHECKING:
    from jax.scipy.linalg import *  # noqa: F403
