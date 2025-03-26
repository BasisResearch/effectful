# side effect: register defdata for jax.Array
import effectful.handlers.jax.terms  # noqa: F401

from .handlers import jax_getitem as jax_getitem
from .handlers import sizesof as sizesof
from .handlers import to_array as to_array
