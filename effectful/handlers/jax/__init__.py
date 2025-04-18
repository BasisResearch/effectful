try:
    # Dummy import to check if jax is installed
    import jax  # noqa: F401
    import numpyro  # noqa: F401
except ImportError:
    raise ImportError("Jax and Numpyro are required to use effectful.handlers.jax")

# side effect: register defdata for jax.Array
import effectful.handlers.jax.terms  # noqa: F401

from .handlers import jax_getitem as jax_getitem
from .handlers import jit as jit
from .handlers import sizesof as sizesof
