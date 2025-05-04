try:
    # Dummy import to check if jax is installed
    import jax  # noqa: F401
except ImportError:
    raise ImportError("Jax is required to use effectful.handlers.jax")

# side effect: register defdata for jax.Array
import effectful.handlers.jax._terms  # noqa: F401

from ._handlers import bind_dims as bind_dims
from ._handlers import jax_getitem as jax_getitem
from ._handlers import jit as jit
from ._handlers import sizesof as sizesof
from ._handlers import unbind_dims as unbind_dims
