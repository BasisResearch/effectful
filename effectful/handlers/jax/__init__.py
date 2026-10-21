"""Mirrored JAX effects, representation joins, and interpretation-aware transforms."""

try:
    import jax as _jax  # noqa: F401 -- check the optional dependency
except ImportError:
    raise ImportError("JAX is required to use effectful.handlers.jax")

# Register symbolic array terms before exposing numerical APIs.
from . import _terms as _terms
from . import lax as lax
from . import numpy as numpy
from . import scipy as scipy
from ._callables import wrap_callable as wrap_callable
from ._custom_array import CustomArrayControlFlow as CustomArrayControlFlow
from ._custom_array import is_custom_array as is_custom_array
from ._custom_array import to_jax_array as to_jax_array
from ._custom_array import upcast as upcast
from ._custom_array import upcast_many as upcast_many
from ._handlers import bind_dims as bind_dims
from ._handlers import jax_getitem as jax_getitem
from ._handlers import sizesof as sizesof
from ._handlers import unbind_dims as unbind_dims
from ._transforms import checkpoint as checkpoint
from ._transforms import custom_jvp as custom_jvp
from ._transforms import custom_vjp as custom_vjp
from ._transforms import eval_shape as eval_shape
from ._transforms import grad as grad
from ._transforms import hessian as hessian
from ._transforms import jacfwd as jacfwd
from ._transforms import jacrev as jacrev
from ._transforms import jit as jit
from ._transforms import jvp as jvp
from ._transforms import linear_transpose as linear_transpose
from ._transforms import linearize as linearize
from ._transforms import make_jaxpr as make_jaxpr
from ._transforms import remat as remat
from ._transforms import value_and_grad as value_and_grad
from ._transforms import vjp as vjp
from ._transforms import vmap as vmap
