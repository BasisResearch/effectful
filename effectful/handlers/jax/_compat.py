"""Private JAX APIs used by the experimental primitive interpreter.

Tested against JAX 0.9.1 and 0.9.2. Keeping this import boundary small makes
upgrading the interpreter independent of effectful's existing array handlers.
"""

import jax
from jax._src import core
from jax._src import linear_util as lu
from jax._src.ad_checkpoint import remat_p
from jax._src.custom_batching import custom_vmap_p
from jax._src.custom_dce import custom_dce_p
from jax._src.custom_derivatives import (
    custom_jvp_call_p,
    custom_vjp_call_p,
    linear_call_p,
    remat_opt_p,
)
from jax._src.custom_transpose import custom_transpose_p
from jax._src.export._export import call_exported_p
from jax._src.interpreters.pxla import xla_pmap_p
from jax._src.lax.control_flow.loops import eval_jaxpr_p
from jax._src.lax.control_flow.solves import linear_solve_p
from jax._src.literals import TypedNdArray
from jax._src.shard_map import shard_map_p
from jax._src.state.discharge import run_state_p

# No recursive interception defaults are provided for these boundaries.
# Keep supported primitives out of this list so their defaults can forward freely.
UNSUPPORTED_HIGHER_ORDER_PRIMITIVES = (
    core.call_p,
    core.closed_call_p,
    remat_p,
    remat_opt_p,
    custom_vmap_p,
    custom_dce_p,
    custom_transpose_p,
    linear_call_p,
    linear_solve_p,
    shard_map_p,
    xla_pmap_p,
    run_state_p,
    eval_jaxpr_p,
    call_exported_p,
    jax.lax.reduce_p,
    jax.lax.reduce_window_p,
    jax.lax.select_and_scatter_p,
    jax.lax.scatter_p,
    jax.lax.scatter_add_p,
    jax.lax.scatter_sub_p,
    jax.lax.scatter_mul_p,
    jax.lax.scatter_min_p,
    jax.lax.scatter_max_p,
)


def check_version():
    if jax.__version__ not in {"0.9.1", "0.9.2"}:
        raise RuntimeError(
            "effectful JAX interception requires JAX 0.9.1 or 0.9.2; "
            f"found {jax.__version__}"
        )


def bind_primitive(primitive, *args, **kwargs):
    binding = primitive.get_bind_params(kwargs)
    if isinstance(binding, tuple):  # 0.9.1: positional subfunctions
        subfuns, keywords = binding
        return primitive.bind(*subfuns, *args, **keywords)
    return primitive.bind(*args, **binding)


def custom_bind_params(primitive, params):
    """Return custom-call subfunctions and parameters uniformly across versions."""
    binding = primitive.get_bind_params(params)
    if isinstance(binding, tuple):  # 0.9.1: positional subfunctions
        return binding
    keywords = dict(binding)
    return keywords.pop("subfuns"), keywords


def bind_custom(primitive, args, subfuns, params):
    if jax.__version__ == "0.9.1":
        return primitive.bind(*subfuns, *args, **params)
    return primitive.bind(*args, subfuns=subfuns, **params)


__all__ = [
    "UNSUPPORTED_HIGHER_ORDER_PRIMITIVES",
    "core",
    "lu",
    "custom_jvp_call_p",
    "custom_vjp_call_p",
    "check_version",
    "bind_primitive",
    "bind_custom",
    "custom_bind_params",
    "TypedNdArray",
    "with_constants",
    "eval_program",
]


def with_constants(jaxpr, constants):
    return core.ClosedJaxpr(jaxpr, constants)


def eval_program(program, values):
    return core.eval_jaxpr(program.jaxpr, program.consts, *values)
