"""Private JAX APIs used by the experimental primitive interpreter.

Tested against JAX 0.9.1 and 0.9.2. Keeping this import boundary small makes
upgrading the interpreter independent of effectful's existing array handlers.
"""

import jax
from jax._src import core
from jax._src.literals import TypedNdArray


def check_version():
    if jax.__version__ not in {"0.9.1", "0.9.2"}:
        raise RuntimeError(
            "effectful JAX interception requires JAX 0.9.1 or 0.9.2; "
            f"found {jax.__version__}"
        )


def bind_primitive(primitive, args, params):
    binding = primitive.get_bind_params(params)
    if isinstance(binding, tuple):  # 0.9.1: positional subfunctions
        subfuns, keywords = binding
        return primitive.bind(*subfuns, *args, **keywords)
    return primitive.bind(*args, **binding)


def bind_custom(primitive, args, subfuns, params):
    if jax.__version__ == "0.9.1":
        return primitive.bind(*subfuns, *args, **params)
    return primitive.bind(*args, subfuns=subfuns, **params)


__all__ = ["core", "check_version", "bind_primitive", "bind_custom", "TypedNdArray"]
