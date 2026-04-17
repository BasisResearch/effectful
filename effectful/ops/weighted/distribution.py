import jax

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled


@defop
def D(*args: tuple[tuple[int, ...], jax.Array]) -> jax.Array:
    if not all(isinstance(kv, tuple) and len(kv) == 2 for kv in args):
        raise ValueError("Expected a sequence of key-value pairs")
    raise NotHandled
