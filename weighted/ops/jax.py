from collections.abc import Iterable

import jax
from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled


@defop
def key() -> jax.Array:
    return jax.random.key(0)


@defop
def reals(*, shape: tuple[int, ...] = ()) -> Iterable[jax.Array]:
    raise NotHandled
