from collections.abc import Iterable

import jax
from effectful.ops.syntax import defop


@defop
def key() -> jax.Array:
    return jax.random.key(0)


@defop
def reals(*, shape: tuple[int, ...] = ()) -> Iterable[jax.Array]:
    raise NotImplementedError
