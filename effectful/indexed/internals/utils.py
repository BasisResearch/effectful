import functools

from effectful.ops.syntax import defop
from effectful.ops.types import Operation


@functools.lru_cache(maxsize=None)
def name_to_sym(name: str) -> Operation[[], int]:
    return defop(int, name=name)
