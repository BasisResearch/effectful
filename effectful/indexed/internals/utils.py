import functools

from effectful.ops.core import Operation, gensym


@functools.lru_cache(maxsize=None)
def name_to_sym(name: str) -> Operation[[], int]:
    return gensym(int, name=name)
