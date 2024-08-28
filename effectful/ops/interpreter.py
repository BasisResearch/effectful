import contextlib
from typing import TypeVar

from typing_extensions import ParamSpec

from effectful.ops.core import Interpretation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


def union(
    intp: Interpretation[S, T],
    *intps: Interpretation[S, T],
) -> Interpretation[S, T]:
    if len(intps) == 0:  # unit
        return intp
    elif len(intps) > 1:  # associativity
        return union(intp, union(*intps))

    (intp2,) = intps

    res = dict(intp)

    for op, i2 in intp2.items():
        res[op] = i2

    return res


@contextlib.contextmanager
def interpreter(intp: Interpretation):
    from ..internals.runtime import swap_interpretation

    old_intp = swap_interpretation(intp)
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)
