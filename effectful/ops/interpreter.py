import contextlib
from typing import TypeVar

from typing_extensions import ParamSpec

from effectful.ops.core import Interpretation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@contextlib.contextmanager
def interpreter(intp: Interpretation):
    from ..internals.runtime import swap_interpretation

    old_intp: Interpretation = swap_interpretation(intp)
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)
