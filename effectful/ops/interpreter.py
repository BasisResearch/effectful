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
def interpreter(intp: Interpretation, *, unset: bool = True):
    from ..internals.runtime import get_interpretation, swap_interpretation

    old_intp = get_interpretation()
    try:
        new_intp = {
            op: intp[op] if op in intp else old_intp[op]
            for op in set(intp.keys()) | set(old_intp.keys())
        }
        old_intp = swap_interpretation(new_intp)
        yield intp
    finally:
        if unset:
            _ = swap_interpretation(old_intp)
        else:
            if len(list(old_intp.keys())) == 0 and len(list(intp.keys())) > 0:
                raise RuntimeError(f"Dangling interpretation on stack: {intp}")
