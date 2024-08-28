import contextlib
from typing import Callable, TypeVar, cast

from typing_extensions import ParamSpec

from effectful.ops.core import Interpretation, Operation
from effectful.ops.interpreter import interpreter
from effectful.ops.handler import coproduct

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


def product(
    intp: Interpretation[S, T],
    *intps: Interpretation[S, T],
) -> Interpretation[S, T]:
    if len(intps) == 0:  # unit
        return intp
    elif len(intps) > 1:  # associativity
        return product(intp, product(*intps))

    (intp2,) = intps

    # on prompt, jump to the outer interpretation and interpret it using itself
    return coproduct(
        {op: interpreter(intp)(intp[op]) for op in intp if op in intp2},
        {op: interpreter(intp)(intp2[op]) for op in intp2},
    )


@contextlib.contextmanager
def runner(intp: Interpretation[S, T]):
    from effectful.internals.runtime import get_interpretation

    with interpreter(product(get_interpretation(), intp)):
        yield intp
