import contextlib
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.prompts import bind_continuation, compose_continuation
from effectful.internals.runtime import get_interpretation
from effectful.ops.core import Interpretation
from effectful.ops.interpreter import interpreter

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@bind_continuation
def fwd(cont: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return cont(*args, **kwargs)


def coproduct(
    intp: Interpretation[S, T],
    *intps: Interpretation[S, T],
) -> Interpretation[S, T]:
    if len(intps) == 0:  # unit
        return intp
    elif len(intps) > 1:  # associativity
        return coproduct(intp, coproduct(*intps))

    (intp2,) = intps

    res = dict(intp)

    for op, i2 in intp2.items():
        i1 = intp.get(op)
        if i1:
            res[op] = compose_continuation(i1, i2)
        else:
            res[op] = i2

    return res


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
def handler(intp: Interpretation[S, T], *, closed: bool = False):
    with interpreter((union if closed else coproduct)(get_interpretation(), intp)):
        yield intp
