import contextlib
from copy import copy
from typing import Optional, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.prompts import Prompt, bind_prompts
from effectful.internals.runtime import get_interpretation
from effectful.ops.core import Interpretation, Operation, define
from effectful.ops.interpreter import interpreter

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@define(Operation)
def fwd(__result: Optional[T]) -> T:
    return __result


@define(Operation)
def coproduct(
    intp: Interpretation[S, T],
    *intps: Interpretation[S, T],
    prompt: Prompt[T] = fwd,
) -> Interpretation[S, T]:
    if len(intps) == 0:  # unit
        return intp
    elif len(intps) > 1:  # associativity
        return coproduct(intp, coproduct(*intps, prompt=prompt), prompt=prompt)

    (intp2,) = intps

    res = copy(intp)

    for op, i2 in intp2.items():
        if i1 := intp.get(op):
            res[op] = bind_prompts({prompt: i1})(i2)
        else:
            res[op] = i2

    return res


@contextlib.contextmanager
def handler(intp: Interpretation[S, T], *, prompt: Prompt[T] = fwd):
    with interpreter(coproduct(get_interpretation(), intp, prompt=prompt)):
        yield intp
