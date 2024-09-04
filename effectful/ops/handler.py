import contextlib
from typing import Optional, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.prompts import Prompt, bind_prompt, bind_result
from effectful.internals.runtime import get_interpretation, interpreter
from effectful.ops.core import Interpretation, Operation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@Operation
def fwd(__result: Optional[S]) -> S:
    return __result  # type: ignore


@Operation
def coproduct(
    intp: Interpretation[S, T],
    *intps: Interpretation[S, T],
    prompt: Prompt[T] = fwd,  # type: ignore
) -> Interpretation[S, T]:
    if len(intps) == 0:  # unit
        return intp
    elif len(intps) > 1:  # associativity
        return coproduct(intp, coproduct(*intps, prompt=prompt), prompt=prompt)

    (intp2,) = intps

    res = dict(intp)

    for op, i2 in intp2.items():
        i1 = intp.get(op)
        if i1:
            res[op] = bind_prompt(prompt, i1, i2)
        else:
            res[op] = i2

    return res


@contextlib.contextmanager
def handler(
    intp: Interpretation[S, T],
    *,
    prompt: Prompt[T] = fwd,  # type: ignore
):
    with interpreter(coproduct(get_interpretation(), intp, prompt=prompt)):
        yield intp


@contextlib.contextmanager
def closed_handler(intp: Interpretation[S, T]):
    with interpreter({**get_interpretation(), **intp}):
        yield intp
