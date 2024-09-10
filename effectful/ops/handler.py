import contextlib
from typing import Optional, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.prompts import Prompt, bind_prompt
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


def coproduct(
    intp: Interpretation[S, T],
    intp2: Interpretation[S, T],
    prompt: Prompt[T] = fwd,  # type: ignore
) -> Interpretation[S, T]:
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
