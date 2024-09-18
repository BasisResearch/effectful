from typing import Callable, Optional, TypeVar, cast

from typing_extensions import ParamSpec

from effectful.internals.prompts import bind_prompt
from effectful.ops.core import Interpretation, Operation
from effectful.ops.handler import closed_handler

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


@Operation
def reflect(__result: Optional[S], *args, **kwargs) -> S:
    return __result  # type: ignore


def product(
    intp: Interpretation[S, T],
    intp2: Interpretation[S, T],
    prompt: Operation[..., T] = reflect,  # type: ignore
) -> Interpretation[S, T]:
    # on prompt, jump to the outer interpretation and interpret it using itself
    refls = {op: closed_handler(intp)(op) for op in intp}

    return {
        op: closed_handler(refls)(
            intp2[op]
            if op not in intp
            else bind_prompt(
                prompt,
                closed_handler(intp)(cast(Callable[..., T], op)),
                intp2[op],
            )
        )
        for op in intp2
    }
