from typing import Callable, Optional, TypeVar, cast

from typing_extensions import ParamSpec

from effectful.internals.prompts import Prompt, bind_prompt
from effectful.ops.core import Interpretation, Operation
from effectful.ops.interpreter import interpreter

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


@Operation
def reflect(__result: Optional[S]) -> S:
    return __result  # type: ignore


@Operation
def product(
    intp: Interpretation[S, T],
    *intps: Interpretation[S, T],
    prompt: Prompt[T] = reflect,  # type: ignore
) -> Interpretation[S, T]:
    if len(intps) == 0:  # unit
        return intp
    elif len(intps) > 1:  # associativity
        return product(intp, product(*intps, prompt=prompt), prompt=prompt)

    (intp2,) = intps

    # on prompt, jump to the outer interpretation and interpret it using itself
    refls = {op: interpreter(intp)(op) for op in intp}

    return {
        op: interpreter(refls)(
            intp2[op]
            if op not in intp
            else bind_prompt(
                prompt,
                interpreter(intp)(cast(Callable[..., T], op)),
                intp2[op],
            )
        )
        for op in intp2
    }
