import contextlib
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


@contextlib.contextmanager
def runner(
    intp: Interpretation[S, T],
    *,
    prompt: Prompt[T] = reflect,  # type: ignore
    handler_prompt: Optional[Prompt[T]] = None,
):
    from effectful.internals.runtime import get_interpretation

    curr_intp, next_intp = get_interpretation(), intp

    if handler_prompt is not None:
        assert (
            prompt is not handler_prompt
        ), f"runner prompt and handler prompt must be distinct, but got {handler_prompt}"
        h2r = {handler_prompt: prompt}
        curr_intp = {op: interpreter(h2r)(curr_intp[op]) for op in curr_intp.keys()}
        next_intp = {op: interpreter(h2r)(next_intp[op]) for op in next_intp.keys()}

    with interpreter(product(curr_intp, next_intp, prompt=prompt)):
        yield intp
