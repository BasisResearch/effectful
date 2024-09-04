import contextlib
from typing import Callable, TypeVar, cast

from typing_extensions import ParamSpec

from effectful.internals.prompts import bind_prompt
from effectful.internals.runtime import get_interpretation
from effectful.ops.core import Interpretation, Operation
from effectful.ops.handler import closed_handler, fwd

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


@Operation
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
    refls = {op: closed_handler(intp)(op) for op in intp}

    return {
        op: closed_handler(refls)(
            intp2[op]
            if op not in intp
            else bind_prompt(
                fwd,  # type: ignore
                closed_handler(intp)(cast(Callable[..., T], op)),
                intp2[op],
            )
        )
        for op in intp2
    }


@contextlib.contextmanager
def runner(intp: Interpretation[S, T]):

    with closed_handler(product(get_interpretation(), intp)):
        yield intp
