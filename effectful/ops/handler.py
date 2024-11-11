import contextlib
from typing import Callable, Optional, TypeVar, cast

from typing_extensions import ParamSpec

from effectful.internals.prompts import bind_prompt
from effectful.internals.runtime import get_interpretation, interpreter
from effectful.ops.core import Interpretation, Operation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@Operation
def fwd(__result: Optional[S], *args, **kwargs) -> S:
    return __result  # type: ignore


def coproduct(
    intp: Interpretation[S, T],
    intp2: Interpretation[S, T],
) -> Interpretation[S, T]:
    res = dict(intp)

    for op, i2 in intp2.items():
        i1 = intp.get(op)
        if i1:
            res[op] = bind_prompt(fwd, i1, i2)  # type: ignore
        else:
            res[op] = bind_prompt(fwd, op.__default_rule__, i2)

    return res


def product(
    intp: Interpretation[S, T],
    intp2: Interpretation[S, T],
) -> Interpretation[S, T]:
    # on prompt, jump to the outer interpretation and interpret it using itself
    refls = {op: closed_handler(intp)(op) for op in intp}

    return {
        op: closed_handler(refls)(
            bind_prompt(fwd, op.__default_rule__, intp2[op])
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
def handler(intp: Interpretation[S, T]):
    with interpreter(coproduct(get_interpretation(), intp)):
        yield intp


@contextlib.contextmanager
def closed_handler(intp: Interpretation[S, T]):
    with interpreter(coproduct({}, {**get_interpretation(), **intp})):
        yield intp
