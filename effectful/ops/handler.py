import contextlib
from typing import Callable, Optional, TypeVar, cast

from typing_extensions import ParamSpec

from effectful.internals.prompts import bind_prompt
from effectful.internals.runtime import get_interpretation, interpreter
from effectful.internals.sugar import gensym
from effectful.ops.core import Interpretation, Operation, apply

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
            res[op] = i2

    return res


def product(
    intp: Interpretation[S, T],
    intp2: Interpretation[S, T],
) -> Interpretation[S, T]:
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


def product2(
    intp: Interpretation[S, T],
    intp2: Interpretation[S, T],
) -> Interpretation[S, T]:
    if not any(op in intp for op in intp2):
        refls2 = {op: op.__default_rule__ for op in intp2}
        intp_ = {op: handler(refls2)(intp[op]) for op in intp}
        return {op: runner(intp_)(intp2[op]) for op in intp2}
    else:
        renaming = {op: gensym(op) for op in intp2 if op in intp}
        intp_ = {renaming.get(op, op): handler(renaming)(intp[op]) for op in intp}
        return product2(intp_, coproduct(renaming, intp2))


@contextlib.contextmanager
def runner(intp: Interpretation[S, T]):

    @interpreter(get_interpretation())
    def _apply(_, op: Operation[P, S], *args: P.args, **kwargs: P.kwargs):
        return op(*args, **kwargs)

    with interpreter({apply: _apply, **intp}):
        yield


@contextlib.contextmanager
def handler(intp: Interpretation[S, T]):
    with interpreter(coproduct(get_interpretation(), intp)):
        yield intp


@contextlib.contextmanager
def closed_handler(intp: Interpretation[S, T]):
    with interpreter({**get_interpretation(), **intp}):
        yield intp
