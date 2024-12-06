import contextlib
from typing import Callable, Optional, TypeVar

from typing_extensions import Concatenate, ParamSpec

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


def bind_result(fn: Callable[Concatenate[Optional[T], P], T]) -> Callable[P, T]:
    from effectful.internals.runtime import _get_result

    return lambda *a, **k: fn(_get_result(), *a, **k)


def bind_result_to_method(
    fn: Callable[Concatenate[V, Optional[T], P], T]
) -> Callable[Concatenate[V, P], T]:
    return bind_result(lambda r, s, *a, **k: fn(s, r, *a, **k))


def coproduct(
    intp: Interpretation[S, T],
    intp2: Interpretation[S, T],
) -> Interpretation[S, T]:
    from effectful.internals.runtime import (
        _get_args,
        _get_result,
        _set_args,
        _set_prompt,
        _set_result,
    )

    res = dict(intp)
    for op, i2 in intp2.items():
        if op is fwd or op is _get_result or op is _get_args:
            res[op] = i2  # fast path for special cases, should be equivalent if removed
        else:
            i1 = intp.get(op, op.__default_rule__)  # type: ignore
            res[op] = _set_prompt(fwd, _set_result(_set_args(i1)), _set_args(i2))  # type: ignore

    return res


def product(
    intp: Interpretation[S, T],
    intp2: Interpretation[S, T],
) -> Interpretation[S, T]:
    if any(op in intp for op in intp2):  # alpha-rename
        renaming = {op: gensym(op) for op in intp2 if op in intp}
        intp_fresh = {renaming.get(op, op): handler(renaming)(intp[op]) for op in intp}
        return product(intp_fresh, intp2)
    else:
        refls2 = {op: op.__default_rule__ for op in intp2}
        intp_ = coproduct({}, {op: runner(refls2)(intp[op]) for op in intp})
        return {op: runner(intp_)(intp2[op]) for op in intp2}


@contextlib.contextmanager
def runner(intp: Interpretation[S, T]):
    from effectful.internals.runtime import get_interpretation, interpreter

    @interpreter(get_interpretation())
    def _reapply(_, op: Operation[P, S], *args: P.args, **kwargs: P.kwargs):
        return op(*args, **kwargs)

    with interpreter({apply: _reapply, **intp}):
        yield intp


@contextlib.contextmanager
def handler(intp: Interpretation[S, T]):
    from effectful.internals.runtime import get_interpretation, interpreter

    with interpreter(coproduct(get_interpretation(), intp)):
        yield intp


@contextlib.contextmanager
def closed_handler(intp: Interpretation[S, T]):
    from effectful.internals.runtime import get_interpretation, interpreter

    with interpreter(coproduct({}, {**get_interpretation(), **intp})):
        yield intp
