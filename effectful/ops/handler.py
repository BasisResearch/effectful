import contextlib
import functools
from typing import Callable, Optional, TypeVar

from typing_extensions import Concatenate, ParamSpec

from effectful.internals.runtime import (
    _flatten_args,
    get_interpretation,
    get_runtime,
    interpreter,
    set_continuation,
)
from effectful.ops.core import Interpretation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


def bind_continuation(
    fn: Callable[Concatenate[Callable[[], T], P], T]
) -> Callable[P, T]:
    cc = get_runtime()._CONTINUATION_STATE.gets(lambda c: c(), default=bind_result(lambda r: r))  # type: ignore
    return functools.wraps(fn)(functools.partial(fn, cc))


def bind_args(fn: Callable[P, T]) -> Callable[[], T]:
    return get_runtime()._ARG_STATE.gets(_flatten_args(fn), default=((), {}))


def bind_result(fn: Callable[Concatenate[Optional[S], P], T]) -> Callable[P, T]:
    return get_runtime()._RESULT_STATE.gets(fn, default=None)


def bind_result_to_method(fn):
    return bind_result(lambda r, self, *a, **k: fn(self, r, *a, **k))


@bind_continuation
def fwd(cont: Callable[[], T], s: Optional[S], *args, **kwargs) -> T:
    cont_: Callable[[], T]
    cont_ = (
        functools.partial(get_runtime()._ARG_STATE.sets(cont), (args, kwargs))  # type: ignore
        if args or kwargs
        else cont
    )
    return get_runtime()._RESULT_STATE.sets(cont_)(s)


def union(
    intp: Interpretation[S, T], intp2: Interpretation[S, T]
) -> Interpretation[S, T]:
    return {**intp, **intp2}


def coproduct(
    intp: Interpretation[S, T], intp2: Interpretation[S, T]
) -> Interpretation[S, T]:
    res = dict(intp)

    for op, i2 in intp2.items():
        i1 = intp.get(op)
        if i1:
            res[op] = set_continuation(i1, i2)
        else:
            res[op] = i2

    return res


def product(
    intp: Interpretation[S, T], intp2: Interpretation[S, T]
) -> Interpretation[S, T]:
    # implicit fixpoint semantics
    intp_ = {op: handler(intp, closed=True)(intp[op]) for op in intp}
    intp2_ = {op: handler(coproduct(intp_, intp2), closed=True)(intp2[op]) for op in intp2}
    # # explicit fixpoint semantics
    # h_outer = lambda f: lambda *a, **k: handler(coproduct(intp2_, intp_), closed=True)(f)(*a, **k)
    # h_inner = lambda f: lambda *a, **k: handler(coproduct(intp_, intp2_), closed=True)(f)(*a, **k)
    # intp_: Interpretation[S, T] = {op: h_outer(intp[op]) for op in intp}
    # intp2_: Interpretation[S, T] = {op: h_inner(intp2[op]) for op in intp2}
    return {op: fn for op, fn in coproduct(intp_, intp2_).items() if op in intp2_}
    # previous definition (incorrect?)
    # return coproduct(
    #     {op: handler(intp, closed=True)(intp[op]) for op in intp if op in intp2},
    #     {op: handler(intp, closed=True)(intp2[op]) for op in intp2},
    # )


@contextlib.contextmanager
def handler(intp: Interpretation[S, T], *, closed: bool = False):
    with interpreter((union if closed else coproduct)(get_interpretation(), intp)):
        yield intp
