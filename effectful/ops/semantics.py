import contextlib
import functools
import typing
from typing import Callable, Optional, Set, Type, TypeVar

import tree
from typing_extensions import ParamSpec

from effectful.ops.syntax import NoDefaultRule, deffn, defop, defterm
from effectful.ops.types import Expr, Interpretation, MaybeResult, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@defop  # type: ignore
def apply(
    intp: Interpretation[S, T], op: Operation[P, S], *args: P.args, **kwargs: P.kwargs
) -> T:
    if op in intp:
        return intp[op](*args, **kwargs)
    elif apply in intp:
        return intp[apply](intp, op, *args, **kwargs)
    else:
        return op.__default_rule__(*args, **kwargs)  # type: ignore


@defop  # type: ignore
def call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    match defterm(fn):
        case Term(defun_, (body_, *argvars_), kwvars_) if defun_ == deffn:
            body: Expr[Callable[P, T]] = body_
            argvars: tuple[Operation, ...] = typing.cast(
                tuple[Operation, ...], argvars_
            )
            kwvars: dict[str, Operation] = typing.cast(
                dict[str, Operation], dict(kwvars_)
            )
            subs = {
                **{v: functools.partial(lambda x: x, a) for v, a in zip(argvars, args)},
                **{
                    kwvars[k]: functools.partial(lambda x: x, kwargs[k]) for k in kwargs
                },
            }
            with handler(subs):
                return evaluate(body)  # type: ignore
        case _:
            raise NoDefaultRule


@defop
def fwd(__result: MaybeResult[S], *args, **kwargs) -> S:
    return __result  # type: ignore


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
        renaming = {op: defop(op) for op in intp2 if op in intp}
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


def evaluate(expr: Expr[T], *, intp: Optional[Interpretation[S, T]] = None) -> Expr[T]:
    if intp is None:
        from effectful.internals.runtime import get_interpretation

        intp = get_interpretation()

    match defterm(expr):
        case Term(op, args, kwargs):
            (args, kwargs) = tree.map_structure(
                functools.partial(evaluate, intp=intp), (args, dict(kwargs))
            )
            return apply.__default_rule__(intp, op, *args, **kwargs)  # type: ignore
        case literal:
            return literal


def typeof(term: Expr[T]) -> Type[T]:
    from effectful.internals.runtime import interpreter

    with interpreter({apply: lambda _, op, *a, **k: op.__type_rule__(*a, **k)}):  # type: ignore
        return evaluate(term)  # type: ignore


def strof(term: Expr[S]) -> str:
    from effectful.internals.runtime import interpreter

    def _to_str(_, op, *args, **kwargs):
        return f"{op}({', '.join(str(a) for a in args)}, {', '.join(f'{k}={v}' for k, v in kwargs.items())})"

    with interpreter({apply: _to_str}):  # type: ignore
        return evaluate(defterm(term))  # type: ignore


def fvsof(term: Expr[S]) -> Set[Operation]:
    from effectful.internals.runtime import interpreter

    _fvs: Set[Operation] = set()

    def _update_fvs(_, op, *args, **kwargs):
        _fvs.add(op)
        for bound_var in op.__fvs_rule__(*args, **kwargs):
            _fvs.pop(bound_var, None)

    with interpreter({apply: _update_fvs}):  # type: ignore
        evaluate(defterm(term))  # type: ignore

    return _fvs
