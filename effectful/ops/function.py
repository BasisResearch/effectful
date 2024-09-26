import collections
import collections.abc
import dataclasses
import functools
import inspect
import numbers
import operator
import typing
from typing import (
    Annotated,
    Callable,
    Generic,
    Mapping,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter
from effectful.internals.sugar import (
    Annotation,
    Bound,
    Scoped,
    _BaseNeutral,
    _NeutralExpr,
    _StuckNeutral,
    embed,
    gensym,
    hoas,
    unembed,
)
from effectful.ops.core import (
    Expr,
    Interpretation,
    Operation,
    Term,
    apply,
    evaluate,
    typeof,
)
from effectful.ops.handler import handler

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@Operation
def defun(
    body: T,
    *args: Annotated[Operation, Bound()],
    **kwargs: Annotated[Operation, Bound()],
) -> Callable[..., T]:
    raise NotImplementedError


@Operation
def call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    match unembed(fn):
        case Term(defun_, (body, *argvars), kwvars) if defun_ == defun:
            kwvars = dict(kwvars)
            subs = {
                **{v: functools.partial(lambda x: x, a) for v, a in zip(argvars, args)},
                **{
                    kwvars[k]: functools.partial(lambda x: x, kwargs[k]) for k in kwargs
                },
            }
            with handler(subs):
                return embed(evaluate(unembed(body)))
        case fn_literal if not isinstance(fn_literal, Term):
            return fn_literal(*args, **kwargs)
        case _:
            raise NotImplementedError


@embed.register(collections.abc.Callable)
class _CallableNeutral(
    Generic[P, T],
    collections.abc.Callable[P, T],
    _BaseNeutral[collections.abc.Callable[P, T]],
):
    def __call__(self, *args: _NeutralExpr, **kwargs: _NeutralExpr) -> _NeutralExpr[T]:
        return call(
            self, *[embed(a) for a in args], **{k: embed(v) for k, v in kwargs.items()}
        )


@unembed.register(collections.abc.Callable)
def _unembed_callable(value: Callable[P, T]) -> Term[Callable[P, T]]:
    assert not isinstance(value, _StuckNeutral)

    sig = inspect.signature(value)

    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise NotImplementedError(
                f"cannot unembed {value}: parameter {name} is variadic"
            )

    bound_sig = sig.bind(
        **{name: gensym(param.annotation) for name, param in sig.parameters.items()}
    )
    bound_sig.apply_defaults()

    with interpreter(
        {
            apply: lambda _, op, *a, **k: embed(op.__free_rule__(*a, **k)),
            call: call.__default_rule__,
        }
    ):
        body = value(
            *[a() for a in bound_sig.args],
            **{k: v() for k, v in bound_sig.kwargs.items()},
        )

    return defun(body, *bound_sig.args, **bound_sig.kwargs)
