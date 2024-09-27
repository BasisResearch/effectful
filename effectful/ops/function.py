import collections
import collections.abc
import functools
import inspect
import typing
from typing import Annotated, Callable, Generic, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter
from effectful.internals.sugar import Bound, NoDefaultRule, _BaseNeutral, gensym
from effectful.ops.core import (
    Box,
    BoxExpr,
    Expr,
    Neutral,
    Operation,
    Term,
    apply,
    embed,
    evaluate,
    unembed,
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
    raise NoDefaultRule


@Operation  # type: ignore
def call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    match unembed(fn):
        case Term(defun_, (body_, *argvars_), kwvars_) if defun_ == defun:
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
                return embed(evaluate(body) if isinstance(body, Term) else body)  # type: ignore
        case fn_literal if not isinstance(fn_literal, Term):
            return fn_literal(*args, **kwargs)
        case _:
            raise NoDefaultRule


@embed.register(collections.abc.Callable)  # type: ignore
class _CallableNeutral(Generic[P, T], _BaseNeutral[collections.abc.Callable[P, T]]):
    def __call__(self, *args: BoxExpr, **kwargs: BoxExpr) -> Box[T]:
        return call(
            self,  # type: ignore
            *[embed(a) for a in args],  # type: ignore
            **{k: embed(v) for k, v in kwargs.items()},  #  type: ignore
        )


@unembed.register(collections.abc.Callable)  # type: ignore
def _unembed_callable(value: Callable[P, T]) -> Expr[Callable[P, T]]:
    assert not isinstance(value, Neutral)

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
