import functools
import typing
from typing import Annotated, Callable, TypeVar

import tree
from typing_extensions import ParamSpec

from effectful.internals.sugar import Bound, NoDefaultRule
from effectful.ops.core import Expr, Operation, Term, as_term, evaluate
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
def funcall(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    match as_term(fn):
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
                return evaluate(body)  # type: ignore
        case fn_expr if not isinstance(fn_expr, (Term, Operation)):

            @Operation
            @functools.wraps(fn_expr)
            def _fn_expr_op(*args: P.args, **kwargs: P.kwargs) -> T:
                if not any(
                    tree.flatten(
                        tree.map_structure(
                            lambda x: isinstance(x, Term), (args, kwargs)
                        )
                    )
                ):
                    return fn_expr(*args, **kwargs)
                else:
                    raise NoDefaultRule

            return _fn_expr_op(*args, **kwargs)
        case _:
            raise NoDefaultRule