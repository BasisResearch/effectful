import functools
import typing
from typing import Annotated, Callable, TypeVar

from typing_extensions import ParamSpec
import tree

import torch

from effectful.internals.sugar import (
    Bound,
    NoDefaultRule,
    torch_getitem,
    _register_torch_op,
    EagerTensorTerm,
)
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
        case _:
            raise NoDefaultRule


def grad(func, has_aux=False, **kwargs):
    """Compute the gradient of a function with respect to its arguments. This is
    a wrapper around `torch.func.grad` that allows for the function to be called
    with indexed arguments.

    """

    def add_indexes(func, unindexed, indexed):
        def add_single_index(u, i):
            """Given an unindexed tensor and an index expression, expand the
            tensor to have leading dimensions corresponding to the named indexes
            and create an index expression.

            """
            if isinstance(i, EagerTensorTerm):
                indexes = i.indices()
                return torch_getitem(
                    u[(None,) * len(indexes) + (slice(None),) * len(i.shape)],
                    tuple(indexes),
                )
            return u

        reindexed = tree.map_structure(add_single_index, unindexed, indexed)
        reindexed_args, reindexed_kwargs = reindexed
        ret = func(*reindexed_args, **reindexed_kwargs)
        return ret.to_tensor().reshape(ret.shape)

    return lambda *indexed_args, **indexed_kwargs: _register_torch_op(
        torch.func.grad(
            lambda *unindexed_args, **unindexed_kwargs: add_indexes(
                func, (unindexed_args, unindexed_kwargs), (indexed_args, indexed_kwargs)
            ),
            has_aux=has_aux,
            **kwargs,
        )
    )(*indexed_args, **indexed_kwargs)
