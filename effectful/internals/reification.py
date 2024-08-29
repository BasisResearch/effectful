import dataclasses
import functools
import typing
import weakref
from typing import (
    Callable,
    Generic,
    Mapping,
    MutableMapping,
    NamedTuple,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import ParamSpec

from effectful.ops.core import Context, Operation, Term, apply, define, gensym
from effectful.ops.interpreter import interpreter

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@dataclasses.dataclass
class Neutral(Generic[T]):
    value: Term[T] | T
    env: Context[T, Term[T]]


@interpreter({apply: lambda op, *args, **kwargs: Term(op, args, tuple(kwargs.items()))})
def reflect(op: Operation[P, T], *args: P.args, **kwargs: P.kwargs) -> Neutral[T]:
    return reify(Neutral(Term(op, args, tuple(kwargs.items())), {}))


@functools.singledispatch
def reify(x) -> Neutral:
    return Neutral(x, {})


@reify.register
def _reify_neutral(x: Neutral):
    reified_args = [reify(a) for a in x.value.args]
    reified_kwargs = [(k, reify(v)) for k, v in x.value.kwargs]
    val = Term(
        x.value.op,
        [a.value for a in reified_args],
        [(k, v.value) for k, v in reified_kwargs]
    )
    env = {**x.env}
    for v in (*reified_args, *(v for _, v in reified_kwargs)):
        env.update(v.env)
    return Neutral(val, env)


@reify.register
def _reify_term(x: Term):
    syn = gensym(object)
    return Neutral(syn(), {syn: x})


@reify.register
def _reify_tuple(xs: tuple):
    xs_reified = [reify(x) for x in xs]
    val = define(tuple)(*[x.value for x in xs_reified])
    env = {syn: sem for x in xs_reified for syn, sem in x.env.items()}
    return Neutral(val, env)


@reify.register
def _reify_dict(xs: dict):
    xs_reified = [(reify(k), reify(v)) for k, v in xs.items()]
    val = define(dict)(*((k.value, v.value) for k, v in xs_reified))
    env = {}
    for k, v in xs_reified:
        env.update(k.env)
        env.update(v.env)
    return Neutral(val, env)
