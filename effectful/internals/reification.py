import dataclasses
import functools
from typing import Generic, MutableMapping, Sequence, Tuple, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter
from effectful.ops.core import Context, Operation, Term, apply, define, gensym

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@dataclasses.dataclass
class _Neutral(Generic[T]):
    value: Term[T] | T
    env: Context[T, Term[T]]


@interpreter({apply: lambda op, *args, **kwargs: Term(op, args, tuple(kwargs.items()))})
def reflect(
    op: Operation[P, T], *args: P.args, **kwargs: P.kwargs
) -> Tuple[Term[T], Context[T, Term[T]]]:
    args_: Sequence[_Neutral[T]] = [reify(a) for a in args]
    kwargs_: Sequence[Tuple[str, _Neutral[T]]] = [
        (k, reify(v)) for k, v in kwargs.items()
    ]
    val: Term[T] = Term(
        op, [a.value for a in args_], [(k, v.value) for k, v in kwargs_]
    )
    env = {k: v for a in (*args_, *(v for _, v in kwargs_)) for k, v in a.env.items()}
    return val, env


@functools.singledispatch
def reify(x) -> _Neutral:
    return _Neutral(x, {})


@reify.register
def _reify_term(x: Term):
    syn = gensym(object)
    return _Neutral(syn(), {syn: x})


@reify.register
def _reify_tuple(xs: tuple):
    xs_reified = [reify(x) for x in xs]
    val = define(tuple)(*[x.value for x in xs_reified])
    env = {syn: sem for x in xs_reified for syn, sem in x.env.items()}
    return _Neutral(val, env)


@reify.register
def _reify_dict(xs: dict):
    xs_reified = [(reify(k), reify(v)) for k, v in xs.items()]
    val = define(dict)(*((k.value, v.value) for k, v in xs_reified))
    env: MutableMapping = {}
    for k, v in xs_reified:
        env.update(k.env)
        env.update(v.env)
    return _Neutral(val, env)