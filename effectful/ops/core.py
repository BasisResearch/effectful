import collections.abc
import dataclasses
import functools
import typing
from typing import Callable, Generic, Iterable, Mapping, Optional, Protocol, Type, TypeVar

from typing_extensions import Concatenate, ParamSpec, TypeGuard, dataclass_transform


P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)


@dataclass_transform()
@functools.cache
def define(m: Type[T]) -> "Operation[..., T]":
    """
    Scott encoding of a type as its constructor.
    """
    if issubclass(m, Operation):
        return dataclasses.dataclass()(m)
    else:
        return define(Operation)(m)


class Operation(Generic[P, T_co]):
    default: Callable[P, T_co]

    def __call__(self: "Operation[P, S]", *args: P.args, **kwargs: P.kwargs) -> S:
        return apply(self, *args, **kwargs)  # type: ignore


Operation = define(Operation)
Interpretation = Mapping[Operation[..., T], Callable[..., V]]


@define
class Symbol:
    name: str


@define
class Constant(Generic[T]):
    value: T


@define
class Variable(Generic[T]):
    name: Symbol
    type: Type[T]


@define
class Term(Generic[T]):
    op: Operation[..., T]
    args: Iterable["Term[T]" | Constant[T] | Variable[T]]
    kwargs: Mapping[str, "Term[T]" | Constant[T] | Variable[T]]


Context = collections.abc.MutableMapping[Symbol, T]
TypeContext = Context[Type[T]]
TermContext = Context[Term[T]]


@Operation
def register(
    op: Operation[P, T],
    intp: Optional[Interpretation[T, V]],
    interpret_op: Callable[Q, V],
) -> Callable[Q, V]:
    if intp is None:
        setattr(op, "default", interpret_op)
        return interpret_op
    elif isinstance(intp, collections.abc.MutableMapping):
        intp.__setitem__(op, interpret_op)
        return interpret_op
    raise NotImplementedError(f"Cannot register {op} in {intp}")


@Operation
def apply(op: Operation[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    from ..internals.runtime import runtime_apply
    return runtime_apply(op, *args, **kwargs)


@Operation
def evaluate(term: Term[T]) -> T:
    return apply(
        term.op,  # type: ignore
        *(evaluate(a) if isinstance(a, Term) else a for a in term.args),
        **{
            k: (evaluate(v) if isinstance(v, Term) else v)
            for k, v in term.kwargs.items()
        },
    )
