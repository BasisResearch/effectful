import collections.abc
import typing
from typing import Callable, Mapping, Optional, Protocol, Type, TypeVar, Union

from typing_extensions import ParamSpec

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)


@typing.runtime_checkable
class Operation(Protocol[P, T_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        ...

    def default(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        ...


Interpretation = Mapping[Operation[P, T_co], Callable[P, T_co]]


Symbolic = Union["Term[T]", T, "Variable[T]"]

@typing.runtime_checkable
class Term(Protocol[T]):
    op: Operation[P, T]
    args: P.args
    kwargs: P.kwargs


Symbol = str  # TODO replace with extensional protocol type
Context = typing.MutableMapping[Symbol, T]
TypeContext = Context[Type[T]]
TermContext = Context[Term[T]]


@typing.runtime_checkable
class Variable(Protocol[T]):
    name: Symbol
    type: Type[T]


def define(m: Union[Type[T], Callable[Q, T]]) -> Operation[Q, T]:
    """
    Scott encoding of a type as its constructor.
    """
    from ..internals.bootstrap import base_define

    return base_define(m)


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


def apply(
    intp: Interpretation[P, S], op: Operation[P, S], *args: P.args, **kwargs: P.kwargs
) -> T:
    try:
        interpret = intp[op]
    except KeyError:
        interpret = op.default
    return interpret(*args, **kwargs)


@define(Operation)
def evaluate(term: Term[T]) -> T:
    return term.op(
        *(evaluate(a) if isinstance(a, Term) else a for a in term.args),
        **{
            k: (evaluate(v) if isinstance(v, Term) else v)
            for k, v in term.kwargs.items()
        },
    )
