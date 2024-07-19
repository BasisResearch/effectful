import collections.abc
from typing import (Any, Callable, Generic, Iterable, Mapping, NoReturn, Optional, Type, TypeVar, Union)

from typing_extensions import ParamSpec, dataclass_transform

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@dataclass_transform()
def define(m: Type[T]) -> "Operation[..., T]":
    """
    Scott encoding of a type as its constructor.
    """
    from effectful.internals.bootstrap import base_define

    return base_define(m)  # type: ignore


@define
class Operation(Generic[Q, V]):
    default: Callable[Q, V]

    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        return apply(self, *args, **kwargs)  # type: ignore


Interpretation = Mapping[Operation[..., T], Callable[..., V]]


@define
class Symbol:
    name: str


@define
class Variable(Generic[T]):
    symbol: Symbol
    type: Type[T]


@define
class Constant(Generic[T]):
    value: T


@define
class Term(Generic[T]):
    op: Operation[..., T]
    args: Iterable[Union["Term[T]", Constant[T], Variable[T]]]
    kwargs: Mapping[str, Union["Term[T]", Constant[T], Variable[T]]]


Context = Mapping[Symbol, T]
TypeContext = Context[Type[T]]
TermContext = Context[Term[T]]


def apply(op: Operation[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    from effectful.internals.runtime import get_interpretation

    return get_interpretation().get(op, op.default)(*args, **kwargs)


@Operation
def evaluate(term: Term[T]) -> T:
    return apply(
        term.op,
        *(evaluate(a) if isinstance(a, Term) else a for a in term.args),
        **{
            k: (evaluate(v) if isinstance(v, Term) else v)
            for k, v in term.kwargs.items()
        },
    )


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


class NoDefaultImplementationError(RuntimeError):
    """
    A :class:`RuntimeError` raised when an :py:class:`Operation` does not have
    a default implementation.
    """


def explicit_operation(*args: Any, **kwargs: Any) -> Operation[..., Any]:
    """
    Returns a new :class:`Operation` without a default implementation.

    If the returned operation is called without first being bound, it will raise
    a :class:`NoDefaultImplementaitonError`. Both the arguments passed to this
    function and to the resulting operation will be passed to the :py:class:`NoDefaultImplementaitonError`.

    :param args: Arguments which are passed to the :class:`NoDefaultImplementationError` constructor
    :param kwargs: Arguments which are passed to the :class:`NoDefaultImplementationError` constructor

    :return: A new, explicit :class:`Operation`
    """

    def callback(*args2: Any, **kwargs2: Any) -> NoReturn:
        raise NoDefaultImplementationError(*(args + args2), **{**kwargs, **kwargs2})

    return Operation(callback)
