import dataclasses
import typing
from typing import Callable, Generic, Iterable, Mapping, Type, TypeVar, Union

from typing_extensions import ParamSpec

from effectful.internals.runtime import weak_memoize

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@weak_memoize
def define(m: Type[T]) -> "Operation[..., T]":
    """
    Scott encoding of a type as its constructor.
    """
    if not typing.TYPE_CHECKING:
        if typing.get_origin(m) not in (m, None):
            return define(typing.get_origin(m))

    return m(m) if m is Operation else define(Operation[..., m])(m)  # type: ignore


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class Operation(Generic[Q, V]):
    default: Callable[Q, V]

    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        return apply(self, *args, **kwargs)  # type: ignore


Interpretation = Mapping[Operation[..., T], Callable[..., V]]


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class Symbol:
    name: str


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class Variable(Generic[T]):
    symbol: Symbol
    type: Type[T]


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class Constant(Generic[T]):
    value: T


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
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
