import collections.abc
from typing import (
    Callable,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

from typing_extensions import ParamSpec

from effectful.internals.bootstrap import InjectedDataclass, Operation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


Interpretation = Mapping[Operation[..., T], Callable[..., V]]


class Symbol(InjectedDataclass):
    name: str


Expr: TypeAlias = Union["Variable[T]", "Constant[T]", "Term[T]"]


class Variable(Generic[T], InjectedDataclass):
    symbol: Symbol
    type: Type[T]


class Constant(Generic[T], InjectedDataclass):
    value: T


class Term(Generic[T], InjectedDataclass):
    op: Operation[..., T]
    args: Iterable[Expr]
    kwargs: Mapping[str, Expr]


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
