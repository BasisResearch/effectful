import dataclasses
import typing
from typing import Callable, Generic, Mapping, Sequence, Tuple, Type, TypeVar, Union

from typing_extensions import ParamSpec

from effectful.internals.runtime import (
    bind_interpretation,
    get_interpretation,
    weak_memoize,
)

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
        intp = get_interpretation()
        return apply.default(intp, apply, intp, self, *args, **kwargs)  # type: ignore


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class Term(Generic[T]):
    op: Operation[..., T]
    args: Sequence[Union["Term[T]", T]]
    kwargs: Sequence[Tuple[str, Union["Term[T]", T]]]


Context = Mapping[Operation[..., T], V]
Interpretation = Context[T, Callable[..., V]]


def gensym(t: Type[T]) -> Operation[[], T]:
    op: Operation[[], T] = Operation(lambda: Term(op, (), ()))  # type: ignore
    return op


@Operation  # type: ignore
def apply(
    intp: Interpretation[S, T], op: Operation[P, S], *args: P.args, **kwargs: P.kwargs
) -> T:
    return intp.get(op, op.default)(*args, **kwargs)  # type: ignore


@bind_interpretation
def evaluate(intp: Interpretation[S, T], term: Term[S]) -> Term[T] | T:
    args = [evaluate(a) if isinstance(a, Term) else a for a in term.args]  # type: ignore
    kwargs = {k: evaluate(v) if isinstance(v, Term) else v for k, v in term.kwargs}  # type: ignore
    return apply.default(intp, apply, intp, term.op, *args, **kwargs)  # type: ignore
