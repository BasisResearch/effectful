import dataclasses
import functools
import typing
import weakref
from typing import (
    Callable,
    Generic,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import ParamSpec, TypeAlias

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@dataclasses.dataclass(eq=True, repr=True, unsafe_hash=True)
class Operation(Generic[Q, V]):
    default: Callable[Q, V]

    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        return evaluate(Term(self, args, kwargs))  # type: ignore


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class Term(Generic[T]):
    op: Operation[..., T]
    args: Sequence[Union["Term[T]", T]]
    kwargs: Sequence[Tuple[str, Union["Term[T]", T]]]


Context: TypeAlias = Mapping[Operation[..., S], T]
Interpretation: TypeAlias = Context[S, Callable[..., T]]


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class TypeInContext(Generic[T]):
    context: Context[T, Type[T]]
    type: Type[T]


@functools.cache
def define(m: Type[T]) -> Operation[..., T]:
    """
    Scott encoding of a type as its constructor.
    """
    if not typing.TYPE_CHECKING:
        if typing.get_origin(m) not in (m, None):
            return define(typing.get_origin(m))

    return m(m) if m is Operation else define(Operation[..., m])(m)  # type: ignore


@Operation
def apply(op: Operation[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return op.default(*args, **kwargs)  # type: ignore


def evaluate(term: Term[T]) -> T:
    from effectful.internals.runtime import get_interpretation

    intp = get_interpretation()
    op = term.op
    args = [evaluate(a) if isinstance(a, Term) else a for a in term.args]
    kwargs = {k: (evaluate(v) if isinstance(v, Term) else v) for k, v in term.kwargs}
    if op in intp:
        return intp[op](*args, **kwargs)
    elif apply in intp:
        return intp[apply](op, *args, **kwargs)
    else:
        return op.default(*args, **kwargs)


def gensym(t: Type[T]) -> Operation[[], T]:
    op: Operation[[], T] = define(Operation)(lambda: Term(op, (), ()))
    JUDGEMENTS[op] = lambda: TypeInContext(context={op: t}, type=t)
    return op


JUDGEMENTS: MutableMapping[Operation, Callable[..., TypeInContext]] = (
    weakref.WeakKeyDictionary()
)
BINDINGS: MutableMapping[Operation, Callable] = weakref.WeakKeyDictionary()
