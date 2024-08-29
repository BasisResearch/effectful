import collections.abc
import dataclasses
import weakref
from typing import Callable, Generic, Iterable, Mapping, Optional, Tuple, Type, TypeVar, Union

from typing_extensions import ParamSpec, TypeAlias

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


def define(m: Type[T]) -> "Operation[..., T]":
    """
    Scott encoding of a type as its constructor.
    """
    from effectful.internals.bootstrap import base_define

    return base_define(m)  # type: ignore


@dataclasses.dataclass(eq=True, repr=True, unsafe_hash=True)
class Operation(Generic[Q, V]):
    default: Callable[Q, V]

    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        return evaluate(Term(self, args, kwargs))


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class Term(Generic[T]):
    op: Operation[..., T]
    args: Iterable[Union["Term[T]", T]]
    kwargs: Mapping[str, Union["Term[T]", T]]


Context: TypeAlias = Mapping[Operation[..., S], T]
Interpretation: TypeAlias = Context[S, Callable[..., T]]


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class TypeInContext(Generic[T]):
    context: Context[T, Type[T]]
    type: Type[T]


@Operation
def apply(op: Operation[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return op.default(*args, **kwargs)


def evaluate(term: Term[T]) -> T:
    from effectful.internals.runtime import get_interpretation

    intp = get_interpretation()
    op = term.op
    args = [evaluate(a) if isinstance(a, Term) else a for a in term.args]
    kwargs = {k: (evaluate(v) if isinstance(v, Term) else v) for k, v in term.kwargs.items()}
    if op in intp:
        return intp[op](*args, **kwargs)
    elif apply in intp:
        return intp[apply](op, *args, **kwargs)
    else:
        return op.default(*args, **kwargs)


def gensym(t: Type[T] = object) -> Operation[[], T]:
    op = define(Operation)(lambda: Term(op, (), {}))
    JUDGEMENTS[op] = lambda: TypeInContext(context={op: t}, type=t)
    return op


JUDGEMENTS: Interpretation[T, TypeInContext[T]] = weakref.WeakKeyDictionary()
BINDINGS: Interpretation[T, T] = weakref.WeakKeyDictionary()
