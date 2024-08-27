import collections.abc
import dataclasses
import weakref
from typing import Callable, Generic, Iterable, Mapping, Optional, Type, TypeVar, Union

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

    # judgement: Callable[..., tuple[Type[V], Mapping[str, Type[V]]]]

    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        # from effectful.internals.runtime import get_interpretation
        # return evaluate(Term(self, *(reify(a) for a in args), **{k: reify(v) for k, v in kwargs.items()})))
        # return evaluate(reify(Term(self, args, kwargs)))
        # return apply(self, *args, **kwargs)  # type: ignore
        return evaluate(Term(self, args, kwargs))


Interpretation = Mapping[Operation[..., T], Callable[..., V]]


# @define
@dataclasses.dataclass(frozen=True, eq=True, order=True, repr=True, unsafe_hash=True)
class Term(Generic[T]):
    op: Operation[..., T]
    args: Iterable[Union["Term[T]", T]]
    kwargs: Mapping[str, Union["Term[T]", T]]


Context = Mapping[Operation[..., S], T]
TypeInContext = tuple[Context[T, Type[T]], Type[T]]


JUDGEMENTS: Interpretation[T, TypeInContext[T]] = weakref.WeakKeyDictionary()

BINDINGS: Interpretation[T, T] = weakref.WeakKeyDictionary()


def gensym(t: Type[T] = object) -> Operation[[], T]:
    op = Operation(lambda: Term(op, (), {}))
    JUDGEMENTS[op] = lambda: ({op: t}, t)
    return op


@Operation
def apply(op: Operation[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return op.default(*args, **kwargs)


# @Operation
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
