import collections.abc
from dataclasses import dataclass
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
from weakref import WeakKeyDictionary

from typing_extensions import ParamSpec, dataclass_transform

from effectful.internals.runtime import get_interpretation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


class InjectedMeta(type):
    def __call__(self: type[T], *a, **k) -> T:
        return define(self)(*a, **k)


class InjectedType(metaclass=InjectedMeta):
    pass


@dataclass(frozen=True, repr=False)
class Operation(Generic[P, V], InjectedType):
    default: Callable[P, V]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> V:
        return get_interpretation().get(self, self.default)(*args, **kwargs)

    def __repr__(self):
        if hasattr(self.default, "__name__"):
            return f"<Operation {self.default.__name__} at {hex(id(self))}>"
        else:
            return f"<Operation at {hex(id(self))}>"


def _blank_con(ty: type[T]) -> Callable[P, T]:
    sup = type(ty)
    if issubclass(sup, InjectedMeta):
        sup = type

    def make(*a: P.args, **k: P.kwargs) -> T:
        return sup.__call__(ty, *a, **k)

    make.__name__ = f"define_{ty.__name__}"
    return make


def _blank_op(default: Callable[P, T]) -> Operation[P, T]:
    return _blank_con(Operation)(default)


_CONSTRUCTOR_MAP: WeakKeyDictionary[type, Operation] = WeakKeyDictionary()
_CONSTRUCTOR_MAP[Operation] = _blank_op(_blank_op)


def define(ty: type[T]) -> Operation[..., T]:
    if ty not in _CONSTRUCTOR_MAP:
        _CONSTRUCTOR_MAP[ty] = Operation(_blank_con(ty))

    return _CONSTRUCTOR_MAP[ty]


@dataclass_transform()
class InjectedDataclass(InjectedType):
    def __init_subclass__(sub, **kwargs):
        dataclass(sub, unsafe_hash=True)


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


Interpretation = Mapping[Operation[..., T], Callable[..., V]]


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
