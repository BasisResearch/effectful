import dataclasses
import typing
from typing import Callable, Generic, Mapping, Sequence, Tuple, Type, TypeVar, Union

from typing_extensions import ParamSpec

from effectful.internals.runtime import (
    _CTXOF_RULES,
    _TYPEOF_RULES,
    get_interpretation,
    interpreter,
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
        return apply(self, *args, **kwargs)  # type: ignore


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


def apply(op: Operation[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return get_interpretation().get(op, op.default)(*args, **kwargs)


@Operation
def evaluate(term: Term[T]) -> T:
    return apply(
        term.op,
        *(evaluate(a) if isinstance(a, Term) else a for a in term.args),
        **{k: (evaluate(v) if isinstance(v, Term) else v) for k, v in term.kwargs},
    )


def ctxof(term: Term[T]) -> set[Operation[..., T]]:

    _scope = set()

    def make_scope_rule(op, ctxof_rule):

        def scope_rule(*args, **kwargs):
            bound = ctxof_rule(*args, **kwargs)
            _scope.add(op)
            for v in bound:
                _scope.remove(v)
            return Term(op, args, tuple(kwargs.items()))

        return scope_rule

    with interpreter(
        {op: make_scope_rule(op, rule) for op, rule in _CTXOF_RULES.items()}
    ):
        evaluate(term)

    return _scope


def typeof(term: Term[T]) -> Type[T]:
    with interpreter(_TYPEOF_RULES):
        return evaluate(term)  # type: ignore
