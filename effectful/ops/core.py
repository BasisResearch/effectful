import dataclasses
import typing
from typing import Callable, Generic, Mapping, Sequence, Tuple, Type, TypeVar, Union

from typing_extensions import ParamSpec

from effectful.internals.runtime import (
    _CTXOF_RULES,
    _TYPEOF_RULES,
    bind_interpretation,
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

    def __str__(self):
        return self.default.__name__

    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        return apply.default(get_interpretation(), self, *args, **kwargs)  # type: ignore


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class Term(Generic[T]):
    op: Operation[..., T]
    args: Sequence[Union["Term[T]", T]]
    kwargs: Sequence[Tuple[str, Union["Term[T]", T]]]

    def __str__(self):
        params_str = ""
        if len(self.args) > 0:
            params_str += ", ".join(str(x) for x in self.args)
        if len(self.kwargs) > 0:
            params_str += ", " + ", ".join(f"{k}={str(v)}" for (k, v) in self.kwargs)
        return f"{str(self.op)}({params_str})"


Context = Mapping[Operation[..., T], V]
Interpretation = Context[T, Callable[..., V]]


def gensym(t: Type[T]) -> Operation[[], T]:
    op: Operation[[], T] = Operation(lambda: Term(op, (), ()))  # type: ignore
    return op


@Operation  # type: ignore
def apply(
    intp: Interpretation[S, T], op: Operation[P, S], *args: P.args, **kwargs: P.kwargs
) -> T:
    if op in intp:
        return intp[op](*args, **kwargs)
    elif apply in intp:
        return intp[apply](intp, op, *args, **kwargs)
    else:
        return op.default(*args, **kwargs)  # type: ignore


@bind_interpretation
def evaluate(intp: Interpretation[S, T], term: Term[S]) -> Term[T] | T:
    args = [evaluate(a) if isinstance(a, Term) else a for a in term.args]  # type: ignore
    kwargs = {k: evaluate(v) if isinstance(v, Term) else v for k, v in term.kwargs}  # type: ignore
    return apply.default(intp, term.op, *args, **kwargs)  # type: ignore


def ctxof(term: Term[T]) -> set[Operation[..., T]]:

    _ctx_set = set()

    def _update_ctx(_, op, *args, **kwargs):
        _ctx_set.add(op)
        for bound_var in _CTXOF_RULES[op](*args, **kwargs):
            _ctx_set.remove(bound_var)
        return Term(op, args, tuple(kwargs.items()))

    with interpreter({apply: _update_ctx}):
        evaluate(term)  # type: ignore

    return _ctx_set


def typeof(term: Term[T]) -> Type[T]:
    with interpreter(_TYPEOF_RULES):
        return evaluate(term)  # type: ignore
