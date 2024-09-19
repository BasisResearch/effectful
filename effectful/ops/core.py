import dataclasses
import inspect
import tree
from typing import (
    Callable,
    Dict,
    Generic,
    Mapping,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import ParamSpec

from effectful.internals.runtime import (
    bind_interpretation,
    get_interpretation,
    interpreter,
)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@dataclasses.dataclass(eq=True, repr=True, unsafe_hash=True)
class Operation(Generic[Q, V]):
    signature: Callable[Q, V]

    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        return self.signature(*args, **kwargs)

    def __free_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Term[V]":
        return Term(self, args, tuple(kwargs.items()))

    def __type_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Type[V]:
        return object

    def __scope_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Interpretation[T, Type[T]]":
        return {}

    def __post_init__(self):
        try:
            from effectful.internals.sugar import infer_free_rule, infer_scope_rule, infer_type_rule
            self.__free_rule__ = infer_free_rule(self)
            self.__scope_rule__ = infer_scope_rule(self)
            self.__type_rule__ = infer_type_rule(self)
        except ImportError:
            pass

    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        return apply.__default_rule__(get_interpretation(), self, *args, **kwargs)  # type: ignore


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class Term(Generic[T]):
    op: Operation[..., T]
    args: Sequence[Union["Term[T]", T]]
    kwargs: Sequence[Tuple[str, Union["Term[T]", T]]]


Interpretation = Mapping[Operation[..., T], Callable[..., V]]


@Operation  # type: ignore
def apply(
    intp: Interpretation[S, T], op: Operation[P, S], *args: P.args, **kwargs: P.kwargs
) -> T:
    if op in intp:
        return intp[op](*args, **kwargs)
    elif apply in intp:
        return intp[apply](intp, op, *args, **kwargs)
    else:
        return op.__default_rule__(*args, **kwargs)  # type: ignore


@bind_interpretation
def evaluate(intp: Interpretation[S, T], term: Term[S]) -> Term[T] | T:
    _eval_leaf = lambda v: evaluate(v) if isinstance(v, Term) else v
    args = [tree.map_structure(_eval_leaf, arg) for arg in term.args]
    kwargs = {k: tree.map_structure(_eval_leaf, v) for k, v in term.kwargs}
    return apply.__default_rule__(intp, term.op, *args, **kwargs)  # type: ignore


def ctxof(term: Term[S]) -> Interpretation[Operation[..., T], Type[T]]:

    _ctx: Dict[Operation[..., T], Callable[..., Type[T]]] = {}

    def _update_ctx(_, op, *args, **kwargs):
        _ctx.setdefault(op, op.__type_rule__)
        for bound_var in op.__scope_rule__(*args, **kwargs):
            _ctx.pop(bound_var, None)
        return Term(op, args, tuple(kwargs.items()))

    with interpreter({apply: _update_ctx}):  # type: ignore
        evaluate(term)

    return _ctx


def typeof(term: Term[T]) -> Type[T]:
    with interpreter({apply: lambda _, op, *a, **k: op.__type_rule__(*a, **k)}):  # type: ignore
        return evaluate(term)  # type: ignore
