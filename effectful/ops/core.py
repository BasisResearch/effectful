import dataclasses
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import tree
from typing_extensions import ParamSpec

from effectful.internals.runtime import (
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
    signature: Callable[Q, V]

    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        from effectful.internals.sugar import infer_default_rule

        return infer_default_rule(self)(*args, **kwargs)

    def __free_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        from effectful.internals.sugar import infer_free_rule

        return infer_free_rule(self)(*args, **kwargs)

    def __type_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Type[V]:
        from effectful.internals.sugar import infer_type_rule

        return infer_type_rule(self)(*args, **kwargs)

    def __scope_rule__(
        self, *args: Q.args, **kwargs: Q.kwargs
    ) -> "Interpretation[T, Type[T]]":
        from effectful.internals.sugar import infer_scope_rule

        return infer_scope_rule(self)(*args, **kwargs)  # type: ignore

    def __str__(self):
        return self.signature.__name__

    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        return apply.__default_rule__(get_interpretation(), self, *args, **kwargs)  # type: ignore


@typing.runtime_checkable
class Term(Protocol[T]):
    op: Operation[..., T]
    args: Sequence["Expr[Any]"]
    kwargs: Sequence[Tuple[str, "Expr[Any]"]]

    __match_args__: tuple[str, str, str] = ("op", "args", "kwargs")


Expr = Union[T, Term[T]]


def syntactic_eq(x: Expr[T], other: Expr[T]) -> bool:
    """Syntactic equality, ignoring the interpretation of the terms."""
    match x, other:
        case Term(op, args, kwargs), Term(op2, args2, kwargs2):
            kwargs, kwargs2 = dict(kwargs), dict(kwargs2)
            try:
                tree.assert_same_structure(
                    (op, args, kwargs), (op2, args2, kwargs2), check_types=True
                )
            except (TypeError, ValueError):
                return False
            return all(
                tree.flatten(
                    tree.map_structure(
                        syntactic_eq, (op, args, kwargs), (op2, args2, kwargs2)
                    )
                )
            )
        case Term(_, _, _), _:
            return False
        case _, Term(_, _, _):
            return False
        case _, _:
            return x == other
    return False


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
def evaluate(intp: Interpretation[S, T], expr: Expr[T]) -> Expr[T]:
    match unembed(expr):
        case Term(op, args, kwargs):
            (args, kwargs) = tree.map_structure(evaluate, (args, dict(kwargs)))
            return apply.__default_rule__(intp, op, *args, **kwargs)  # type: ignore
        case literal:
            return literal


def ctxof(term: Expr[S]) -> Interpretation[T, Type[T]]:
    _ctx: Dict[Operation[..., T], Callable[..., Type[T]]] = {}

    def _update_ctx(_, op, *args, **kwargs):
        _ctx.setdefault(op, op.__type_rule__)
        for bound_var in op.__scope_rule__(*args, **kwargs):
            _ctx.pop(bound_var, None)

    with interpreter({apply: _update_ctx}):  # type: ignore
        evaluate(term)  # type: ignore

    return _ctx


def typeof(term: Expr[T]) -> Type[T]:
    with interpreter({apply: lambda _, op, *a, **k: op.__type_rule__(*a, **k)}):  # type: ignore
        return evaluate(term)  # type: ignore


def unembed(value: Expr[T]) -> Expr[T]:
    from effectful.internals.sugar import _unembed_registry

    if isinstance(value, Term):
        return value  # type: ignore
    else:
        impl: Callable[[T], Expr[T]]
        impl = _unembed_registry.dispatch(type(value))  # type: ignore
        return impl(value)
