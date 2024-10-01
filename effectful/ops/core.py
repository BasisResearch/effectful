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


def trace(func):
    def traced(*args, **kwargs):
        ret = func(*args, **kwargs)
        print(func.__name__, args, kwargs, ret)
        return ret

    return traced


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

    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Box[V]":
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

    def __str__(self: "Term[T]") -> str:
        params_str = ""
        if len(self.args) > 0:
            params_str += ", ".join(str(x) for x in self.args)
        if len(self.kwargs) > 0:
            params_str += ", " + ", ".join(f"{k}={str(v)}" for (k, v) in self.kwargs)
        return f"{str(self.op)}({params_str})"

    def syntactic_eq(self, other: "Term[T]") -> bool:
        """Syntactic equality, ignoring the interpretation of the terms."""
        return (self.op, self.args, self.kwargs) == (other.op, other.args, other.kwargs)


Expr = Union[T, Term[T]]


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
def evaluate(intp: Interpretation[S, T], expr: Expr[S]) -> Expr[T]:
    match expr:
        case Term(op, args, kwargs):
            args = [evaluate(a) for a in args]  # type: ignore
            kwargs = {k: evaluate(v) for k, v in kwargs}  # type: ignore
            return apply.__default_rule__(intp, op, *args, **kwargs)  # type: ignore
        case _:
            return expr


def ctxof(term: Union[Expr[S], Term[S]]) -> Interpretation[T, Type[T]]:

    _ctx: Dict[Operation[..., T], Callable[..., Type[T]]] = {}

    def _update_ctx(_, op, *args, **kwargs):
        _ctx.setdefault(op, op.__type_rule__)
        for bound_var in op.__scope_rule__(*args, **kwargs):
            _ctx.pop(bound_var, None)

    with interpreter({apply: _update_ctx}):  # type: ignore
        evaluate(term if isinstance(term, Term) else unembed(term))  # type: ignore

    return _ctx


@trace
def typeof(term: Union[Expr[T], Term[T]]) -> Type[T]:
    with interpreter({apply: lambda _, op, *a, **k: op.__type_rule__(*a, **k)}):  # type: ignore
        return evaluate(term if isinstance(term, Term) else unembed(term))  # type: ignore


@trace
def unembed(value: Expr[T]) -> Term[T]:
    from effectful.internals.sugar import _unembed_registry

    if isinstance(value, Term):
        return value  # type: ignore
    else:
        impl: Callable[[T], Expr[T]]
        impl = _unembed_registry.dispatch(type(value))  # type: ignore
        return impl(value)


@trace
def embed(expr: Expr[T]) -> Expr[T]:
    from effectful.internals.sugar import _embed_registry

    if isinstance(expr, Term):
        impl: Callable[[Term[T]], Expr[T]]
        impl = _embed_registry.dispatch(typeof(expr))  # type: ignore
        return impl(expr.op, expr.args, expr.kwargs)
    else:
        return expr


def hoas(value: Expr[T]):
    return embed(unembed(value))
