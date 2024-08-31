import dataclasses
import functools
import typing
from typing import Callable, Generic, Mapping, Sequence, Tuple, Type, TypeVar, Union

from typing_extensions import ParamSpec, TypeAlias

from effectful.internals.runtime import get_interpretation, get_runtime, weak_memoize

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class Operation(Generic[Q, V]):
    default: Callable[Q, V]

    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        from effectful.internals.reification import reflect
        from effectful.ops.handler import handler

        term, env = reflect(self, *args, **kwargs)
        with handler({k: functools.partial(lambda x: x, v) for k, v in env.items()}):
            with handler(get_runtime()._BINDINGS):
                return evaluate(term)  # type: ignore


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


@weak_memoize
def define(m: Type[T]) -> Operation[..., T]:
    """
    Scott encoding of a type as its constructor.
    """
    if not typing.TYPE_CHECKING:
        if typing.get_origin(m) not in (m, None):
            return define(typing.get_origin(m))

    return m(m) if m is Operation else define(Operation[..., m])(m)  # type: ignore


@Operation
def apply(op: Operation[P, T], *args, **kwargs) -> T:
    return op.default(*args, **kwargs)


def evaluate(term: T | Term[T]) -> T:
    if not isinstance(term, Term):
        from effectful.internals.reification import reify
        from effectful.ops.handler import handler

        tm_env = reify(term)
        with handler(
            {k: functools.partial(evaluate, v) for k, v in tm_env.env.items()}
        ):
            return (
                evaluate(tm_env.value)
                if isinstance(tm_env.value, Term)
                else tm_env.value
            )

    op = term.op
    args = [evaluate(a) if isinstance(a, Term) else a for a in term.args]
    kwargs = {k: (evaluate(v) if isinstance(v, Term) else v) for k, v in term.kwargs}

    intp = get_interpretation()
    if op in intp:
        return intp[op](*args, **kwargs)
    elif apply in intp:
        return intp[apply](op, *args, **kwargs)
    else:
        return op.default(*args, **kwargs)


def gensym(t: Type[T]) -> Operation[[], T]:
    op: Operation[[], T] = Operation(lambda: Term(op, (), ()))  # type: ignore
    get_runtime()._JUDGEMENTS[op] = lambda: TypeInContext(context={op: t}, type=t)
    return op
