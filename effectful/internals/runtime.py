import dataclasses
import functools
import typing
import weakref
from typing import Callable, Generic, Mapping, Protocol, TypeVar
from typing_extensions import Concatenate, ParamSpec

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)


@typing.runtime_checkable
class Operation(Protocol[P, T_co]):
    def default(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        ...


Interpretation = Mapping[Operation[..., S], Callable[..., T]]


@dataclasses.dataclass
class Runtime(Generic[S, T]):
    interpretation: Interpretation[S, T]


@functools.lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return Runtime(interpretation={})


def get_interpretation():
    return get_runtime().interpretation


def bind_interpretation(
    fn: Callable[Concatenate[Interpretation[S, T], Operation[P, S], P], T]
) -> Callable[Concatenate[Operation[P, S], P], T]:
    """
    Bind the interpretation to the function.
    """
    from ..internals.runtime import get_interpretation

    @functools.wraps(fn)
    def _wrapper(op: Operation[P, S], *args: P.args, **kwargs: P.kwargs) -> T:
        intp = get_interpretation()  # type: ignore
        return fn(typing.cast(Interpretation[S, T], intp), op, *args, **kwargs)

    return _wrapper


@bind_interpretation
def swap_interpretation(old_intp: Interpretation[S, T], intp: Interpretation[S, V]) -> Interpretation[S, T]:
    get_runtime().interpretation = intp
    return old_intp


@bind_interpretation
def runtime_apply(intp: Interpretation[S, T], op: Operation[P, S], *args: P.args, **kwargs: P.kwargs) -> T:
    return intp[op](*args, **kwargs)


def weak_memoize(f: Callable[[S], T]) -> Callable[[S], T]:
    """
    Memoize a one-argument function using a dictionary
    whose keys are weak references to the arguments.
    """

    cache: weakref.WeakKeyDictionary[S, T] = weakref.WeakKeyDictionary()

    @functools.wraps(f)
    def wrapper(x: S) -> T:
        try:
            return cache[x]
        except KeyError:
            result = f(x)
            cache[x] = result
            return result

    return wrapper
