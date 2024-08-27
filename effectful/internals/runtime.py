import dataclasses
import functools
import typing
import weakref
from typing import Callable, Generic, Mapping, Optional, Tuple, TypeVar

from typing_extensions import Concatenate, ParamSpec

Q = ParamSpec("Q")
P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)

if typing.TYPE_CHECKING:
    from ..ops.core import Interpretation


@dataclasses.dataclass
class Runtime(Generic[S, T]):
    interpretation: "Interpretation[S, T]"
    call_state: "LinearState"
    cont_state: "LinearState"


@functools.lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    call_state = LinearState(None)
    cont_state = LinearState(lambda r, *_, **__: r)
    return Runtime(interpretation={}, call_state=call_state, cont_state=cont_state)


def get_interpretation():
    return get_runtime().interpretation


def swap_interpretation(intp: "Interpretation[S, V]") -> "Interpretation[S, T]":
    old_intp = get_runtime().interpretation
    get_runtime().interpretation = intp
    return old_intp


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


class LinearState(Generic[S]):
    _state: list[S]

    def __init__(self, default: S):
        self._default = default
        self._state = []

    def sets(self, fn: Callable[P, T]) -> Callable[Concatenate[S, P], T]:
        @functools.wraps(fn)
        def _wrapper(state: S, *args: P.args, **kwargs: P.kwargs) -> T:
            self._state.append(state)
            try:
                return fn(*args, **kwargs)
            finally:
                if self._state:
                    self._state.pop()

        return _wrapper

    def __call__(self, fn: Callable[Concatenate[S, P], T]) -> Callable[P, T]:
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return fn(self._default if not self._state else self._state.pop(), *args, **kwargs)

        return functools.wraps(fn)(_wrapper)


def bind_result(fn: Callable[Concatenate[Optional[T], P], T]) -> Callable[P, T]:
    return get_runtime().call_state(fn)


def bind_cont(fn: Callable[Concatenate[V, P], T]) -> Callable[Concatenate[V, P], T]:
    return get_runtime().cont_state(fn)


def compose_continuation(cont: Callable[P, T], fn: Callable[P, T]) -> Callable[P, T]:
    r = get_runtime()
    return functools.wraps(fn)(functools.partial(r.cont_state.sets(fn), r.call_state.sets(cont)))