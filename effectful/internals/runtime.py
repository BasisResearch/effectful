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
    result_state: "_LinearState"
    continuation_state: "_LinearState"


@functools.lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return Runtime(interpretation={}, result_state=_LinearState(), continuation_state=_LinearState())


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


class _LinearState(Generic[S]):
    _state: list[S]

    def __init__(self):
        self._state = []

    def sets(self, fn: Callable[P, T]) -> Callable[Concatenate[S, P], T]:
        def _wrapper(state: S, *args: P.args, **kwargs: P.kwargs) -> T:
            self._state.append(state)
            try:
                return fn(*args, **kwargs)
            finally:
                if self._state:
                    self._state.pop()

        return functools.wraps(fn)(_wrapper)

    def gets(self, fn: Callable[Concatenate[S, P], T], *, default: S) -> Callable[P, T]:
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return fn(default if not self._state else self._state.pop(), *args, **kwargs)

        return functools.wraps(fn)(_wrapper)


def bind_result(
    fn: Optional[Callable[Concatenate[S, P], T]] = None,
    *,
    default: Optional[S] = None,
):
    if fn is None:
        return functools.partial(bind_result, default=default)
    else:
        return get_runtime().result_state.gets(fn, default=default)


def bind_continuation(
    fn: Optional[Callable[Concatenate[Callable[Concatenate[S, P], T], P], T]] = None,
    *,
    default: Callable[Concatenate[S, P], T] = lambda r, *_, **__: r,
):
    if fn is None:
        return functools.partial(bind_continuation, default=default)
    else:
        cc = get_runtime().continuation_state.gets(lambda c, *a, **k: c(*a, **k), default=default)
        return functools.wraps(fn)(functools.partial(fn, cc))


def compose_continuation(cont: Callable[P, T], fn: Callable[P, T]) -> Callable[P, T]:
    r = get_runtime()
    return functools.wraps(fn)(functools.partial(r.continuation_state.sets(fn), r.result_state.sets(cont)))