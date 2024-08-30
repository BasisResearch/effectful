import dataclasses
import functools
import typing
import weakref
from typing import Callable, Generic, MutableMapping, Tuple, TypeVar

from typing_extensions import Concatenate, ParamSpec

Q = ParamSpec("Q")
P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)
ArgSet = TypeVar("ArgSet", bound=Tuple[Tuple, MutableMapping])


if typing.TYPE_CHECKING:
    from ..ops.core import Interpretation


@dataclasses.dataclass
class Runtime(Generic[S, T]):
    interpretation: "Interpretation[S, T]"


@functools.lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return Runtime(interpretation={})


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
            return fn(
                default if not self._state else self._state.pop(), *args, **kwargs
            )

        return functools.wraps(fn)(_wrapper)


def _flatten_args(fn: Callable[Q, V]) -> Callable[[ArgSet], V]:
    return lambda ak: fn(*ak[0], **typing.cast(MutableMapping, ak[1]))


def _unflatten_args(fn: Callable[[ArgSet], V]) -> Callable[Q, V]:
    return lambda *a, **k: fn(typing.cast(ArgSet, (a, k)))


def _dup_arg(fn: Callable[[S, S], V]) -> Callable[[S], V]:
    return lambda x: fn(x, x)


def set_continuation(cont: Callable[P, T], fn: Callable[P, T]) -> Callable[P, T]:
    fn_ = _unflatten_args(_dup_arg(_ARG_STATE.sets(_flatten_args(fn))))
    cont_ = _ARG_STATE.gets(_flatten_args(cont), default=((), {}))
    return functools.wraps(fn)(functools.partial(_CONTINUATION_STATE.sets(fn_), cont_))


_ARG_STATE: _LinearState = _LinearState()
_RESULT_STATE: _LinearState = _LinearState()
_CONTINUATION_STATE: _LinearState = _LinearState()

_JUDGEMENTS = weakref.WeakKeyDictionary()  # type: ignore
_BINDINGS = weakref.WeakKeyDictionary()  # type: ignore
