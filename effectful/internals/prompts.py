import functools
import typing
from typing import Callable, Generic, MutableMapping, Optional, Tuple, TypeVar

from typing_extensions import Concatenate, ParamSpec, TypeAlias

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


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


_ARG_STATE: _LinearState = _LinearState()
_RESULT_STATE: _LinearState = _LinearState()
_CONTINUATION_STATE: _LinearState = _LinearState()

ArgSet = TypeVar("ArgSet", bound=Tuple[Tuple, MutableMapping])


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
