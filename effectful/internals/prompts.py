import functools
from typing import Callable, Generic, Optional, TypeVar

from typing_extensions import Concatenate, ParamSpec

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


_RESULT_STATE: _LinearState = _LinearState()
_CONTINUATION_STATE: _LinearState = _LinearState()


def bind_result(
    fn: Optional[Callable[Concatenate[S, P], T]] = None,
    *,
    default: Optional[S] = None,
):
    if fn is None:
        return functools.partial(bind_result, default=default)
    else:
        return _RESULT_STATE.gets(fn, default=default)


def bind_result_to_method(fn):
    return bind_result(lambda r, self, *a, **k: fn(self, r, *a, **k))


def bind_continuation(
    fn: Optional[Callable[Concatenate[Callable[Concatenate[S, P], T], P], T]] = None,
    *,
    default: Callable[Concatenate[S, P], T] = lambda r, *_, **__: r,
):
    if fn is None:
        return functools.partial(bind_continuation, default=default)
    else:
        cc = _CONTINUATION_STATE.gets(lambda c, *a, **k: c(*a, **k), default=default)
        return functools.wraps(fn)(functools.partial(fn, cc))


def compose_continuation(cont: Callable[P, T], fn: Callable[P, T]) -> Callable[P, T]:
    return functools.wraps(fn)(
        functools.partial(_CONTINUATION_STATE.sets(fn), _RESULT_STATE.sets(cont))
    )
