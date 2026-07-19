import contextlib
import contextvars
import functools
import inspect
from collections.abc import Callable, Mapping

from effectful.ops.types import Interpretation, Operation

_INTERPRETATION: "contextvars.ContextVar[Interpretation]" = contextvars.ContextVar(
    "effectful_interpretation", default={}
)


def get_interpretation() -> "Interpretation":
    return _INTERPRETATION.get()


@contextlib.contextmanager
def interpreter(intp: "Interpretation"):
    token = _INTERPRETATION.set(intp)
    try:
        yield intp
    finally:
        _INTERPRETATION.reset(token)


@Operation.define
def _get_args() -> tuple[tuple, Mapping]:
    return ((), {})


def _restore_args[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    sig = inspect.signature(fn)
    if not sig.parameters:
        return fn

    @functools.wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        a, k = (a, k) if a or k else _get_args()
        return fn(*a, **k)

    return _cont_wrapper


def _save_args[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    from effectful.ops.semantics import handler

    sig = inspect.signature(fn)
    if not sig.parameters:
        return fn

    @functools.wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        with handler({_get_args: lambda: (a, k)}):
            return fn(*a, **k)

    return _cont_wrapper


def _set_prompt[**P, T](
    prompt: Operation[P, T], cont: Callable[P, T], body: Callable[P, T]
) -> Callable[P, T]:
    from effectful.ops.semantics import handler

    @functools.wraps(body)
    def bound_body(*a: P.args, **k: P.kwargs) -> T:
        next_cont = get_interpretation().get(prompt, prompt.__default_rule__)
        with handler({prompt: handler({prompt: next_cont})(cont)}):
            return body(*a, **k)

    return bound_body
