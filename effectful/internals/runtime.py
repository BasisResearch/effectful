import contextlib
import contextvars
import functools
import typing
from collections.abc import Callable, Mapping

from effectful.ops.syntax import defop
from effectful.ops.types import Interpretation, Operation

INTERPRETER: contextvars.ContextVar[Interpretation] = contextvars.ContextVar(
    "interpretation", default=typing.cast(Interpretation, {})
)


def get_interpretation():
    return INTERPRETER.get()


@contextlib.contextmanager
def interpreter(intp: "Interpretation"):
    token = INTERPRETER.set(intp)
    try:
        yield intp
    finally:
        INTERPRETER.reset(token)


@defop
def _get_args() -> tuple[tuple, Mapping]:
    return ((), {})


def _restore_args[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        a, k = (a, k) if a or k else _get_args()  # type: ignore
        return fn(*a, **k)

    return _cont_wrapper


def _save_args[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    from effectful.ops.semantics import handler

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
