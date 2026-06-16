import contextlib
import dataclasses
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import local

from effectful.ops.types import Interpretation, NotHandled, Operation


@dataclasses.dataclass
class Runtime[S, T](local):
    interpretation: "Interpretation[S, T]"


@functools.lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return Runtime(interpretation={})


def get_interpretation():
    return get_runtime().interpretation


@contextlib.contextmanager
def interpreter(intp: "Interpretation"):
    r = get_runtime()
    old_intp = r.interpretation
    try:
        old_intp, r.interpretation = r.interpretation, dict(intp)
        yield intp
    finally:
        r.interpretation = old_intp


@dataclass
class _FwdContext:
    op: Operation
    next_handler: Callable | None
    args: tuple
    kwargs: Mapping


@Operation.define
def _get_context() -> _FwdContext:
    raise NotHandled


def _save_context[**P, T](
    fn: Callable[P, T], op: Operation[P, T], next_handler: Callable[P, T] | None = None
) -> Callable[P, T]:
    from effectful.ops.semantics import handler

    intp = {_get_op: lambda: op} | (
        {} if next_handler is None else {_get_next_handler: lambda: next_handler}
    )

    @functools.wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        with handler(intp | {_get_args: lambda: (a, k)}):
            return fn(*a, **k)

    return _cont_wrapper


@Operation.define
def _get_args() -> tuple[tuple, Mapping]:
    return ((), {})


@Operation.define
def _get_op() -> Operation:
    raise NotHandled


@Operation.define
def _get_next_handler() -> Callable | None:
    return None


def _restore_args[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        a, k = (a, k) if a or k else _get_args()
        return fn(*a, **k)

    return _cont_wrapper


def _save_args[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    from effectful.ops.semantics import handler

    @functools.wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        with handler({_get_args: lambda: (a, k)}):
            return fn(*a, **k)

    return _cont_wrapper


def _save_op[**P, T](fn: Callable[P, T], op: Operation[P, T]) -> Callable[P, T]:
    from effectful.ops.semantics import handler

    @functools.wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        with handler({_get_op: lambda: op}):
            return fn(*a, **k)

    return _cont_wrapper


def _save_next_handler[**P, T](
    fn: Callable[P, T], next: Callable[P, T]
) -> Callable[P, T]:
    from effectful.ops.semantics import handler

    @functools.wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        with handler({_get_next_handler: lambda: next}):
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
