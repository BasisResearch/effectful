import contextlib
import functools
import threading
from collections.abc import Callable, Mapping

from effectful.ops.syntax import defop
from effectful.ops.types import Interpretation, Operation

# Global reentrant lock for mutual exclusion of handler execution
_handler_lock = threading.RLock()


class Runtime[S, T](threading.local):
    """Thread-local runtime for effectful interpretations."""

    interpretation: "Interpretation[S, T]"

    def __init__(self):
        super().__init__()
        self.interpretation = {}


def get_handler_lock() -> threading.RLock:
    """Get the global handler execution lock.

    This lock ensures mutual exclusion for handler execution by default.
    Use release_handler_lock() to temporarily release it for concurrent operations.
    """
    return _handler_lock


@contextlib.contextmanager
def release_handler_lock():
    """Context manager to temporarily release the handler lock.

    Use this when performing I/O operations or other blocking calls
    that should allow other threads to execute handlers concurrently.

    Example:
        @defop
        def llm_call(prompt: str) -> str:
            with release_handler_lock():
                # HTTP request can run while other threads execute handlers
                response = requests.post(url, json={"prompt": prompt})
            return response.json()["result"]
    """
    lock = get_handler_lock()
    lock.release()
    try:
        yield
    finally:
        lock.acquire()


@contextlib.contextmanager
def acquire_handler_lock():
    """Context manager to acquire the handler lock.

    This is called automatically by apply() for handler execution.
    Most users won't need to call this directly.
    """
    lock = get_handler_lock()
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


@functools.lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return Runtime()


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
