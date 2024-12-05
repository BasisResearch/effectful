from contextlib import contextmanager
from functools import wraps
from typing import Callable, Mapping, Optional, Tuple, TypeVar

from typing_extensions import Concatenate, ParamSpec

from effectful.internals.runtime import get_interpretation, interpreter
from effectful.ops.core import Operation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@Operation
def _get_result() -> Optional[T]:
    return None


@Operation
def _get_args() -> Tuple[Tuple, Mapping]:
    return ((), {})


def _set_state(fn: Callable[P, T]) -> Callable[Concatenate[Optional[S], P], T]:

    @wraps(fn)
    def _cont_wrapper(res: Optional[S], *a: P.args, **k: P.kwargs) -> T:
        a, k = (a, k) if a or k else _get_args()  # type: ignore
        res = res if res is not None else _get_result()
        with interpreter({**get_interpretation(), _get_result: lambda: res, _get_args: lambda: (a, k)}):  # type: ignore
            return fn(*a, **k)

    return _cont_wrapper


def _set_args(fn: Callable[P, T]) -> Callable[P, T]:

    @wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        with interpreter({**get_interpretation(), _get_args: lambda: (a, k)}):  # type: ignore
            return fn(*a, **k)

    return _cont_wrapper


def bind_result(fn: Callable[Concatenate[Optional[T], P], T]) -> Callable[P, T]:
    return lambda *a, **k: fn(_get_result(), *a, **k)


def bind_result_to_method(
    fn: Callable[Concatenate[V, Optional[T], P], T]
) -> Callable[Concatenate[V, P], T]:
    return bind_result(lambda r, s, *a, **k: fn(s, r, *a, **k))


def bind_prompt(
    prompt: Operation[Concatenate[S, P], T],
    prompt_impl: Callable[P, T],
    wrapped: Callable[P, T],
) -> Callable[P, T]:
    """
    Bind a :py:type:`Prompt` to a particular implementation in a particular function.

    Within the body of the wrapped function, calling ``prompt`` will forward the
    arguments passed to the wrapped function to the prompt's implementation.
    The value passed to ``prompt`` will be bound to the :class:`State` ``result``,
    which can be accessed either directly or through the :fn:``bind_result`` wrapper.

    :param prompt: The prompt to be bound
    :param prompt_impl: The implementation of that prompt
    :param wrapped: The function in which the prompt will be bound.
    :return: A wrapper which calls the wrapped function with the prompt bound.

    >>> @Operation
    ... def call_my_manager(has_receit: bool) -> bool:
    ...     raise RuntimeError
    >>> def clerk(problem: str) -> str:
    ...     if "refund" in problem:
    ...         print("Clerk: Let me get my manager.")
    ...         refunded = call_my_manager("receit" in problem)
    ...         if refunded:
    ...             print("Clerk: Great, here's your refund.")
    ...     else:
    ...         print("Clerk: Let me help you with that.")
    >>> @bind_result
    ... def manager(has_receit, problem: str) -> str:
    ...     if has_receit:
    ...         print("Manager: You can have a refund.")
    ...         return True
    ...     elif "corporate" in problem:
    ...         print("Manager: No need to be hasty, have your refund!")
    ...         return True
    ...     else:
    ...         print("Manager: Sorry, but you're ineligable for a refund.")
    ...         return False
    >>> storefront = bind_prompt(call_my_manager, manager, clerk)
    >>> storefront("Do you have this in black?")
    Clerk: Let me help you with that.
    >>> storefront("Can I refund this purchase?")
    Clerk: Let me get my manager.
    Manager: Sorry, but you're ineligable for a refund.
    >>> storefront("Can I refund this purchase? I have a receit")
    Clerk: Let me get my manager.
    Manager: You can have a refund.
    Clerk: Great, here's your refund.
    >>> storefront("Can I refund this purchase? I'll tell corporate!")
    Clerk: Let me get my manager.
    Manager: No need to be hasty, have your refund!
    Clerk: Great, here's your refund.
    """

    @contextmanager
    def _handler(intp):
        with interpreter({**get_interpretation(), **intp}):
            yield intp

    cont = _set_state(prompt_impl)
    body = _set_args(wrapped)

    @wraps(wrapped)
    def wrapper(*a: P.args, **k: P.kwargs) -> T:
        prev = get_interpretation().get(prompt, prompt.__default_rule__)
        with _handler({prompt: _handler({prompt: prev})(cont)}):  # type: ignore
            return body(*a, **k)

    return wrapper
