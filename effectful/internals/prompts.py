from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Tuple, TypeAlias, TypeVar

from typing_extensions import Concatenate, ParamSpec

from effectful.internals.state import State
from effectful.ops.core import Interpretation, Operation
from effectful.ops.interpreter import interpreter

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


Prompt: TypeAlias = Operation[[Optional[S]], S]
Args: TypeAlias = Tuple[Tuple, Mapping]


@contextmanager
def shallow_interpreter(intp: Interpretation):
    from effectful.internals.runtime import get_interpretation

    # destructive update: calling any op in intp should remove intp from active
    active_intp = get_interpretation()
    prev_intp = {op: active_intp.get(op, op.default) for op in intp}
    next_intp = {
        op: interpreter(prev_intp, unset=False)(impl) for op, impl in intp.items()
    }

    with interpreter(next_intp):
        yield intp


result = State[Any](None)
args = State[Args](State._Empty())


def bind_result(fn: Callable[Concatenate[Optional[T], P], T]) -> Callable[P, T]:
    return lambda *a, **k: fn(result(), *a, **k)


def bind_result_to_method(
    fn: Callable[Concatenate[V, Optional[T], P], T]
) -> Callable[Concatenate[V, P], T]:
    return lambda s, *a, **k: fn(s, result(), *a, **k)


def bind_prompt(
    prompt: Prompt[S], prompt_impl: Callable[P, T], wrapped: Callable[P, T]
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

    @wraps(prompt)
    def prompt_wrapper(res: Optional[T]) -> T:
        with interpreter(result.bound_to(res)):
            return prompt_impl(*args()[0], **args()[1])

    @wraps(wrapped)
    def wrapper(*a: P.args, **k: P.kwargs) -> T:
        with shallow_interpreter({prompt: prompt_wrapper}):
            with interpreter(args.bound_to((a, k))):
                return wrapped(*a, **k)

    return wrapper
