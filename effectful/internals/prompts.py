from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Tuple, TypeVar

from typing_extensions import ParamSpec

from effectful.handlers.state import State
from effectful.ops.core import Interpretation, Operation
from effectful.ops.interpreter import interpreter

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


Prompt = Operation[[Optional[S]], S]
Args = Tuple[Tuple, Mapping]


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


def bind_prompt(
    prompt: Prompt[S],
    prompt_impl: Callable[P, T],
    wrapped: Callable[P, T],
) -> Callable[P, T]:
    """
    Bind a prompt to a particular implementation in a particular function.

    Within the body of the wrapped function, calling `prompt` will forward the
    arguments passed to the wrapped function to the prompt's implementation.
    The value passed to `prompt` will be bound to the state effect `result`.

    :param prompt: The prompt to be bound
    :param prompt_impl: The implementation of that prompt
    :param wrapped: The function in which the prompt will be bound
    :return: A wrapper which calls the wrapped function with the prompt bound

    """

    @wraps(wrapped)
    def wrapper(*a: P.args, **k: P.kwargs) -> T:
        @wraps(prompt)
        def wrapped_prompt(res: Optional[T]) -> T:
            with interpreter(result.bound_to(res)):
                return prompt_impl(*args()[0], **args()[1])

        with shallow_interpreter({prompt: wrapped_prompt}):
            with interpreter(args.bound_to((a, k))):
                return wrapped(*a, **k)

    return wrapper
