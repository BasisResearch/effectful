import contextlib
import functools
from typing import Any, Callable, Mapping, Optional, Tuple, TypeVar

from typing_extensions import Concatenate, ParamSpec

from effectful.handlers.state import State
from effectful.ops.core import Interpretation, Operation
from effectful.ops.interpreter import interpreter

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@contextlib.contextmanager
def shallow_interpreter(intp: Interpretation):
    from effectful.internals.runtime import get_interpretation

    # destructive update: calling any op in intp should remove intp from active
    active_intp = get_interpretation()
    prev_intp = {
        op: active_intp[op] if op in active_intp else op.default for op in intp.keys()
    }

    with interpreter(
        {op: interpreter(prev_intp, unset=False)(intp[op]) for op in intp.keys()}
    ):
        yield intp


def value_or_result(fn: Callable[P, T]) -> Callable[Concatenate[Optional[T], P], T]:
    """
    Return either the value or the result of calling the function.
    """

    @functools.wraps(fn)
    def _wrapper(__result: Optional[T], *args: P.args, **kwargs: P.kwargs) -> T:
        return fn(*args, **kwargs) if __result is None else __result

    return _wrapper


result = State[Any](None)


def _set_result(
    fn: Callable[Concatenate[Optional[T], P], T]
) -> Callable[Concatenate[Optional[T], P], T]:
    @functools.wraps(fn)
    def _wrapper(res: Optional[T], *args: P.args, **kwargs: P.kwargs) -> T:
        return shallow_interpreter(result.bound_to(res))(fn)(res, *args, **kwargs)

    return _wrapper


def bind_result(fn: Callable[Concatenate[Optional[T], P], T]) -> Callable[P, T]:
    @functools.wraps(fn)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return interpreter(result.bound_to(None))(fn)(result.get(), *args, **kwargs)

    return _wrapper


Prompt = Operation[[Optional[S]], S]


def bind_prompts(
    unbound_conts: Mapping[Prompt[S], Callable[P, T]],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    Args = Tuple[Tuple, Mapping]

    args = State[Args]()

    def set_args(fn: Callable[Q, V]) -> Callable[Q, V]:
        @functools.wraps(fn)
        def _wrapper(*a: Q.args, **ks: Q.kwargs) -> V:
            return interpreter(args.bound_to((a, ks)))(fn)(*a, **ks)

        return _wrapper

    def bind_args(fn: Callable[Q, V]) -> Callable[Q, V]:
        bound_conts = {
            p: _set_result(
                functools.partial(
                    lambda k, _: k(*args.get()[0], **args.get()[1]),
                    unbound_conts[p],
                )
            )
            for p in unbound_conts.keys()
        }
        return shallow_interpreter(bound_conts)(set_args(fn))

    return bind_args
