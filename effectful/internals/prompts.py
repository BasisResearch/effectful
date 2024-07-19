import contextlib
import functools
from typing import Callable, Generic, Mapping, Optional, Tuple, TypeVar

from typing_extensions import Concatenate, ParamSpec

from effectful.ops.core import Interpretation, Operation
from effectful.ops.interpreter import interpreter

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@contextlib.contextmanager
def shallow_interpreter(intp: Interpretation):
    from ..internals.runtime import get_interpretation

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


class LinearState(Generic[S]):
    _get_state: Operation[[], S]

    def __init__(self, initial_state: S):

        @Operation
        def _get_state() -> S:
            raise NotImplementedError("No state stored")
        
        self._initial_state = initial_state
        self._get_state = _get_state

    def sets(self, fn: Callable[P, T]) -> Callable[Concatenate[S, P], T]:
        @functools.wraps(fn)
        def _wrapper(state: S, *args: P.args, **kwargs: P.kwargs) -> T:
            return shallow_interpreter({self._get_state: lambda: state})(fn)(*args, **kwargs)

        return _wrapper

    def gets(self, fn: Callable[Concatenate[S, P], T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                state = self._get_state()
            except NotImplementedError:
                state = self._initial_state
            return fn(state, *args, **kwargs)

        return _wrapper


Result = Optional[T]
ArgSet = TypeVar("ArgSet", bound=Tuple[Tuple, Mapping])

class LinearResult(Generic[T], LinearState[Result[T]]):
    pass


class LinearArgs(Generic[ArgSet], LinearState[ArgSet]):
    pass


_result = LinearResult(None)
_set_result = _result.sets
bind_result = _result.gets

_arg_state = LinearArgs(None)
_set_args = _arg_state.sets
_bind_args = _arg_state.gets

Prompt = Operation[[Result[S]], S]


def bind_prompts(
    unbound_conts: Mapping[Prompt[S], Callable[P, T]],
) -> Callable[[Callable[P, T]], Callable[P, T]]:

    def _dup_args(fn: Callable[[ArgSet], V]) -> Callable[[ArgSet], V]:
        return lambda ak: _set_args(lambda: fn(ak))(ak)

    def _flatten_args(fn: Callable[Q, V]) -> Callable[[ArgSet], V]:
        return lambda ak: fn(*ak[0], **ak[1])

    def _bind_local_state(fn: Callable[Q, V]) -> Callable[Q, V]:
        bound_conts = {
            p: _set_result(_bind_args(_flatten_args(unbound_conts[p]))) for p in unbound_conts.keys()
        }
        return shallow_interpreter(bound_conts)(lambda *a, **k: _dup_args(_flatten_args(fn))((a, k)))

    return _bind_local_state
