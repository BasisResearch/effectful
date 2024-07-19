import contextlib
import functools
import typing
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


class _NoStateError(Exception):
    pass


class LinearState(Generic[S]):
    _get_state: Operation[[], S]

    def __init__(self):

        @Operation
        def _get_state() -> S:
            raise _NoStateError("No state stored")

        self._get_state = _get_state

    def sets(self, fn: Callable[P, T]) -> Callable[Concatenate[S, P], T]:
        @functools.wraps(fn)
        def _wrapper(state: S, *args: P.args, **kwargs: P.kwargs) -> T:
            return shallow_interpreter({self._get_state: lambda: state})(fn)(
                *args, **kwargs
            )

        return _wrapper

    def gets(self, fn: Callable[Concatenate[S, P], T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return fn(self._get_state(), *args, **kwargs)

        return _wrapper

    def dups(
        self, fn: Callable[Concatenate[S, P], T]
    ) -> Callable[Concatenate[S, P], T]:
        @functools.wraps(fn)
        def _wrapper(s: S, *a: P.args, **k: P.kwargs) -> T:
            return self.sets(lambda *a, **k: fn(s, *a, **k))(s, *a, **k)

        return _wrapper


Result = Optional[T]
ArgSet = TypeVar("ArgSet", bound=Tuple[Tuple, Mapping])


class LinearResult(Generic[T], LinearState[Result[T]]):
    pass


class LinearArgs(Generic[ArgSet], LinearState[ArgSet]):
    pass


_arg_state = LinearArgs()
_result_state = LinearResult()


def bind_result(fn: Callable[Concatenate[Result[T], P], T]) -> Callable[P, T]:

    bound_fn = _result_state.gets(fn)
    default_bound_fn = _result_state.dups(fn)

    @functools.wraps(fn)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return bound_fn(*args, **kwargs)
        except _NoStateError:
            return default_bound_fn(None, *args, **kwargs)

    return _wrapper


Prompt = Operation[[Result[S]], S]


def bind_prompts(
    unbound_conts: Mapping[Prompt[S], Callable[P, T]],
) -> Callable[[Callable[P, T]], Callable[P, T]]:

    def _flatten_args(fn: Callable[Q, V]) -> Callable[[ArgSet], V]:
        return lambda ak: fn(*ak[0], **typing.cast(Mapping, ak[1]))

    def _unflatten_args(fn: Callable[[ArgSet], V]) -> Callable[Q, V]:
        return lambda *a, **k: fn(typing.cast(ArgSet, (a, k)))

    def _bind_local_state(fn: Callable[Q, V]) -> Callable[Q, V]:
        bound_conts = {
            p: _result_state.sets(_arg_state.gets(_flatten_args(unbound_conts[p])))
            for p in unbound_conts.keys()
        }
        return shallow_interpreter(bound_conts)(
            _unflatten_args(_arg_state.dups(_flatten_args(fn)))
        )

    return _bind_local_state
