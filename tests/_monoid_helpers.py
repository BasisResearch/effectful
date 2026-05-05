from collections.abc import Callable, Mapping, Sequence
from typing import Any, get_args, get_origin

from hypothesis import strategies as st

from effectful.ops.syntax import deffn
from effectful.ops.types import Operation


def _value_strategy_for(annotation: Any) -> st.SearchStrategy[Any]:
    """Strategy for the value an *0-arg* Operation should return."""
    if annotation is int:
        return st.integers()
    if annotation is float:
        return st.floats(allow_nan=False)
    if get_origin(annotation) is list and get_args(annotation) == (int,):
        return st.lists(st.integers())
    raise NotImplementedError(
        f"No value strategy for return annotation {annotation!r}; "
        "supported: int, list[int]"
    )


_UNARY_INT_FNS: list[Callable[[int], int]] = [
    lambda x: x,
    lambda x: x + 1,
    lambda x: x - 1,
    lambda x: -x,
    lambda x: 2 * x,
    lambda x: 3 * x + 1,
]

_BINARY_INT_FNS: list[Callable[[int, int], int]] = [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x + 2 * y,
    lambda x, y: 2 * x - y,
]

_UNARY_LIST_FNS: list[Callable[[int], list[int]]] = [
    lambda _x: [],
    lambda x: [x],
    lambda x: [x, x + 1],
    lambda x: [x, -x],
    lambda x: [0, x, x + 1],
]


def _strategy_for_op(op: Operation) -> st.SearchStrategy[Callable[..., Any]]:
    """Pick a strategy producing a callable suitable for binding `op` in an
    interpretation. Inspects the operation's signature.
    """
    sig = op.__signature__
    params = list(sig.parameters.values())
    ret = sig.return_annotation
    param_types = tuple(p.annotation for p in params)

    if not params:
        return _value_strategy_for(ret).map(deffn)
    if ret is int and param_types == (int,):
        return st.sampled_from(_UNARY_INT_FNS)
    if ret is int and param_types == (int, int):
        return st.sampled_from(_BINARY_INT_FNS)
    if get_origin(ret) is list and get_args(ret) == (int,) and param_types == (int,):
        return st.sampled_from(_UNARY_LIST_FNS)
    raise NotImplementedError(
        f"Function-typed free var must return int or list[int]; got {ret!r} for {op}"
    )


@st.composite
def random_interpretation(
    draw: st.DrawFn, free_vars: Sequence[Operation]
) -> Mapping[Operation, Callable[..., Any]]:
    """Draw an Interpretation binding every Operation in `case.free_vars` to
    a randomly chosen value/callable. Keys are Operation identities.
    """
    intp: dict[Operation, Callable[..., Any]] = {}
    for op in free_vars:
        intp[op] = draw(_strategy_for_op(op))
    return intp


__all__ = ["random_interpretation"]
