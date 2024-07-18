import itertools
import logging
from typing import List, TypeVar

import pytest
from typing_extensions import ParamSpec

from effectful.handlers.state import State
from effectful.internals.prompts import bind_result, value_or_result
from effectful.ops.core import Interpretation, Operation, define
from effectful.ops.handler import coproduct, fwd, handler
from effectful.ops.interpreter import interpreter
from effectful.ops.runner import product, reflect, runner

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@define(Operation)
def plus_1(x: int) -> int:
    return x + 1


@define(Operation)
def plus_2(x: int) -> int:
    return x + 2


@define(Operation)
def times_plus_1(x: int, y: int) -> int:
    return x * y + 1


def block(*ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(lambda v, *args, **kwargs: reflect(v)) for op in ops}


def defaults(*ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(value_or_result(op.default)) for op in ops}


def times_n_handler(n: int, *ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(lambda v, *args, **kwargs: fwd(v) * n) for op in ops}


OPERATION_CASES = (
    [[plus_1, (i,)] for i in range(5)]
    + [[plus_2, (i,)] for i in range(5)]
    + [[times_plus_1, (i, j)] for i, j in itertools.product(range(5), range(5))]
)
N_CASES = [1, 2, 3]


@pytest.mark.parametrize("op,args", OPERATION_CASES)
def test_affine_continuation_product(op, args):
    def f():
        return op(*args)

    h_twice = {op: bind_result(lambda v, *a, **k: reflect(reflect(v)))}

    assert (
        interpreter(defaults(op))(f)()
        == interpreter(product(defaults(op), h_twice))(f)()
    )


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_product_block_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = coproduct(block(op), times_n_handler(n1, op))
    h2 = coproduct(block(op), times_n_handler(n2, op))

    intp1 = product(h0, product(h1, h2))
    intp2 = product(product(h0, h1), h2)

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()


def test_runner_scopes():
    @define(Operation)
    def double(v):
        raise RuntimeError("No Defaults")

    @define(Operation)
    def triple(v):
        raise RuntimeError("No Defaults")

    @define(Operation)
    def sextuple(v):
        raise RuntimeError("No Defaults")

    def multiply_in_length():
        return handler(
            {
                double: lambda v: [v, v],
                triple: lambda v: [v, v, v],
            }
        )

    def multiply_in_value():
        return handler(
            {
                double: lambda v: v + v,
                triple: lambda v: v + v + v,
            }
        )

    def sextuple_as_double_triple(mode):
        interp = {sextuple: lambda v: double(triple(v))}
        if mode == "runner":
            return runner(interp)
        elif mode == "handler":
            return handler(interp)

    with multiply_in_length():
        with sextuple_as_double_triple("runner"):
            with multiply_in_value():
                assert double(2) == 4
                assert triple(3) == 9
                assert sextuple(6) == [[6, 6, 6], [6, 6, 6]]
        with sextuple_as_double_triple("handler"):
            with multiply_in_value():
                assert double(2) == 4
                assert triple(3) == 9
                assert sextuple(6) == 36


def test_using_runner_to_implement_trailing_state():
    def trailing_state(st: State[List[T]]):
        def trailing_set(new_value: T) -> None:
            st.set(st.get() + [new_value])

        interp: Interpretation = {
            st.get: lambda: st.get()[-1],
            st.set: trailing_set,
        }

        return runner(interp)

    st = State([])

    with handler(defaults(st.get, st.bound_to, st.set)):
        with trailing_state(st):
            st.set(3)
            assert st.get() == 3
            st.set(4)
            assert st.get() == 4
        assert st.get() == [3, 4]


def test_runner_outer_reflect():
    def plus_two_calling_plus_one():
        def plus_minus_one_then_reflect(v):
            r = plus_1(v)
            return reflect(r - 1)

        return {plus_2: plus_minus_one_then_reflect}

    defs = {plus_1: plus_1.default, plus_2: plus_2.default}
    with interpreter(product(defs, plus_two_calling_plus_one())):
        assert plus_1(1) == 2
        assert plus_2(2) == 4


def test_runner_outer_reflect_1():
    @bind_result
    def plus_two_impl_inner(res, v):
        assert res is None
        r = plus_1(v)
        return reflect(r + 1)

    @bind_result
    def plus_two_impl_outer(res, v):
        if res is None:
            return v + 2
        else:
            return res

    @bind_result
    def plus_one_to_plus_five(res, v):
        assert res is None
        return plus_2(v) + 3

    intp_inner = {plus_2: plus_two_impl_inner}
    intp_outer = {plus_1: plus_one_to_plus_five, plus_2: plus_two_impl_outer}

    with interpreter(intp_inner):
        assert plus_2(2) == 4

    with interpreter(product(intp_outer, intp_inner)):
        assert plus_1(1) == 2
        assert plus_2(2) == 2 + (2 + 3) + 1

    with interpreter(intp_outer):
        assert plus_1(1) == 1 + 2 + 3
        with runner(intp_inner):
            assert plus_2(2) == 2 + (2 + 3) + 1
