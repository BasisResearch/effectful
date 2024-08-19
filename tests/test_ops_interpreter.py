import contextlib
import functools
import itertools
import logging
from typing import TypeVar

import pytest
from typing_extensions import ParamSpec

from effectful.internals.prompts import bind_result
from effectful.internals.sugar import ObjectInterpretation, implements
from effectful.ops.core import (
    InjectedDataclass,
    Interpretation,
    Operation,
    define,
    register,
)
from effectful.ops.interpreter import interpreter

logger = logging.getLogger(__name__)


P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@Operation
def plus_1(x: int) -> int:
    return x + 1


@Operation
def plus_2(x: int) -> int:
    return x + 2


@Operation
def times_plus_1(x: int, y: int) -> int:
    return x * y + 1


def times_n(n: int, *ops: Operation[..., int]) -> Interpretation[int, int]:
    @bind_result
    def _op_times_n(res, n: int, op: Operation[..., int], *args: int) -> int:
        return (res or op.default(*args)) * n

    return {op: functools.partial(_op_times_n, n, op) for op in ops}


OPERATION_CASES = (
    [[plus_1, (i,)] for i in range(5)]
    + [[plus_2, (i,)] for i in range(5)]
    + [[times_plus_1, (i, j)] for i, j in itertools.product(range(5), range(5))]
)
N_CASES = [1, 2, 3]
DEPTH_CASES = [1, 2, 3]


@pytest.mark.parametrize("op,args", OPERATION_CASES)
def test_op_default(op, args):
    assert op(*args) == op.default(*args)


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_times_n_interpretation(op, args, n):
    new_op = Operation(lambda *args: op(*args) + 3)

    assert op in times_n(n, op)
    assert new_op not in times_n(n, op)

    assert op(*args) * n == times_n(n, op)[op](*args)


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_register_new_op(op, args, n):
    new_op = Operation(lambda *args: op(*args) + 3)
    intp = times_n(n, op)

    with interpreter(intp):
        new_value = new_op(*args)
        assert new_value == op.default(*args) * n + 3

        register(new_op, intp, times_n(n, new_op)[new_op])
        assert new_op(*args) == new_value

    with interpreter(intp):
        assert new_op(*args) == (op.default(*args) * n + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_1(op, args, n):
    new_op = Operation(lambda *args: op(*args) + 3)

    with interpreter(times_n(n, new_op)):
        assert op(*args) == op.default(*args)
        assert new_op(*args) == (op.default(*args) + 3) * n == (op(*args) + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_2(op, args, n):
    new_op = Operation(lambda *args: op(*args) + 3)

    with interpreter(times_n(n, op)):
        assert op(*args) == op.default(*args) * n
        assert new_op(*args) == op.default(*args) * n + 3


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_3(op, args, n):
    new_op = Operation(lambda *args: op(*args) + 3)

    with interpreter(times_n(n, op, new_op)):
        assert op(*args) == op.default(*args) * n
        assert new_op(*args) == (op.default(*args) * n + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_1(op, args, n_outer, n_inner):
    new_op = Operation(lambda *args: op(*args) + 3)

    with interpreter(times_n(n_outer, op, new_op)):
        with interpreter(times_n(n_inner, op)):
            assert op(*args) == op.default(*args) * n_inner
            assert new_op(*args) == (op.default(*args) * n_inner + 3) * n_outer


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_2(op, args, n_outer, n_inner):
    new_op = Operation(lambda *args: op(*args) + 3)

    with interpreter(times_n(n_outer, op, new_op)):
        with interpreter(times_n(n_inner, new_op)):
            assert op(*args) == op.default(*args) * n_outer
            assert new_op(*args) == (op.default(*args) * n_outer + 3) * n_inner


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_3(op, args, n_outer, n_inner):
    new_op = Operation(lambda *args: op(*args) + 3)

    with interpreter(times_n(n_outer, op, new_op)):
        with interpreter(times_n(n_inner, op, new_op)):
            assert op(*args) == op.default(*args) * n_inner
            assert new_op(*args) == (op.default(*args) * n_inner + 3) * n_inner


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", DEPTH_CASES)
def test_op_repeat_nest_interpreter(op, args, n, depth):
    new_op = Operation(lambda *args: op(*args) + 3)

    intp = times_n(n, new_op)
    with contextlib.ExitStack() as stack:
        for _ in range(depth):
            stack.enter_context(interpreter(intp))

        assert op(*args) == op.default(*args)
        assert new_op(*args) == intp[new_op](*args)


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", DEPTH_CASES)
def test_op_fail_nest_interpreter(op, args, n, depth):
    def _fail_op(*args: int) -> int:
        raise ValueError("oops")

    fail_op = Operation(_fail_op)
    intp = times_n(n, op, fail_op)

    with pytest.raises(ValueError, match="oops"):
        try:
            with contextlib.ExitStack() as stack:
                for _ in range(depth):
                    stack.enter_context(interpreter(intp))

                try:
                    fail_op(*args)
                except ValueError as e:
                    assert op(*args) == op.default(*args) * n
                    raise e
        except ValueError as e:
            assert op(*args) == op.default(*args)
            raise e


def test_object_interpretation_inheretance():
    @Operation
    def op1():
        return "op1"

    @Operation
    def op2():
        return "op2"

    @Operation
    def op3():
        return "op3"

    @Operation
    def op4():
        return "op4"

    class MyHandler(ObjectInterpretation):
        @implements(op1)
        def op1_impl(self):
            return "MyHandler.op1_impl"

        @implements(op2)
        def op2_impl(self):
            return "MyHandler.op2_impl"

        @implements(op3)
        def an_op_impl(self):
            return "MyHandler.an_op_impl"

        @implements(op4)
        def another_op_impl(self):
            return "MyHandler.another_op_impl"

    class MyHandlerSubclass(MyHandler):
        @implements(op1)
        def op1_impl(self):  # same method name, same op
            return "MyHandlerSubclass.op1_impl"

        @implements(op2)
        def another_op2_impl(self):  # different method name, same op
            return "MyHandlerSubclass.another_op2_impl"

        @implements(op3)
        def another_op_impl(
            self,
        ):  # reusing method name from parent impl of different op
            return "MyHandlerSubclass.another_op_impl"

        # no new implementation of op4, but will its behavior change through redefinition of another_op_impl?

    my_handler = MyHandler()
    with interpreter(my_handler):
        assert op1() == "MyHandler.op1_impl"
        assert op2() == "MyHandler.op2_impl"
        assert op3() == "MyHandler.an_op_impl"
        assert op4() == "MyHandler.another_op_impl"

    my_handler_subclass = MyHandlerSubclass()
    with interpreter(my_handler_subclass):
        assert op1() == "MyHandlerSubclass.op1_impl"
        assert op2() == "MyHandlerSubclass.another_op2_impl"
        assert op3() == "MyHandlerSubclass.another_op_impl"
        assert op4() == "MyHandler.another_op_impl"


def test_injected_types() -> None:
    class Foo(InjectedDataclass):
        foo: int

    assert isinstance(Foo(1), Foo)
    assert Foo(1).foo == 1

    assert isinstance(define(Foo), Operation)
    assert define(Foo) is not Foo
    assert define(Foo)(1) == Foo(1)

    with interpreter({define(Foo): lambda *_, **__: 1}):
        assert Foo(1) == 1

    with interpreter({define(Operation): lambda *_, **__: 1}):
        assert Operation(lambda: 1) == 1

    assert isinstance(define(list), Operation)
    assert isinstance(define(list)((1, 2)), list)
    assert define(list)((1, 2)) == [1, 2]

    assert define(Foo) is define(Foo)
    assert define(list) is define(list)
