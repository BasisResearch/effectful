import logging
from typing import TypeVar

from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter
from effectful.internals.sugar import ObjectInterpretation, implements
from effectful.ops.core import Interpretation, Operation, define

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


def test_memoized_define():
    assert define(Interpretation) is define(Interpretation)
    assert define(Interpretation[int, int]) is define(Interpretation[int, int])
    assert define(Interpretation[int, int]) is define(Interpretation[int, float])
    assert define(Interpretation[int, int]) is define(Interpretation)

    assert define(Operation) is define(Operation)
    assert define(Operation[P, int]) is define(Operation[P, int])
    assert define(Operation[P, int]) is define(Operation[P, float])
    assert define(Operation[P, int]) is define(Operation)

    assert define(Operation) is not define(Interpretation)


def test_object_interpretation_inheritance():
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


def test_sugar_subclassing():

    class ScaleBy(ObjectInterpretation):
        def __init__(self, scale):
            self._scale = scale

        @implements(plus_1)
        def plus_1(self, v):
            return v + self._scale

        @implements(plus_2)
        def plus_2(self, v):
            return v + 2 * self._scale

    class ScaleAndShiftBy(ScaleBy):
        def __init__(self, scale, shift):
            super().__init__(scale)
            self._shift = shift

        @implements(plus_1)
        def plus_1(self, v):
            return super().plus_1(v) + self._shift

        # plus_2 inhereted from ScaleBy

    with interpreter(ScaleBy(4)):
        assert plus_1(4) == 8
        assert plus_2(4) == 12

    with interpreter(ScaleAndShiftBy(4, 1)):
        assert plus_1(4) == 9
        assert plus_2(4) == 12
