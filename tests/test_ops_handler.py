import contextlib
import itertools
import logging
from typing import TypeVar

import pytest
from typing_extensions import ParamSpec

from effectful.internals.sugar import NoDefaultRule, ObjectInterpretation, implements
from effectful.ops.core import Interpretation, Operation
from effectful.ops.handler import bind_result, coproduct, fwd, handler, product

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


def defaults(*ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: op.__default_rule__ for op in ops}  # type: ignore


def times_n_handler(n: int, *ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(lambda r, *args, **kwargs: fwd(r) * n) for op in ops}


OPERATION_CASES = (
    [[plus_1, (i,)] for i in range(5)]
    + [[plus_2, (i,)] for i in range(5)]
    + [[times_plus_1, (i, j)] for i, j in itertools.product(range(5), range(5))]
)
N_CASES = [1, 2, 3]


def test_fwd_simple():
    def plus_1_fwd(x):
        # do nothing and just fwd
        return fwd(None)

    with handler({plus_1: plus_1_fwd}):
        assert plus_1(1) == 2


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_compose_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, op)

    intp1 = coproduct(h0, coproduct(h1, h2))
    intp2 = coproduct(coproduct(h0, h1), h2)

    assert handler(intp1)(f)() == handler(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_compose_commute_orthogonal(op, args, n1, n2):
    def f():
        return op(*args) + new_op(*args)

    new_op = Operation(lambda *args: op(*args) + 3)

    h0 = defaults(op, new_op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, new_op)

    intp1 = coproduct(h0, coproduct(h1, h2))
    intp2 = coproduct(h0, coproduct(h2, h1))

    assert handler(intp1)(f)() == handler(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_handler_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, op)

    expected = handler(coproduct(h0, coproduct(h1, h2)))(f)()

    with handler(h0), handler(h1), handler(h2):
        assert f() == expected

    with handler(coproduct(h0, h1)), handler(h2):
        assert f() == expected

    with handler(h0), handler(coproduct(h1, h2)):
        assert f() == expected


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_stop_without_fwd(op, args, n, depth):
    def f():
        return op(*args)

    expected = f()

    with contextlib.ExitStack() as stack:
        for _ in range(depth):
            stack.enter_context(handler(times_n_handler(n, op)))

        stack.enter_context(handler(defaults(op)))

        assert f() == expected


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

    with handler(ScaleBy(4)):
        assert plus_1(4) == 8
        assert plus_2(4) == 12

    with handler(ScaleAndShiftBy(4, 1)):
        assert plus_1(4) == 9
        assert plus_2(4) == 12


def test_fwd_default():
    """
    Test that forwarding with no outer handler defers to the default rule.
    """

    @Operation
    def do_stuff():
        return "default stuff"

    def do_more_stuff():
        return fwd(None) + " and more"

    def fancy_stuff():
        return "fancy stuff"

    # forwarding with no outer handler defers to the default rule
    with handler({do_stuff: do_more_stuff}):
        assert do_stuff() == "default stuff and more"

    # forwarding with an outer handler uses the outer handler
    with handler(coproduct({do_stuff: fancy_stuff}, {do_stuff: do_more_stuff})):
        assert do_stuff() == "fancy stuff and more"

    # empty coproducts allow forwarding to the default implementation
    with handler(coproduct({}, {do_stuff: do_more_stuff})):
        assert do_stuff() == "default stuff and more"

    # ditto products
    with handler(product({}, {do_stuff: do_more_stuff})):
        assert do_stuff() == "default stuff and more"


def test_product_resets_fwd():
    @Operation
    def do_stuff():
        raise NoDefaultRule

    @Operation
    def do_other_stuff():
        return "other stuff"

    h_outer = {
        do_stuff: lambda: "default stuff",
        do_other_stuff: lambda: fwd(None) + " and more " + do_stuff(),
    }
    h_inner = {do_stuff: lambda: "fancy " + do_other_stuff()}
    h_topmost = {do_stuff: lambda: "should not be called"}

    with handler(product(h_topmost, product(h_outer, h_inner))):
        assert do_stuff() == "fancy other stuff and more default stuff"


@Operation
def op0():
    raise NoDefaultRule


@Operation
def op1():
    raise NoDefaultRule


@Operation
def op2():
    raise NoDefaultRule


def f_op2():
    return op2()


def test_product_alpha_equivalent():

    h0 = {op0: lambda: (op1(), 0), op1: lambda: 2}
    h1 = {op0: lambda: (op2(), 0), op2: lambda: 2}
    h2 = {op2: lambda: (op0(), 2)}

    h_lhs = product(h0, h2)
    h_rhs = product(h1, h2)

    lhs = handler(h_lhs)(f_op2)()
    rhs = handler(h_rhs)(f_op2)()

    assert lhs == rhs


def test_product_associative():

    h0 = {op0: lambda: 0}
    h1 = {op1: lambda: (op0(), 1)}
    h2 = {op2: lambda: (op1(), 2)}

    h_lhs = product(h0, product(h1, h2))
    h_rhs = product(product(h0, h1), h2)

    lhs = handler(h_lhs)(f_op2)()
    rhs = handler(h_rhs)(f_op2)()

    assert lhs == rhs


def test_product_commute_orthogonal():

    h0 = {op0: lambda: 0}
    h1 = {op1: lambda: 1}
    h2 = {op2: lambda: (op1(), op0(), 2)}

    h_lhs = product(h0, product(h1, h2))
    h_rhs = product(h1, product(h0, h2))

    lhs = handler(h_lhs)(f_op2)()
    rhs = handler(h_rhs)(f_op2)()

    assert lhs == rhs


def test_product_distributive():

    h0 = {op0: lambda: 0, op1: lambda: (op0(), 1)}
    h1 = {op2: lambda: (op0(), op1(), 1)}
    h2 = {op2: lambda: (fwd(None), op0(), op1(), 2)}

    h_lhs = product(h0, coproduct(h1, h2))
    h_rhs = coproduct(product(h0, h1), product(h0, h2))

    h00 = {op0: lambda: 5, op1: lambda: (op0(), 6)}
    h_invalid_1 = product(h00, coproduct(product(h0, h1), h2))
    h_invalid_2 = product(h00, coproduct(h1, product(h0, h2)))

    lhs = handler(h_lhs)(f_op2)()
    rhs = handler(h_rhs)(f_op2)()
    invalid_1 = handler(h_invalid_1)(f_op2)()
    invalid_2 = handler(h_invalid_2)(f_op2)()

    assert lhs == rhs
    assert invalid_1 != invalid_2 and invalid_1 != lhs and invalid_2 != lhs
