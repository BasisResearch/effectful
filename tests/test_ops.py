import contextlib
import functools
import itertools
import logging
from typing import TypeVar

import pytest
from typing_extensions import ParamSpec

from effectful.internals.runtime import get_interpretation, interpreter
from effectful.ops.core import Interpretation, Operation, define
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


def block(*ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(lambda r, *a, **k: fwd(r)) for op in ops}


def defaults(*ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: op.default for op in ops}


def times_n_handler(n: int, *ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(lambda r, *args, **kwargs: fwd(r) * n) for op in ops}


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
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    assert op in times_n(n, op)
    assert new_op not in times_n(n, op)

    assert op(*args) * n == times_n(n, op)[op](*args)


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_1(op, args, n):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with interpreter(times_n(n, new_op)):
        assert op(*args) == op.default(*args)
        assert new_op(*args) == (op.default(*args) + 3) * n == (op(*args) + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_2(op, args, n):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with interpreter(times_n(n, op)):
        assert op(*args) == op.default(*args) * n
        assert new_op(*args) == op.default(*args) * n + 3


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_3(op, args, n):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with interpreter(times_n(n, op, new_op)):
        assert op(*args) == op.default(*args) * n
        assert new_op(*args) == (op.default(*args) * n + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_1(op, args, n_outer, n_inner):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with handler(times_n(n_outer, op, new_op), closed=True):
        with handler(times_n(n_inner, op), closed=True):
            assert op(*args) == op.default(*args) * n_inner
            assert new_op(*args) == (op.default(*args) * n_inner + 3) * n_outer


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_2(op, args, n_outer, n_inner):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with handler(times_n(n_outer, op, new_op), closed=True):
        with handler(times_n(n_inner, new_op), closed=True):
            assert op(*args) == op.default(*args) * n_outer
            assert new_op(*args) == (op.default(*args) * n_outer + 3) * n_inner


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_3(op, args, n_outer, n_inner):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with handler(times_n(n_outer, op, new_op), closed=True):
        with handler(times_n(n_inner, op, new_op), closed=True):
            assert op(*args) == op.default(*args) * n_inner
            assert new_op(*args) == (op.default(*args) * n_inner + 3) * n_inner


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", DEPTH_CASES)
def test_op_repeat_nest_interpreter(op, args, n, depth):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    intp = times_n(n, new_op)
    with contextlib.ExitStack() as stack:
        for _ in range(depth):
            stack.enter_context(handler(intp, closed=True))

        assert op(*args) == op.default(*args)
        assert new_op(*args) == intp[new_op](*args)


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", DEPTH_CASES)
def test_op_fail_nest_interpreter(op, args, n, depth):
    def _fail_op(*args: int) -> int:
        raise ValueError("oops")

    fail_op = define(Operation)(_fail_op)
    intp = times_n(n, op, fail_op)

    with pytest.raises(ValueError, match="oops"):
        try:
            with contextlib.ExitStack() as stack:
                for _ in range(depth):
                    stack.enter_context(handler(intp, closed=True))

                try:
                    fail_op(*args)
                except ValueError as e:
                    assert op(*args) == op.default(*args) * n
                    raise e
        except ValueError as e:
            assert op(*args) == op.default(*args)
            raise e


@pytest.mark.parametrize("op,args", OPERATION_CASES)
def test_affine_continuation(op, args):
    def f():
        return op(*args)

    h_twice = {op: bind_result(lambda r, *a, **k: fwd(fwd(r)))}

    assert (
        interpreter(defaults(op))(f)()
        == interpreter(coproduct(defaults(op), h_twice))(f)()
        == interpreter(product(defaults(op), h_twice))(f)()
    )


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_coproduct_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, op)

    intp1 = coproduct(h0, coproduct(h1, h2))
    intp2 = coproduct(coproduct(h0, h1), h2)

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()


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


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_coproduct_commute_orthogonal(op, args, n1, n2):
    def f():
        return op(*args) + new_op(*args)

    new_op = define(Operation)(lambda *args: op(*args) + 3)

    h0 = defaults(op, new_op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, new_op)

    intp1 = coproduct(h0, coproduct(h1, h2))
    intp2 = coproduct(h0, coproduct(h2, h1))

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_handler_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, op)

    expected = interpreter(coproduct(h0, coproduct(h1, h2)))(f)()

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

    @contextlib.contextmanager
    def sextuple_as_double_triple(mode):
        interp = {sextuple: lambda v: double(triple(v))}
        if mode == "runner":
            with interpreter(product(get_interpretation(), interp)):
                yield
        elif mode == "handler":
            with handler(interp):
                yield

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


def test_runner_outer_reflect():
    def plus_two_calling_plus_one():
        def plus_minus_one_then_reflect(v):
            r = plus_1(v)
            return fwd(r - 1)

        return {plus_2: plus_minus_one_then_reflect}

    defs = {plus_1: plus_1.default, plus_2: plus_2.default}
    with interpreter(product(defs, plus_two_calling_plus_one())):
        assert plus_1(1) == 2
        assert plus_2(2) == 4


def test_runner_outer_reflect_1():
    @bind_result
    def plus_two_impl_inner(r, v):
        assert r is None
        r = plus_1(v)
        return fwd(r + 1)

    @bind_result
    def plus_two_impl_outer(res, v):
        return res or v + 2

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
        with interpreter(product(intp_outer, intp_inner)):
            assert plus_2(2) == 2 + (2 + 3) + 1


def test_runner_outer_reflect_2():
    log = []

    def logging(fn):
        def wrapped(*a, **k):
            log.append(fn.__name__)
            return fn(*a, **k)

        return wrapped

    def check_log(*expected):
        assert log == list(expected)
        log.clear()

    @bind_result
    @logging
    def plus_two_impl_inner(r, v):
        return f"+2-impl_inner({v})"

    @bind_result
    @logging
    def plus_two_impl_outer(res, v):
        return plus_1(f"+2-impl_outer({v})")

    @bind_result
    @logging
    def plus_one_impl_inner(res, v):
        assert res is None
        return f"+1-impl_inner({plus_2(v)})"

    @bind_result
    @logging
    def plus_one_even_outer(res, v):
        return f"+1-even_outer({v})"

    intp_even_more_outer = {plus_1: plus_one_even_outer}
    intp_inner = {plus_1: plus_one_impl_inner, plus_2: plus_two_impl_inner}
    intp_outer = {plus_2: plus_two_impl_outer}

    with interpreter(intp_inner):
        assert plus_2(2) == "+2-impl_inner(2)"
        check_log("plus_two_impl_inner")

    with interpreter(product(intp_even_more_outer, product(intp_outer, intp_inner))):
        assert plus_1(1) == "+1-impl_inner(+1-even_outer(+2-impl_outer(1)))"
        check_log(
            "plus_one_impl_inner",
            "plus_two_impl_outer",
            "plus_one_even_outer",
        )
        assert plus_2(2) == "+2-impl_inner(2)"
        check_log("plus_two_impl_inner")
