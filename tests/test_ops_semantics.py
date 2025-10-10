import contextlib
import functools
import itertools
import logging
from collections.abc import Callable
from typing import Annotated, Any, Union

import pytest

from effectful.ops.semantics import (
    coproduct,
    evaluate,
    fvsof,
    fwd,
    handler,
    product,
    runner,
    typeof,
)
from effectful.ops.syntax import ObjectInterpretation, Scoped, deffn, defop, implements
from effectful.ops.types import Interpretation, NotHandled, Operation

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def closed_handler[S, T](intp: Interpretation[S, T]):
    from effectful.internals.runtime import get_interpretation, interpreter

    with interpreter(coproduct({}, {**get_interpretation(), **intp})):
        yield intp


@defop
def plus_1(x: int) -> int:
    return x + 1


@defop
def plus_2(x: int) -> int:
    return x + 2


@defop
def times_plus_1(x: int, y: int) -> int:
    return x * y + 1


def times_n(n: int, *ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: (lambda *args: (fwd() * n)) for op in ops}


OPERATION_CASES = (
    [[plus_1, (i,)] for i in range(5)]
    + [[plus_2, (i,)] for i in range(5)]
    + [[times_plus_1, (i, j)] for i, j in itertools.product(range(5), range(5))]
)
N_CASES = [1, 2, 3]
DEPTH_CASES = [1, 2, 3]


@pytest.mark.parametrize("op,args", OPERATION_CASES)
def test_op_default(op, args):
    assert op(*args) == op.__default_rule__(*args)


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_times_n_interpretation(op, args, n):
    new_op = defop(lambda *args: op(*args) + 3)

    assert op in times_n(n, op)
    assert new_op not in times_n(n, op)

    with handler(times_n(n, op)):
        assert op(*args) == op.__default_rule__(*args) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_register_new_op(op, args, n):
    new_op = defop(lambda *args: op(*args) + 3)
    intp = times_n(n, op)

    with closed_handler(intp):
        new_value = new_op(*args)
        assert new_value == op.__default_rule__(*args) * n + 3

        intp[new_op] = times_n(n, new_op)[new_op]
        assert new_op(*args) == new_value

    with closed_handler(intp):
        assert new_op(*args) == (op.__default_rule__(*args) * n + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_1(op, args, n):
    new_op = defop(lambda *args: op(*args) + 3)

    with closed_handler(times_n(n, new_op)):
        assert op(*args) == op.__default_rule__(*args)
        assert (
            new_op(*args) == (op.__default_rule__(*args) + 3) * n == (op(*args) + 3) * n
        )


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_2(op, args, n):
    new_op = defop(lambda *args: op(*args) + 3)

    with closed_handler(times_n(n, op)):
        assert op(*args) == op.__default_rule__(*args) * n
        assert new_op(*args) == op.__default_rule__(*args) * n + 3


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_3(op, args, n):
    new_op = defop(lambda *args: op(*args) + 3)

    with closed_handler(times_n(n, op, new_op)):
        assert op(*args) == op.__default_rule__(*args) * n
        assert new_op(*args) == (op.__default_rule__(*args) * n + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_1(op, args, n_outer, n_inner):
    new_op = defop(lambda *args: op(*args) + 3)

    with closed_handler(times_n(n_outer, op, new_op)):
        with closed_handler(times_n(n_inner, op)):
            assert op(*args) == op.__default_rule__(*args) * n_inner
            assert new_op(*args) == (op.__default_rule__(*args) * n_inner + 3) * n_outer


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_2(op, args, n_outer, n_inner):
    new_op = defop(lambda *args: op(*args) + 3)

    with closed_handler(times_n(n_outer, op, new_op)):
        with closed_handler(times_n(n_inner, new_op)):
            assert op(*args) == op.__default_rule__(*args) * n_outer
            assert new_op(*args) == (op.__default_rule__(*args) * n_outer + 3) * n_inner


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_3(op, args, n_outer, n_inner):
    new_op = defop(lambda *args: op(*args) + 3)

    with closed_handler(times_n(n_outer, op, new_op)):
        with closed_handler(times_n(n_inner, op, new_op)):
            assert op(*args) == op.__default_rule__(*args) * n_inner
            assert new_op(*args) == (op.__default_rule__(*args) * n_inner + 3) * n_inner


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", DEPTH_CASES)
def test_op_repeat_nest_interpreter(op, args, n, depth):
    new_op = defop(lambda *args: op(*args) + 3)

    intp = times_n(n, new_op)
    with contextlib.ExitStack() as stack:
        for _ in range(depth):
            stack.enter_context(closed_handler(intp))

        # intp does not bind op, so it should execute unchanged
        assert op(*args) == op.__default_rule__(*args)

        # however, intp does bind new_op, so it should execute with the new rule
        assert new_op(*args) == (op(*args) + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", DEPTH_CASES)
def test_op_fail_nest_interpreter(op, args, n, depth):
    def _fail_op(*args: int) -> int:
        raise ValueError("oops")

    fail_op = defop(_fail_op)
    intp = times_n(n, op, fail_op)

    with pytest.raises(ValueError, match="oops"):
        try:
            with contextlib.ExitStack() as stack:
                for _ in range(depth):
                    stack.enter_context(closed_handler(intp))

                try:
                    fail_op(*args)
                except ValueError as e:
                    assert op(*args) == op.__default_rule__(*args) * n
                    raise e
        except ValueError as e:
            assert op(*args) == op.__default_rule__(*args)
            raise e


def test_object_interpretation_inheretance():
    @defop
    def op1():
        return "op1"

    @defop
    def op2():
        return "op2"

    @defop
    def op3():
        return "op3"

    @defop
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
    with closed_handler(my_handler):
        assert op1() == "MyHandler.op1_impl"
        assert op2() == "MyHandler.op2_impl"
        assert op3() == "MyHandler.an_op_impl"
        assert op4() == "MyHandler.another_op_impl"

    my_handler_subclass = MyHandlerSubclass()
    with closed_handler(my_handler_subclass):
        assert op1() == "MyHandlerSubclass.op1_impl"
        assert op2() == "MyHandlerSubclass.another_op2_impl"
        assert op3() == "MyHandlerSubclass.another_op_impl"
        assert op4() == "MyHandler.another_op_impl"


def defaults(*ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: op.__default_rule__ for op in ops}  # type: ignore


def test_fwd_simple():
    def plus_1_fwd(x):
        # do nothing and just fwd
        return fwd()

    with handler({plus_1: plus_1_fwd}):
        assert plus_1(1) == 2


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_compose_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n(n1, op)
    h2 = times_n(n2, op)

    intp1 = coproduct(h0, coproduct(h1, h2))
    intp2 = coproduct(coproduct(h0, h1), h2)

    assert handler(intp1)(f)() == handler(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_compose_commute_orthogonal(op, args, n1, n2):
    def f():
        return op(*args) + new_op(*args)

    new_op = defop(lambda *args: op(*args) + 3)

    h0 = defaults(op, new_op)
    h1 = times_n(n1, op)
    h2 = times_n(n2, new_op)

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
    h1 = times_n(n1, op)
    h2 = times_n(n2, op)

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
            stack.enter_context(handler(times_n(n, op)))

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

    @defop
    def do_stuff():
        return "default stuff"

    def do_more_stuff():
        return fwd() + " and more"

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
    @defop
    def do_stuff():
        raise NotHandled

    @defop
    def do_other_stuff():
        return "other stuff"

    h_outer = {
        do_stuff: lambda: "default stuff",
        do_other_stuff: lambda: fwd() + " and more " + do_stuff(),
    }
    h_inner = {do_stuff: lambda: "fancy " + do_other_stuff()}
    h_topmost = {do_stuff: lambda: "should not be called"}

    with handler(product(h_topmost, product(h_outer, h_inner))):
        assert do_stuff() == "fancy other stuff and more default stuff"


@defop
def op0():
    raise NotHandled


@defop
def op1():
    raise NotHandled


@defop
def op2():
    raise NotHandled


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
    h2 = {op2: lambda: (fwd(), op0(), op1(), 2)}

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


def test_evaluate():
    @defop
    def Nested(*args, **kwargs):
        raise NotHandled

    x = defop(int, name="x")
    y = defop(int, name="y")
    t = Nested([{"a": y()}, x(), (x(), y())], x(), arg1={"b": x()})

    with handler({x: lambda: 1, y: lambda: 2}):
        assert evaluate(t) == Nested([{"a": 2}, 1, (1, 2)], 1, arg1={"b": 1})


def test_ctxof():
    x = defop(object)
    y = defop(object)

    @defop
    def Nested(*args, **kwargs):
        raise NotHandled

    assert fvsof(Nested(x(), y())) >= {x, y}
    assert fvsof(Nested([x()], y())) >= {x, y}
    assert fvsof(Nested([x()], [y()])) >= {x, y}
    assert fvsof(Nested((x(), y()))) >= {x, y}


def test_handler_typing() -> None:
    """This test is for the type checker; it doesn't do anything interesting
    when run.

    """

    @defop
    def f(x: int) -> int:
        raise NotHandled

    @defop
    def g(x: str, y: bool) -> str:
        return "test" if y else x

    # Note: this annotation is required. Without annotation, mypy joins the two
    # operator types to `object`.
    i: Interpretation = {f: lambda x: x + 1, g: lambda x, y: x + str(y)}

    handler(i)
    runner(i)
    product(i, i)
    coproduct(i, i)
    evaluate(0, intp=i)

    # include tests with inlined interpretation, because mypy might do inference
    # differently
    handler({f: lambda x: x + 1, g: lambda x, y: x + str(y)})
    runner({f: lambda x: x + 1, g: lambda x, y: x + str(y)})
    product(
        {f: lambda x: x + 1, g: lambda x, y: x + str(y)},
        {f: lambda x: x + 1, g: lambda x, y: x + str(y)},
    )
    coproduct(
        {f: lambda x: x + 1, g: lambda x, y: x + str(y)},
        {f: lambda x: x + 1, g: lambda x, y: x + str(y)},
    )
    evaluate(0, intp={f: lambda x: x + 1, g: lambda x, y: x + str(y)})


def test_typeof_basic():
    """Test typeof with basic operations that have simple return types."""

    @defop
    def add(x: int, y: int) -> int:
        raise NotHandled

    @defop
    def is_positive(x: int) -> bool:
        raise NotHandled

    @defop
    def get_name() -> str:
        raise NotHandled

    assert typeof(add(1, 2)) is int
    assert typeof(is_positive(5)) is bool
    assert typeof(get_name()) is str


def test_typeof_nested():
    """Test typeof with nested operations."""

    @defop
    def add(x: int, y: int) -> int:
        raise NotHandled

    @defop
    def multiply(x: int, y: int) -> int:
        raise NotHandled

    @defop
    def is_even(x: int) -> bool:
        raise NotHandled

    assert typeof(add(multiply(2, 3), 4)) is int
    assert typeof(is_even(add(1, 2))) is bool


def test_typeof_polymorphic():
    """Test typeof with operations that have polymorphic return types."""

    @defop
    def identity[T](x: T) -> T:
        raise NotHandled

    @defop
    def first[T, U](x: T, y: U) -> T:
        raise NotHandled

    @defop
    def if_then_else[T](cond: bool, then_val: T, else_val: T) -> T:
        raise NotHandled

    assert typeof(identity(42)) is int
    assert typeof(identity("hello")) is str
    assert typeof(first(42, "hello")) is int
    assert typeof(first("hello", 42)) is str
    assert typeof(if_then_else(True, 42, 43)) is int
    assert typeof(if_then_else(False, "hello", "world")) is str


def test_typeof_none():
    """Test typeof with operations that return None."""

    @defop
    def do_nothing() -> None:
        raise NotHandled

    @defop
    def print_value(x: Any) -> None:
        raise NotHandled

    assert typeof(do_nothing()) is type(None)
    assert typeof(print_value(42)) is type(None)


def test_typeof_scoped():
    """Test typeof with operations that have scoped annotations."""

    @defop
    def Lambda[S, T, A, B](
        var: Annotated[Operation[[], S], Scoped[A]], body: Annotated[T, Scoped[A | B]]
    ) -> Annotated[Callable[[S], T], Scoped[B]]:
        raise NotHandled

    x = defop(int, name="x")

    # Lambda that adds 1 to its argument
    lambda_term = Lambda(x, x() + 1)
    assert typeof(lambda_term) is Callable


def test_typeof_no_annotations():
    """Test typeof with operations that lack type annotations."""

    @defop
    def untyped_op(x, y):
        raise NotHandled

    @defop
    def partially_typed_op(x: int, y):
        raise NotHandled

    # Without annotations, the default is object
    assert typeof(untyped_op(1, 2)) is object
    assert typeof(partially_typed_op(1, 2)) is object


@pytest.mark.xfail(reason="Union types are not yet supported")
def test_typeof_union():
    """Test typeof with union types."""

    @defop
    def maybe_int(b: bool) -> int | str:
        raise NotHandled

    # Union types are simplified to their origin type
    assert typeof(maybe_int(True)) is Union


@pytest.mark.xfail(reason="Union types are not yet supported")
def test_typeof_optional():
    """Test typeof with Optional types."""

    @defop
    def maybe_value(b: bool) -> int | None:
        raise NotHandled

    # Optional[int] is Union[int, None], so it simplifies to Union
    assert typeof(maybe_value(True)) is Union


def test_typeof_generic():
    """Test typeof with generic classes."""

    class Box[T]:
        def __init__(self, value: T):
            self.value = value

    @defop
    def box_value[T](x: T) -> Box[T]:
        raise NotHandled

    # Generic types are simplified to their origin type
    assert typeof(box_value(42)) is Box


def test_defdata_large(benchmark):
    """Test defdata with large nested operations that form a binary tree of arbitrary size."""
    import random

    @defop
    def f[T, A, B](
        v: Annotated[Operation[[], int], Scoped[A]],
        x: Annotated[T, Scoped[A | B]],
        y: Annotated[T, Scoped[A | B]],
    ) -> Annotated[T, Scoped[B]]:
        """Generic operation that takes two arguments of the same type and returns that type."""
        raise NotHandled

    def build_tree(depth: int) -> Any:
        """
        Recursively build a binary tree of f operations with the specified depth.

        Args:
            depth: The depth of the tree (0 means just a leaf)
            leaf_type: The type of values at the leaves (int, str, etc.)
            start_value: The starting value for leaf generation

        Returns:
            A nested tree of f operations with leaves of the specified type
        """
        if depth == 0:
            if random.random() < 0.5:
                return 0
            else:
                return defop(int)()

        # Recursively build left and right subtrees
        left = build_tree(depth - 1)
        right = build_tree(depth - 1)

        return f(defop(int), left, right)

    # Test a very large tree (depth 8 = 255 leaf nodes)
    benchmark(functools.partial(build_tree, 7))


def test_evaluate_deep():
    x, y, z = defop(int), defop(int), defop(int)
    intp = {x: deffn(1), y: deffn(2), z: deffn(x() + y())}

    with handler(intp):
        assert z() == 3

    assert handler(intp)(z)() == 3

    assert evaluate(evaluate(z(), intp=intp), intp=intp) == 3

    assert evaluate(z(), intp=intp) == 3


def test_fvsof_binder():
    x, y, z = defop(int, name="x"), defop(int, name="y"), defop(int, name="z")

    @defop
    def add(a: int, b: int) -> int:
        raise NotHandled

    @defop
    def Lam2[A, B](
        body: Annotated[int, Scoped[A | B]],
        var1: Annotated[Operation[[], int], Scoped[A]],
        var2: Annotated[Operation[[], int], Scoped[A]],
    ) -> Annotated[Callable[[int, int], int], Scoped[B]]:
        raise NotHandled

    term = Lam2(add(x(), add(y(), z())), x, y)
    assert not {x, y} <= fvsof(term)
    assert fvsof(term) == {z, Lam2, add}
