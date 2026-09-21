import collections.abc
import contextlib
import dataclasses
import itertools
import logging
import operator
from collections.abc import Callable, Mapping, MutableSequence
from typing import Annotated, Any, Literal, TypeVar, Union

import pytest

from effectful.ops.semantics import (
    apply,
    coproduct,
    evaluate,
    fvsof,
    fwd,
    handler,
    typeof,
)
from effectful.ops.syntax import (
    ObjectInterpretation,
    Scoped,
    deffn,
    defop,
    implements,
)
from effectful.ops.types import (
    ApplyOperation,
    Interpretation,
    NotHandled,
    Operation,
    Term,
)

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
    return {op: (lambda *args: fwd() * n) for op in ops}


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


@pytest.mark.parametrize(
    "make_intp",
    [
        pytest.param(
            lambda f: coproduct(coproduct({}, {f: lambda x: x}), {f: lambda _: fwd()}),
            id="right-assoc-empty-left",
        ),
        pytest.param(
            lambda f: coproduct(coproduct({f: lambda x: x}, {}), {f: lambda _: fwd()}),
            id="right-assoc-empty-mid",
        ),
        pytest.param(
            lambda f: coproduct({f: lambda x: x}, coproduct({}, {f: lambda _: fwd()})),
            id="left-assoc-empty-mid",
        ),
        pytest.param(
            lambda f: coproduct({f: lambda x: x}, coproduct({f: lambda _: fwd()}, {})),
            id="left-assoc-empty-right",
        ),
    ],
)
def test_coproduct_identity(make_intp) -> None:
    @Operation.define
    def f(x) -> int:
        raise NotHandled

    intp = make_intp(f)
    assert handler(intp)(evaluate)(f(42)) == 42


def test_object_interpretation_inheritance():
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


def test_fwd_from_operation_handler_to_apply_handler():
    @Operation.define
    def f(x: int) -> int:
        return x + 1

    calls = []

    def f_handler(x):
        calls.append(("f", x))
        return fwd()

    def apply_handler(op, *args, **kwargs):
        calls.append(("apply", op, args, kwargs))
        return fwd()

    with handler({apply: apply_handler}), handler({f: f_handler}):
        assert f(1) == 2

    assert calls == [("f", 1), ("apply", f, (1,), {})]


def test_fwd_from_operation_handler_to_apply_handler_with_replacement_args():
    @Operation.define
    def f(x: int) -> int:
        return x

    apply_args = []

    def apply_handler(op, *args, **kwargs):
        apply_args.append((op, args, kwargs))
        return fwd(op, *args, **kwargs)

    with handler({apply: apply_handler}), handler({f: lambda x: fwd(x + 1)}):
        assert f(1) == 2

    assert apply_args == [(f, (2,), {})]


def test_fwd_through_apply_handlers_is_associative():
    calls = []

    @Operation.define
    def f() -> int:
        calls.append("default")
        return 1

    def forwarding(name):
        def impl(*args, **kwargs):
            calls.append(name)
            return fwd()

        return impl

    h0 = {apply: forwarding("left apply")}
    h1 = {f: forwarding("exact")}
    h2 = {apply: forwarding("right apply")}
    expected = ["exact", "right apply", "left apply", "default"]

    for intp in (
        coproduct(coproduct(h0, h1), h2),
        coproduct(h0, coproduct(h1, h2)),
    ):
        calls.clear()
        with handler(intp):
            assert f() == 1
        assert calls == expected

    calls.clear()
    with handler(h0), handler(h1), handler(h2):
        assert f() == 1
    assert calls == expected


def test_fwd_through_apply_operation_subtypes():
    calls = []

    class BaseOperation(Operation):
        pass

    class DerivedOperation(BaseOperation):
        pass

    @DerivedOperation.define
    def f(x: int) -> int:
        calls.append("default")
        return x + 1

    def forwarding(name):
        def impl(*args, **kwargs):
            calls.append(name)
            return fwd()

        return impl

    assert isinstance(Operation.__apply__, ApplyOperation)
    assert isinstance(BaseOperation.__apply__, ApplyOperation)
    assert isinstance(DerivedOperation.__apply__, ApplyOperation)
    assert not isinstance(f, ApplyOperation)

    with handler(
        {
            Operation.__apply__: forwarding("apply"),
            BaseOperation.__apply__: forwarding("base apply"),
            DerivedOperation.__apply__: forwarding("derived apply"),
            f: forwarding("exact"),
        }
    ):
        assert f(1) == 2

    assert calls == ["exact", "derived apply", "base apply", "apply", "default"]

    # Unhandled intermediate apply operations proceed directly to their defaults,
    # so the base apply handler sees the original operation exactly once.
    calls.clear()
    with handler(
        {
            Operation.__apply__: forwarding("apply"),
            f: forwarding("exact"),
        }
    ):
        assert f(1) == 2

    assert calls == ["exact", "apply", "default"]


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


def test_evaluate():
    @defop
    def Nested(*args, **kwargs):
        raise NotHandled

    x = defop(int, name="x")
    y = defop(int, name="y")
    t = Nested([{"a": y()}, x(), (x(), y())], x(), arg1={"b": x()})

    with handler({x: lambda: 1, y: lambda: 2}):
        assert evaluate(t) == Nested([{"a": 2}, 1, (1, 2)], 1, arg1={"b": 1})


def test_memoized_interpretation():
    from effectful.internals.runtime import cache, interpreter

    @defop
    def node(x: object) -> object:
        raise NotHandled

    term = node(node(1))

    class Intp(ObjectInterpretation):
        def __init__(self):
            self.calls = 0

        @implements(apply)
        def _(self, op, *args, **kwargs):
            self.calls += 1
            return (op.__name__, args, kwargs)

    intp = Intp()
    expected = ("node", (("node", (1,), {}),), {})

    # ``evaluate`` installs a cache for the duration of a call when none is
    # active, so results are shared between separate calls only while a scope
    # holds one open.
    with cache():
        assert interpreter(intp)(evaluate)(term) == expected
        assert intp.calls == 2

        # The root cache is checked before its children are traversed, including
        # when evaluation is expressed directly through a handler.
        with interpreter(intp):
            assert evaluate(term) == expected
        assert intp.calls == 2

        # Child results are cached independently and can be reused directly.
        assert interpreter(intp)(evaluate)(term.args[0]) == expected[1][0]
        assert intp.calls == 2

        # A composition has a distinct identity even when its added handler is not
        # used while evaluating this term.
        combined_intp = coproduct(intp, {plus_1: lambda x: x})
        assert interpreter(combined_intp)(evaluate)(term) == expected
        assert intp.calls == 4

        # A separate interpretation has its own cache namespace.
        other_intp = Intp()
        assert interpreter(other_intp)(evaluate)(term) == expected
        assert other_intp.calls == 2


def test_memoized_interpretation_does_not_cache_failures():
    from effectful.internals.runtime import cache, interpreter

    @defop
    def node() -> object:
        raise NotHandled

    term = node()

    class Intp(ObjectInterpretation):
        def __init__(self):
            self.calls = 0

        @implements(apply)
        def _(self, op, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("failed analysis")
            return "success"

    intp = Intp()
    with cache():
        with pytest.raises(ValueError, match="failed analysis"):
            interpreter(intp)(evaluate)(term)

        # The failure left nothing behind, so the retry recomputes; the result of
        # that retry is what gets cached and reused.
        assert interpreter(intp)(evaluate)(term) == "success"
        assert intp.calls == 2
        assert interpreter(intp)(evaluate)(term) == "success"
        assert intp.calls == 2


@pytest.mark.parametrize(
    "build_args",
    [
        lambda x, y: (x(), y()),
        lambda x, y: ([x()], y()),
        lambda x, y: ([x()], [y()]),
        lambda x, y: (([x()], [y()]),),
    ],
)
def test_ctxof(build_args):
    x = defop(object, name="x")
    y = defop(object, name="y")

    @defop
    def Nested(*args, **kwargs):
        raise NotHandled

    term = Nested(*build_args(x, y))
    actual = fvsof(term)
    expected = {x, y, Nested}
    assert actual >= expected


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
    coproduct(i, i)
    evaluate(0, intp=i)

    # include tests with inlined interpretation, because mypy might do inference
    # differently
    handler({f: lambda x: x + 1, g: lambda x, y: x + str(y)})
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


def test_typeof_variadic_typevartuple_operation():
    """An operation may be variadic over a ``TypeVarTuple``.

    Its ``*args`` annotation is an unpacking, which cannot carry the ``Scoped``
    annotation inferred for every other parameter, so the parameter is left
    unannotated and sits in the root scope: it binds nothing, and the argument
    types still reach the return type.
    """

    @defop
    def pack[*Ts](*args: *Ts) -> tuple[*Ts]:
        raise NotHandled

    assert typeof(pack(1, "a"), keep_params=True) == tuple[int, str]
    assert typeof(pack(1), keep_params=True) == tuple[int]
    assert typeof(pack(), keep_params=True) == tuple[()]
    assert typeof(pack(1, "a")) is tuple

    # Nothing is bound by the variadic, so a free variable stays free.
    x = defop(int, name="x")
    assert x in fvsof(pack(x()))


def test_typeof_application_of_polymorphic_callable():
    """Applying a callable term resolves the callee's own type variables.

    The arguments of the call are matched against the components of the ``P`` in
    ``__call__``'s ``self: Callable[P, T]``, so a polymorphic callee is
    instantiated by the call rather than left at its bound.
    """
    S = TypeVar("S")

    mono = defop(Callable[[int], str], name="mono")
    poly = defop(Callable[[S], S], name="poly")
    elem = defop(Callable[[list[S]], S], name="elem")
    curried = defop(Callable[[int], Callable[[str], bool]], name="curried")
    gradual = defop(Callable[..., str], name="gradual")

    assert typeof(mono()(3)) is str
    assert typeof(poly()(3)) is int
    assert typeof(poly()("a")) is str
    assert typeof(elem()([1, 2, 3])) is int

    # The parameter types of an intermediate result survive the first call.
    assert typeof(curried()(1), keep_params=True) == Callable[[str], bool]
    assert typeof(curried()(1)("a")) is bool

    # ``...`` is consistent with any signature, so it constrains nothing.
    assert typeof(gradual()(1, 2, k=3)) is str


def test_typeof_application_conflicting_argument():
    """A conflicting argument is rejected, as it is for an ordinary parameter."""
    mono = defop(Callable[[int], str], name="mono")

    with pytest.raises(TypeError, match="Cannot unify"):
        mono()("a")


def test_typeof_keep_params_generic():
    """``keep_params`` returns the inferred type with its parameters intact."""

    @defop
    def digits(n: int) -> list[int]:
        raise NotHandled

    @defop
    def counts(key: str) -> dict[str, int]:
        raise NotHandled

    assert typeof(digits(1)) is list
    assert typeof(digits(1), keep_params=True) == list[int]

    assert typeof(counts("a")) is dict
    assert typeof(counts("a"), keep_params=True) == dict[str, int]


def test_typeof_keep_params_polymorphic():
    """Parameters resolved from the arguments survive into the returned type."""

    @defop
    def wrap[T](value: T) -> list[T]:
        raise NotHandled

    @defop
    def pair[T, U](key: T, value: U) -> dict[T, U]:
        raise NotHandled

    assert typeof(wrap(1), keep_params=True) == list[int]
    assert typeof(wrap("a"), keep_params=True) == list[str]
    assert typeof(pair(1, "a"), keep_params=True) == dict[int, str]


def test_typeof_keep_params_unresolved_typevar():
    """A parameter the arguments don't determine comes back as the variable itself.

    So ``keep_params=True`` can return something that is not a runtime class,
    where the simplified form always collapses to one.
    """
    S = TypeVar("S")

    @defop
    def unknown(n: int) -> list[S]:
        raise NotHandled

    assert typeof(unknown(1)) is list
    assert typeof(unknown(1), keep_params=True) == list[S]


def test_typeof_type_argument_binds_typevar():
    """An operation taking ``type[T]`` is instantiated by the class it is given.

    The class arrives as a value, so it is ``nested_type`` that has to report it
    as ``type[C]`` for ``T`` to have anything to bind to.
    """

    @defop
    def make[T](cls: type[T]) -> T:
        raise NotHandled

    class Foo:
        def __init__(self, a: int) -> None:
            pass

    assert typeof(make(int)) is int
    assert typeof(make(int), keep_params=True) is int
    assert typeof(make(Foo), keep_params=True) is Foo


def test_typeof_class_argument_to_callable_parameter_is_unconstrained():
    """A class object satisfies a ``Callable`` pattern but constrains nothing.

    Synthesizing the constructor's signature is out of scope, so ``T`` stays
    free rather than picking up ``__init__``'s ``-> None``.
    """
    S = TypeVar("S")

    @defop
    def build(f: Callable[..., S]) -> S:
        raise NotHandled

    class Foo:
        def __init__(self, a: int) -> None:
            pass

    assert typeof(build(Foo), keep_params=True) == S
    assert typeof(build(int), keep_params=True) == S


def test_typeof_keep_params_literal():
    """``keep_params`` skips the collapse of a ``Literal`` to its value type."""

    @defop
    def get_mode() -> Literal["read", "write"]:
        raise NotHandled

    assert typeof(get_mode()) is str
    assert typeof(get_mode(), keep_params=True) == Literal["read", "write"]


def test_typeof_keep_params_values():
    """For a value rather than a term, the parameters come from its contents.

    Note the asymmetry with the term cases above: a value's type is read off the
    value, so it is canonicalized to an abstract base (``MutableSequence``),
    where a term's is read off the annotations that produced it (``list``).
    """
    x = defop(int, name="x")

    assert typeof(1, keep_params=True) is int

    assert typeof([1, 2]) is list
    assert typeof([1, 2], keep_params=True) == MutableSequence[int]

    # A collection of terms is described by the types of its elements.
    assert typeof([x()], keep_params=True) == MutableSequence[int]

    assert typeof(x) is Operation
    assert typeof(x, keep_params=True) == Operation[[], int]


def test_typeof_dataclass_does_not_run_constructor_with_inferred_types():
    @dataclasses.dataclass
    class AbsoluteValue:
        value: int

        def __init__(self, value: Any):
            # This works for both concrete numbers and numeric Terms, but not for
            # the internal boxes used by typeof to represent inferred types.
            self.value = abs(value)

    value = defop(int, name="value")

    assert typeof(AbsoluteValue(value())) is AbsoluteValue


def test_defdata_large(benchmark):
    """Test defdata with large nested operations that form a binary tree of arbitrary size."""
    import random

    from effectful.internals.runtime import cache

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
    def run():
        # A scope per iteration rather than one around ``benchmark``: each round
        # builds fresh objects, so a shared cache would only accumulate entries
        # that can never be hit.
        with cache():
            return build_tree(7)

    benchmark(run)


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
    actual = fvsof(term)
    assert not ({x, y} & actual)
    assert actual >= {z, Lam2, add}


def test_fvsof_collection_binder():
    a, b, c, d = (
        defop(int, name="a"),
        defop(int, name="b"),
        defop(int, name="c"),
        defop(int, name="d"),
    )

    @defop
    def add(x: int, y: int) -> int:
        raise NotHandled

    @defop
    def let_many[A, B](
        body: Annotated[int, Scoped[A | B]],
        bindings: Annotated[dict[Operation[[], int], int], Scoped[A]],
    ) -> Annotated[int, Scoped[B]]:
        raise NotHandled

    term = let_many(add(a(), b()), {a: c(), c: d()})
    actual = fvsof(term)
    assert actual == {b, d, let_many, add}


def test_fvsof_collection_does_not_include_apply():
    x = defop(int, name="x")

    assert fvsof((x(),)) == {x}


def test_interpretation_typing():
    @defop
    def f[T](m: Mapping[Operation, T], x: T) -> T:
        raise NotHandled

    x = defop(int)
    t1 = f({x: x()}, 2)

    assert isinstance(t1, Term) and typeof(t1) == int


def test_typeof_literal():
    """Test typeof with Literal type annotations."""

    @defop
    def get_mode() -> Literal["read", "write"]:
        raise NotHandled

    @defop
    def get_status() -> Literal[200, 404]:
        raise NotHandled

    @defop
    def get_flag() -> Literal[True]:
        raise NotHandled

    @defop
    def bad_definition() -> Literal[()]:
        raise NotHandled

    @defop
    def get_mixed() -> Literal[1, "a"]:
        raise NotHandled

    assert typeof(get_mode()) is str
    assert typeof(get_status()) is int
    assert typeof(get_flag()) is bool

    with pytest.raises(
        TypeError,
        match="Literal annotations must be supplied with at least one argument",
    ):
        bad_definition()

    with pytest.raises(TypeError, match="Union types are not supported"):
        typeof(get_mixed())


@pytest.mark.timeout(20)
def test_evaluate_dag_no_exponential_blowup():
    """A DAG of nested tuples sharing the same Term is O(n), not O(2^n).

    Bounded by a timeout because the regression this guards against does not
    fail, it hangs: losing memoization turns the depth-20 DAG below into 2**20
    evaluations.
    """
    from effectful.internals.runtime import cache

    call_count = 0

    @defop
    def counted() -> int:
        raise NotHandled

    def counted_handler():
        nonlocal call_count
        call_count += 1
        return 42

    # Build a DAG of nested tuples: each level shares the same child object.
    # As a tree this would have 2^depth leaves; as a DAG it's depth+1 objects.
    depth = 20
    node = counted()
    for _ in range(depth):
        node = (node, node)

    # One cache scope spanning both the evaluation and the term construction
    # below. Without it each would install its own, so nothing computed by the
    # first would be available to the second.
    with cache():
        call_count = 0
        with handler({counted: counted_handler}):
            result = evaluate(node)

        deffn(node, counted)(0)

    # The handler should only be called once (the shared Term)
    assert call_count == 1
    # The result should be nested tuples of 42
    leaf = result
    for _ in range(depth):
        assert isinstance(leaf, tuple) and len(leaf) == 2
        assert leaf[0] is leaf[1]  # memoization returns same object
        leaf = leaf[0]
    assert leaf == 42


def test_evaluate_dag_cache_isolation():
    """Different interpretations produce different results for the same expr."""
    x = defop(int, name="x")
    shared = x()
    expr = (shared, shared)

    assert evaluate(expr, intp={x: lambda: 1}) == (1, 1)
    assert evaluate(expr, intp={x: lambda: 99}) == (99, 99)


def test_evaluate_dag_nested_different_intp():
    """evaluate(expr, intp=...) inside a handler gets its own cache."""
    x = defop(int, name="x")
    y = defop(int, name="y")

    shared = x()
    inner_expr = (shared, shared)

    result = evaluate(y(), intp={y: lambda: evaluate(inner_expr, intp={x: lambda: 7})})
    assert result == (7, 7)


def test_evaluate_dag_matches_tree():
    """DAG evaluation produces the same result as evaluating an equivalent tree."""
    x = defop(int, name="x")

    @defop
    def mul(a: int, b: int) -> int:
        raise NotHandled

    shared = x()
    dag = (mul(shared, shared), mul(shared, shared))

    # Equivalent tree with distinct Term objects
    tree = (mul(x(), x()), mul(x(), x()))

    intp = {x: lambda: 3, mul: lambda a, b: a * b}
    assert evaluate(dag, intp=intp) == evaluate(tree, intp=intp) == (9, 9)


def test_fvsof_dataclass() -> None:
    @dataclasses.dataclass
    class A:
        x: int

        def __init__(self, x: int):
            assert x is not None
            self.x = x

    v = Operation.define(int)
    actual = fvsof(A(v()))
    assert actual == {v}


def test_defdata_dataclass_init_effects() -> None:
    @Operation.define
    def f(x: int):
        raise NotHandled

    @dataclasses.dataclass
    class A:
        x: int

        def __init__(self, x: int):
            self.x = f(x)

    @Operation.define
    def g(a: A):
        raise NotHandled

    v = Operation.define(int)
    t = g(A(v()))
    assert isinstance(t.args[0].x, Term)


def test_instanceop_super() -> None:
    class A:
        @Operation.define
        def f(self):
            return "A"

    class B(A):
        @Operation.define
        def f(self):
            return super().f() + " and B"

    assert isinstance(A.f, Operation)
    assert isinstance(A().f, Operation)
    assert isinstance(B.f, Operation)
    assert isinstance(B().f, Operation)

    assert A.f != A().f != B.f != B().f

    assert A().f() == "A"
    assert B().f() == "A and B"
    with handler({A.f: lambda self: "*A*"}):
        assert A().f() == "*A*"
        assert B().f() == "*A* and B"
    with handler({B.f: lambda self: super(B, self).f() + " and *B*"}):
        assert A().f() == "A"
        assert B().f() == "A and *B*"
    b = B()
    with handler({b.f: lambda: "*B*"}):
        assert b.f() == "*B*"


def test_instanceop_dataclass() -> None:
    """Dataclasses with no free variables get instance operations."""

    @dataclasses.dataclass
    class A:
        @Operation.define
        def f(self):
            raise NotHandled

    assert isinstance(A.f, Operation)
    assert isinstance(A().f, Operation)

    @dataclasses.dataclass
    class B:
        x: int

        @Operation.define
        def g(self):
            raise NotHandled

    assert isinstance(B.g, Operation)
    fv = Operation.define(int)()
    assert not isinstance(B(fv).g, Operation)


def test_coproduct_fwd_chain(benchmark):
    """Benchmark coproduct + fwd over a deep chain of forwarding handlers.

    Compose n - 1 interpretations that simply ``fwd()`` on top of a single
    base interpretation that returns 0, then measure the cost of dispatching
    through the whole chain.
    """
    n = 50

    @defop
    def op() -> int:
        raise NotHandled

    base: Interpretation[int, int] = {op: lambda: 0}
    intp = base
    for _ in range(n - 1):
        intp = coproduct(intp, {op: lambda: fwd()})

    def run():
        with handler(intp):
            return op()

    assert run() == 0
    assert benchmark(run) == 0


def test_fwd_in_definition_raises():
    @Operation.define
    def f():
        return fwd()

    with pytest.raises(RuntimeError):
        f()


# ---------------------------------------------------------------------------
# Identity-preserving evaluation.
#
# Evaluating a term hands back the objects it already held wherever nothing
# changed, rather than a structurally equal copy. Everything keyed on node
# identity depends on it: the evaluation cache, and any memo table a client
# builds on top of one.


@defop
def _add(x: int, y: int) -> int:
    raise NotHandled


@defop
def _mul(x: int, y: int) -> int:
    raise NotHandled


_ARITH: Interpretation = {_add: operator.add, _mul: operator.mul}


def _reachable(expr) -> list[Term]:
    """Every distinct :class:`Term` reachable from ``expr``, compared by identity."""
    seen: dict[int, Term] = {}
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Term) and id(node) not in seen:
            seen[id(node)] = node
            stack.extend(node.args)
            stack.extend(node.kwargs.values())
        elif isinstance(node, tuple | list):
            stack.extend(node)
    return list(seen.values())


def test_evaluate_of_an_unchanged_term_is_the_term():
    from effectful.internals.runtime import cache

    x = defop(int, name="x")
    term = _add(x(), _mul(2, 3))

    with cache():
        assert evaluate(term) is term


def test_evaluate_compares_containers_without_their_protocols():
    """Recognizing an unchanged container must not depend on ``len`` or on key equality.

    A value need only support what the rule that rebuilt it used. ``jax``'s
    ``Rotation`` is a tuple subclass whose ``__len__`` raises, and a term used as a
    mapping key has ``__eq__`` and ``__hash__`` that are themselves operations, so
    asking either of them a question builds a term rather than answering it.
    """
    from effectful.internals.runtime import cache

    x = defop(int, name="x")

    class NoLen(tuple):
        def __len__(self):
            raise TypeError("no len")

    key = _add(x(), 1)

    for value in (NoLen((_mul(2, 3),)), {key: _mul(2, 3)}):
        with cache():
            assert evaluate(value) is value


def test_discarded_rebuilds_do_not_accumulate_in_the_cache():
    """Re-evaluating a term in one cache scope leaves nothing behind.

    Evaluating a lambda builds a renamed copy, compares it, and throws it away. The
    copy is typed on the way out, so it briefly holds cache entries of its own. They
    have to go with it: the cache keys terms weakly, so nothing is left once the copy
    is collected. A term that could not be weakly referenced, or a rebuild cached
    against something that outlives it, would turn every re-evaluation into a leak
    for as long as the scope is open.
    """
    import gc

    from effectful.internals.runtime import cache

    def entries(store):
        """Every ``(expression, interpretation)`` pair the store holds.

        Counted rather than just the expressions: a discarded rebuild is reached as
        the *value* of an entry recorded under the throwaway interpretation that built
        it, so counting expressions alone stays flat while the store grows.
        """
        gc.collect()
        return sum(len(inner.data) for inner in store.data.values())

    x, y, z = defop(int, name="x"), defop(int, name="y"), defop(int, name="z")
    # Nested, so that rebuilding the outer lambda renames it while the inner ones
    # keep binders of their own.
    f = deffn(deffn(deffn(_add(_add(x(), y()), z()), z), y), x)

    with cache() as store:
        for _ in range(10):
            assert evaluate(f) is f
        warm = entries(store)

        for _ in range(100):
            assert evaluate(f) is f
        assert entries(store) == warm

        # Nothing in the store is a lambda the result does not contain.
        reachable, stack = set(), [f]
        while stack:
            node = stack.pop()
            if isinstance(node, Term) and id(node) not in reachable:
                reachable.add(id(node))
                stack.extend(node.args)
                stack.extend(node.kwargs.values())
        assert not [
            k
            for k in store
            if isinstance(k, Term) and k.op is deffn and id(k) not in reachable
        ]


def test_evaluate_compares_unordered_containers_without_regard_to_order():
    """Mappings and sets are unordered, so recognizing a rebuild must not read order.

    A rebuilt mapping happens to be built in the original's iteration order today, so
    comparing pairwise would pass while depending on that. Sequences are ordered and
    do compare pairwise.
    """
    from effectful.ops.semantics import _is_rebuild

    x = defop(int, name="x")
    a, b = _add(x(), 1), _add(x(), 2)

    assert _is_rebuild({"p": a, "q": b}, {"q": b, "p": a})
    assert not _is_rebuild({"p": a}, {"p": b})
    assert not _is_rebuild({"p": a}, {"q": a})
    assert not _is_rebuild({"p": a, "q": b}, {"p": a})

    assert _is_rebuild({a, b}, {b, a})
    assert not _is_rebuild({a}, {b})

    assert not _is_rebuild([a, b], [b, a])


def test_evaluate_of_an_unchanged_dataclass_is_the_dataclass():
    from effectful.internals.runtime import cache

    @dataclasses.dataclass(frozen=True)
    class Box:
        inner: object
        tag: str = "t"

    x = defop(int, name="x")
    box = Box(_add(x(), 1))

    with cache():
        assert evaluate(box) is box


def test_beta_reduction_shares_subterms_that_do_not_mention_the_binder():
    from effectful.internals.runtime import cache

    x, y = defop(int, name="x"), defop(int, name="y")
    closed = _mul(y(), 3)  # mentions y, not x
    f = deffn(_add(x(), closed), x)

    with cache():
        result = f(1)

    assert isinstance(result, Term) and result.op is _add
    assert result.args[0] == 1
    assert result.args[1] is closed


def test_beta_reduction_still_substitutes_and_computes_under_handlers():
    from effectful.internals.runtime import cache

    x = defop(int, name="x")
    f = deffn(_add(x(), _mul(2, 3)), x)

    with handler(_ARITH), cache():
        assert f(1) == 7


def test_an_unchanged_lambda_is_handed_back_not_renamed():
    from effectful.internals.runtime import cache

    x = defop(int, name="x")
    # Construction renamed x once; evaluating the result must not rename it again.
    f = deffn(_add(x(), 1), x)

    with cache():
        assert evaluate(f) is f


def test_a_changed_lambda_gets_a_fresh_binder():
    from effectful.internals.runtime import cache

    x, y = defop(int, name="x"), defop(int, name="y")
    g = deffn(deffn(_add(x(), y()), x), y)  # substituting y changes the inner body

    with cache():
        result = g(5)

    assert isinstance(result, Term) and result.op is deffn
    assert result.args[1] is not x and x not in fvsof(result)


def test_substitution_stops_at_a_binder_for_the_substituted_variable():
    """A variable may be free in one part of a term and bound in another.

    Constructing a binder renames it, so reaching this needs the fresh binder taken
    back out of the node that introduced it. Substituting that variable must replace
    the free occurrence and leave the bound one alone.
    """
    from effectful.internals.runtime import cache

    x = defop(int, name="x")
    f = deffn(_mul(x(), 2), x)
    var = f.args[1]  # f's own binder, made fresh when f was built
    term = _add(var(), f)
    assert var in fvsof(term)

    with cache():
        result = evaluate(term, intp={var: lambda: 5})

    assert result.args[0] == 5  # the free occurrence is substituted
    assert result.args[1] is f  # the binder and its body are untouched


def test_substitution_does_not_capture_a_shared_lambdas_binder():
    from effectful.internals.runtime import cache
    from effectful.ops.syntax import defdata

    call = defdata.dispatch(collections.abc.Callable).__call__

    x = defop(int, name="x")
    f = deffn(_mul(x(), 2), x)
    body = _add(x(), call(f, 1))

    with handler(_ARITH), cache():
        assert deffn(body, x)(5) == 7  # 5 + 2; capturing substitution would give 12


def test_analyses_see_past_shadowed_binders():
    x, y = defop(int, name="x"), defop(int, name="y")
    f = deffn(_add(x(), y()), x)

    assert x not in fvsof(f) and y in fvsof(f)
    assert typeof(f) is collections.abc.Callable
    assert typeof(_add(x(), 1)) is int


def test_beta_reduction_under_an_apply_handler_reaches_every_node():
    """Sharing must not hide nodes from an ``apply`` handler.

    An ``apply`` handler is not a term constructor, so none of the shortcuts may fire
    on its behalf: it has to be offered every node, as ``fvsof`` and ``typeof`` are.
    """
    from effectful.internals.runtime import cache

    x, y = defop(int, name="x"), defop(int, name="y")
    f = deffn(_add(x(), _mul(y(), 3)), x)

    class Recording(ObjectInterpretation):
        def __init__(self):
            self.ops: list[Operation] = []

        @implements(apply)
        def _(self, op, *args, **kwargs):
            self.ops.append(op)
            return op.__default_rule__(*args, **kwargs)

    recording = Recording()
    with handler(recording), cache():
        result = f(1)

    assert isinstance(result, Term)
    assert {_add, _mul, y} <= set(recording.ops)


@pytest.mark.timeout(20)
def test_evaluate_dag_under_binders_no_exponential_blowup():
    """A DAG shared beneath binders is evaluated once per node, not once per path.

    Evaluating an operand of a binder adds handlers for the variables bound in it,
    which makes another interpretation. Evaluation is memoized on the identity of the
    interpretation it runs under, so every node below a binder is a miss against what
    was cached for it outside, and a child reached along both operands of ``depth``
    nested binders would be evaluated ``2 ** depth`` times. Two things prevent that:
    the added interpretation is shared between operands that bind the same variables,
    and a term none of those variables can occur in is evaluated under the enclosing
    interpretation instead, where it is already cached.

    ``test_evaluate_dag_no_exponential_blowup`` shares nodes through tuples, which
    bind nothing, so it does not reach this path.
    """
    from effectful.internals.runtime import cache

    call_count = 0

    @defop
    def counted() -> int:
        raise NotHandled

    def counted_handler():
        nonlocal call_count
        call_count += 1
        return 42

    @defop
    def Bind[S, T, A, B](
        var: Annotated[Operation[[], S], Scoped[A]],
        left: Annotated[T, Scoped[A | B]],
        right: Annotated[T, Scoped[A | B]],
    ) -> Annotated[T, Scoped[B]]:
        raise NotHandled

    depth = 20
    node = counted()
    for _ in range(depth):
        node = Bind(defop(int), node, node)  # both operands are the same object

    with cache(), handler({counted: counted_handler}):
        evaluate(node)

    assert call_count == 1


@pytest.mark.timeout(20)
def test_deep_sharing_stays_linear():
    """A chain of lambdas over a shared closed term never copies that term.

    Bounded by a timeout rather than asserted on time: the regression this guards
    against is quadratic growth in both nodes and work, which shows up as the chain
    getting longer.
    """
    from effectful.internals.runtime import cache

    y = defop(int, name="y")
    closed = _mul(y(), 3)

    term = closed
    with cache():
        for _ in range(50):
            x = defop(int, name="x")
            term = deffn(_add(x(), term), x)(1)

    found = [node for node in _reachable(term) if node.op is _mul]
    assert len(found) == 1 and found[0] is closed


# A binding operation defined as a method reaches terms through
# `Operation.__get__`, which binds it to the instance. Such a node is built in two
# steps -- once under the class operation, then re-headed under the bound one -- and
# the sharing above has to survive both.


@dataclasses.dataclass(frozen=True)
class _Folder:
    """A binding operation defined as a method, in the shape of a fold."""

    name: str

    @Operation.define
    def reduce[A, B, U](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Mapping[Operation[[], int], list[int]], Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        raise NotHandled


_TOTAL = _Folder("total")


def _reduction(z: Operation[[], int], body=None):
    return _TOTAL.reduce(_add(z(), 1) if body is None else body, {z: [0, 1, 2]})


def test_a_bound_operation_binds_the_variables_of_its_streams():
    z = defop(int, name="z")
    node = _reduction(z)

    assert node.op is _TOTAL.reduce
    assert z not in fvsof(node)
    assert next(iter(node.args[1])) is not z


def test_evaluate_of_an_unchanged_bound_operation_node_is_the_node():
    from effectful.internals.runtime import cache

    z = defop(int, name="z")
    node = _reduction(z)

    with cache():
        assert [evaluate(node) is node for _ in range(3)] == [True, True, True]


def test_substitution_keeps_a_bound_operation_node_it_does_not_mention():
    from effectful.internals.runtime import cache

    z, s = defop(int, name="z"), defop(int, name="s")
    node = _reduction(z)

    with cache():
        row = deffn((node, s()), s)(7)

    assert row[0] is node and row[1] == 7


def test_a_bound_operation_node_mentioned_twice_stays_one_object():
    from effectful.internals.runtime import cache

    z, s = defop(int, name="z"), defop(int, name="s")
    node = _reduction(z)

    with cache():
        row = deffn((node, node, s()), s)(7)

    assert row[0] is node and row[1] is node


def test_a_bound_operation_node_survives_a_surrounding_binder():
    from effectful.internals.runtime import cache

    z, w = defop(int, name="z"), defop(int, name="w")
    node = _reduction(z)
    f = deffn((node, w()), w)

    with cache():
        assert evaluate(f) is f
        assert f(9)[0] is node


def test_substitution_reaching_inside_a_bound_operation_node_rewrites_it():
    from effectful.internals.runtime import cache

    z, s = defop(int, name="z"), defop(int, name="s")
    node = _reduction(z, body=_add(z(), s()))

    with cache():
        result = deffn(node, s)(5)

    assert result is not node
    assert result.op is _TOTAL.reduce
    assert s not in fvsof(result)


def test_handlers_on_the_class_and_the_bound_operation_both_fire():
    z = defop(int, name="z")
    body, streams = _add(z(), 1), {z: [0, 1, 2]}

    with handler({_Folder.reduce: lambda *a, **k: "class"}):
        assert _TOTAL.reduce(body, streams) == "class"

    with handler({_TOTAL.reduce: lambda *a, **k: "bound"}):
        assert _TOTAL.reduce(body, streams) == "bound"
