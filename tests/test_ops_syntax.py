import collections
import collections.abc
import dataclasses
import functools
import inspect
import logging
import os
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Annotated, ClassVar

import pytest

from docs.source.lambda_ import App, Lam, Let, eager_mixed
from effectful.ops.semantics import evaluate, fvsof, handler, typeof
from effectful.ops.syntax import (
    Scoped,
    _CustomSingleDispatchCallable,
    defdata,
    deffn,
    defop,
    defstream,
    defterm,
    iter_,
    next_,
    syntactic_eq,
    trace,
)
from effectful.ops.types import NotHandled, Operation, Term

logger = logging.getLogger(__name__)

T = typing.TypeVar("T")

call = defdata.dispatch(collections.abc.Callable).__call__


def test_always_fresh():
    x = defop(int)
    y = defop(int)
    assert x != y

    x = defop(int, name="x")
    y = defop(int, name="y")
    assert x != y
    assert x.__name__ == "x"
    assert y.__name__ == "y"

    x1 = defop(int, name="x")
    x2 = defop(int, name="x")
    assert x1 != x2
    assert x1.__name__ == "x"
    assert x2.__name__ == "x"


def f(x: int) -> int:
    return x


def test_gensym_operation():
    def g(x: int) -> int:
        return x

    assert defop(f) != f != defop(f)

    assert defop(f) != defop(g) != defop(f)

    assert defop(f).__name__ == f.__name__
    assert defop(f, name="f2").__name__ == "f2"
    assert str(defop(f)) == f.__name__
    assert str(defop(f, name="f2")) == "f2"


def test_gensym_operation_2():
    @defop
    def op(x: int) -> int:
        return x

    # passing an operation to gensym should return a new operation
    g_op = defop(op)
    assert g_op != defop(g_op) != defop(op, name=op.__name__) != op

    # the new operation should have no default rule
    t = g_op(0)
    assert isinstance(t, Term)
    assert t.op == g_op
    assert t.args == (0,)


def test_gensym_annotations():
    """Test that gensym respects annotations."""

    @defop
    def Lam[S, T, A](
        var: Annotated[Operation[[], S], Scoped[A]],
        body: Annotated[T, Scoped[A]],
    ) -> Callable[[S], T]:
        raise NotHandled

    x = defop(int)
    y = defop(int)
    lam = defop(Lam)

    assert x not in fvsof(Lam(x, x()))
    assert y in fvsof(Lam(x, y()))

    # binding annotations must be preserved for ctxof to work properly
    assert x not in fvsof(lam(x, x()))
    assert y in fvsof(lam(x, y()))


def test_operation_metadata():
    """Test that Operation instances preserve decorated function metadata."""

    def f(x):
        """Docstring for f"""
        return x + 1

    f_op = defop(f)
    ff_op = defop(f)

    assert f.__doc__ == f_op.__doc__
    assert f.__name__ == f_op.__name__
    assert hash(f) == hash(f_op)
    assert f_op != ff_op


def test_scoped_collections():
    """Test that Scoped annotations work with tree-structured collections containing Operations."""

    # Test let_many operation with Mapping[Operation, T]
    @defop
    def let_many[S, T, A, B](
        bindings: Annotated[Mapping[Operation[[], T], T], Scoped[A]],
        body: Annotated[S, Scoped[A | B]],
    ) -> Annotated[S, Scoped[B]]:
        raise NotHandled

    x = defop(int, name="x")
    y = defop(int, name="y")
    z = defop(int, name="z")

    # Variables in bindings should be bound
    bindings = {x: 1, y: 2}
    body = x() + y() + z()
    term = let_many(bindings, body)
    free_vars = fvsof(term)

    new_x = list(term.args[0].keys())[0]
    new_y = list(term.args[0].keys())[1]
    assert new_x == term.args[1].args[0].args[0].op and new_x != x
    assert new_y == term.args[1].args[0].args[1].op and new_y != y

    assert x not in free_vars
    assert y not in free_vars
    assert z in free_vars

    # Test with nested collections
    @defop
    def let_nested[S, T, A, B](
        bindings: Annotated[list[tuple[Operation[[], T], T]], Scoped[A]],
        body: Annotated[S, Scoped[A | B]],
    ) -> Annotated[S, Scoped[B]]:
        raise NotHandled

    w = defop(int, name="w")
    nested_bindings = [(x, 1), (y, 2)]
    term2 = let_nested(nested_bindings, x() + y() + w())
    free_vars2 = fvsof(term2)

    assert x not in free_vars2
    assert y not in free_vars2
    assert w in free_vars2

    # Test empty collections
    empty_bindings = {}
    term3 = let_many(empty_bindings, z())
    free_vars3 = fvsof(term3)

    assert z in free_vars3


def test_no_default_tracing():
    x, y = defop(int), defop(int)

    @defop
    def add(x: int, y: int) -> int:
        raise NotHandled

    def f1(x: int) -> int:
        return add(x, add(y(), 1))

    f1_term = defterm(f1)

    f1_app = call(f1, x())
    f1_term_app = f1_term(x())

    assert y in fvsof(f1_term_app)
    assert y not in fvsof(f1_app)

    assert y not in fvsof(evaluate(f1_app))

    assert isinstance(f1_app, Term)
    assert f1_app.op is call
    assert f1_app.args[0] is f1


def test_term_str():
    x1 = defop(int, name="x")
    x2 = defop(int, name="x")
    x3 = defop(x1)

    assert str(x1) == str(x2) == str(x3) == "x"
    assert repr(x1) != repr(x2) != repr(x3)
    assert str(x1() + x2()) == "__add__(x(), x!1())"
    assert str(x1() + x1()) == "__add__(x(), x())"
    assert str(deffn(x1() + x1(), x1)) == "deffn(__add__(x(), x()), x)"
    assert str(deffn(x1() + x1(), x2)) == "deffn(__add__(x(), x()), x!1)"
    assert str(deffn(x1() + x2(), x1)) == "deffn(__add__(x(), x!1()), x)"


def test_defdata_renaming():
    @defop
    def Let[S, T, A, B](
        var: Annotated[Operation[[], S], Scoped[A]],
        val: Annotated[S, Scoped[B]],
        body: Annotated[T, Scoped[A | B]],
    ) -> Annotated[T, Scoped[B]]:
        raise NotHandled

    x, y = defop(int, name="x"), defop(int, name="y")

    # Constructing the term should rename the bound variable x in the right hand
    # side of the let only.
    let2 = Let(x, y() + x(), x() + y())
    assert let2.args[0] != x
    assert let2.args[1].args[0].op == y
    assert let2.args[1].args[1].op == x
    assert let2.args[2].args[0].op == let2.args[0]
    assert let2.args[2].args[1].op == y


def test_defop_singledispatch():
    """Test that defop can be used with singledispatch functions."""

    @defop
    @functools.singledispatch
    def process(x: object) -> object:
        raise NotHandled("Unsupported type")

    @process.register(int)
    def _(x: int):
        return x + 1

    @process.register(str)
    def _(x: str):
        return x.upper()

    assert process(1) == 2
    assert process("hello") == "HELLO"

    assert process.__signature__ == inspect.signature(process)


def test_defop_customsingledispatch():
    """Test that defop can be used with CustomSingleDispatch functions."""

    @defop
    @_CustomSingleDispatchCallable
    def process(__dispatch: Callable, x: object) -> object:
        return __dispatch(type(x))(x)

    @process.register(int)
    def _(x: int):
        return x + 1

    @process.register(str)
    def _(x: str):
        return x.upper()

    assert process(1) == 2
    assert process("hello") == "HELLO"

    assert process.__signature__ == inspect.signature(process)


def test_defop_method():
    """Test that defop can be used as a method decorator."""

    class MyClass:
        @defop
        def my_method(self, x: int) -> int:
            raise NotHandled

    instance = MyClass()
    term = instance.my_method(5)

    assert isinstance(MyClass.my_method, Operation)

    # check signature
    assert MyClass.my_method.__signature__ == inspect.signature(
        MyClass.my_method._default
    )

    assert isinstance(term, Term)
    assert isinstance(term.op, Operation)
    assert term.op.__name__ == "my_method"
    assert term.args == (
        instance,
        5,
    )
    assert term.kwargs == {}

    # Ensure the operation is unique
    another_instance = MyClass()
    assert instance.my_method is not another_instance.my_method

    # Test that the method can be called with a handler
    with handler({MyClass.my_method: lambda self, x: x + 2}):
        assert instance.my_method(5) == 7
        assert another_instance.my_method(10) == 12


def test_defop_bound_method() -> None:
    """Test that defop can be used as a bound method decorator."""

    class MyClass:
        def my_bound_method(self, x: int) -> int:
            raise NotHandled

    instance = MyClass()
    my_bound_method_op = defop(instance.my_bound_method)

    assert isinstance(my_bound_method_op, Operation)

    # Test that the bound method can be called with a handler
    with handler({my_bound_method_op: lambda x: x + 1}):
        assert my_bound_method_op(5) == 6


def test_defop_setattr() -> None:
    class MyClass:
        def __init__(self, my_op: Operation):
            self.my_op = my_op

    @defop
    def my_op(x: int) -> int:
        raise NotHandled

    instance = MyClass(my_op)
    assert isinstance(instance.my_op, Operation)
    assert instance.my_op is my_op

    tm = instance.my_op(5)
    assert isinstance(tm, Term)
    assert isinstance(tm.op, Operation)
    assert tm.op is my_op


def test_defop_setattr_class() -> None:
    class MyClass:
        my_op: ClassVar[Operation]

    @defop
    def my_op(x: int) -> int:
        raise NotHandled

    MyClass.my_op = my_op

    tm = MyClass.my_op(5)
    assert isinstance(tm, Term)
    assert isinstance(tm.op, Operation)
    assert tm.op is my_op
    assert tm.args == (5,)

    with pytest.raises(TypeError):
        MyClass().my_op(5)


@pytest.mark.xfail(reason="defop does not support classmethod yet")
def test_defop_classmethod():
    """Test that defop can be used as a classmethod decorator."""

    class MyClass:
        @defop
        @classmethod
        def my_classmethod(cls, x: int) -> int:
            raise NotHandled

    term = MyClass.my_classmethod(5)

    assert isinstance(MyClass.my_classmethod, Operation)
    # check signature
    assert MyClass.my_classmethod.__signature__ == inspect.signature(
        MyClass.my_classmethod._default
    )

    assert isinstance(term, Term)
    assert isinstance(term.op, Operation)
    assert term.op.__name__ == "my_classmethod"
    assert term.args == (
        MyClass,
        5,
    )
    assert term.kwargs == {}

    # Ensure the operation is unique
    another_term = MyClass.my_classmethod(10)
    assert term.op is another_term.op

    # Test that the classmethod can be called with a handler
    with handler({MyClass.my_classmethod: lambda cls, x: x + 3}):
        assert MyClass.my_classmethod(5) == 8
        assert MyClass.my_classmethod(10) == 13


def test_defop_staticmethod():
    """Test that defop can be used as a staticmethod decorator."""

    class MyClass:
        @defop
        @staticmethod
        def my_staticmethod(x: int) -> int:
            raise NotHandled

    term = MyClass.my_staticmethod(5)

    assert isinstance(MyClass.my_staticmethod, Operation)
    # check signature
    assert MyClass.my_staticmethod.__signature__ == inspect.signature(
        MyClass.my_staticmethod._default
    )

    assert isinstance(term, Term)
    assert isinstance(term.op, Operation)
    assert term.op.__name__ == "my_staticmethod"
    assert term.args == (5,)
    assert term.kwargs == {}

    # Ensure the operation is unique
    another_term = MyClass.my_staticmethod(10)
    assert term.op is another_term.op

    # Test that the staticmethod can be called with a handler
    with handler({MyClass.my_staticmethod: lambda x: x + 4}):
        assert MyClass.my_staticmethod(5) == 9
        assert MyClass.my_staticmethod(10) == 14


def test_defop_property():
    """Test that defop can be used as a property decorator."""

    class MyClass:
        @defop
        @property
        def my_property(self) -> int:
            raise NotHandled

    instance = MyClass()
    term = instance.my_property

    assert isinstance(MyClass.my_property, Operation)
    assert MyClass.my_property.__signature__ == inspect.signature(
        MyClass.my_property._default
    )

    assert isinstance(term, Term)
    assert isinstance(term.op, Operation)
    assert term.op.__name__ == "my_property"
    assert term.args == (instance,)
    assert term.kwargs == {}

    # Ensure the operation is unique
    another_instance = MyClass()
    assert instance.my_property is not another_instance.my_property

    # Test that the property can be called with a handler
    with handler({MyClass.my_property: lambda self: 42}):
        assert instance.my_property == 42
        assert another_instance.my_property == 42


def test_defop_singledispatchmethod():
    """Test that defop can be used as a singledispatchmethod decorator."""

    class MyClass:
        @defop
        @functools.singledispatchmethod
        def my_singledispatch(self, x: object) -> object:
            raise NotHandled

        @my_singledispatch.register
        def _(self, x: int) -> int:
            return x + 1

        @my_singledispatch.register
        def _(self, x: str) -> str:
            return x + "!"

    class MySubClass(MyClass):
        @MyClass.my_singledispatch.register
        def _(self, x: bool) -> bool:
            return x

    instance = MyClass()
    assert instance.my_singledispatch is not MyClass().my_singledispatch
    assert MySubClass.my_singledispatch is MyClass.my_singledispatch

    term_float = instance.my_singledispatch(1.5)

    assert isinstance(MyClass.my_singledispatch, Operation)
    assert MyClass.my_singledispatch.__signature__ == inspect.signature(
        MyClass.my_singledispatch._default
    )

    assert isinstance(term_float, Term)
    assert term_float.op.__name__ == "my_singledispatch"
    assert term_float.args == (
        instance,
        1.5,
    )
    assert term_float.kwargs == {}

    # Test that the method can be called with a handler
    with handler({MyClass.my_singledispatch: lambda self, x: x + 6}):
        assert instance.my_singledispatch(5) == 11


def test_defdata_iterable():
    @defop
    def cons_iterable(*args: int) -> Iterable[int]:
        raise NotHandled

    tm = cons_iterable(1, 2, 3)
    assert isinstance(tm, Term)
    assert isinstance(tm, Iterable)
    assert issubclass(typeof(tm), Iterable)
    assert tm.op is cons_iterable
    assert tm.args == (1, 2, 3)

    tm_iter = iter(tm)
    assert isinstance(tm_iter, Term)
    assert isinstance(tm_iter, Iterator)
    assert issubclass(typeof(tm_iter), Iterator)
    assert tm_iter.op is iter_

    tm_iter_next = next(tm_iter)
    assert isinstance(tm_iter_next, Term)
    # assert isinstance(tm_iter_next, numbers.Number)  # TODO
    # assert issubclass(typeof(tm_iter_next), numbers.Number)
    assert tm_iter_next.op is next_

    assert list(tm.args) == [1, 2, 3]


def test_defstream_1():
    x = defop(int, name="x")
    y = defop(int, name="y")
    tm = defstream(x() + y(), {x: [1, 2, 3], y: [x() + 1, x() + 2, x() + 3]})

    assert isinstance(tm, Term)
    assert isinstance(tm, Iterable)
    assert issubclass(typeof(tm), Iterable)
    assert tm.op is defstream

    assert x not in fvsof(tm)
    assert y not in fvsof(tm)

    tm_iter = iter(tm)
    assert isinstance(tm_iter, Term)
    assert isinstance(tm_iter, Iterator)
    assert issubclass(typeof(tm_iter), Iterator)
    assert tm_iter.op is iter_

    tm_iter_next = next(tm_iter)
    assert isinstance(tm_iter_next, Term)
    # assert isinstance(tm_iter_next, numbers.Number)  # TODO
    # assert issubclass(typeof(tm_iter_next), numbers.Number)
    assert tm_iter_next.op is next_


def test_eval_dataclass() -> None:
    @dataclasses.dataclass
    class Point:
        x: int
        y: int

    @dataclasses.dataclass
    class Line:
        start: Point
        end: Point

    @dataclasses.dataclass
    class Lines:
        origin: Point
        lines: list[Line]

    x, y = defop(int, name="x"), defop(int, name="y")
    p1 = Point(x(), y())
    p2 = Point(x() + 1, y() + 1)
    line = Line(p1, p2)
    lines = Lines(p1, [line])

    assert {x, y} <= fvsof(lines)

    assert p1 == lines.origin

    with handler({x: lambda: 3, y: lambda: 4}):
        evaluated_lines = evaluate(lines)

    assert isinstance(evaluated_lines, Lines)
    assert evaluated_lines == Lines(
        origin=Point(3, 4),
        lines=[Line(Point(3, 4), Point(4, 5))],
    )


def test_eval_namedtuple() -> None:
    Point = collections.namedtuple("Point", ["x", "y"])
    Line = collections.namedtuple("Line", ["start", "end"])
    Lines = collections.namedtuple("Lines", ["origin", "lines"])

    x, y = defop(int, name="x"), defop(int, name="y")
    p1 = Point(x(), y())
    p2 = Point(x() + 1, y() + 1)
    line = Line(p1, p2)
    lines = Lines(p1, [line])

    assert {x, y} <= fvsof(lines)

    assert p1 == lines.origin

    with handler({x: lambda: 3, y: lambda: 4}):
        evaluated_lines = evaluate(lines)

    assert isinstance(evaluated_lines, Lines)
    assert evaluated_lines == Lines(
        origin=Point(3, 4),
        lines=[Line(Point(3, 4), Point(4, 5))],
    )


def test_lambda_calculus_1():
    x, y = defop(int), defop(int)

    with handler(eager_mixed):
        e1 = x() + 1
        f1 = Lam(x, e1)

        assert syntactic_eq(App(f1, 1), 2)
        assert syntactic_eq(Lam(y, f1), f1)
        assert syntactic_eq(Lam(x, f1.args[1]), f1.args[1])

        assert fvsof(e1) == fvsof(x() + 1)
        assert fvsof(Lam(x, e1).args[1]) != fvsof(Lam(x, e1).args[1])

        assert typeof(e1) is int
        assert typeof(f1) is collections.abc.Callable


def test_lambda_calculus_2():
    x, y = defop(int), defop(int)

    with handler(eager_mixed):
        f2 = Lam(x, Lam(y, (x() + y())))
        assert syntactic_eq(App(App(f2, 1), 2), 3)
        assert syntactic_eq(Lam(y, f2), f2)


def test_lambda_calculus_3():
    x, y, f = (
        defop(int),
        defop(int),
        defop(collections.abc.Callable[[int], collections.abc.Callable[[int], int]]),
    )

    with handler(eager_mixed):
        f2 = Lam(x, Lam(y, (x() + y())))
        app2 = Lam(f, Lam(x, Lam(y, App(App(f(), x()), y()))))
        assert syntactic_eq(App(App(App(app2, f2), 1), 2), 3)


def test_lambda_calculus_4():
    x, f, g = (
        defop(int),
        defop(collections.abc.Callable[[T], T]),
        defop(collections.abc.Callable[[T], T]),
    )

    with handler(eager_mixed):
        add1 = Lam(x, (x() + 1))
        compose = Lam(f, Lam(g, Lam(x, App(f(), App(g(), x())))))
        f1_twice = App(App(compose, add1), add1)
        assert syntactic_eq(App(f1_twice, 1), 3)


def test_lambda_calculus_5():
    x = defop(int)

    with handler(eager_mixed):
        e_add1 = Let(x, x(), (x() + 1))
        f_add1 = Lam(x, e_add1)

        assert x in fvsof(e_add1)
        assert e_add1.args[0] != x

        assert x not in fvsof(f_add1)
        assert f_add1.args[0] != f_add1.args[1].args[0]

        assert syntactic_eq(App(f_add1, 1), 2)
        assert syntactic_eq(Let(x, 1, e_add1), 2)


def test_arithmetic_1():
    x_, y_ = defop(int), defop(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert syntactic_eq((1 + 2) + x, x + 3)
        assert not syntactic_eq(x + 1, y + 1)
        assert syntactic_eq(x + 0, 0 + x) and syntactic_eq(0 + x, x)


def test_arithmetic_2():
    x_, y_ = defop(int), defop(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert syntactic_eq(x + y, y + x)
        assert syntactic_eq(3 + x, x + 3)
        assert syntactic_eq(1 + (x + 2), x + 3)
        assert syntactic_eq((x + 1) + 2, x + 3)


def test_arithmetic_3():
    x_, y_ = defop(int), defop(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert syntactic_eq((1 + (y + 1)) + (1 + (x + 1)), (y + x) + 4)
        assert syntactic_eq(1 + ((x + y) + 2), (x + y) + 3)
        assert syntactic_eq(1 + ((x + (y + 1)) + 1), (x + y) + 3)


def test_arithmetic_4():
    x_, y_ = defop(int), defop(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        expr1 = ((x + x) + (x + x)) + ((x + x) + (x + x))
        expr2 = x + (x + (x + (x + (x + (x + (x + x))))))
        expr3 = ((((((x + x) + x) + x) + x) + x) + x) + x
        assert syntactic_eq(expr1, expr2) and syntactic_eq(expr2, expr3)

        expr4 = (x + y) + (y + x)
        expr5 = (y + (x + x)) + y
        expr6 = y + (x + (y + x))
        assert syntactic_eq(expr4, expr5) and syntactic_eq(expr5, expr6)


def test_arithmetic_5():
    x, y = defop(int), defop(int)

    with handler(eager_mixed):
        assert syntactic_eq(Let(x, x() + 3, x() + 1), x() + 4)
        assert syntactic_eq(Let(x, x() + 3, x() + y() + 1), y() + x() + 4)

        assert syntactic_eq(Let(x, x() + 3, Let(x, x() + 4, x() + y())), x() + y() + 7)


def test_defun_1():
    x, y = defop(int), defop(int)

    with handler(eager_mixed):

        @trace
        def f1(x: int) -> int:
            return x + y() + 1

        assert typeof(f1) is collections.abc.Callable
        assert y in fvsof(f1)
        assert x not in fvsof(f1)

        assert syntactic_eq(f1(1), y() + 2)
        assert syntactic_eq(f1(x()), x() + y() + 1)


def test_defun_2():
    with handler(eager_mixed):

        @trace
        def f1(x: int, y: int) -> int:
            return x + y

        @trace
        def f2(x: int, y: int) -> int:
            @trace
            def f2_inner(y: int) -> int:
                return x + y

            return f2_inner(y)

        assert syntactic_eq(f1(1, 2), 3) and syntactic_eq(f2(1, 2), 3)


def test_defun_3():
    with handler(eager_mixed):

        @trace
        def f2(x: int, y: int) -> int:
            return x + y

        @trace
        def app2(f: collections.abc.Callable[[int, int], int], x: int, y: int) -> int:
            return f(x, y)

        assert syntactic_eq(app2(f2, 1, 2), 3)


@pytest.mark.xfail(condition=os.getenv("CI") == "true", reason="Fails on CI")
def test_defun_4():
    x = defop(int)

    with handler(eager_mixed):

        @trace
        def compose(
            f: collections.abc.Callable[[int], int],
            g: collections.abc.Callable[[int], int],
        ) -> collections.abc.Callable[[int], int]:
            @trace
            def fg(x: int) -> int:
                assert callable(f), f"f is not callable: {f}"
                assert callable(g), f"g is not callable: {g}"
                return f(g(x))

            return fg

        assert callable(compose), f"compose is not callable: {compose}"

        @trace
        def add1(x: int) -> int:
            return x + 1

        assert callable(add1), f"add1 is not callable: {add1}"

        @trace
        def add1_twice(x: int) -> int:
            return compose(add1, add1)(x)

        assert callable(add1_twice), f"add1_twice is not callable: {add1_twice}"

        assert syntactic_eq(add1_twice(1), 3) and syntactic_eq(
            compose(add1, add1)(1), 3
        )
        assert syntactic_eq(add1_twice(x()), x() + 2) and syntactic_eq(
            compose(add1, add1)(x()), x() + 2
        )


def test_defun_5():
    with pytest.raises(ValueError, match="variadic"):
        trace(lambda *xs: None)

    with pytest.raises(ValueError, match="variadic"):
        trace(lambda **ys: None)

    with pytest.raises(ValueError, match="variadic"):
        trace(lambda y=1, **ys: None)

    with pytest.raises(ValueError, match="variadic"):
        trace(lambda x, *xs, y=1, **ys: None)


def test_evaluate_2():
    x = defop(int, name="x")
    y = defop(int, name="y")
    t = x() + y()
    assert isinstance(t, Term)
    with handler({x: lambda: 1, y: lambda: 3}):
        assert evaluate(t) == 4

    t = x() * y()
    assert isinstance(t, Term)
    with handler({x: lambda: 2, y: lambda: 3}):
        assert evaluate(t) == 6

    t = x() - y()
    assert isinstance(t, Term)
    with handler({x: lambda: 2, y: lambda: 3}):
        assert evaluate(t) == -1

    t = x() ^ y()
    assert isinstance(t, Term)
    with handler({x: lambda: 1, y: lambda: 2}):
        assert evaluate(t) == 3


def test_syntactic_eq() -> None:
    l = defop(list[int])()
    assert syntactic_eq("test", "test")
    assert syntactic_eq([1, 2, 3], [1, 2, 3])
    assert syntactic_eq(set([1, 2, 3]), set([1, 2, 3]))
    assert syntactic_eq({"a": 1, "b": 2}, {"b": 2, "a": 1})
    assert syntactic_eq(l, l)
    assert not syntactic_eq(1, defop(int)())
    assert not syntactic_eq(defop(int)(), 1)
    assert not syntactic_eq([], l)
    assert not syntactic_eq(1, [])


def test_arg_positioning():
    @defop
    def f(x):
        raise NotHandled

    assert isinstance(f(0), Term) and isinstance(f(x=0), Term)
    assert f(0).args == f(x=0).args == (0,)
    assert f(0).kwargs == f(x=0).kwargs == {}
