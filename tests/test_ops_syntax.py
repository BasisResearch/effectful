import functools
import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Annotated, ClassVar

import pytest

import effectful.handlers.numbers  # noqa: F401
from effectful.ops.semantics import call, evaluate, fvsof, handler, typeof
from effectful.ops.syntax import (
    Scoped,
    _CustomSingleDispatchCallable,
    deffn,
    defop,
    defstream,
    defterm,
    iter_,
    next_,
)
from effectful.ops.types import Operation, Term


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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
    assert str(x1() + x2()) == "add(x(), x!1())"
    assert str(x1() + x1()) == "add(x(), x())"
    assert str(deffn(x1() + x1(), x1)) == "deffn(add(x(), x()), x)"
    assert str(deffn(x1() + x1(), x2)) == "deffn(add(x(), x()), x!1)"
    assert str(deffn(x1() + x2(), x1)) == "deffn(add(x(), x!1()), x)"


def test_defop_singledispatch():
    """Test that defop can be used with singledispatch functions."""

    @defop
    @functools.singledispatch
    def process(x: object) -> object:
        raise NotImplementedError("Unsupported type")

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
            raise NotImplementedError

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


def test_defop_bound_method():
    """Test that defop can be used as a bound method decorator."""

    class MyClass:
        def my_bound_method(self, x: int) -> int:
            raise NotImplementedError

    instance = MyClass()
    my_bound_method_op = defop(instance.my_bound_method)

    assert isinstance(my_bound_method_op, Operation)

    # Test that the bound method can be called with a handler
    with handler({my_bound_method_op: lambda x: x + 1}):
        assert my_bound_method_op(5) == 6


def test_defop_setattr():
    class MyClass:
        def __init__(self, my_op: Operation):
            self.my_op = my_op

    @defop
    def my_op(x: int) -> int:
        raise NotImplementedError

    instance = MyClass(my_op)
    assert isinstance(instance.my_op, Operation)
    assert instance.my_op is my_op

    tm = instance.my_op(5)
    assert isinstance(tm, Term)
    assert isinstance(tm.op, Operation)
    assert tm.op is my_op


def test_defop_setattr_class():
    class MyClass:
        my_op: ClassVar[Operation]

    @defop
    def my_op(x: int) -> int:
        raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
        raise NotImplementedError

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
