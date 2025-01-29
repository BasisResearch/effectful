from typing import Annotated, Callable, TypeVar

from effectful.ops.semantics import call, evaluate, fvsof
from effectful.ops.syntax import Scoped, defop, defterm, deffn
from effectful.ops.types import Operation, Term
import effectful.handlers.numbers  # noqa: F401


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
    S, T = TypeVar("S"), TypeVar("T")
    A = TypeVar("A")

    @defop
    def Lam(
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
