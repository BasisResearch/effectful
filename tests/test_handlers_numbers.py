import collections
import logging

import pytest

from docs.source.lambda_ import App, Lam, Let, eager_mixed
from effectful.ops.semantics import evaluate, fvsof, handler, typeof
from effectful.ops.syntax import defop, defterm
from effectful.ops.types import Term

logger = logging.getLogger(__name__)


def test_lambda_calculus_1():

    x, y = defop(int), defop(int)

    with handler(eager_mixed):
        e1 = x() + 1
        f1 = Lam(x, e1)

        assert App(f1, 1) == 2
        assert Lam(y, f1) == f1
        assert Lam(x, f1.args[1]) == f1.args[1]

        assert fvsof(e1) == fvsof(x() + 1)
        assert fvsof(Lam(x, e1).args[1]) != fvsof(Lam(x, e1).args[1])

        assert typeof(e1) is int
        assert typeof(f1) is collections.abc.Callable


def test_lambda_calculus_2():

    x, y = defop(int), defop(int)

    with handler(eager_mixed):
        f2 = Lam(x, Lam(y, (x() + y())))
        assert App(App(f2, 1), 2) == 3
        assert Lam(y, f2) == f2


def test_lambda_calculus_3():

    x, y, f = defop(int), defop(int), defop(collections.abc.Callable)

    with handler(eager_mixed):
        f2 = Lam(x, Lam(y, (x() + y())))
        app2 = Lam(f, Lam(x, Lam(y, App(App(f(), x()), y()))))
        assert App(App(App(app2, f2), 1), 2) == 3


def test_lambda_calculus_4():

    x, f, g = (
        defop(int),
        defop(collections.abc.Callable),
        defop(collections.abc.Callable),
    )

    with handler(eager_mixed):
        add1 = Lam(x, (x() + 1))
        compose = Lam(f, Lam(g, Lam(x, App(f(), App(g(), x())))))
        f1_twice = App(App(compose, add1), add1)
        assert App(f1_twice, 1) == 3


def test_lambda_calculus_5():

    x = defop(int)

    with handler(eager_mixed):
        e_add1 = Let(x, x(), (x() + 1))
        f_add1 = Lam(x, e_add1)

        assert x in fvsof(e_add1)
        assert e_add1.args[0] != x

        assert x not in fvsof(f_add1)
        assert f_add1.args[0] != f_add1.args[1].args[0]

        assert App(f_add1, 1) == 2
        assert Let(x, 1, e_add1) == 2


def test_arithmetic_1():

    x_, y_ = defop(int), defop(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert (1 + 2) + x == x + 3
        assert not (x + 1 == y + 1)
        assert x + 0 == 0 + x == x


def test_arithmetic_2():

    x_, y_ = defop(int), defop(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert x + y == y + x
        assert 3 + x == x + 3
        assert 1 + (x + 2) == x + 3
        assert (x + 1) + 2 == x + 3


def test_arithmetic_3():

    x_, y_ = defop(int), defop(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert (1 + (y + 1)) + (1 + (x + 1)) == (y + x) + 4
        assert 1 + ((x + y) + 2) == (x + y) + 3
        assert 1 + ((x + (y + 1)) + 1) == (x + y) + 3


def test_arithmetic_4():

    x_, y_ = defop(int), defop(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert (
            ((x + x) + (x + x)) + ((x + x) + (x + x))
            == x + (x + (x + (x + (x + (x + (x + x))))))
            == ((((((x + x) + x) + x) + x) + x) + x) + x
        )

        assert (x + y) + (y + x) == (y + (x + x)) + y == y + (x + (y + x))


def test_arithmetic_5():

    x, y = defop(int), defop(int)

    with handler(eager_mixed):
        assert Let(x, x() + 3, x() + 1) == x() + 4
        assert Let(x, x() + 3, x() + y() + 1) == y() + x() + 4

        assert Let(x, x() + 3, Let(x, x() + 4, x() + y())) == x() + y() + 7


def test_defun_1():

    x, y = defop(int), defop(int)

    with handler(eager_mixed):

        @defterm
        def f1(x: int) -> int:
            return x + y() + 1

        assert typeof(f1) is collections.abc.Callable
        assert y in fvsof(f1)
        assert x not in fvsof(f1)

        assert f1(1) == y() + 2
        assert f1(x()) == x() + y() + 1


def test_defun_2():

    with handler(eager_mixed):

        @defterm
        def f1(x: int, y: int) -> int:
            return x + y

        @defterm
        def f2(x: int, y: int) -> int:

            @defterm
            def f2_inner(y: int) -> int:
                return x + y

            return f2_inner(y)

        assert f1(1, 2) == f2(1, 2) == 3


def test_defun_3():

    with handler(eager_mixed):

        @defterm
        def f2(x: int, y: int) -> int:
            return x + y

        @defterm
        def app2(f: collections.abc.Callable, x: int, y: int) -> int:
            return f(x, y)

        assert app2(f2, 1, 2) == 3


def test_defun_4():

    x = defop(int)

    with handler(eager_mixed):

        @defterm
        def compose(
            f: collections.abc.Callable[[int], int],
            g: collections.abc.Callable[[int], int],
        ) -> collections.abc.Callable[[int], int]:

            @defterm
            def fg(x: int) -> int:
                return f(g(x))

            return fg

        @defterm
        def add1(x: int) -> int:
            return x + 1

        @defterm
        def add1_twice(x: int) -> int:
            return compose(add1, add1)(x)

        assert add1_twice(1) == compose(add1, add1)(1) == 3
        assert add1_twice(x()) == compose(add1, add1)(x()) == x() + 2


def test_defun_5():

    with pytest.raises(ValueError, match="variadic"):
        defterm(lambda *xs: None)

    with pytest.raises(ValueError, match="variadic"):
        defterm(lambda **ys: None)

    with pytest.raises(ValueError, match="variadic"):
        defterm(lambda y=1, **ys: None)

    with pytest.raises(ValueError, match="variadic"):
        defterm(lambda x, *xs, y=1, **ys: None)


def test_evaluate_2():
    x = defop(int, name="x")
    y = defop(int, name="y")
    t = x() + y()
    assert isinstance(t, Term)
    assert t.op.__name__ == "add"
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
