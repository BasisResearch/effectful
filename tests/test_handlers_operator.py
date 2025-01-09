import collections
import functools
import logging
import operator
from typing import Annotated, Callable, TypeVar

import pytest
from typing_extensions import ParamSpec

from effectful.handlers.operator import OPERATORS
from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import Bound, NoDefaultRule, Scoped, defop, defterm
from effectful.ops.types import Expr, Interpretation, Operation, Term

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


add = OPERATORS[operator.add]


@defop
def App(f: Callable[[S], T], arg: S) -> T:
    raise NoDefaultRule


@defop
def Lam(var: Annotated[Operation[[], S], Bound()], body: T) -> Callable[[S], T]:
    raise NoDefaultRule


@defop
def Let(
    var: Annotated[Operation[[], S], Bound(0)],
    val: Annotated[S, Scoped(1)],
    body: Annotated[T, Scoped(0)],
) -> T:
    raise NoDefaultRule


def beta_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    """integer addition"""
    match x, y:
        case int(), int():
            return x + y
        case _:
            return fwd(None)


def beta_app(f: Expr[Callable[[S], T]], arg: Expr[S]) -> Expr[T]:
    """beta reduction"""
    match f, arg:
        case Term(op, (var, body)), _ if op == Lam:
            return handler({var: lambda: arg})(evaluate)(body)  # type: ignore
        case _:
            return fwd(None)


def beta_let(var: Operation[[], S], val: Expr[S], body: Expr[T]) -> Expr[T]:
    """let binding"""
    return handler({var: lambda: val})(evaluate)(body)  # type: ignore


def eta_lam(var: Operation[[], S], body: Expr[T]) -> Expr[Callable[[S], T]] | Expr[T]:
    """eta reduction"""
    if var not in fvsof(body):  # type: ignore
        return body
    else:
        return fwd(None)


def eta_let(var: Operation[[], S], val: Expr[S], body: Expr[T]) -> Expr[T]:
    """eta reduction"""
    if var not in fvsof(body):  # type: ignore
        return body
    else:
        return fwd(None)


def commute_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case Term(), int():
            return y + x  # type: ignore
        case _:
            return fwd(None)


def assoc_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case _, Term(op, (a, b)) if op == add:
            return (x + a) + b  # type: ignore
        case _:
            return fwd(None)


def unit_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case _, 0:
            return x
        case 0, _:
            return y
        case _:
            return fwd(None)


def sort_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case Term(vx, ()), Term(vy, ()) if id(vx) > id(vy):
            return y + x  # type: ignore
        case Term(add_, (a, Term(vx, ()))), Term(vy, ()) if add_ == add and id(vx) > id(
            vy
        ):
            return (a + vy()) + vx()  # type: ignore
        case _:
            return fwd(None)


eta_rules: Interpretation = {
    Lam: eta_lam,
    Let: eta_let,
}
beta_rules: Interpretation = {
    add: beta_add,
    App: beta_app,
    Let: beta_let,
}
commute_rules: Interpretation = {
    add: commute_add,
}
assoc_rules: Interpretation = {
    add: assoc_add,
}
unit_rules: Interpretation = {
    add: unit_add,
}
sort_rules: Interpretation = {
    add: sort_add,
}

eager_mixed = functools.reduce(
    coproduct,
    (
        eta_rules,
        beta_rules,
        commute_rules,
        assoc_rules,
        unit_rules,
        sort_rules,
    ),
)


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

            return f2_inner(y)  # type: ignore

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

            return fg  # type: ignore

        @defterm
        def add1(x: int) -> int:
            return x + 1

        @defterm
        def add1_twice(x: int) -> int:
            return compose(add1, add1)(x)

        assert add1_twice(1) == compose(add1, add1)(1) == 3
        assert add1_twice(x()) == compose(add1, add1)(x()) == x() + 2


def test_defun_5():

    with pytest.raises(NotImplementedError, match="variadic"):
        defterm(lambda *xs: None)

    with pytest.raises(NotImplementedError, match="variadic"):
        defterm(lambda **ys: None)

    with pytest.raises(NotImplementedError, match="variadic"):
        defterm(lambda y=1, **ys: None)

    with pytest.raises(NotImplementedError, match="variadic"):
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
