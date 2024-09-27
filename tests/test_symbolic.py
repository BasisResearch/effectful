import collections
import functools
import logging
import operator
from typing import Annotated, Callable, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.sugar import OPERATORS, Bound, NoDefaultRule, Scoped, gensym
from effectful.ops.core import (
    Expr,
    Interpretation,
    Operation,
    Term,
    ctxof,
    embed,
    evaluate,
    typeof,
    unembed,
)
from effectful.ops.handler import coproduct, fwd, handler

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


add = OPERATORS[operator.add]


@Operation
def App(f: Callable[[S], T], arg: S) -> T:
    raise NoDefaultRule


@Operation
def Lam(var: Annotated[Operation[[], S], Bound()], body: T) -> Callable[[S], T]:
    raise NoDefaultRule


@Operation
def Let(
    var: Annotated[Operation[[], S], Bound(0)],
    val: Annotated[S, Scoped(1)],
    body: Annotated[T, Scoped(0)],
) -> T:
    raise NoDefaultRule


def beta_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    """integer addition"""
    match unembed(x), unembed(y):
        case int(_), int(_):
            return x + y
        case _:
            return fwd(None)


def beta_app(f: Expr[Callable[[S], T]], arg: Expr[S]) -> Expr[T]:
    """beta reduction"""
    match unembed(f), arg:
        case Term(op, (var, body), ()), _ if op == Lam:
            return handler({var: lambda: arg})(evaluate)(unembed(body))  # type: ignore
        case _:
            return fwd(None)


def beta_let(var: Operation[[], S], val: Expr[S], body: Expr[T]) -> Expr[T]:
    """let binding"""
    return handler({var: lambda: val})(evaluate)(unembed(body))  # type: ignore


def eta_lam(var: Operation[[], S], body: Expr[T]) -> Expr[Callable[[S], T]] | Expr[T]:
    """eta reduction"""
    if var not in ctxof(body):  # type: ignore
        return body
    else:
        return fwd(None)


def eta_let(var: Operation[[], S], val: Expr[S], body: Expr[T]) -> Expr[T]:
    """eta reduction"""
    if var not in ctxof(body):  # type: ignore
        return body
    else:
        return fwd(None)


def commute_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match unembed(x), unembed(y):
        case Term(_, _, _), int(_):
            return y + x
        case _:
            return fwd(None)


def assoc_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match unembed(x), unembed(y):
        case _, Term(op, (a, b), ()) if op == add:
            return (x + embed(a)) + embed(b)
        case _:
            return fwd(None)


def unit_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match unembed(x), unembed(y):
        case _, 0:
            return x
        case 0, _:
            return y
        case _:
            return fwd(None)


def sort_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match unembed(x), unembed(y):
        case Term(vx, (), ()), Term(vy, (), ()) if id(vx) > id(vy):
            return y + x
        case Term(add_, (a, Term(vx, (), ())), ()), Term(
            vy, (), ()
        ) if add_ == add and id(vx) > id(vy):
            return (embed(a) + vy()) + vx()
        case _:
            return fwd(None)


alpha_rules: Interpretation = {
    add: add.__default_rule__,
    App: App.__default_rule__,
    Lam: Lam.__default_rule__,
    Let: Let.__default_rule__,
}
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
        alpha_rules,
        eta_rules,
        beta_rules,
        commute_rules,
        assoc_rules,
        unit_rules,
        sort_rules,
    ),
)


def test_lambda_calculus_1():

    x, y = gensym(int), gensym(int)

    with handler(eager_mixed):
        e1 = x() + 1
        f1 = Lam(x, e1)

        assert App(f1, 1) == 2
        assert Lam(y, f1) == f1
        assert Lam(x, unembed(f1).args[1]) == unembed(f1).args[1]

        assert typeof(e1) is int
        assert typeof(f1) is collections.abc.Callable


def test_lambda_calculus_2():

    x, y = gensym(int), gensym(int)

    with handler(eager_mixed):
        f2 = Lam(x, Lam(y, (x() + y())))
        assert App(App(f2, 1), 2) == 3
        assert Lam(y, f2) == f2


def test_lambda_calculus_3():

    x, y, f = gensym(int), gensym(int), gensym(Callable)

    with handler(eager_mixed):
        f2 = Lam(x, Lam(y, (x() + y())))
        app2 = Lam(f, Lam(x, Lam(y, App(App(f(), x()), y()))))
        assert App(App(App(app2, f2), 1), 2) == 3


def test_lambda_calculus_4():

    x, f, g = gensym(int), gensym(Callable), gensym(Callable)

    with handler(eager_mixed):
        add1 = Lam(x, (x() + 1))
        compose = Lam(f, Lam(g, Lam(x, App(f(), App(g(), x())))))
        f1_twice = App(App(compose, add1), add1)
        assert App(f1_twice, 1) == 3


def test_lambda_calculus_5():

    x = gensym(int)

    with handler(eager_mixed):
        e_add1 = Let(x, x(), (x() + 1))
        f_add1 = Lam(x, e_add1)

        assert x in ctxof(e_add1)
        assert unembed(e_add1).args[0] != x

        assert x not in ctxof(f_add1)
        assert unembed(f_add1).args[0] != unembed(f_add1).args[1].args[0]

        assert App(f_add1, 1) == 2
        assert Let(x, 1, e_add1) == 2


def test_arithmetic_1():

    x_, y_ = gensym(int), gensym(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert (1 + 2) + x == x + 3
        assert not (x + 1 == y + 1)
        assert x + 0 == 0 + x == x


def test_arithmetic_2():

    x_, y_ = gensym(int), gensym(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert x + y == y + x
        assert 3 + x == x + 3
        assert 1 + (x + 2) == x + 3
        assert (x + 1) + 2 == x + 3


def test_arithmetic_3():

    x_, y_ = gensym(int), gensym(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert (1 + (y + 1)) + (1 + (x + 1)) == (y + x) + 4
        assert 1 + ((x + y) + 2) == (x + y) + 3
        assert 1 + ((x + (y + 1)) + 1) == (x + y) + 3


def test_arithmetic_4():

    x_, y_ = gensym(int), gensym(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert (
            ((x + x) + (x + x)) + ((x + x) + (x + x))
            == x + (x + (x + (x + (x + (x + (x + x))))))
            == ((((((x + x) + x) + x) + x) + x) + x) + x
        )

        assert (x + y) + (y + x) == (y + (x + x)) + y == y + (x + (y + x))


def test_arithmetic_5():

    x, y = gensym(int), gensym(int)

    with handler(eager_mixed):
        assert Let(x, x() + 3, x() + 1) == x() + 4
        assert Let(x, x() + 3, x() + y() + 1) == y() + x() + 4

        assert Let(x, x() + 3, Let(x, x() + 4, x() + y())) == x() + y() + 7
