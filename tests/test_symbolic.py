import collections
import logging
import operator
from typing import Annotated, Callable, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.sugar import OPERATORS, Bound, Scoped, embed, gensym, unembed
from effectful.ops.core import (
    Expr,
    Interpretation,
    Operation,
    Term,
    ctxof,
    evaluate,
    typeof,
)
from effectful.ops.handler import coproduct, fwd, handler

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


add = OPERATORS[operator.add]


@Operation
def App(f: Callable[[S], T], arg: S) -> T:
    raise NotImplementedError


@Operation
def Lam(var: Annotated[Operation[[], S], Bound()], body: T) -> Callable[[S], T]:
    raise NotImplementedError


@Operation
def Let(
    var: Annotated[Operation[[], S], Bound(0)],
    val: Annotated[S, Scoped(1)],
    body: Annotated[T, Scoped(0)],
) -> T:
    raise NotImplementedError


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


def eager_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    """integer addition"""
    match unembed(x), unembed(y):
        case int(_), int(_):
            return x + y
        case _:
            return fwd(None)


def eager_app(f: Expr[Callable[[S], T]], arg: Expr[S]) -> Expr[T]:
    """beta reduction"""
    match unembed(f), arg:
        case Term(op, (var, body), ()), _ if op == Lam:
            return handler({var: lambda: arg})(evaluate)(body)  # type: ignore
        case _:
            return fwd(None)


def eager_let(var: Operation[[], S], val: Expr[S], body: Expr[T]) -> Expr[T]:
    """let binding"""
    return handler({var: lambda: val})(evaluate)(body)  # type: ignore


free: Interpretation = {
    add: add.__default_rule__,
    App: App.__default_rule__,
    Lam: Lam.__default_rule__,
    Let: Let.__default_rule__,
}
lazy: Interpretation = {
    Lam: eta_lam,
    Let: eta_let,
}
eager: Interpretation = {
    add: eager_add,
    App: eager_app,
    Let: eager_let,
}

eager_mixed = coproduct(free, coproduct(lazy, eager))


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

    x, y, z = gensym(int), gensym(int), gensym(object)

    with handler(eager_mixed):
        f2 = Lam(x, Lam(y, (x() + y())))
        app2 = Lam(z, Lam(x, Lam(y, App(App(z(), x()), y()))))
        assert App(App(App(app2, f2), 1), 2) == 3


def test_lambda_calculus_4():

    x, y, z = gensym(int), gensym(object), gensym(object)

    with handler(eager_mixed):
        add1 = Lam(x, (x() + 1))
        compose = Lam(x, Lam(y, Lam(z, App(x(), App(y(), z())))))
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

    def simplify_add(x, y):
        match unembed(x), unembed(y):
            case Term(_, _, _), int(_):
                return y + x
            case _, Term(_, (a, b), ()):
                return (x + embed(a)) + embed(b)
            case _:
                return fwd(None)

    def unit_add(x, y):
        match unembed(x), unembed(y):
            case _, 0:
                return x
            case 0, _:
                return y
            case _:
                return fwd(None)

    eager_mixed = coproduct(
        coproduct({add: add.__default_rule__}, {add: eager_add}),
        coproduct({add: simplify_add}, {add: unit_add}),
    )

    x_, y_ = gensym(int), gensym(int)
    x, y = x_(), y_()

    with handler(eager_mixed):
        assert (1 + 2) + x == x + 3

        assert not (x + 1 == y + 1)

        assert 3 + x == x + 3
        assert 1 + (x + 2) == x + 3
        assert (x + 1) + 2 == x + 3
        assert (1 + (y + 1)) + (1 + (x + 1)) == (y + x) + 4

        assert 1 + ((x + y) + 2) == (x + y) + 3
        assert 1 + ((x + (y + 1)) + 1) == (x + y) + 3

        assert (
            ((x + x) + (x + x)) + ((x + x) + (x + x))
            == x + (x + (x + (x + (x + (x + (x + x))))))
            == ((((((x + x) + x) + x) + x) + x) + x) + x
        )

        assert x + 0 == 0 + x == x
