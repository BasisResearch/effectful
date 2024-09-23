import collections
import logging
from typing import Annotated, Callable, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.sugar import Bound, Scoped, gensym, unembed
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


@Operation
def Add(x: int, y: int) -> int:
    raise NotImplementedError


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
    match x, y:
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
    Add: Add.__free_rule__,
    App: App.__free_rule__,
    Lam: Lam.__free_rule__,
    Let: Let.__free_rule__,
}
lazy: Interpretation = {
    Lam: eta_lam,
    Let: eta_let,
}
eager: Interpretation = {
    Add: eager_add,
    App: eager_app,
    Let: eager_let,
}

eager_mixed = coproduct(free, coproduct(lazy, eager))


def test_lambda_calculus_1():

    x, y = gensym(int), gensym(int)

    with handler(eager_mixed):
        e1 = Add(x(), 1)
        f1 = Lam(x, e1)

        assert App(f1, 1) == 2
        assert Lam(y, f1) == f1
        assert Lam(x, unembed(f1).args[1]) == unembed(f1).args[1]

        assert typeof(e1) is int
        assert typeof(f1) is collections.abc.Callable


def test_lambda_calculus_2():

    x, y = gensym(int), gensym(int)

    with handler(eager_mixed):
        f2 = Lam(x, Lam(y, Add(x(), y())))
        assert App(App(f2, 1), 2) == 3
        assert Lam(y, f2) == f2


def test_lambda_calculus_3():

    x, y, z = gensym(int), gensym(int), gensym(object)

    with handler(eager_mixed):
        f2 = Lam(x, Lam(y, Add(x(), y())))
        app2 = Lam(z, Lam(x, Lam(y, App(App(z(), x()), y()))))
        assert App(App(App(app2, f2), 1), 2) == 3


def test_lambda_calculus_4():

    x, y, z = gensym(int), gensym(object), gensym(object)

    with handler(eager_mixed):
        add1 = Lam(x, Add(x(), 1))
        compose = Lam(x, Lam(y, Lam(z, App(x(), App(y(), z())))))
        f1_twice = App(App(compose, add1), add1)
        assert App(f1_twice, 1) == 3


def test_lambda_calculus_5():

    x = gensym(int)

    with handler(eager_mixed):
        e_add1 = Let(x, x(), Add(x(), 1))
        f_add1 = Lam(x, e_add1)

        assert x in ctxof(e_add1)
        assert unembed(e_add1).args[0] != x

        assert x not in ctxof(f_add1)
        assert unembed(f_add1).args[0] != unembed(f_add1).args[1].args[0]

        assert App(f_add1, 1) == 2
        assert Let(x, 1, e_add1) == 2
