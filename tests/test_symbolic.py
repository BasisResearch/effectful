import functools
import logging
import operator
from typing import Annotated, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.runtime import get_runtime, interpreter
from effectful.internals.sugar import Bound, Box, defop
from effectful.ops.core import Operation, Term, evaluate, gensym
from effectful.ops.handler import coproduct, fwd, handler

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def test_lazy_addition():

    add = defop(operator.add)

    def eager_add(x, y):
        match x, y:
            case int(x_), int(y_):
                return x_ + y_
            case _:
                return fwd(None)

    eager = {add: eager_add}

    def simplify_add(x, y):
        match x, y:
            case Term(_, _, _), int(_):
                return y + Box(x)
            case _, Term(_, (a, b), ()):
                return (Box(x) + a) + b
            case _:
                return fwd(None)

    simplify_assoc_commut = {add: simplify_add}

    def unit_add(x, y):
        match x, y:
            case _, 0:
                return x
            case 0, _:
                return y
            case _:
                return fwd(None)

    simplify_unit = {add: unit_add}

    lazy = {add: lambda x, y: Term(add, (x, y), ())}
    mixed = coproduct(lazy, eager)
    simplified = coproduct(simplify_assoc_commut, simplify_unit)
    mixed_simplified = coproduct(mixed, simplified)

    x_, y_ = gensym(int), gensym(int)
    x, y = Box(x_()), Box(y_())
    one, two, three = Box(1), Box(2), Box(3)

    with interpreter(eager):
        assert one + two == three

    with interpreter(lazy):
        assert one + two == Term(add, (one, two), ())
        assert add(one, add(two, three)) == Term(
            add, (one, Term(add, (two, three), ())), ()
        )
        assert x + y == Term(add, (x, y), ())
        assert not (x + 1 == y + 1)

    with interpreter(mixed):
        assert one + two == three
        assert (one + two) + x == 3 + x

    with interpreter(mixed_simplified):
        assert one + two == three
        assert (one + two) + x == x + 3

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


def test_lambda_calculus():
    @defop
    def Add(x: int, y: int) -> int:
        raise NotImplementedError

    @defop
    def App(f: Term, arg: Term) -> Term:
        raise NotImplementedError

    @defop
    def Lam(var: Annotated[Term, Bound()], body: Term) -> Term:
        raise NotImplementedError

    def eager_add(x, y):
        """integer addition"""
        match x, y:
            case int(_), int(_):
                return x + y
            case _:
                return fwd(None)

    def eager_app(f: Term, arg: Term | int):
        """beta reduction"""
        match f, arg:
            case Term(op, (var, body), ()), _ if op == Lam:
                return handler({var: lambda: arg})(evaluate)(body)  # type: ignore
            case _:
                return fwd(None)

    def eta_lam(var: Operation, body: Term):
        """eta reduction"""
        if var not in interpreter(get_runtime()._JUDGEMENTS)(evaluate)(body).context:
            return body
        else:
            return fwd(None)

    free = {
        op: functools.partial(lambda op, *a, **k: Term(op, a, tuple(k.items())), op)
        for op in (Add, App, Lam)
    }
    eager = {Add: eager_add, App: eager_app, Lam: eta_lam}
    eager_mixed = coproduct(free, eager)

    x, y, z = gensym(object), gensym(object), gensym(object)
    one, two, three = 1, 2, 3

    with interpreter(eager_mixed):
        f1 = Lam(x, Add(x(), one))
        assert handler({x: lambda: one})(evaluate)(f1) == f1
        assert handler({y: lambda: one})(evaluate)(f1) == f1
        assert App(f1, one) == two
        assert Lam(y, f1) == f1

        f2 = Lam(x, Lam(y, Add(x(), y())))
        assert App(App(f2, one), two) == three
        assert Lam(y, f2) == f2

        app2 = Lam(z, Lam(x, Lam(y, App(App(z(), x()), y()))))
        assert App(App(App(app2, f2), one), two) == three

        compose = Lam(x, Lam(y, Lam(z, App(x(), App(y(), z())))))
        f1_twice = App(App(compose, f1), f1)
        assert App(f1_twice, one) == three
