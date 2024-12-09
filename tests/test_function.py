import collections
import functools
import logging
import operator
from typing import Callable, TypeVar

import pytest
from typing_extensions import ParamSpec

from effectful.internals.sugar import OPERATORS
from effectful.ops.core import (
    Expr,
    Interpretation,
    Term,
    as_term,
    ctxof,
    gensym,
    typeof,
)
from effectful.ops.handler import coproduct, fwd, handler

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


add = OPERATORS[operator.add]


def beta_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case int(x_), int(y_):
            return x_ + y_
        case _:
            return fwd(None)


def commute_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case Term(_, _, _), int(y_):
            return y_ + x  # type: ignore
        case _:
            return fwd(None)


def assoc_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case _, Term(op, (a, b), ()) if op == add:
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
        case Term(vx, (), ()), Term(vy, (), ()) if id(vx) > id(vy):
            return y + x  # type: ignore
        case Term(add_, (a, Term(vx, (), ())), ()), Term(
            vy, (), ()
        ) if add_ == add and id(vx) > id(vy):
            return (a + vy()) + vx()  # type: ignore
        case _:
            return fwd(None)


alpha_rules: Interpretation = {
    add: add.__default_rule__,
}
beta_rules: Interpretation = {
    add: beta_add,
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
        beta_rules,
        commute_rules,
        assoc_rules,
        unit_rules,
        sort_rules,
    ),
)


def test_defun_1():

    x, y = gensym(int), gensym(int)

    with handler(eager_mixed):

        @as_term
        def f1(x: int) -> int:
            return x + y() + 1

        assert typeof(f1) is collections.abc.Callable
        assert y in ctxof(f1)
        assert x not in ctxof(f1)

        assert f1(1) == y() + 2
        assert f1(x()) == x() + y() + 1


def test_defun_2():

    with handler(eager_mixed):

        @as_term
        def f1(x: int, y: int) -> int:
            return x + y

        @as_term
        def f2(x: int, y: int) -> int:

            @as_term
            def f2_inner(y: int) -> int:
                return x + y

            return f2_inner(y)  # type: ignore

        assert f1(1, 2) == f2(1, 2) == 3


def test_defun_3():

    with handler(eager_mixed):

        @as_term
        def f2(x: int, y: int) -> int:
            return x + y

        @as_term
        def app2(f: Callable, x: int, y: int) -> int:
            return f(x, y)

        assert app2(f2, 1, 2) == 3


def test_defun_4():

    x = gensym(int)

    with handler(eager_mixed):

        @as_term
        def compose(
            f: Callable[[int], int], g: Callable[[int], int]
        ) -> Callable[[int], int]:

            @as_term
            def fg(x: int) -> int:
                return f(g(x))

            return fg  # type: ignore

        @as_term
        def add1(x: int) -> int:
            return x + 1

        @as_term
        def add1_twice(x: int) -> int:
            return compose(add1, add1)(x)

        assert add1_twice(1) == compose(add1, add1)(1) == 3
        assert add1_twice(x()) == compose(add1, add1)(x()) == x() + 2


def test_defun_5():

    with pytest.raises(NotImplementedError, match="variadic"):
        as_term(lambda *xs: None)

    with pytest.raises(NotImplementedError, match="variadic"):
        as_term(lambda **ys: None)

    with pytest.raises(NotImplementedError, match="variadic"):
        as_term(lambda y=1, **ys: None)

    with pytest.raises(NotImplementedError, match="variadic"):
        as_term(lambda x, *xs, y=1, **ys: None)
