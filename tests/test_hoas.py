import collections
import functools
import logging
import operator
from typing import Callable, TypeVar

import pytest
from typing_extensions import ParamSpec

from effectful.internals.sugar import OPERATORS, gensym
from effectful.ops.core import (
    Box,
    Interpretation,
    Term,
    ctxof,
    embed,
    hoas,
    typeof,
    unembed,
)
from effectful.ops.function import call, defun
from effectful.ops.handler import coproduct, fwd, handler

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


add = OPERATORS[operator.add]


def beta_add(x: Box[int], y: Box[int]) -> Box[int]:
    match unembed(x), unembed(y):
        case int(x_), int(y_):
            return x_ + y_
        case _:
            return fwd(None)


def commute_add(x: Box[int], y: Box[int]) -> Box[int]:
    match unembed(x), unembed(y):
        case Term(_, _, _), int(y_):
            return y_ + x
        case _:
            return fwd(None)


def assoc_add(x: Box[int], y: Box[int]) -> Box[int]:
    match unembed(x), unembed(y):
        case _, Term(op, (a, b), ()) if op == add:
            return (x + embed(a)) + embed(b)
        case _:
            return fwd(None)


def unit_add(x: Box[int], y: Box[int]) -> Box[int]:
    match unembed(x), unembed(y):
        case _, 0:
            return x
        case 0, _:
            return y
        case _:
            return fwd(None)


def sort_add(x: Box[int], y: Box[int]) -> Box[int]:
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


def test_hoas_1():

    x, y = gensym(int), gensym(int)

    with handler(eager_mixed):

        @hoas
        def f1(x: int) -> int:
            return x + y() + 1

        assert typeof(f1) is collections.abc.Callable
        assert y in ctxof(f1)
        assert x not in ctxof(f1)

        assert f1(1) == y() + 2
        assert f1(x()) == x() + y() + 1


def test_hoas_2():

    with handler(eager_mixed):

        @hoas
        def f1(x: int, y: int) -> int:
            return x + y

        @hoas
        def f2(x: int, y: int) -> int:

            @hoas
            def f2_inner(y: int) -> int:
                return x + y

            return f2_inner(y)

        assert f1(1, 2) == f2(1, 2) == 3


def test_hoas_3():

    with handler(eager_mixed):

        @hoas
        def f2(x: int, y: int) -> int:
            return x + y

        @hoas
        def app2(f: Callable, x: int, y: int) -> int:
            return f(x, y)

        assert app2(f2, 1, 2) == 3


def test_hoas_4():

    x = gensym(int)

    with handler(eager_mixed):

        @hoas
        def compose(
            f: Callable[[int], int], g: Callable[[int], int]
        ) -> Callable[[int], int]:

            @hoas
            def fg(x: int) -> int:
                return f(g(x))

            return fg

        @hoas
        def add1(x: int) -> int:
            return x + 1

        @hoas
        def add1_twice(x: int) -> int:
            return compose(add1, add1)(x)

        assert add1_twice(1) == compose(add1, add1)(1) == 3
        assert add1_twice(x()) == compose(add1, add1)(x()) == x() + 2


def test_hoas_5():

    with pytest.raises(NotImplementedError, match="variadic"):
        hoas(lambda *xs: None)

    with pytest.raises(NotImplementedError, match="variadic"):
        hoas(lambda **ys: None)

    with pytest.raises(NotImplementedError, match="variadic"):
        hoas(lambda y=1, **ys: None)

    with pytest.raises(NotImplementedError, match="variadic"):
        hoas(lambda x, *xs, y=1, **ys: None)
