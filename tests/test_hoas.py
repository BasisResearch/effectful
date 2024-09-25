import collections
import functools
import logging
import operator
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.sugar import (
    OPERATORS,
    Bound,
    Scoped,
    call,
    defun,
    embed,
    gensym,
    hoas,
    unembed,
)
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


def beta_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match unembed(x), unembed(y):
        case int(_), int(_):
            return x + y
        case _:
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

        @hoas
        def f1(x: int) -> int:
            return x + y() + 1

        assert typeof(f1) is collections.abc.Callable
        assert y in ctxof(f1)
        assert x not in ctxof(f1)

        assert f1(1) == y() + 2
        assert f1(x()) == x() + y() + 1
