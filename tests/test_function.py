import collections
import functools
import logging
import operator
from typing import Callable, TypeVar

import pytest
from typing_extensions import ParamSpec

from effectful.internals.sugar import OPERATORS, gensym
from effectful.ops.core import Expr, Interpretation, Operation, Term, as_term, ctxof, evaluate, typeof
from effectful.ops.function import defun, funcall
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
            return (x + a) + b
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
            return y + x
        case Term(add_, (a, Term(vx, (), ())), ()), Term(
            vy, (), ()
        ) if add_ == add and id(vx) > id(vy):
            return (a + vy()) + vx()
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


def test_tpe_1():
    import torch
    from effectful.internals.sugar import TORCH_OPS, Sized

    getitem = TORCH_OPS[torch.ops.aten.index]

    with handler(eager_mixed):
        i, j = gensym(Sized(2)), gensym(Sized(3))
        xval, y1_val, y2_val = torch.rand(2, 3), torch.rand(2), torch.rand(3)
        expected = torch.add(torch.add(xval, y1_val[..., None]), y2_val[None])

        x_ij = getitem(xval, (i(), j()))
        x_plus_y1_ij = torch.add(x_ij, getitem(y1_val, (i(),)))
        x_plus_y1_plus_y2_ij = torch.add(x_plus_y1_ij, getitem(y2_val, (j(),)))
        f_actual = defun(x_plus_y1_plus_y2_ij, i, j)
        for ii in range(2):
            for jj in range(3):
                assert f_actual(torch.tensor(ii), torch.tensor(jj)) == expected[ii, jj]


def test_tpe_2():
    import torch
    from effectful.internals.sugar import TORCH_OPS, Sized

    getitem = TORCH_OPS[torch.ops.aten.index]

    with handler(eager_mixed):
        xval, ival = torch.rand(2, 3), torch.arange(2)
        expected = torch.sum(xval[ival, :], dim=0)

        i, j = gensym(Sized(2)), gensym(Sized(3))
        x_j = getitem(xval, (ival, j(),))
        sum_x_j = torch.sum(x_j, dim=0)
        f_actual = defun(sum_x_j, j)
        for jj in range(3):
            assert f_actual(torch.tensor(jj)) == expected[jj]


def test_tpe_3():
    import torch
    from effectful.internals.sugar import TORCH_OPS, Sized

    getitem = TORCH_OPS[torch.ops.aten.index]

    with handler(eager_mixed):
        xval, ival = torch.rand(4, 2, 3), torch.arange(2)
        expected = torch.sum(xval, dim=1)

        i, j, k = gensym(Sized(2)), gensym(Sized(3)), gensym(Sized(4))
        x_j = getitem(xval, (k(), ival, j(),))
        sum_x_j = torch.sum(x_j, dim=0)
        f_actual = defun(sum_x_j, j, k)
        for jj in range(3):
            for kk in range(4):
                assert f_actual(torch.tensor(jj), torch.tensor(kk)) == expected[kk, jj]


def test_tpe_4():
    import torch
    from effectful.internals.sugar import Sized

    with handler(eager_mixed):
        xval, ival = torch.rand(4, 2, 3), torch.arange(2)
        expected = torch.sum(xval, dim=1)

        @as_term
        def f_actual(x: torch.Tensor, j: Sized(3), k: Sized(4)) -> torch.Tensor:  # type: ignore
            return torch.sum(x[k, ival, j], dim=0)

        for jj in range(3):
            for kk in range(4):
                assert f_actual(xval, torch.tensor(jj), torch.tensor(kk)) == expected[kk, jj]
