import collections
import functools
import logging
import operator
from typing import Callable, TypeVar

import pytest
import torch
from typing_extensions import ParamSpec

from effectful.internals.sugar import OPERATORS, Sized, gensym, torch_getitem
from effectful.ops.core import (
    Expr,
    Interpretation,
    Operation,
    Term,
    as_term,
    ctxof,
    evaluate,
    typeof,
)
from effectful.ops.function import defun, funcall
from effectful.ops.handler import coproduct, fwd, handler

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def test_tpe_1():
    i, j = gensym(Sized(2)), gensym(Sized(3))
    xval, y1_val, y2_val = torch.rand(2, 3), torch.rand(2), torch.rand(3)
    expected = torch.add(torch.add(xval, y1_val[..., None]), y2_val[None])

    x_ij = torch_getitem(xval, (i(), j()))
    x_plus_y1_ij = torch.add(x_ij, torch_getitem(y1_val, (i(),)))
    x_plus_y1_plus_y2_ij = torch.add(x_plus_y1_ij, torch_getitem(y2_val, (j(),)))
    f_actual = defun(x_plus_y1_plus_y2_ij, i, j)
    for ii in range(2):
        for jj in range(3):
            assert f_actual(torch.tensor(ii), torch.tensor(jj)) == expected[ii, jj]


def test_tpe_2():
    xval, ival = torch.rand(2, 3), torch.arange(2)
    expected = torch.sum(xval[ival, :], dim=0)

    i, j = gensym(Sized(2)), gensym(Sized(3))
    x_j = torch_getitem(
        xval,
        (
            ival,
            j(),
        ),
    )
    sum_x_j = torch.sum(x_j, dim=0)
    f_actual = defun(sum_x_j, j)
    for jj in range(3):
        assert f_actual(torch.tensor(jj)) == expected[jj]


def test_tpe_3():
    xval, ival = torch.rand(4, 2, 3), torch.arange(2)
    expected = torch.sum(xval, dim=1)

    i, j, k = gensym(Sized(2)), gensym(Sized(3)), gensym(Sized(4))
    x_j = torch_getitem(
        xval,
        (
            k(),
            ival,
            j(),
        ),
    )
    sum_x_j = torch.sum(x_j, dim=0)
    f_actual = defun(sum_x_j, j, k)
    for jj in range(3):
        for kk in range(4):
            assert f_actual(torch.tensor(jj), torch.tensor(kk)) == expected[kk, jj]


def test_tpe_4():
    xval, ival = torch.rand(4, 2, 3), torch.arange(2)
    expected = torch.sum(xval, dim=1)

    @as_term
    def f_actual(x: torch.Tensor, j: Sized(3), k: Sized(4)) -> torch.Tensor:  # type: ignore
        return torch.sum(x[k, ival, j], dim=0)

    for jj in range(3):
        for kk in range(4):
            assert (
                f_actual(xval, torch.tensor(jj), torch.tensor(kk)) == expected[kk, jj]
            )
