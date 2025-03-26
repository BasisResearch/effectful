import logging
from typing import TypeVar

import jax
import pytest
from typing_extensions import ParamSpec

from effectful.handlers.jax import jax_getitem, sizesof, to_tensor
from effectful.ops.semantics import evaluate, fvsof, handler
from effectful.ops.syntax import deffn, defop, defterm
from effectful.ops.types import Term

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def test_tpe_1():
    i, j = defop(jax.Array), defop(jax.Array)
    key = jax.random.PRNGKey(0)
    xval, y1_val, y2_val = (
        jax.random.normal(key, (2, 3)),
        jax.random.normal(key, (2)),
        jax.random.normal(key, (3)),
    )

    expected = xval + y1_val[..., None] + y2_val[None]

    x_ij = xval[i(), j()]
    x_plus_y1_ij = x_ij + y1_val[i()]
    actual = x_plus_y1_ij + y2_val[j()]

    assert actual.op == jax_getitem
    assert set(a.op for a in actual.args[1]) == {i, j}
    assert actual.shape == ()
    assert actual.size == 1
    assert actual.ndim == 0

    assert (to_tensor(actual, i, j) == expected).all()
