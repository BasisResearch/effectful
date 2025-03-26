import logging
from typing import TypeVar

import jax
import pytest
from typing_extensions import ParamSpec

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import jax_getitem, sizesof, to_array
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

    assert (to_array(actual, i, j) == expected).all()


def test_to_array():
    """Test to_array's handling of free variables and tensor shapes"""
    i, j, k = (
        defop(jax.Array, name="i"),
        defop(jax.Array, name="j"),
        defop(jax.Array, name="k"),
    )
    key = jax.random.PRNGKey(0)
    t = jax.random.normal(key, (2, 3, 4))

    # Test case 1: Converting all named dimensions to positional
    t_ijk = t[i(), j(), k()]
    assert fvsof(t_ijk) >= {i, j, k}

    t1 = to_array(t_ijk, i, j, k)
    assert not (fvsof(t1) & {i, j, k})
    assert t1.shape == (2, 3, 4)

    # Test case 2: Different ordering of dimensions
    t2 = to_array(t_ijk, k, j, i)
    assert not (fvsof(t1) & {i, j, k})
    assert t2.shape == (4, 3, 2)

    # Test case 3: Keeping some dimensions as free variables
    t3 = to_array(t_ijk, i)  # Convert only i to positional
    assert fvsof(t3) >= {j, k}  # j and k remain free
    assert isinstance(t3, Term)
    assert t3.shape == (2,)

    t4 = to_array(t_ijk, i, j)  # Convert i and j to positional
    assert fvsof(t4) >= {k} and not (fvsof(t4) & {i, j})  # only k remains free
    assert isinstance(t4, Term)
    assert t4.shape == (2, 3)

    # Test case 4: Empty order list keeps all variables free
    t5 = to_array(t_ijk)
    assert fvsof(t5) >= {i, j, k}  # All variables remain free
    assert isinstance(t5, Term)
    assert t5.shape == tuple()

    # Test case 5: Verify permuted tensors maintain correct relationships
    t_kji = jnp.permute_dims(t, (2, 1, 0))[k(), j(), i()]
    t6 = to_array(t_kji, i, j, k)
    t7 = to_array(t_ijk, i, j, k)
    assert jnp.allclose(t6, t7)

    # Test case 6: Mixed operations with free variables
    x = jnp.sin(t_ijk)  # Apply operation to indexed tensor
    x1 = to_array(x, i, j)  # Convert some dimensions
    assert fvsof(x1) >= {k}  # k remains free
    assert isinstance(x1, Term)
    assert x1.shape == (2, 3)

    # Test case 7: Multiple tensors sharing variables
    t2_ijk = jax.random.normal(key, (2, 3, 4))[i(), j(), k()]
    sum_t = t_ijk + t2_ijk
    sum1 = to_array(sum_t, i, j)
    assert fvsof(sum1) >= {k}  # k remains free
    assert isinstance(sum1, Term)
    assert sum1.shape == (2, 3)

    # Test case 8: Tensor term with non-sized free variables
    w = defop(jax.Array, name="w")
    t_ijk = t[i(), j(), k()] + w()
    t8 = to_array(t_ijk, i, j, k)
    assert fvsof(t8) >= {w}

    # Test case 9: Eliminate remaining free variables in result
    with handler({w: lambda: jnp.array(1.0)}):
        t9 = evaluate(t8)
    assert not (fvsof(t9) & {i, j, k, w})
