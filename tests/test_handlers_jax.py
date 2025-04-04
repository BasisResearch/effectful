from typing import TypeVar

import jax
import pytest
from typing_extensions import ParamSpec

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import jax_getitem, jit, to_array
from effectful.ops.semantics import evaluate, fvsof, handler
from effectful.ops.syntax import deffn, defop, defterm
from effectful.ops.types import Term

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


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
    assert fvsof(actual) >= {i, j}
    assert actual.shape == ()
    assert actual.size == 1
    assert actual.ndim == 0

    assert (to_array(actual, i, j) == expected).all()


def test_tpe_2():
    key = jax.random.PRNGKey(0)
    xval = jax.random.normal(key, (2, 3))
    ival = jnp.arange(2)
    expected = jnp.sum(xval[ival, :], axis=0)

    j = defop(jax.Array)
    x_j = xval[ival, j()]

    assert x_j.shape == (2,)
    actual = jnp.sum(x_j, axis=0)

    assert actual.op == jax_getitem
    assert fvsof(actual) >= {j}
    assert actual.shape == ()
    assert actual.size == 1

    assert (to_array(actual, j) == expected).all()


def test_tpe_3():
    key = jax.random.PRNGKey(0)
    xval = jax.random.normal(key, (4, 2, 3))
    ival = jnp.arange(2)
    expected = jnp.sum(xval, axis=1)

    j, k = defop(jax.Array), defop(jax.Array)
    x_j = xval[k(), ival, j()]
    actual = jnp.sum(x_j, axis=0)

    assert actual.op == jax_getitem
    assert fvsof(actual) >= {j, k}
    assert actual.shape == ()
    assert actual.size == 1

    assert (to_array(actual, k, j) == expected).all()


def test_tpe_known_index():
    """Constant indexes are partially evaluated away."""
    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")

    cases = [
        jnp.ones((2, 3))[i(), 1],
        jnp.ones((2, 3))[0, i()],
        jnp.ones((2, 3, 4))[0, i(), 1],
        jnp.ones((2, 3, 4))[0, i(), j()],
        jnp.ones((2, 3, 4))[i(), j(), 3],
    ]

    for case_ in cases:
        assert all(isinstance(a, Term) for a in case_.args[1])
        assert not any(isinstance(a, int) for a in case_.args[1])


def test_tpe_constant_eval():
    """Constant indexes are partially evaluated away."""
    height, width = (
        defop(jax.Array, name="height"),
        defop(jax.Array, name="width"),
    )
    t = jnp.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
    A = t[height(), width()]

    layer = defop(jax.Array, name="layer")
    with handler(
        {
            height: lambda: layer() // jnp.array(3),
            width: lambda: layer() % jnp.array(3),
        }
    ):
        A_layer = evaluate(A)
    with handler({layer: lambda: jnp.array(2)}):
        A_final = evaluate(A_layer)

    assert not isinstance(A_final, Term)


def test_tpe_stack():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    xval = jax.random.normal(key1, (10, 5))
    yval = jax.random.normal(key2, (10, 5))

    i = defop(jax.Array)
    j = defop(jax.Array)
    x_ij = xval[i(), j()]
    y_ij = yval[i(), j()]
    actual = jnp.stack((x_ij, y_ij))
    assert actual.shape == (2,)
    assert (
        jnp.transpose(to_array(actual, i, j), [2, 0, 1]) == jnp.stack((xval, yval))
    ).all()


INDEXING_CASES = [
    # Simple integer indexing
    (jax.random.normal(jax.random.PRNGKey(0), (4, 5, 6)), (0,)),
    # Simple slice indexing
    (jax.random.normal(jax.random.PRNGKey(1), (4, 5, 6)), (slice(1, 3),)),
    # Advanced indexing with arrays
    (jax.random.normal(jax.random.PRNGKey(2), (4, 5, 6)), (jnp.array([0, 2]),)),
    (
        jax.random.normal(jax.random.PRNGKey(3), (4, 5, 6)),
        (jnp.array([0, 2]), slice(None), jnp.array([0, 2])),
    ),
    # Mixed indexing
    (
        jax.random.normal(jax.random.PRNGKey(4), (4, 5, 6)),
        (slice(None), jnp.array([1, 3]), 2),
    ),
    # Indexing with None (newaxis)
    (
        jax.random.normal(jax.random.PRNGKey(5), (4, 5, 6)),
        (None, slice(None), None, slice(1, 3)),
    ),
    # Indexing with Ellipsis
    (
        jax.random.normal(jax.random.PRNGKey(6), (4, 5, 6, 7)),
        (Ellipsis, jnp.array([1, 3])),
    ),
    # Integer and array indexing
    (jax.random.normal(jax.random.PRNGKey(7), (4, 5, 6)), (2, jnp.array([1, 3, 4]))),
    # Indexing with negative indices
    (jax.random.normal(jax.random.PRNGKey(8), (4, 5, 6)), (-1,)),
    # Indexing with step in slice (currently supports only slice(None))
    # (jax.random.normal(jax.random.PRNGKey(9), (4, 5, 6)), (slice(None, None, 2),)),
    # Indexing with empty array
    (
        jax.random.normal(jax.random.PRNGKey(10), (4, 5, 6)),
        (jnp.array([], dtype=jnp.int32),),
    ),
    # Complex mixed indexing
    (
        jax.random.normal(jax.random.PRNGKey(11), (4, 5, 6)),
        (slice(None), jnp.array([0, 2]), None, Ellipsis),
    ),
    # Indexing with multiple None
    (
        jax.random.normal(jax.random.PRNGKey(12), (4, 5, 6)),
        (None, None, 1, slice(None), None),
    ),
    # Additional complex cases
    (
        jax.random.normal(jax.random.PRNGKey(13), (4, 5, 6)),
        (jnp.array([[0, 1], [2, 3]]), jnp.array([[1, 2], [3, 4]]), slice(None)),
    ),
    (
        jax.random.normal(jax.random.PRNGKey(14), (4, 5, 6)),
        (Ellipsis, None, jnp.array([0, 2])),
    ),
    (
        jax.random.normal(jax.random.PRNGKey(15), (4, 5, 6)),
        (jnp.arange(4)[..., None, None],),
    ),
    (
        jax.random.normal(jax.random.PRNGKey(16), (4, 5, 6)),
        (jnp.arange(4)[..., None, None], None, slice(None)),
    ),
    (
        jax.random.normal(jax.random.PRNGKey(17), (4, 5, 6)),
        (None, jnp.arange(4)[..., None, None], None, slice(None)),
    ),
    (
        jax.random.normal(jax.random.PRNGKey(18), (4, 5, 6)),
        (jnp.arange(4)[..., None, None], jnp.arange(5)[..., None]),
    ),
    (
        jax.random.normal(jax.random.PRNGKey(19), (4, 5, 6)),
        (jnp.arange(4)[..., None, None], jnp.arange(5)[..., None], None, 1),
    ),
    (
        jax.random.normal(jax.random.PRNGKey(20), (4, 5, 6)),
        (
            jnp.arange(4)[..., None, None],
            jnp.arange(5)[..., None],
            None,
            slice(None),
        ),
    ),
    (
        jax.random.normal(jax.random.PRNGKey(21), (3, 4, 5, 6)),
        (
            Ellipsis,
            jnp.arange(4)[..., None, None],
            jnp.arange(5)[..., None],
            slice(None),
        ),
    ),
]


@pytest.mark.parametrize("tensor, idx", INDEXING_CASES)
def test_custom_getitem(tensor, idx):
    expected = tensor[idx]
    result = jax_getitem(tensor, idx)
    assert result.shape == expected.shape, (
        f"Shape mismatch for idx: {idx}. Expected: {expected.shape}, Got: {result.shape}"
    )
    assert jnp.allclose(result, expected, equal_nan=True), f"Failed for idx: {idx}"


def test_jax_jit_1():
    @jit
    def f(x, y):
        return to_array(jax_getitem(x, [i(), j()]) + jax_getitem(y, [j()]), i, j)

    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")
    x, y = jnp.ones((5, 4)), jnp.ones((4,))

    assert (f(x, y) == x + y).all()


def test_jax_jit_2():
    @jit
    def f(x, y):
        return jax_getitem(x, [i(), j()]) + jax_getitem(y, [j()])

    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")
    x, y = jnp.ones((5, 4)), jnp.ones((4,))

    assert (to_array(f(x, y), i, j) == x + y).all()


def test_jax_jit_3():
    @jit
    def f(x, y):
        return jax_getitem(x, [i(), j()]) + jax_getitem(y, [j()])

    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")
    x, y = jnp.ones((5, 4)), jnp.ones((4,))

    assert (to_array(f(x, y), i, j) == x + y).all()
