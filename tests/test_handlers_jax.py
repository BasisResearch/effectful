from typing import TypeVar

import jax
from typing_extensions import ParamSpec

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, jax_getitem, jit, sizesof
from effectful.ops.semantics import evaluate, fvsof, handler
from effectful.ops.syntax import defdata, defop
from effectful.ops.types import Term

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def test_bind_dims():
    """Test bind_dims's handling of free variables and tensor shapes"""
    i, j, k = (
        defop(jax.Array, name="i"),
        defop(jax.Array, name="j"),
        defop(jax.Array, name="k"),
    )
    key = jax.random.PRNGKey(0)
    t = jax.random.normal(key, (2, 3, 4))

    # Test case 1: Converting all named dimensions to positional
    t_ijk = jax_getitem(t, [i(), j(), k()])
    assert fvsof(t_ijk) >= {i, j, k}

    t1 = bind_dims(t_ijk, i, j, k)
    assert not (fvsof(t1) & {i, j, k})
    assert t1.shape == (2, 3, 4)

    # Test case 2: Different ordering of dimensions
    t2 = bind_dims(t_ijk, k, j, i)
    assert not (fvsof(t1) & {i, j, k})
    assert t2.shape == (4, 3, 2)

    # Test case 3: Keeping some dimensions as free variables
    t3 = bind_dims(t_ijk, i)  # Convert only i to positional
    assert fvsof(t3) >= {j, k}  # j and k remain free
    assert isinstance(t3, Term)
    assert t3.shape == (2,)

    t4 = bind_dims(t_ijk, i, j)  # Convert i and j to positional
    assert fvsof(t4) >= {k} and not (fvsof(t4) & {i, j})  # only k remains free
    assert isinstance(t4, Term)
    assert t4.shape == (2, 3)

    # Test case 4: Empty order list keeps all variables free
    t5 = bind_dims(t_ijk)
    assert fvsof(t5) >= {i, j, k}  # All variables remain free
    assert isinstance(t5, Term)
    assert t5.shape == tuple()

    # Test case 5: Verify permuted tensors maintain correct relationships
    t_kji = jax_getitem(jnp.permute_dims(t, (2, 1, 0)), [k(), j(), i()])
    t6 = bind_dims(t_kji, i, j, k)
    t7 = bind_dims(t_ijk, i, j, k)
    assert jnp.allclose(t6, t7)

    # Test case 6: Mixed operations with free variables
    x = jnp.sin(t_ijk)  # Apply operation to indexed tensor
    x1 = bind_dims(x, i, j)  # Convert some dimensions
    assert fvsof(x1) >= {k}  # k remains free
    assert isinstance(x1, Term)
    assert x1.shape == (2, 3)

    # Test case 7: Multiple tensors sharing variables
    t2_ijk = jax_getitem(jax.random.normal(key, (2, 3, 4)), [i(), j(), k()])
    sum_t = t_ijk + t2_ijk
    sum1 = bind_dims(sum_t, i, j)
    assert fvsof(sum1) >= {k}  # k remains free
    assert isinstance(sum1, Term)
    assert sum1.shape == (2, 3)

    # Test case 8: Tensor term with non-sized free variables
    w = defop(jax.Array, name="w")
    t_ijk = jax_getitem(t, [i(), j(), k()]) + w()
    t8 = bind_dims(t_ijk, i, j, k)
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

    x_ij = jax_getitem(xval, [i(), j()])
    x_plus_y1_ij = x_ij + jax_getitem(y1_val, [i()])
    actual = x_plus_y1_ij + jax_getitem(y2_val, [j()])

    assert actual.op == jax_getitem
    assert fvsof(actual) >= {i, j}
    assert actual.shape == ()
    assert actual.size == 1
    assert actual.ndim == 0

    assert (bind_dims(actual, i, j) == expected).all()


def test_tpe_2():
    key = jax.random.PRNGKey(0)
    xval = jax.random.normal(key, (2, 3))
    ival = jnp.arange(2)
    expected = jnp.sum(xval[ival, :], axis=0)

    j = defop(jax.Array)
    x_j = jax_getitem(xval, [ival, j()])

    assert x_j.shape == (2,)
    actual = jnp.sum(x_j, axis=0)

    assert actual.op == jax_getitem
    assert fvsof(actual) >= {j}
    assert actual.shape == ()
    assert actual.size == 1

    assert (bind_dims(actual, j) == expected).all()


def test_tpe_3():
    key = jax.random.PRNGKey(0)
    xval = jax.random.normal(key, (4, 2, 3))
    ival = jnp.arange(2)
    expected = jnp.sum(xval, axis=1)

    j, k = defop(jax.Array), defop(jax.Array)
    x_j = jax_getitem(xval, [k(), ival, j()])
    actual = jnp.sum(x_j, axis=0)

    assert actual.op == jax_getitem
    assert fvsof(actual) >= {j, k}
    assert actual.shape == ()
    assert actual.size == 1

    assert (bind_dims(actual, k, j) == expected).all()


def test_tpe_known_index():
    """Constant indexes are partially evaluated away."""
    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")

    cases = [
        jax_getitem(jnp.ones((2, 3)), [i(), 1]),
        jax_getitem(jnp.ones((2, 3)), [0, i()]),
        jax_getitem(jnp.ones((2, 3, 4)), [0, i(), 1]),
        jax_getitem(jnp.ones((2, 3, 4)), [0, i(), j()]),
        jax_getitem(jnp.ones((2, 3, 4)), [i(), j(), 3]),
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
    A = jax_getitem(t, [height(), width()])

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
    x_ij = jax_getitem(xval, [i(), j()])
    y_ij = jax_getitem(yval, [i(), j()])
    actual = jnp.stack((x_ij, y_ij))
    assert actual.shape == (2,)
    assert (
        jnp.transpose(bind_dims(actual, i, j), [2, 0, 1]) == jnp.stack((xval, yval))
    ).all()


def test_jax_jit_1():
    @jit
    def f(x, y):
        bound = bind_dims(jax_getitem(x, [i(), j()]) + jax_getitem(y, [j()]), i, j)
        return bound

    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")
    x, y = jnp.ones((5, 4)), jnp.ones((4,))

    z = f(x, y)
    assert (z == x + y).all()


def test_jax_jit_2():
    @jit
    def f(x, y):
        return jax_getitem(x, [i(), j()]) + jax_getitem(y, [j()])

    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")
    x, y = jnp.ones((5, 4)), jnp.ones((4,))

    assert (bind_dims(f(x, y), i, j) == x + y).all()


def test_jax_jit_3():
    @jit
    def f(x, y):
        return jax_getitem(x, [i(), j()]) + jax_getitem(y, [j()])

    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")
    x, y = jnp.ones((5, 4)), jnp.ones((4,))

    assert (bind_dims(f(x, y), i, j) == x + y).all()


def test_jax_broadcast_to():
    i = defop(jax.Array, name="i")
    t = jnp.broadcast_to(jax_getitem(jnp.ones((2, 3)), [i(), slice(None)]), (3,))
    assert not isinstance(t.shape, Term) and t.shape == (3,)


def test_jax_nested_getitem():
    t = jnp.ones((2, 3))
    i, j = defop(jax.Array), defop(jax.Array)
    t_i = jax_getitem(t, [i()])

    t_ij = defdata(jax_getitem, t_i, [j()])
    assert sizesof(t_ij) == {i: 2, j: 3}

    t_ij = jax_getitem(t_i, [j()])
    assert sizesof(t_ij) == {i: 2, j: 3}


def test_jax_at_updates():
    """Test .at array update functionality for indexed arrays."""
    i, j, k = defop(jax.Array), defop(jax.Array), defop(jax.Array)

    # Test the exact case from the original issue
    a = jax_getitem(jnp.ones((5, 4, 3)), [i(), j()])
    a = a.at[1].set(0)
    b = jax_getitem(jnp.array([0, 1]), [k()])
    a = a.at[b].set(0)

    # Verify the result has the expected properties
    assert isinstance(a, Term)
    assert a.shape == (3,)

    # Test with 1D remaining dimension
    arr_2d = jnp.ones((3, 5))
    indexed_2d = jax_getitem(arr_2d, [i()])  # Shape (5,)
    updated_2d = indexed_2d.at[2].set(99.0)
    assert isinstance(updated_2d, Term)
    assert updated_2d.shape == (5,)

    # Test with 2D remaining dimensions
    arr_3d = jnp.ones((2, 3, 4))
    indexed_3d = jax_getitem(arr_3d, [i()])  # Shape (3, 4)
    updated_3d = indexed_3d.at[1, 2].set(99.0)
    assert isinstance(updated_3d, Term)
    assert updated_3d.shape == (3, 4)

    # Test using term as index
    arr = jnp.ones((5, 3))
    a = jax_getitem(arr, [i()])  # Shape (3,)
    k = defop(jax.Array)
    b = jax_getitem(jnp.array([0, 1, 2]), [k()])  # Shape ()
    updated = a.at[b].set(99.0)
    assert isinstance(updated, Term)
    assert updated.shape == (3,)


def test_jax_len():
    i = defop(jax.Array, name="i")
    t = jnp.ones((2, 3, 4))
    t_i = jax_getitem(t, [i()])
    assert len(t_i) == 3

    for row in t_i:
        assert len(row) == 4
