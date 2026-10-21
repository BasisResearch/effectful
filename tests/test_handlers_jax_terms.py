"""Symbolic array methods implemented by jax._terms."""

import jax
import pytest

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import jax_getitem
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import defop
from effectful.ops.types import Term


def test_jax_at_updates():
    """Test .at array update functionality for indexed arrays."""
    i, j, k = defop(jax.Array), defop(jax.Array), defop(jax.Array)

    # Test the exact case from the original issue
    a = jax_getitem(jnp.ones((5, 4, 3)), (i(), j()))
    a = a.at[1].set(0)
    b = jax_getitem(jnp.array([0, 1]), (k(),))
    a = a.at[b].set(0)

    # Verify the result has the expected properties
    assert isinstance(a, Term)
    assert a.shape == (3,)

    # Test with 1D remaining dimension
    arr_2d = jnp.ones((3, 5))
    indexed_2d = jax_getitem(arr_2d, (i(),))  # Shape (5,)
    updated_2d = indexed_2d.at[2].set(99.0)
    assert isinstance(updated_2d, Term)
    assert updated_2d.shape == (5,)

    # Test with 2D remaining dimensions
    arr_3d = jnp.ones((2, 3, 4))
    indexed_3d = jax_getitem(arr_3d, (i(),))  # Shape (3, 4)
    updated_3d = indexed_3d.at[1, 2].set(99.0)
    assert isinstance(updated_3d, Term)
    assert updated_3d.shape == (3, 4)

    # Test using term as index
    arr = jnp.ones((5, 3))
    a = jax_getitem(arr, (i(),))  # Shape (3,)
    k = defop(jax.Array)
    b = jax_getitem(jnp.array([0, 1, 2]), (k(),))  # Shape ()
    updated = a.at[b].set(99.0)
    assert isinstance(updated, Term)
    assert updated.shape == (3,)


def test_jax_len():
    i = defop(jax.Array, name="i")
    with pytest.raises(TypeError):
        len(i())

    t = jnp.ones((2, 3, 4))
    t_i = jax_getitem(t, (i(),))
    assert len(t_i) == 3

    for row in t_i:
        assert len(row) == 4


def test_jax_iter():
    i = defop(jax.Array, name="i")
    with pytest.raises(TypeError):
        tuple(i())


def test_arrayterm_all():
    """Test .all() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[True, False, True], [True, True, True]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.all()
    jnp_result = jnp.all(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_any():
    """Test .any() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[False, False, True], [False, False, False]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.any()
    jnp_result = jnp.any(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_argmax():
    """Test .argmax() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 5, 3], [4, 2, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.argmax()
    jnp_result = jnp.argmax(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_argmin():
    """Test .argmin() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[5, 1, 3], [4, 6, 2]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.argmin()
    jnp_result = jnp.argmin(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_argpartition():
    """Test .argpartition() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[5, 1, 3, 2], [4, 6, 2, 8]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.argpartition(2)
    jnp_result = jnp.argpartition(array_term, 2)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        # argpartition results can vary, so just check they have the same shape
        assert eval_result.shape == eval_jnp_result.shape


def test_arrayterm_argsort():
    """Test .argsort() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[5, 1, 3], [4, 6, 2]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.argsort()
    jnp_result = jnp.argsort(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_astype():
    """Test .astype() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.astype(jnp.float32)
    jnp_result = jnp.astype(array_term, jnp.float32)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert eval_result.dtype == jnp.float32
        assert eval_jnp_result.dtype == jnp.float32


def test_arrayterm_choose():
    """Test .choose() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[0, 1, 2], [1, 0, 2]])
    array_term = jax_getitem(base_array, (i(),))
    choices = [
        jnp.array([10, 20, 30]),
        jnp.array([40, 50, 60]),
        jnp.array([70, 80, 90]),
    ]

    result = array_term.choose(choices, mode="wrap")
    jnp_result = jnp.choose(array_term, choices, mode="wrap")

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_clip():
    """Test .clip() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 5, 3], [4, 2, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.clip(min=2, max=5)
    jnp_result = jnp.clip(array_term, min=2, max=5)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_compress():
    """Test .compress() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))
    condition = jnp.array([True, False, True])

    result = array_term.compress(condition)
    jnp_result = jnp.compress(condition, array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_conj():
    """Test .conj() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.conj()
    jnp_result = jnp.conj(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_conjugate():
    """Test .conjugate() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.conjugate()
    jnp_result = jnp.conjugate(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_copy():
    """Test .copy() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.copy()
    jnp_result = jnp.copy(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_cumprod():
    """Test .cumprod() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [2, 3, 4]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.cumprod()
    jnp_result = jnp.cumprod(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_cumsum():
    """Test .cumsum() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.cumsum()
    jnp_result = jnp.cumsum(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_diagonal():
    """Test .diagonal() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.diagonal()
    jnp_result = jnp.diagonal(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_dot():
    """Test .dot() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))
    other = jnp.array([1, 1, 1])

    result = array_term.dot(other)
    jnp_result = jnp.dot(array_term, other)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_max():
    """Test .max() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 5, 3], [4, 2, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.max()
    jnp_result = jnp.max(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_mean():
    """Test .mean() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.mean()
    jnp_result = jnp.mean(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_min():
    """Test .min() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 5, 3], [4, 2, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.min()
    jnp_result = jnp.min(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_nonzero():
    """Test .nonzero() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[0, 1, 0], [2, 0, 3]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.nonzero(size=4)
    jnp_result = jnp.nonzero(array_term, size=4)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        for r, j in zip(eval_result, eval_jnp_result):
            assert jnp.allclose(r, j)


def test_arrayterm_prod():
    """Test .prod() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [2, 3, 4]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.prod()
    jnp_result = jnp.prod(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_ptp():
    """Test .ptp() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 5, 3], [4, 2, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.ptp()
    jnp_result = jnp.ptp(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_ravel():
    """Test .ravel() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.ravel()
    jnp_result = jnp.ravel(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_repeat():
    """Test .repeat() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.repeat(2)
    jnp_result = jnp.repeat(array_term, 2)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_reshape():
    """Test .reshape() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.reshape((3,))
    jnp_result = jnp.reshape(array_term, (3,))

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_round():
    """Test .round() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1.234, 2.567], [3.891, 4.123]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.round(decimals=1)
    jnp_result = jnp.round(array_term, decimals=1)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_searchsorted():
    """Test .searchsorted() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 3, 5], [2, 4, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.searchsorted(4)
    jnp_result = jnp.searchsorted(array_term, 4)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_sort():
    """Test .sort() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[3, 1, 5], [6, 2, 4]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.sort()
    jnp_result = jnp.sort(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_squeeze():
    """Test .squeeze() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1], [2]], [[3], [4]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.squeeze()
    jnp_result = jnp.squeeze(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_std():
    """Test .std() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.std()
    jnp_result = jnp.std(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_sum():
    """Test .sum() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.sum()
    jnp_result = jnp.sum(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_swapaxes():
    """Test .swapaxes() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.swapaxes(0, 1)
    jnp_result = jnp.swapaxes(array_term, 0, 1)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_take():
    """Test .take() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[10, 20, 30], [40, 50, 60]])
    array_term = jax_getitem(base_array, (i(),))
    indices = jnp.array([0, 2])

    result = array_term.take(indices)
    jnp_result = jnp.take(array_term, indices)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_trace():
    """Test .trace() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.trace()
    jnp_result = jnp.trace(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_transpose():
    """Test .transpose() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.transpose()
    result_T = array_term.T
    jnp_result = jnp.transpose(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_result_T = evaluate(result_T)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)
        assert jnp.allclose(eval_result_T, eval_jnp_result)


def test_arrayterm_var():
    """Test .var() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.var()
    jnp_result = jnp.var(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)
