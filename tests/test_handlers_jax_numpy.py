"""Numerical effects and dtype exports in the mirrored NumPy namespace."""

import jax
import jax.numpy as jnp
import numpy as np

from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax import numpy as enp
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import defop
from effectful.ops.types import Term


def test_jax_broadcast_to():
    i = defop(jax.Array, name="i")
    t = enp.broadcast_to(jax_getitem(enp.ones((2, 3)), (i(), slice(None))), (3,))
    assert not isinstance(t.shape, Term) and t.shape == (3,)


def test_jax_array():
    x, y = defop(jax.Array), defop(jax.Array)

    t1 = enp.array((x(),))
    assert isinstance(t1, Term)

    t2 = enp.array((jax_getitem(enp.array([1, 2, 3]), (y(),)),))
    assert isinstance(t2, Term)


def test_mirrored_dtype_configuration():
    assert enp.float32 is jnp.float32
    assert enp.int32 is jnp.int32
    np.testing.assert_array_equal(enp.asarray([1, 2], dtype=enp.int32), [1, 2])


def test_interception_simple():
    """Mirrored operations dispatch through handlers without a custom trace."""

    def fn(a, b):
        return enp.matmul(a, b)

    a, b = (jnp.eye(3), jnp.arange(3.0))
    np.testing.assert_allclose(fn(a, b), b)
    with handler({enp.matmul: lambda *a, **k: fwd() + 1}):
        np.testing.assert_allclose(fn(a, b), b + 1)


def test_interception_in_handler():
    """Handlers for primitive operations run in interception context."""

    def fn(a, b):
        return enp.matmul(a, b)

    a, b = (jnp.eye(3), jnp.arange(3.0))
    np.testing.assert_allclose(fn(a, b), b)

    def _dot_general(*args, **kwargs):
        return enp.add(fwd(), 1)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    intp = {enp.matmul: _dot_general, enp.add: _add}
    with handler(intp):
        actual = fn(a, b)
        np.testing.assert_allclose(actual, (b + 1) * 2)


def test_interception_interpretation_change():
    """Changing the interpretation context changes behavior."""

    def fn(a, b):
        return enp.matmul(a, b)

    a, b = (jnp.eye(3), jnp.arange(3.0))
    np.testing.assert_allclose(fn(a, b), b)

    def _dot_general_1(*args, **kwargs):
        return enp.add(fwd(), 1)

    def _dot_general_2(*args, **kwargs):
        return enp.add(fwd(), 2)

    with handler({enp.matmul: _dot_general_1}):
        actual = fn(a, b)
        np.testing.assert_allclose(actual, b + 1)
    with handler({enp.matmul: _dot_general_2}):
        actual = fn(a, b)
        np.testing.assert_allclose(actual, b + 2)
