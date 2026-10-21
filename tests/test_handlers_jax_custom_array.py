"""Specification tests for the planned custom-array interception extension."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from effectful.handlers.jax import _custom_array, primitive_op
from effectful.handlers.jax._compat import core
from effectful.handlers.jax._intercept import upcast
from effectful.ops.semantics import fwd, handler


@jax.tree_util.register_dataclass
@dataclass
class Diagonal:
    """A logical square matrix with only its diagonal stored as a buffer."""

    diag: jax.Array

    @property
    def shape(self):
        return (*self.diag.shape[:-1], self.diag.shape[-1], self.diag.shape[-1])

    @property
    def dtype(self):
        return self.diag.dtype


@pytest.fixture
def densifications():
    return []


@pytest.fixture
def custom_array(densifications):
    api = _custom_array

    def is_custom_array(value):
        return isinstance(value, Diagonal) or fwd()

    def aval(value):
        if isinstance(value, Diagonal):
            return core.ShapedArray(value.shape, value.dtype)
        return fwd()

    def to_jax_array(value):
        if isinstance(value, Diagonal):
            # Record no buffers/tracers: this is a benign observation of policy.
            densifications.append(value.shape)
            return jnp.diag(value.diag)
        return fwd()

    def diagonal_upcast(left, right):
        if isinstance(left, Diagonal) and isinstance(right, Diagonal):
            if left.shape == right.shape and left.dtype == right.dtype:
                return lambda x: x, lambda x: x
        elif isinstance(left, Diagonal):
            if left.shape == right.shape and left.dtype == right.dtype:
                return api.to_jax_array, lambda x: x
        elif isinstance(right, Diagonal):
            if left.shape == right.shape and left.dtype == right.dtype:
                return lambda x: x, api.to_jax_array
        return fwd()

    def add(a, b, **params):
        if isinstance(a, Diagonal) and isinstance(b, Diagonal):
            return Diagonal(a.diag + b.diag)
        return fwd()

    def dot_general(a, b, **params):
        if isinstance(a, Diagonal) and len(b.shape) == 1:
            assert params["dimension_numbers"] == (((1,), (0,)), ((), ()))
            return a.diag * b
        return fwd()

    with handler(
        {
            api.is_custom_array: is_custom_array,
            api.aval: aval,
            api.to_jax_array: to_jax_array,
            upcast: diagonal_upcast,
            primitive_op(jax.lax.add_p): add,
            primitive_op(jax.lax.dot_general_p): dot_general,
        }
    ):
        yield api


def test_to_jax_array_ordinary_operands():
    for value in (1, 2.0, True, jnp.arange(3.0)):
        assert _custom_array.to_jax_array(value) is value
    result = jax.jit(lambda x: _custom_array.to_jax_array(x) + 1)(jnp.array(2.0))
    np.testing.assert_allclose(result, 3)


def test_to_jax_array_missing_conversion():
    with handler({_custom_array.is_custom_array: lambda x: isinstance(x, Diagonal)}):
        with pytest.raises(NotImplementedError, match="No conversion from Diagonal"):
            _custom_array.to_jax_array(Diagonal(jnp.ones(2)))


def test_interception_simple(custom_array, densifications):
    """A primitive handler receives the whole matrix, not its diagonal buffer."""

    @custom_array.custom_arrays
    def fn(a, b):
        assert a.shape == (3, 3)
        return a @ b

    a, b = Diagonal(jnp.array([2.0, 3.0, 4.0])), jnp.arange(3.0)
    np.testing.assert_allclose(fn(a, b), a.diag * b)

    def dot(a, b, **params):
        assert isinstance(a, Diagonal)
        return fwd() + 1

    with handler({primitive_op(jax.lax.dot_general_p): dot}):
        np.testing.assert_allclose(fn(a, b), a.diag * b + 1)
    assert not densifications


def test_interception_in_handler(custom_array):
    """JAX operations invoked by a custom-array handler are intercepted too."""

    @custom_array.custom_arrays
    def fn(a, b):
        return a @ b

    def dot(*args, **params):
        return fwd() + 1

    def add(*args, **params):
        return fwd() * 2

    a, b = Diagonal(jnp.array([2.0, 3.0, 4.0])), jnp.arange(3.0)
    with handler(
        {
            primitive_op(jax.lax.dot_general_p): dot,
            primitive_op(jax.lax.add_p): add,
        }
    ):
        np.testing.assert_allclose(fn(a, b), (a.diag * b + 1) * 2)


def test_interception_interpretation_change(custom_array):
    @custom_array.custom_arrays
    def fn(a, b):
        return a @ b

    a, b = Diagonal(jnp.array([2.0, 3.0, 4.0])), jnp.arange(3.0)
    for offset in (1, 2):
        with handler(
            {primitive_op(jax.lax.dot_general_p): lambda *a, **k: fwd() + offset}
        ):
            np.testing.assert_allclose(fn(a, b), a.diag * b + offset)


def test_interception_jit_inner(custom_array, densifications):
    """The source JIT sees a logical matrix; its interpreter sees Diagonal."""

    @custom_array.custom_arrays
    @jax.jit
    def fn(a, b):
        return a + b

    a, b = Diagonal(jnp.arange(3.0)), Diagonal(jnp.ones(3))
    result = fn(a, b)
    assert isinstance(result, Diagonal)
    np.testing.assert_allclose(result.diag, a.diag + b.diag)
    assert not densifications


def test_interception_jit_outer(custom_array, densifications):
    @jax.jit
    @custom_array.custom_arrays
    def fn(a, b):
        return a + b

    for values in ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]):
        a, b = Diagonal(jnp.array(values)), Diagonal(jnp.ones(3))
        result = fn(a, b)
        assert isinstance(result, Diagonal)
        np.testing.assert_allclose(result.diag, a.diag + b.diag)
    assert not densifications


@pytest.mark.parametrize("length", [0, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_interception_scan(custom_array, densifications, length, reverse):
    """A custom carry stays compact; only ordinary outputs are stacked."""

    @custom_array.custom_arrays
    def fn(a, b):
        def body(c, _):
            c = c + c
            return c, c @ b

        return jax.lax.scan(body, a, None, length=length, reverse=reverse)

    a, b = Diagonal(jnp.array([2.0, 3.0])), jnp.array([4.0, 5.0])
    carry, ys = fn(a, b)
    assert isinstance(carry, Diagonal)
    np.testing.assert_allclose(carry.diag, a.diag * 2**length)
    factors = 2.0 ** jnp.arange(1, length + 1)
    if reverse:
        factors = factors[::-1]
    np.testing.assert_allclose(ys, factors[:, None] * (a.diag * b))
    assert not densifications


@pytest.mark.xfail(reason="Stacking custom-array scan outputs is not yet supported")
def test_interception_scan_stacks_custom_arrays(custom_array):
    @custom_array.custom_arrays
    def fn(a):
        def body(c, _):
            c = c + c
            return c, c

        return jax.lax.scan(body, a, None, length=3)

    a = Diagonal(jnp.array([2.0, 3.0]))
    carry, ys = fn(a)
    assert isinstance(carry, Diagonal)
    assert isinstance(ys, Diagonal)
    np.testing.assert_allclose(carry.diag, 8 * a.diag)
    np.testing.assert_allclose(ys.diag, jnp.array([2.0, 4.0, 8.0])[:, None] * a.diag)
    assert ys.shape == (3, 2, 2)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("pred", [False, True])
def test_interception_cond(custom_array, densifications, jit, pred):
    """Branches returning compatible custom arrays preserve their storage."""

    @custom_array.custom_arrays
    def fn(a, pred):
        return jax.lax.cond(pred, lambda x: x + x, lambda x: x, a)

    if jit:
        fn = jax.jit(fn)
    a = Diagonal(jnp.array([2.0, 3.0]))
    result = fn(a, jnp.array(pred))
    assert isinstance(result, Diagonal)
    np.testing.assert_allclose(result.diag, a.diag * (2 if pred else 1))
    assert not densifications


@pytest.mark.parametrize("pred", [False, True])
def test_interception_cond_upcast(custom_array, densifications, pred):
    """Diagonal and dense branch outputs join using upcast conversions."""

    @custom_array.custom_arrays
    def fn(a, pred):
        return jax.lax.cond(pred, lambda x: x + x, jnp.sin, a)

    a = Diagonal(jnp.array([2.0, 3.0]))
    result = jax.jit(fn)(a, jnp.array(pred))
    assert isinstance(result, jax.Array)
    expected = jnp.diag(a.diag * 2) if pred else jnp.sin(jnp.diag(a.diag))
    np.testing.assert_allclose(result, expected)
    assert densifications


@pytest.mark.parametrize("index", [0, 1, 2])
def test_interception_switch_upcast(custom_array, index):
    """Later branches can widen the conversions already selected for others."""

    @custom_array.custom_arrays
    def fn(a, index):
        return jax.lax.switch(index, (lambda x: x, lambda x: x + x, jnp.sin), a)

    a = Diagonal(jnp.array([2.0, 3.0]))
    result = jax.jit(fn)(a, jnp.array(index))
    expected = (jnp.diag(a.diag), jnp.diag(2 * a.diag), jnp.sin(jnp.diag(a.diag)))
    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(result, expected[index])


@pytest.mark.parametrize("pred", [False, True])
def test_interception_grad_cond_upcast(custom_array, pred):
    @custom_array.custom_arrays
    def fn(a, pred):
        return jax.lax.cond(pred, lambda x: x + x, jnp.sin, a)

    data = jnp.array([2.0, 3.0])
    grad = jax.jit(jax.grad(lambda d: fn(Diagonal(d), jnp.array(pred)).sum()))
    expected = jnp.full_like(data, 2) if pred else jnp.cos(data)
    np.testing.assert_allclose(grad(data), expected)


def test_interception_while(custom_array, densifications):
    @custom_array.custom_arrays
    def fn(a, limit):
        def body(state):
            i, c = state
            return i + 1, c + c

        return jax.lax.while_loop(lambda state: state[0] < limit, body, (0, a))[1]

    a = Diagonal(jnp.array([2.0, 3.0]))
    result = jax.jit(fn)(a, jnp.array(3))
    assert isinstance(result, Diagonal)
    np.testing.assert_allclose(result.diag, a.diag * 8)
    assert not densifications


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("limit", [0, 2])
def test_interception_while_upcast(custom_array, densifications, jit, limit):
    """The initial custom carry widens to the dense body-output layout."""

    @custom_array.custom_arrays
    def fn(a, limit):
        return jax.lax.while_loop(
            lambda state: state[0] < limit,
            lambda state: (state[0] + 1, jnp.sin(state[1])),
            (0, a),
        )[1]

    if jit:
        fn = jax.jit(fn)
    a = Diagonal(jnp.array([2.0, 3.0]))
    result = fn(a, jnp.array(limit))
    assert isinstance(result, jax.Array)
    expected = jnp.diag(a.diag)
    for _ in range(limit):
        expected = jnp.sin(expected)
    np.testing.assert_allclose(result, expected)
    assert densifications


@pytest.mark.parametrize("jit", [False, True])
def test_interception_grad(custom_array, densifications, jit):
    """Gradients are with respect to stored nonzeros, not a dense matrix."""
    fn = custom_array.custom_arrays(lambda a, b: a @ b)
    b = jnp.array([4.0, 5.0])

    def loss(data):
        return jnp.sum(fn(Diagonal(data), b) ** 2)

    grad = jax.grad(loss)
    if jit:
        grad = jax.jit(grad)
    data = jnp.array([2.0, 3.0])
    result = grad(data)
    assert result.shape == data.shape
    np.testing.assert_allclose(result, 2 * data * b**2)
    assert not densifications


@pytest.mark.parametrize("jit", [False, True])
def test_interception_jvp(custom_array, densifications, jit):
    fn = custom_array.custom_arrays(lambda a, b: a @ b)
    b = jnp.array([4.0, 5.0])

    def run(data):
        return fn(Diagonal(data), b)

    if jit:
        run = jax.jit(run)
    data, direction = jnp.array([2.0, 3.0]), jnp.array([0.5, 1.5])
    value, tangent = jax.jvp(run, (data,), (direction,))
    np.testing.assert_allclose(value, data * b)
    np.testing.assert_allclose(tangent, direction * b)
    assert not densifications


def test_interception_grad_scan(custom_array):
    @custom_array.custom_arrays
    def fn(a, b):
        def body(c, _):
            c = c + c
            return c, c @ b

        return jax.lax.scan(body, a, None, length=3)[1]

    b = jnp.array([4.0, 5.0])
    grad = jax.jit(jax.grad(lambda data: fn(Diagonal(data), b).sum()))
    np.testing.assert_allclose(grad(jnp.array([2.0, 3.0])), 14 * b)


def test_interception_densification(custom_array, densifications):
    """An unhandled operation gets a dense matrix, not the storage buffer."""
    fn = custom_array.custom_arrays(jnp.exp)
    a = Diagonal(jnp.array([2.0, 3.0]))
    result = fn(a)
    assert isinstance(result, jax.Array)
    # exp(0) is one: applying exp only to the stored diagonal would be wrong.
    np.testing.assert_allclose(result, jnp.exp(jnp.diag(a.diag)))
    assert densifications


def test_interception_densification_after_forward(custom_array, densifications):
    fn = custom_array.custom_arrays(jnp.exp)

    def exp(a, **params):
        assert isinstance(a, Diagonal)
        return fwd() + 1

    a = Diagonal(jnp.array([2.0, 3.0]))
    with handler({primitive_op(jax.lax.exp_p): exp}):
        np.testing.assert_allclose(fn(a), jnp.exp(jnp.diag(a.diag)) + 1)
    assert densifications


@pytest.mark.parametrize("jit", [False, True])
def test_interception_grad_densification(custom_array, densifications, jit):
    fn = custom_array.custom_arrays(jnp.exp)
    grad = jax.grad(lambda data: fn(Diagonal(data)).sum())
    if jit:
        grad = jax.jit(grad)
    data = jnp.array([2.0, 3.0])
    np.testing.assert_allclose(grad(data), jnp.exp(data))
    assert densifications
