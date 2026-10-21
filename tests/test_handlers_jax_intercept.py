from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from effectful.handlers.jax import intercept, primitive_op
from effectful.ops.semantics import fwd, handler


def test_interception_simple():
    """Annotating a function with `intercept` causes its calls to jax primitives
    to raise effects.

    """

    @intercept
    def fn(a, b):
        return a @ b

    a, b = jnp.eye(3), jnp.arange(3.0)

    np.testing.assert_allclose(fn(a, b), b)
    with handler({primitive_op(jax.lax.dot_general_p): lambda *a, **k: fwd() + 1}):
        np.testing.assert_allclose(fn(a, b), b + 1)


def test_interception_in_handler():
    """Handlers for primitive operations run in interception context."""

    @intercept
    def fn(a, b):
        return a @ b

    a, b = jnp.eye(3), jnp.arange(3.0)

    np.testing.assert_allclose(fn(a, b), b)

    def _dot_general(*args, **kwargs):
        return fwd() + 1

    def _add(*args, **kwargs):
        return fwd() * 2

    intp = {
        primitive_op(jax.lax.dot_general_p): _dot_general,
        primitive_op(jax.lax.add_p): _add,
    }
    with handler(intp):
        actual = fn(a, b)
        np.testing.assert_allclose(actual, (b + 1) * 2)


def test_interception_interpretation_change():
    """Changing the interpretation context changes behavior."""

    @intercept
    def fn(a, b):
        return a @ b

    a, b = jnp.eye(3), jnp.arange(3.0)

    np.testing.assert_allclose(fn(a, b), b)

    def _dot_general_1(*args, **kwargs):
        return fwd() + 1

    def _dot_general_2(*args, **kwargs):
        return fwd() + 2

    with handler({primitive_op(jax.lax.dot_general_p): _dot_general_1}):
        actual = fn(a, b)
        np.testing.assert_allclose(actual, b + 1)

    with handler({primitive_op(jax.lax.dot_general_p): _dot_general_2}):
        actual = fn(a, b)
        np.testing.assert_allclose(actual, b + 2)


def test_interception_jit_inner():
    """Jitted functions are interceptable."""

    @intercept
    @jax.jit
    def fn(a, b):
        return a + b

    a, b = 1, 2

    np.testing.assert_allclose(fn(a, b), a + b)

    def _add(*args, **kwargs):
        return fwd() * 2

    with handler({primitive_op(jax.lax.add_p): _add}):
        actual = fn(a, b)
        np.testing.assert_allclose(actual, (a + b) * 2)

    with handler({primitive_op(jax.lax.add_p): lambda *a, **k: fwd() * 3}):
        actual = fn(a, b)
        np.testing.assert_allclose(actual, (a + b) * 3)


def test_interception_jit_outer():
    """Jitting an intercepted function stages in the handlers."""

    @jax.jit
    @intercept
    def fn(a, b):
        return a + b

    a, b = 1, 2

    def _add(*args, **kwargs):
        return fwd() * 2

    with handler({primitive_op(jax.lax.add_p): _add}):
        np.testing.assert_allclose(fn(a, b), (a + b) * 2)

    np.testing.assert_allclose(fn(a, b), (a + b) * 2)


def test_interception_scan():
    """Jitted functions are interceptable."""

    def cumsum(cumsum, el):
        cumsum += el
        return cumsum, cumsum  # ('carryover', 'accumulated')

    @intercept
    def fn(a):
        return jax.lax.scan(cumsum, 0, a)

    total, arr = fn(jnp.arange(1, 4))
    np.testing.assert_allclose(arr, jnp.array([1, 3, 6]))
    np.testing.assert_allclose(total, 6)

    def _add(*args, **kwargs):
        return fwd() * 2

    with handler({primitive_op(jax.lax.add_p): _add}):
        total, arr = fn(jnp.arange(1, 4))
        np.testing.assert_allclose(arr, jnp.array([2, 8, 22]))
        np.testing.assert_allclose(total, 22)


def test_interception_cond():
    """Conditional branches are interceptable."""

    @intercept
    def fn(a):
        return jax.lax.cond(a > 0, lambda x: x + 1, lambda x: x + 2, a)

    np.testing.assert_allclose(fn(jnp.array(2)), 3)
    np.testing.assert_allclose(fn(jnp.array(-3)), -1)

    def _add(*args, **kwargs):
        return fwd() * 2

    with handler({primitive_op(jax.lax.add_p): _add}):
        np.testing.assert_allclose(fn(jnp.array(2)), 6)
        np.testing.assert_allclose(fn(jnp.array(-3)), -2)


def test_interception_while():
    """While loop bodies are interceptable."""

    @intercept
    def fn(a, limit):
        return jax.lax.while_loop(lambda c: c < limit, lambda c: c + a, 0)

    np.testing.assert_allclose(fn(jnp.array(1), jnp.array(5)), 5)

    def _add(*args, **kwargs):
        return fwd() * 2

    with handler({primitive_op(jax.lax.add_p): _add}):
        np.testing.assert_allclose(fn(jnp.array(1), jnp.array(5)), 6)


def test_interception_custom_jvp_call():
    """Operations in a custom JVP rule are interceptable."""

    @jax.custom_jvp
    def custom(x):
        return x * x

    @custom.defjvp
    def custom_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        return custom(x), (x + 1) * t

    @intercept
    def fn(a):
        return jax.jvp(custom, (a,), (jnp.ones_like(a),))

    value, tangent = fn(jnp.array(2.0))
    np.testing.assert_allclose(value, 4)
    np.testing.assert_allclose(tangent, 3)

    def _add(*args, **kwargs):
        return fwd() * 2

    with handler({primitive_op(jax.lax.add_p): _add}):
        value, tangent = fn(jnp.array(2.0))
        np.testing.assert_allclose(value, 4)
        np.testing.assert_allclose(tangent, 6)


def test_interception_custom_vjp_call():
    """Operations in a custom VJP rule are interceptable."""

    @jax.custom_vjp
    def custom(x):
        return x * x

    def forward(x):
        return custom(x), x

    def backward(x, g):
        return ((x + 1) * g,)

    custom.defvjp(forward, backward)

    @intercept
    def fn(a):
        return jax.value_and_grad(custom)(a)

    value, gradient = fn(jnp.array(2.0))
    np.testing.assert_allclose(value, 4)
    np.testing.assert_allclose(gradient, 3)

    def _add(*args, **kwargs):
        return fwd() * 2

    with handler({primitive_op(jax.lax.add_p): _add}):
        value, gradient = fn(jnp.array(2.0))
        np.testing.assert_allclose(value, 4)
        np.testing.assert_allclose(gradient, 6)


@pytest.mark.parametrize("jit", [False, True])
def test_interception_custom_jvp_call_outer_jvp(jit):
    """Custom JVP rules remain interceptable when differentiation is outside."""

    @jax.custom_jvp
    def custom(x):
        return x * x

    @custom.defjvp
    def custom_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        return custom(x), (x + 1) * t

    fn = intercept(jax.jit(custom) if jit else custom)
    a, t = jnp.array(2.0), jnp.array(1.0)
    value, tangent = jax.jvp(fn, (a,), (t,))
    np.testing.assert_allclose(value, 4)
    np.testing.assert_allclose(tangent, 3)

    def _add(*args, **kwargs):
        return fwd() * 2

    with handler({primitive_op(jax.lax.add_p): _add}):
        value, tangent = jax.jvp(fn, (a,), (t,))
        np.testing.assert_allclose(value, 4)
        np.testing.assert_allclose(tangent, 6)


@pytest.mark.parametrize("jit", [False, True])
def test_interception_custom_vjp_call_outer_grad(jit):
    """Custom VJP rules remain interceptable when differentiation is outside."""

    @jax.custom_vjp
    def custom(x):
        return x * x

    def forward(x):
        return custom(x), x

    def backward(x, g):
        return ((x + 1) * g,)

    custom.defvjp(forward, backward)
    fn = jax.value_and_grad(intercept(jax.jit(custom) if jit else custom))

    value, gradient = fn(jnp.array(2.0))
    np.testing.assert_allclose(value, 4)
    np.testing.assert_allclose(gradient, 3)

    def _add(*args, **kwargs):
        return fwd() * 2

    with handler({primitive_op(jax.lax.add_p): _add}):
        value, gradient = fn(jnp.array(2.0))
        np.testing.assert_allclose(value, 4)
        np.testing.assert_allclose(gradient, 6)


# def test_interception_custom_array():
#     """Changing the interpretation context changes behavior."""

#     @jax.tree_util.register_dataclass
#     @dataclass
#     class Diagonal:
#         diag: jax.Array

#     def _dot_general_diag(d, a, *args, **kwargs):
#         breakpoint()
#         if not isinstance(d, Diagonal):
#             return fwd()
#         return d.diag * a

#     @intercept
#     def fn(a, b):
#         return a @ b

#     a, b = jnp.eye(3), jnp.arange(3.0)

#     np.testing.assert_allclose(fn(a, b), b)

#     with handler({primitive_op(jax.lax.dot_general_p): _dot_general_diag}):
#         actual = fn(Diagonal(jnp.array([1, 2, 3])), b)
#         np.testing.assert_allclose(actual, b + 1)
