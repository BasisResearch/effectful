"""Mirrored operations and interpretation-aware JAX transforms."""

import jax.numpy as jnp
import numpy as np
import pytest

from effectful.handlers import jax as ejax
from effectful.handlers.jax import lax
from effectful.handlers.jax import numpy as enp
from effectful.ops.semantics import fwd, handler


def test_jit_retraces_source_and_warmed_nested_jit():
    inner_traces, outer_traces = [], []

    @ejax.jit
    def inner(x):
        inner_traces.append("trace")
        return enp.add(x, 1)

    @ejax.jit
    def outer(x):
        outer_traces.append("trace")
        return inner(x)

    np.testing.assert_allclose(inner(jnp.array(2)), 3)
    np.testing.assert_allclose(outer(jnp.array(2)), 3)
    assert len(inner_traces) == len(outer_traces) == 1
    with handler({enp.add: lambda *a, **k: fwd() * 2}):
        np.testing.assert_allclose(outer(jnp.array(2)), 6)
        np.testing.assert_allclose(outer(jnp.array(3)), 8)
        assert len(inner_traces) == len(outer_traces) == 2
    np.testing.assert_allclose(outer(jnp.array(4)), 5)
    assert len(inner_traces) == len(outer_traces) == 2


def test_jit_ignores_transient_forwarding_state():
    from effectful.ops.syntax import defop

    traces = []

    @ejax.jit
    def compiled(x):
        traces.append("trace")
        return enp.add(x, 1)

    @defop
    def request(x):
        return x

    with handler({request: lambda x: compiled(x)}):
        np.testing.assert_allclose(request(jnp.array(2)), 3)
        np.testing.assert_allclose(request(jnp.array(5)), 6)
    assert traces == ["trace"]


@pytest.mark.parametrize(
    "options", [{"static_argnums": 1}, {"static_argnames": "axis"}]
)
def test_jit_static_arguments(options):
    def fn(x, axis=0):
        return enp.sum(x, axis=axis)

    compiled = ejax.jit(fn, **options)
    x = jnp.arange(6.0).reshape(2, 3)
    for axis in (0, 1):
        np.testing.assert_allclose(compiled(x, axis), jnp.sum(x, axis=axis))
        np.testing.assert_allclose(compiled(x, axis=axis), jnp.sum(x, axis=axis))


def test_jit_lower_and_options_decorator():
    @ejax.jit(inline=True)
    def fn(x):
        return enp.add(x, 1)

    executable = fn.lower(jnp.array(2)).compile()
    np.testing.assert_allclose(executable(jnp.array(3)), 4)
    assert fn._cache_size() >= 0


def test_jit_named_dimension_results():
    import jax

    from effectful.handlers.jax import bind_dims, jax_getitem, sizesof
    from effectful.ops.syntax import defop

    i = defop(jax.Array, name="i")

    @ejax.jit
    def fn(x):
        return enp.add(jax_getitem(x, (i(),)), 1)

    result = fn(jnp.arange(3.0))
    assert sizesof(result) == {i: 3}
    np.testing.assert_allclose(bind_dims(result, i), jnp.arange(3.0) + 1)


@pytest.mark.parametrize("transform", [ejax.checkpoint, ejax.eval_shape])
def test_source_transform_policy_change(transform):
    fn = lambda x: enp.add(x, 1)
    run = transform(fn) if transform is ejax.checkpoint else lambda x: transform(fn, x)
    baseline = run(jnp.array(2))
    with handler({enp.add: lambda *a, **k: enp.stack((fwd(), fwd()))}):
        result = run(jnp.array(2))
    assert baseline.shape == ()
    assert result.shape == (2,)


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


def test_interception_jit_inner():
    """Jitted functions are interceptable."""

    @ejax.jit
    def fn(a, b):
        return enp.add(a, b)

    a, b = (1, 2)
    np.testing.assert_allclose(fn(a, b), a + b)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        actual = fn(a, b)
        np.testing.assert_allclose(actual, (a + b) * 2)
    with handler({enp.add: lambda *a, **k: fwd() * 3}):
        actual = fn(a, b)
        np.testing.assert_allclose(actual, (a + b) * 3)


def test_interception_jit_outer():
    """Changing the interpretation retraces the original source callable."""

    @ejax.jit
    def fn(a, b):
        return enp.add(a, b)

    a, b = (1, 2)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        np.testing.assert_allclose(fn(a, b), (a + b) * 2)
    np.testing.assert_allclose(fn(a, b), a + b)


def test_interception_scan():
    """Jitted functions are interceptable."""

    def cumsum(cumsum, el):
        cumsum = enp.add(cumsum, el)
        return (cumsum, cumsum)

    def fn(a):
        return lax.scan(cumsum, 0, a)

    total, arr = fn(jnp.arange(1, 4))
    np.testing.assert_allclose(arr, jnp.array([1, 3, 6]))
    np.testing.assert_allclose(total, 6)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        total, arr = fn(jnp.arange(1, 4))
        np.testing.assert_allclose(arr, jnp.array([2, 8, 22]))
        np.testing.assert_allclose(total, 22)


def test_interception_cond():
    """Conditional branches are interceptable."""

    def fn(a):
        return lax.cond(a > 0, lambda x: enp.add(x, 1), lambda x: enp.add(x, 2), a)

    np.testing.assert_allclose(fn(jnp.array(2)), 3)
    np.testing.assert_allclose(fn(jnp.array(-3)), -1)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        np.testing.assert_allclose(fn(jnp.array(2)), 6)
        np.testing.assert_allclose(fn(jnp.array(-3)), -2)


def test_interception_checkpoint():
    """Checkpoint bodies retain their mirrored effects during source tracing."""

    @ejax.checkpoint
    def fn(a):
        return enp.add(a, 1)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        result = fn(jnp.array(2))
    np.testing.assert_allclose(result, 6)


def test_interception_while():
    """While loop bodies are interceptable."""

    def fn(a, limit):
        return lax.while_loop(lambda c: c < limit, lambda c: enp.add(c, a), 0)

    np.testing.assert_allclose(fn(jnp.array(1), jnp.array(5)), 5)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        np.testing.assert_allclose(fn(jnp.array(1), jnp.array(5)), 6)


def test_interception_custom_jvp_call():
    """Operations in a custom JVP rule are interceptable."""

    @ejax.custom_jvp
    def custom(x):
        return enp.multiply(x, x)

    @custom.defjvp
    def custom_jvp(primals, tangents):
        (x,), (t,) = (primals, tangents)
        return (custom(x), enp.multiply(enp.add(x, 1), t))

    def fn(a):
        return ejax.jvp(custom, (a,), (enp.ones_like(a),))

    value, tangent = fn(jnp.array(2.0))
    np.testing.assert_allclose(value, 4)
    np.testing.assert_allclose(tangent, 3)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        value, tangent = fn(jnp.array(2.0))
        np.testing.assert_allclose(value, 4)
        np.testing.assert_allclose(tangent, 6)


def test_interception_custom_vjp_call():
    """Operations in a custom VJP rule are interceptable."""

    @ejax.custom_vjp
    def custom(x):
        return enp.multiply(x, x)

    def forward(x):
        return (custom(x), x)

    def backward(x, g):
        return (enp.multiply(enp.add(x, 1), g),)

    custom.defvjp(forward, backward)

    def fn(a):
        return ejax.value_and_grad(custom)(a)

    value, gradient = fn(jnp.array(2.0))
    np.testing.assert_allclose(value, 4)
    np.testing.assert_allclose(gradient, 3)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        value, gradient = fn(jnp.array(2.0))
        np.testing.assert_allclose(value, 4)
        np.testing.assert_allclose(gradient, 6)


@pytest.mark.parametrize("jit", [False, True])
def test_interception_custom_jvp_call_outer_jvp(jit):
    """Custom JVP rules remain interceptable when differentiation is outside."""

    @ejax.custom_jvp
    def custom(x):
        return enp.multiply(x, x)

    @custom.defjvp
    def custom_jvp(primals, tangents):
        (x,), (t,) = (primals, tangents)
        return (custom(x), enp.multiply(enp.add(x, 1), t))

    fn = ejax.jit(custom) if jit else custom
    a, t = (jnp.array(2.0), jnp.array(1.0))
    value, tangent = ejax.jvp(fn, (a,), (t,))
    np.testing.assert_allclose(value, 4)
    np.testing.assert_allclose(tangent, 3)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        value, tangent = ejax.jvp(fn, (a,), (t,))
        np.testing.assert_allclose(value, 4)
        np.testing.assert_allclose(tangent, 6)


@pytest.mark.parametrize("jit", [False, True])
def test_interception_custom_vjp_call_outer_grad(jit):
    """Custom VJP rules remain interceptable when differentiation is outside."""

    @ejax.custom_vjp
    def custom(x):
        return enp.multiply(x, x)

    def forward(x):
        return (custom(x), x)

    def backward(x, g):
        return (enp.multiply(enp.add(x, 1), g),)

    custom.defvjp(forward, backward)
    fn = ejax.value_and_grad(ejax.jit(custom) if jit else custom)
    value, gradient = fn(jnp.array(2.0))
    np.testing.assert_allclose(value, 4)
    np.testing.assert_allclose(gradient, 3)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        value, gradient = fn(jnp.array(2.0))
        np.testing.assert_allclose(value, 4)
        np.testing.assert_allclose(gradient, 6)
