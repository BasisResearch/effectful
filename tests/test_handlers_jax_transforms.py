"""JIT, differentiation, and transform wrappers in jax._transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from effectful.handlers import jax as ejax
from effectful.handlers.jax import bind_dims, jax_getitem, jit, lax
from effectful.handlers.jax import numpy as enp
from effectful.ops.semantics import cache, fwd, handler
from effectful.ops.syntax import defop


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


def test_jax_jit_1():
    @jit
    def f(x, y):
        bound = bind_dims(jax_getitem(x, (i(), j())) + jax_getitem(y, (j(),)), i, j)
        return bound

    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")
    x, y = jnp.ones((5, 4)), jnp.ones((4,))

    z = f(x, y)
    assert (z == x + y).all()


def test_jax_jit_2():
    @jit
    def f(x, y):
        return jax_getitem(x, (i(), j())) + jax_getitem(y, (j(),))

    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")
    x, y = jnp.ones((5, 4)), jnp.ones((4,))

    assert (bind_dims(f(x, y), i, j) == x + y).all()


def test_jax_jit_3():
    @jit
    def f(x, y):
        return jax_getitem(x, (i(), j())) + jax_getitem(y, (j(),))

    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")
    x, y = jnp.ones((5, 4)), jnp.ones((4,))

    assert (bind_dims(f(x, y), i, j) == x + y).all()


def test_jit_retraces_shared_callback_under_new_interpretation():
    traces = []

    def branch(x):
        traces.append("branch")
        return enp.add(x, 1)

    @ejax.jit
    def compiled(x):
        return lax.cond(True, branch, branch, x)

    with cache():
        np.testing.assert_allclose(compiled(jnp.array(2)), 3)
        np.testing.assert_allclose(compiled(jnp.array(4)), 5)
        assert traces == ["branch"]
        assert compiled._cache_size() == 1
        with handler({enp.add: lambda *a, **k: fwd() * 2}):
            np.testing.assert_allclose(compiled(jnp.array(2)), 6)
            np.testing.assert_allclose(compiled(jnp.array(4)), 10)
            assert traces == ["branch", "branch"]
            assert compiled._cache_size() == 2
        np.testing.assert_allclose(compiled(jnp.array(6)), 7)
        assert traces == ["branch", "branch"]
        assert compiled._cache_size() == 2


@pytest.mark.parametrize("api", ["checkpoint", "eval_shape"])
def test_cached_callbacks_trace_once_per_interpretation(api):
    traces = []

    def step(x):
        traces.append("trace")
        return enp.add(x, 1)

    checkpointed = ejax.checkpoint(step)

    def check(value, expected):
        if api == "eval_shape":
            assert ejax.eval_shape(step, jnp.array(value)).shape == ()
        else:
            np.testing.assert_allclose(checkpointed(jnp.array(value)), expected)

    with cache() as store:
        check(2, 3)
        check(4, 5)
        assert traces == ["trace"]
        with handler({enp.add: lambda *a, **k: fwd() * 2}):
            check(2, 6)
            check(4, 10)
            assert traces == ["trace", "trace"]
        check(6, 7)
        assert traces == ["trace", "trace"]
    with cache(store):
        check(8, 9)
        assert traces == ["trace", "trace"]
