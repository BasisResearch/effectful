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


@pytest.mark.parametrize("compile_mode", ["none", "primal", "gradient"])
def test_custom_jvp_rule_selection(compile_mode):
    @ejax.custom_jvp
    def custom(x):
        return x * x

    @custom.defjvp
    def original(primals, tangents):
        (x,), (t,) = primals, tangents
        return custom(x), 3 * t

    def replacement(primals, tangents):
        (x,), (t,) = primals, tangents
        return custom.fun(x), 7 * t

    def select(instance):
        if instance is custom:
            assert fwd() is original
            assert instance.fun is custom.__wrapped__
            return replacement
        return fwd()

    assert isinstance(custom, jax.custom_jvp)
    assert custom.jvp is original
    primal = ejax.jit(custom) if compile_mode == "primal" else custom
    derivative = ejax.grad(primal)
    if compile_mode == "gradient":
        derivative = ejax.jit(derivative)
    x = jnp.array(2.0)
    with cache():
        np.testing.assert_allclose(derivative(x), 3)
        with handler({ejax.custom_jvp.jvp: select}):
            np.testing.assert_allclose(derivative(x), 7)
            np.testing.assert_allclose(derivative(x), 7)
        np.testing.assert_allclose(derivative(x), 3)
    assert custom.jvp is original


def test_custom_jvp_callbacks_follow_invocation_interpretation():
    @ejax.custom_jvp
    def custom(x):
        return enp.add(x, 1)

    @custom.defjvp
    def original(primals, tangents):
        (x,), (t,) = primals, tangents
        return custom(x), enp.multiply(t, 2)

    compiled = ejax.jit(custom)
    x, t = jnp.array(2.0), jnp.array(1.0)
    with cache():
        np.testing.assert_allclose(ejax.jvp(compiled, (x,), (t,)), (3, 2))
        with handler(
            {
                enp.add: lambda *a, **k: fwd() * 2,
                enp.multiply: lambda *a, **k: fwd() * 3,
            }
        ):
            np.testing.assert_allclose(ejax.jvp(compiled, (x,), (t,)), (6, 6))
        np.testing.assert_allclose(ejax.jvp(compiled, (x,), (t,)), (3, 2))


@pytest.mark.parametrize("option", ["nondiff_argnums", "nondiff_argnames"])
def test_custom_jvp_rule_selection_nondiff_args(option):
    def fun(scale, x):
        return scale * x

    custom = ejax.custom_jvp(
        fun, **{option: (0,) if option == "nondiff_argnums" else ("scale",)}
    )

    @custom.defjvp
    def original(scale, primals, tangents):
        (x,), (t,) = primals, tangents
        return custom(scale, x), scale * t

    def replacement(scale, primals, tangents):
        (x,), (t,) = primals, tangents
        return custom.fun(scale, x), 2 * scale * t

    def select(instance):
        assert instance is custom
        assert instance.nondiff_argnums == (0,)
        assert fwd() is original
        return replacement

    derivative = ejax.jit(ejax.grad(lambda x: custom(scale=3.0, x=x)))
    with handler({ejax.custom_jvp.jvp: select}):
        np.testing.assert_allclose(derivative(jnp.array(2.0)), 6)


def test_custom_jvp_rule_selection_symbolic_zeros():
    @ejax.custom_jvp
    def custom(x, y):
        return x + y

    seen = []

    def replacement(primals, tangents):
        x, y = primals
        tx, ty = tangents
        seen.append(isinstance(ty, jax.custom_derivatives.SymbolicZero))
        return custom.fun(x, y), 7 * tx

    def original(primals, tangents):
        raise AssertionError("The original rule should be replaced")

    custom.defjvp(original, symbolic_zeros=True)

    def select(instance):
        assert instance is custom
        assert fwd() is original
        return replacement

    with handler({ejax.custom_jvp.jvp: select}):
        result = ejax.jit(ejax.grad(lambda x: custom(x, jnp.array(2.0))))(
            jnp.array(3.0)
        )
    np.testing.assert_allclose(result, 7)
    assert seen == [True]


def test_custom_jvp_defjvps():
    @ejax.custom_jvp
    def custom(x, y):
        return x * y

    custom.defjvps(
        lambda tx, out, x, y: tx * y,
        lambda ty, out, x, y: ty * x,
    )
    np.testing.assert_allclose(ejax.grad(custom)(jnp.array(2.0), jnp.array(3.0)), 3)


def test_custom_jvp_missing_rule():
    custom = ejax.custom_jvp(lambda x: x * x)
    assert custom.jvp is None
    with pytest.raises(AttributeError, match="No JVP defined"):
        custom(jnp.array(2.0))


@pytest.mark.parametrize("compile_mode", ["none", "primal", "gradient"])
@pytest.mark.parametrize("rules, expected", [("fwd", 7), ("bwd", 6), ("both", 14)])
def test_custom_vjp_rule_selection(compile_mode, rules, expected):
    @ejax.custom_vjp
    def custom(x):
        return x * x

    def forward(x):
        return custom(x), jnp.full_like(x, 3)

    def backward(slope, cotangent):
        return (slope * cotangent,)

    custom.defvjp(forward, backward)

    def replacement_forward(x):
        return custom.fun(x), jnp.full_like(x, 7)

    def replacement_backward(slope, cotangent):
        return (2 * slope * cotangent,)

    def select_forward(instance):
        assert instance is custom
        assert instance.fun is custom.__wrapped__
        assert fwd() is forward
        return replacement_forward

    def select_backward(instance):
        assert instance is custom
        assert fwd() is backward
        return replacement_backward

    interpretation = {}
    if rules in ("fwd", "both"):
        interpretation[ejax.custom_vjp.fwd] = select_forward
    if rules in ("bwd", "both"):
        interpretation[ejax.custom_vjp.bwd] = select_backward

    assert isinstance(custom, jax.custom_vjp)
    assert custom.fwd is forward
    assert custom.bwd is backward
    primal = ejax.jit(custom) if compile_mode == "primal" else custom
    derivative = ejax.grad(primal)
    if compile_mode == "gradient":
        derivative = ejax.jit(derivative)
    x = jnp.array(2.0)
    with cache():
        np.testing.assert_allclose(derivative(x), 3)
        with handler(interpretation):
            np.testing.assert_allclose(derivative(x), expected)
            np.testing.assert_allclose(derivative(x), expected)
        np.testing.assert_allclose(derivative(x), 3)
    assert custom.fwd is forward
    assert custom.bwd is backward


def test_custom_vjp_callbacks_follow_invocation_interpretation():
    @ejax.custom_vjp
    def custom(x):
        return enp.add(x, 1)

    def forward(x):
        return custom(x), enp.add(x, 1)

    def backward(residual, cotangent):
        return (enp.multiply(residual, cotangent),)

    custom.defvjp(forward, backward)
    run = ejax.value_and_grad(ejax.jit(custom))
    x = jnp.array(2.0)
    with cache():
        np.testing.assert_allclose(run(x), (3, 3))
        with handler(
            {
                enp.add: lambda *a, **k: fwd() * 2,
                enp.multiply: lambda *a, **k: fwd() * 3,
            }
        ):
            np.testing.assert_allclose(run(x), (6, 18))
        np.testing.assert_allclose(run(x), (3, 3))


@pytest.mark.parametrize("option", ["nondiff_argnums", "nondiff_argnames"])
def test_custom_vjp_rule_selection_nondiff_args(option):
    custom = ejax.custom_vjp(
        lambda scale, x: scale * x,
        **{option: (0,) if option == "nondiff_argnums" else ("scale",)},
    )

    def forward(scale, x):
        return custom(scale, x), None

    def backward(scale, residual, cotangent):
        return (scale * cotangent,)

    custom.defvjp(forward, backward)

    def replacement(scale, residual, cotangent):
        return (2 * scale * cotangent,)

    def select(instance):
        assert instance is custom
        assert instance.nondiff_argnums == (0,)
        assert fwd() is backward
        return replacement

    derivative = ejax.jit(ejax.grad(lambda x: custom(scale=3.0, x=x)))
    with handler({ejax.custom_vjp.bwd: select}):
        np.testing.assert_allclose(derivative(jnp.array(2.0)), 6)


def test_custom_vjp_rule_selection_symbolic_zeros():
    @ejax.custom_vjp
    def custom(x, y):
        return x + y, y

    def original_forward(*args):
        raise AssertionError("The forward rule should be replaced")

    def original_backward(*args):
        raise AssertionError("The backward rule should be replaced")

    custom.defvjp(original_forward, original_backward, symbolic_zeros=True)
    seen = []

    def forward(x, y):
        assert isinstance(x, jax.custom_derivatives.CustomVJPPrimal)
        seen.append((x.perturbed, y.perturbed))
        return (x.value + y.value, y.value), None

    def backward(residual, cotangents):
        gx, gy = cotangents
        seen.append(isinstance(gy, jax.custom_derivatives.SymbolicZero))
        return 7 * gx, None

    with handler(
        {
            ejax.custom_vjp.fwd: lambda instance: forward,
            ejax.custom_vjp.bwd: lambda instance: backward,
        }
    ):
        result = ejax.jit(ejax.grad(lambda x: custom(x, jnp.array(2.0))[0]))(
            jnp.array(3.0)
        )
    np.testing.assert_allclose(result, 7)
    assert seen == [(True, False), True]


def test_custom_vjp_rule_selection_optimize_remat():
    @ejax.custom_vjp
    def custom(x):
        return x * x

    def forward(x):
        return custom.fun(x), None

    def backward(residual, cotangent):
        return (3 * cotangent,)

    custom.defvjp(forward, backward, optimize_remat=True)

    def select(instance):
        assert instance.optimize_remat
        assert fwd() is backward
        return lambda residual, cotangent: (7 * cotangent,)

    with handler({ejax.custom_vjp.bwd: select}):
        result = ejax.jit(ejax.grad(ejax.checkpoint(custom)))(jnp.array(2.0))
    np.testing.assert_allclose(result, 7)


def test_custom_vjp_missing_rules():
    custom = ejax.custom_vjp(lambda x: x * x)
    assert custom.fwd is None
    assert custom.bwd is None
    with pytest.raises(AttributeError, match="No VJP defined"):
        custom(jnp.array(2.0))


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
