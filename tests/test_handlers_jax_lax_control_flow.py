"""Control-flow effects and callback wrapping in jax.lax._control_flow."""

import jax.numpy as jnp
import numpy as np
import pytest

from effectful.handlers import jax as ejax
from effectful.handlers.jax import lax
from effectful.handlers.jax import numpy as enp
from effectful.ops.semantics import cache, fwd, handler


@pytest.mark.parametrize("name", ["cond", "switch", "while_loop", "scan"])
def test_control_flow_defaults_wrap_callbacks(name):
    wrapped = []

    def wrap(fun):
        wrapped.append(fun)
        return fwd()

    def step(x):
        return enp.add(x, 1)

    def other(x):
        return x

    def condition(x):
        return x < 3

    def scan_body(c, _):
        return step(c), c

    with handler({ejax.wrap_callable: wrap}):
        if name == "cond":
            result = lax.cond(True, step, other, jnp.array(2))
            assert wrapped == [step, other]
        elif name == "switch":
            result = lax.switch(0, (step, other), jnp.array(2))
            assert wrapped == [step, other]
        elif name == "while_loop":
            result = lax.while_loop(condition, step, jnp.array(2))
            assert wrapped == [condition, step]
        else:
            result, ys = lax.scan(scan_body, jnp.array(2), None, length=1)
            np.testing.assert_allclose(ys, [2])
            assert wrapped == [scan_body]
    np.testing.assert_allclose(result, 3)


@pytest.mark.parametrize("api", ["cond", "switch", "while", "scan"])
def test_control_flow_policy_change(api):
    def step(x):
        return enp.add(x, 1)

    def run(x):
        if api == "cond":
            return lax.cond(True, step, lambda x: x, x)
        if api == "switch":
            return lax.switch(0, (step, lambda x: x), x)
        if api == "while":
            return lax.while_loop(
                lambda state: state[0] < 1,
                lambda state: (state[0] + 1, step(state[1])),
                (0, x),
            )[1]
        return lax.scan(lambda c, _: (step(c), ()), x, None, length=1)[0]

    np.testing.assert_allclose(run(jnp.array(2)), 3)
    with handler({enp.add: lambda *a, **k: fwd() * 2}):
        np.testing.assert_allclose(run(jnp.array(2)), 6)
    np.testing.assert_allclose(run(jnp.array(2)), 3)


def test_empty_outputs_and_operand_keyword():
    assert lax.cond(True, lambda: (), lambda: ()) == ()
    assert lax.switch(0, (lambda: (), lambda: ())) == ()
    np.testing.assert_allclose(
        lax.cond(True, lambda x: enp.add(x, 1), lambda x: x, operand=jnp.array(2)), 3
    )
    np.testing.assert_allclose(
        lax.switch(0, (lambda x: enp.add(x, 1), lambda x: x), operand=jnp.array(2)), 3
    )
    with pytest.raises(TypeError, match="positional operands"):
        lax.cond(True, lambda x: x, lambda x: x, 1, operand=2)


def test_nested_control_flow_grad():
    def fn(xs):
        def body(c, x):
            c = lax.cond(x > 0, lambda: enp.add(c, x), lambda: enp.subtract(c, x))
            return c, c

        return lax.scan(body, 0.0, xs)[1].sum()

    xs = jnp.array([-1.0, 2.0, 3.0])
    np.testing.assert_allclose(ejax.jit(ejax.grad(fn))(xs), jnp.array([-3.0, 2.0, 1.0]))


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


def test_interception_while():
    """While loop bodies are interceptable."""

    def fn(a, limit):
        return lax.while_loop(lambda c: c < limit, lambda c: enp.add(c, a), 0)

    np.testing.assert_allclose(fn(jnp.array(1), jnp.array(5)), 5)

    def _add(*args, **kwargs):
        return enp.multiply(fwd(), 2)

    with handler({enp.add: _add}):
        np.testing.assert_allclose(fn(jnp.array(1), jnp.array(5)), 6)


@pytest.mark.parametrize("api", ["cond", "switch", "while", "scan"])
def test_cached_callbacks_trace_once_per_interpretation(api):
    traces = []

    def step(x):
        traces.append("trace")
        return enp.add(x, 1)

    def condition(state):
        return state[0] < 1

    def while_body(state):
        return state[0] + 1, step(state[1])

    def scan_body(c, _):
        return step(c), ()

    def run(x):
        if api == "cond":
            return lax.cond(True, step, step, x)
        if api == "switch":
            return lax.switch(0, (step, step), x)
        if api == "while":
            return lax.while_loop(condition, while_body, (0, x))[1]
        return lax.scan(scan_body, x, None, length=1)[0]

    with cache() as store:
        np.testing.assert_allclose(run(jnp.array(2)), 3)
        np.testing.assert_allclose(run(jnp.array(4)), 5)
        assert traces == ["trace"]
        with handler({enp.add: lambda *a, **k: fwd() * 2}):
            np.testing.assert_allclose(run(jnp.array(2)), 6)
            np.testing.assert_allclose(run(jnp.array(4)), 10)
            assert traces == ["trace", "trace"]
        np.testing.assert_allclose(run(jnp.array(6)), 7)
        assert traces == ["trace", "trace"]
    with cache(store):
        np.testing.assert_allclose(run(jnp.array(8)), 9)
        assert traces == ["trace", "trace"]
