"""High-level control flow preserves logical leaves and source interpretations."""

import jax.numpy as jnp
import numpy as np
import pytest

from effectful.handlers import jax as ejax
from effectful.handlers.jax import lax
from effectful.handlers.jax import numpy as enp
from effectful.ops.semantics import fwd, handler


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
