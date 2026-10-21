"""Cached callable identities keep JAX traces local to an interpretation."""

import jax.numpy as jnp
import numpy as np
import pytest

from effectful.handlers import jax as ejax
from effectful.handlers.jax import lax
from effectful.handlers.jax import numpy as enp
from effectful.ops.semantics import cache, evaluate, fwd, handler


def test_wrap_callable_uses_effectful_cache():
    def fun(x):
        return enp.add(x, 1)

    with cache() as store:
        # Wrapper entries must not collide with ordinary evaluation entries.
        assert evaluate(fun) is fun
        baseline = ejax.wrap_callable(fun)
        assert baseline is ejax.wrap_callable(fun)
        with handler({enp.add: lambda *a, **k: fwd() * 2}):
            changed = ejax.wrap_callable(fun)
            assert changed is ejax.wrap_callable(fun)
            assert changed is not baseline
            np.testing.assert_allclose(changed(jnp.array(2)), 6)
        assert ejax.wrap_callable(fun) is baseline

    with cache(store):
        assert ejax.wrap_callable(fun) is baseline
    assert ejax.wrap_callable(fun) is not ejax.wrap_callable(fun)


def test_wrap_callable_accepts_unhashable_callables():
    class Callable:
        __hash__ = None

        def __eq__(self, other):
            raise AssertionError("callback keys must use identity")

        def __call__(self, x):
            return enp.add(x, 1)

    fun = Callable()
    with cache():
        wrapped = ejax.wrap_callable(fun)
        assert wrapped is ejax.wrap_callable(fun)
        np.testing.assert_allclose(wrapped(jnp.array(2)), 3)


@pytest.mark.parametrize(
    "api", ["cond", "switch", "while", "scan", "fori_loop", "checkpoint", "eval_shape"]
)
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

    def fori_body(i, x):
        return step(x)

    checkpointed = ejax.checkpoint(step)

    def run(x):
        if api == "cond":
            return lax.cond(True, step, step, x)
        if api == "switch":
            return lax.switch(0, (step, step), x)
        if api == "while":
            return lax.while_loop(condition, while_body, (0, x))[1]
        if api == "scan":
            return lax.scan(scan_body, x, None, length=1)[0]
        if api == "fori_loop":
            return lax.fori_loop(0, 1, fori_body, x)
        if api == "checkpoint":
            return checkpointed(x)
        return ejax.eval_shape(step, x)

    def check(value, expected):
        result = run(jnp.array(value))
        if api == "eval_shape":
            assert result.shape == ()
        else:
            np.testing.assert_allclose(result, expected)

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
