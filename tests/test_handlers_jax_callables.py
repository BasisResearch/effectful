"""Interpretation-scoped wrapper identities in jax._callables."""

import jax.numpy as jnp
import numpy as np

from effectful.handlers import jax as ejax
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
