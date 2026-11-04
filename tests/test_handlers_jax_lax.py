"""Callback-bearing numerical mirrors in the lax namespace."""

import jax.numpy as jnp
import numpy as np

from effectful.handlers.jax import lax
from effectful.handlers.jax import numpy as enp
from effectful.ops.semantics import cache, fwd, handler


def test_fori_loop_cached_callback():
    traces = []

    def body(i, x):
        traces.append("trace")
        return enp.add(x, 1)

    def run(value):
        return lax.fori_loop(0, 1, body, jnp.array(value))

    with cache() as store:
        np.testing.assert_allclose(run(2), 3)
        np.testing.assert_allclose(run(4), 5)
        assert traces == ["trace"]
        with handler({enp.add: lambda *a, **k: fwd() * 2}):
            np.testing.assert_allclose(run(2), 6)
            np.testing.assert_allclose(run(4), 10)
            assert traces == ["trace", "trace"]
        np.testing.assert_allclose(run(6), 7)
        assert traces == ["trace", "trace"]
    with cache(store):
        np.testing.assert_allclose(run(8), 9)
        assert traces == ["trace", "trace"]
