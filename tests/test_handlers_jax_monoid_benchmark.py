"""Benchmark ``effectful.handlers.jax.monoid.einsum`` against ``jnp.einsum``
under ``jax.jit`` on a range of representative subscript patterns.

Both implementations run through the same parametrized test, so grouping by
the ``spec`` parameter puts them side by side (with a relative ratio column)
for each subscript pattern::

    pytest tests/test_handlers_jax_monoid_benchmark.py \\
        --benchmark-only --benchmark-group-by=param:spec

(Run serially — ``pytest-benchmark`` disables itself under ``pytest-xdist``.)

The current implementation is the *unoptimized* baseline (single broadcast over
all operands, no contraction ordering). These benchmarks exist so we can track
the ratio as we add a contraction-ordering normalization rule.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from effectful.handlers.jax.monoid import einsum


def _make_operands(spec: str, sizes: dict[str, int], key: jax.Array) -> list[jax.Array]:
    in_part = spec.split("->")[0]
    in_specs = in_part.split(",")
    keys = random.split(key, len(in_specs))
    return [
        random.normal(k, tuple(sizes[c] for c in s) if s else ())
        for k, s in zip(keys, in_specs, strict=True)
    ]


@pytest.fixture(scope="module")
def rng_key():
    return random.PRNGKey(0)


IMPLEMENTATIONS = {"jnp": jnp.einsum, "effectful": einsum}


@pytest.mark.parametrize("impl", list(IMPLEMENTATIONS), ids=list(IMPLEMENTATIONS))
@pytest.mark.parametrize("spec,sizes", BENCHMARK_CASES)
def test_bench_einsum(benchmark, impl, spec, sizes, rng_key):
    """Time one ``(spec, impl)`` pair. Group by ``spec`` to compare ``jnp``
    against ``effectful`` for the same subscript pattern (see module docstring).
    """
    operands = _make_operands(spec, sizes, rng_key)

    @jax.jit
    def f(*operands):
        return IMPLEMENTATIONS[impl](spec, *operands)

    f(*operands).block_until_ready()  # warm up cache

    @benchmark
    def _run():
        return f(*operands).block_until_ready()
