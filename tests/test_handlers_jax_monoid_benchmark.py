"""Benchmark ``effectful.handlers.jax.monoid.einsum`` against ``jnp.einsum``
under ``jax.jit`` on a range of representative subscript patterns.

Run with ``pytest tests/test_handlers_jax_monoid_benchmark.py --benchmark-only``
(serially — ``pytest-benchmark`` disables itself under ``pytest-xdist``).

The current implementation is the *unoptimized* baseline (single broadcast over
all operands, no contraction ordering). These benchmarks exist so we can track
the ratio as we add a contraction-ordering normalization rule.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from effectful.handlers.jax.monoid import einsum

# A grab-bag of subscript shapes worth tracking. ``shape`` maps each index
# letter in the spec to its dim size for that benchmark — kept small so even
# the unoptimized broadcast intermediate fits comfortably in memory.
BENCHMARK_CASES = [
    pytest.param("ij,jk->ik", {"i": 64, "j": 64, "k": 64}, id="matmul"),
    pytest.param(
        "bij,bjk->bik",
        {"b": 16, "i": 32, "j": 32, "k": 32},
        id="batched_matmul",
    ),
    pytest.param(
        "ab,bc,cd->ad",
        {"a": 16, "b": 16, "c": 16, "d": 16},
        id="chain_3",
    ),
    pytest.param(
        "ab,bc,cd,de->ae",
        {"a": 8, "b": 8, "c": 8, "d": 8, "e": 8},
        id="chain_4",
    ),
    pytest.param(
        "a,abi,bcij,cdij->ij",
        {"a": 4, "b": 4, "c": 4, "d": 4, "i": 8, "j": 8},
        id="mixed_rank",
    ),
]


def _make_operands(spec: str, sizes: dict[str, int], key: jax.Array) -> list:
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


@pytest.mark.parametrize("spec,sizes", BENCHMARK_CASES)
def test_bench_jnp_einsum(benchmark, spec, sizes, rng_key):
    """Baseline: ``jax.numpy.einsum`` under ``jax.jit``."""
    operands = _make_operands(spec, sizes, rng_key)
    f = jax.jit(lambda *xs: jnp.einsum(spec, *xs))
    f(*operands).block_until_ready()  # warm up cache

    @benchmark
    def _run():
        return f(*operands).block_until_ready()


@pytest.mark.parametrize("spec,sizes", BENCHMARK_CASES)
def test_bench_effectful_einsum(benchmark, spec, sizes, rng_key):
    """Our ``einsum`` under ``jax.jit``. Compare against the ``jnp_einsum``
    baseline run for the same spec.
    """
    operands = _make_operands(spec, sizes, rng_key)
    einsum_term = einsum(spec, *(arr.shape for arr in operands))
    einsum_f = lambda *args: einsum_term(*args)
    f = jax.jit(einsum_f)
    f(*operands).block_until_ready()  # warm up cache

    @benchmark
    def _run():
        return f(*operands).block_until_ready()
