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
        "a,abi,bcij,cdij->ij",
        {"a": 4, "b": 4, "c": 4, "d": 4, "i": 8, "j": 8},
        id="mixed_rank",
    ),
    # ───────────────────────── single-operand reshuffles ─────────────────────
    # No contraction across operands — these stress the diagonal/transpose/sum
    # rewrites rather than any pairwise product ordering.
    pytest.param("ij->ji", {"i": 256, "j": 256}, id="transpose"),
    pytest.param("ijk->", {"i": 96, "j": 96, "k": 96}, id="full_reduce"),
    pytest.param("ijk->k", {"i": 96, "j": 96, "k": 96}, id="partial_reduce"),
    # Repeated index *within* one operand — exercises the implicit-diagonal path
    # in ReduceDeltaIndependent (no explicit jnp.diagonal step).
    pytest.param("ii->", {"i": 1024}, id="trace"),
    pytest.param("ii->i", {"i": 1024}, id="diagonal"),
    pytest.param("bii->b", {"b": 256, "i": 128}, id="batched_trace"),
    pytest.param("iij->ij", {"i": 128, "j": 128}, id="diagonal_keep"),
    # ───────────────────────── no-shared-index blowups ───────────────────────
    # Output is the full outer product — nothing contracts, so the result tensor
    # is as large as the dense intermediate. Pure broadcast cost.
    pytest.param("i,j->ij", {"i": 1024, "j": 1024}, id="outer_product"),
    pytest.param("ij,kl->ijkl", {"i": 32, "j": 32, "k": 32, "l": 32}, id="outer_4d"),
    # Element-wise: every index shared, none contracted.
    pytest.param("ij,ij->ij", {"i": 512, "j": 512}, id="hadamard"),
    # ───────────────────────── ordering-sensitive products ───────────────────
    # Skewed matrix chain: contracting middle-first (b,d small) is orders of
    # magnitude cheaper than the left-to-right order, which materializes a big
    # a×c intermediate. The classic "matrix chain order matters" case.
    pytest.param(
        "ab,bc,cd->ad", {"a": 256, "b": 2, "c": 256, "d": 2}, id="skewed_chain"
    ),
    pytest.param(
        "ab,bc,cd,de->ae",
        {"a": 50, "b": 40, "c": 30, "d": 20, "e": 10},
        id="chain_4",
    ),
    pytest.param(
        "ab,bc,cd,de,ef->af",
        {"a": 12, "b": 11, "c": 10, "d": 9, "e": 8, "f": 7},
        id="chain_5",
    ),
    # ───────────────────────── tensor-network shapes ─────────────────────────
    # Cyclic / hyperedge contractions with no tree decomposition into matmuls;
    # every operand shares indices with two others.
    pytest.param("ij,jk,ki->", {"i": 64, "j": 64, "k": 64}, id="trace_of_product"),
    pytest.param("ij,jk,ik->", {"i": 48, "j": 48, "k": 48}, id="triangle"),
    pytest.param("ijk,jl,kl->il", {"i": 24, "j": 24, "k": 24, "l": 24}, id="hyperedge"),
    # Star: many operands share one contracted index, fanning into a large
    # outer-product output.
    pytest.param(
        "ai,bi,ci,di->abcd",
        {"a": 8, "b": 8, "c": 8, "d": 8, "i": 32},
        id="star_contraction",
    ),
    # Bilinear / quadratic form over a batch (attention-score flavored).
    pytest.param("bi,ij,bj->b", {"b": 128, "i": 64, "j": 64}, id="bilinear"),
    # Batched matrix chain — batch axis rides through three contractions.
    pytest.param(
        "bij,bjk,bkl->bil",
        {"b": 16, "i": 24, "j": 24, "k": 24, "l": 24},
        id="batched_chain",
    ),
    # Multi-index contraction surface: a whole axis-group (c) contracts at once.
    pytest.param(
        "abc,cde->abde",
        {"a": 12, "b": 12, "c": 12, "d": 12, "e": 12},
        id="tensor_contraction",
    ),
    # Leading scalar factor plus an element-wise reduce — checks that the
    # rank-0 operand threads through without spawning a degenerate axis.
    pytest.param(",ij,ij->", {"i": 256, "j": 256}, id="scalar_scaled_reduce"),
]


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
