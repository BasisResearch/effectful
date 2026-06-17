import functools

import jax
import jax.dlpack
import pyro.ops.contract
import pytest
import torch
from jax import random as random

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, jax_getitem, unbind_dims
from effectful.handlers.jax.monoid import (
    ARRAY_REDUCTORS,
    ReduceArray,
    ReduceArrayGather,
    ReduceDeltaSimpleRange,
    ReduceDependentRangeMask,
    ReduceSumProductContraction,
    _einsum_expr,
    einsum,
)
from effectful.ops.monoid import (
    CartesianProduct,
    DeltaEmpty,
    EvaluateIntp,
    NormalizeIntp,
    Product,
    Sum,
    Union,
    delta,
)
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import Array, deffn
from effectful.ops.types import Operation
from tests._monoid_helpers import JaxBackend, syntactic_eq_alpha

MONOIDS = [
    pytest.param(monoid, reductor, id=monoid.__name__)
    for (monoid, reductor) in ARRAY_REDUCTORS.items()
]


@pytest.fixture(scope="module")
def rng_key():
    return random.PRNGKey(0)


@pytest.fixture
def backend() -> JaxBackend:
    return JaxBackend()


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_gather(monoid, reductor, backend: JaxBackend):
    (x, k) = backend.define_vars("x", "k", ret="scalar")
    X = jnp.arange(3)

    lhs = monoid.reduce(x(), {x: X})
    rhs = monoid.reduce(unbind_dims(X, k), {k: range(X.shape[0])})
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceArrayGather())


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_gather_dep(monoid, reductor, backend: JaxBackend):
    (x, y) = backend.define_vars("x", "y", ret="scalar")
    f = backend.define_vars("f", arg_types=(backend.scalar_typ,), ret="stream")
    g = backend.define_vars(
        "g", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )
    X = jnp.arange(3)

    lhs = monoid.reduce(g(x(), y()), {y: f(x()), x: X})
    rhs = monoid.reduce(
        g(unbind_dims(X[:3], x), y()), {y: f(x()), x: range(X.shape[0])}
    )
    backend.check_rewrite(
        lhs=lhs, rhs=rhs, rule=coproduct(ReduceArrayGather(), ReduceDeltaSimpleRange())
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_1(monoid, reductor, backend: JaxBackend):
    (x, k) = backend.define_vars("x", "k", ret="scalar")
    X = jnp.arange(5)

    lhs = monoid.reduce(x(), {x: X})
    rhs = reductor(bind_dims(unbind_dims(X, k), k), axis=(0,))
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=functools.reduce(
            coproduct,  # type: ignore[arg-type]
            [ReduceArrayGather(), ReduceArray(), ReduceDeltaSimpleRange()],
        ),
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_2(monoid, reductor, backend: JaxBackend):
    (x, y, k1, k2) = backend.define_vars("x", "y", "k1", "k2", ret="scalar")
    X = jnp.arange(5)
    Y = jnp.arange(7)
    f = backend.define_vars(
        "f", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = monoid.reduce(f(x(), y()), {x: X, y: Y})
    rhs = reductor(
        bind_dims(f(unbind_dims(X, k1), unbind_dims(Y, k2)), k1, k2), axis=(0, 1)
    )
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=functools.reduce(
            coproduct,  # type: ignore[arg-type]
            [ReduceArrayGather(), ReduceArray(), ReduceDeltaSimpleRange()],
        ),
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_3(monoid, reductor, backend: JaxBackend):
    """Stream `y` is `g(x())` — depends on the bound element of X. The reducer
    must inline ``g`` along the same named dim used to unbind `x`."""
    (x, y, k1, k2) = backend.define_vars("x", "y", "k1", "k2", ret="scalar")
    X = jnp.arange(5)

    f = backend.define_vars(
        "f", arg_types=[backend.scalar_typ, backend.scalar_typ], ret="scalar"
    )
    g = backend.define_vars("g", arg_types=[backend.scalar_typ], ret="stream")

    lhs = monoid.reduce(f(x(), y()), {x: X, y: g(x())})
    rhs = reductor(
        bind_dims(
            monoid.reduce(f(unbind_dims(X, x), y()), {y: g(unbind_dims(X, x))}), x
        ),
        axis=(0,),
    )
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=functools.reduce(
            coproduct,  # type: ignore[arg-type]
            [
                ReduceArrayGather(),
                ReduceArray(),
                ReduceDeltaSimpleRange(),
                DeltaEmpty(),
            ],
        ),
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_arange_reduce_direct_full(monoid, reductor, backend: JaxBackend):
    """A full-range direct index ``A[v()]`` over ``v: arange(N)`` slices the
    whole axis (``A[0:N:1]``) and reduces it -- no materialized-arange gather.
    """
    (v, k) = backend.define_vars("v", "k", ret="scalar")
    A = backend.define_vars("A", ret="stream")

    lhs = monoid.reduce(jax_getitem(A(), [v()]), {v: range(7)})
    rhs = reductor(
        bind_dims(jax_getitem(jax_getitem(A(), [slice(0, 7, 1)]), [k()]), k),
        axis=(0,),
    )
    backend.check_rewrite(
        lhs=lhs, rhs=rhs, rule=coproduct(ReduceArray(), ReduceDeltaSimpleRange())
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_arange_reduce_indirect(monoid, reductor, backend: JaxBackend):
    """When the range var is used both as a direct index and as a value
    (``A[v()] + v()``), the direct use slices and the indirect use materializes
    the range, both aligned on the same fresh dim."""
    (v, k) = backend.define_vars("v", "k", ret="scalar")
    A = jnp.arange(10)

    lhs = monoid.reduce(jax_getitem(A, [v()]) + v(), {v: range(5)})
    rhs = reductor(
        bind_dims(
            jax_getitem(jax_getitem(A, [slice(0, 5, 1)]), [k()])
            + unbind_dims(jnp.arange(5), k),
            k,
        ),
        axis=(0,),
    )
    backend.check_rewrite(
        lhs=lhs, rhs=rhs, rule=coproduct(ReduceArray(), ReduceDeltaSimpleRange())
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_arange_reduce_two_streams(monoid, reductor, backend: JaxBackend):
    """Two arange streams indexing a 2-D array slice both axes and reduce over
    both at once."""
    (u, w, k1, k2) = backend.define_vars("u", "w", "k1", "k2", ret="scalar")
    A = jnp.arange(8 * 9).reshape((8, 9))

    lhs = monoid.reduce(jax_getitem(A, [u(), w()]), {u: range(4), w: range(5)})
    rhs = reductor(
        bind_dims(
            jax_getitem(jax_getitem(A, [slice(0, 4, 1), slice(0, 5, 1)]), [k1(), k2()]),
            k1,
            k2,
        ),
        axis=(0, 1),
    )
    backend.check_rewrite(
        lhs=lhs, rhs=rhs, rule=coproduct(ReduceArray(), ReduceDeltaSimpleRange())
    )


# ---------------------------------------------------------------------------
# Delta rules. All tests use the operation form ``delta(idx, body)`` rather
# than the ``Delta`` dataclass; the delta op is the user-facing surface.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_delta_empty(monoid, reductor, backend: JaxBackend):
    """An empty-index delta unwraps to its body.

    reduce(M, streams, delta((), body)) ≡ reduce(M, streams, body)
    """
    x = backend.define_vars("x", ret="scalar")
    X = backend.define_vars("X", ret="stream")

    lhs = monoid.reduce(delta((), x()), {x: X()})
    rhs = monoid.reduce(x(), {x: X()})
    backend.check_rewrite(
        lhs=lhs, rhs=rhs, rule=coproduct(ReduceDeltaSimpleRange(), DeltaEmpty())
    )


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_delta_empty_arange(monoid, reductor, backend: JaxBackend):
    x = backend.define_vars("x", ret="scalar")
    f = backend.define_vars("f", arg_types=[backend.scalar_typ], ret="scalar")

    lhs = monoid.reduce(delta((x(),), f(x())), {x: range(0)})
    rhs = bind_dims(f(unbind_dims(jnp.array([]), x)), x)
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDeltaSimpleRange())


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_delta_independent_one(monoid, reductor, backend: JaxBackend):
    """One R1 step: peel the final preserved index off a delta.

    reduce(M, {y: Y()}, delta((y(),), f(y()))) ≡ bind_dims(f(unbind_dims(Y(), k)), k)
    """
    (y, k) = backend.define_vars("y", "k", ret="scalar")
    f = backend.define_vars("f", arg_types=[backend.scalar_typ], ret="scalar")

    # We use a concrete range here instead of an abstract one, because
    # unbind_dims is undefined on empty arrays (and the rewrite produces a
    # different rhs in this case)
    lhs = monoid.reduce(delta((y(),), f(y())), {y: range(3)})
    rhs = bind_dims(f(unbind_dims(jnp.arange(3), k)), k)
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDeltaSimpleRange())


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_delta_independent_preserves_others(
    monoid, reductor, backend: JaxBackend
):
    """R1 peels only the final index. Streams not matching the peeled index op
    stay untouched, as do earlier entries in the index tuple.

    reduce(M, {x: X(), y: Y()}, delta((x(), y()), f(x(), y())))
    ≡ reduce(M, {x: X()}, delta((x(),), bind_dims(f(x(), unbind_dims(Y(), k)), k)))
    """
    (x, y, k) = backend.define_vars("x", "y", "k", ret="scalar")
    f = backend.define_vars(
        "f", arg_types=[backend.scalar_typ, backend.scalar_typ], ret="scalar"
    )

    lhs = monoid.reduce(delta((x(), y()), f(x(), y())), {x: range(2), y: range(3)})
    rhs = bind_dims(
        bind_dims(f(unbind_dims(jnp.arange(2), x), unbind_dims(jnp.arange(3), k)), k), x
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDeltaSimpleRange())


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_delta_simple_dep(monoid, reductor, backend: JaxBackend):
    (x, y) = backend.define_vars("x", "y", ret="scalar")
    X = jnp.arange(3)

    lhs = monoid.reduce(
        delta((x(),), unbind_dims(X, x) + y()),
        {x: range(3), y: jnp.stack([x(), x() + 1])},
    )
    rhs = bind_dims(
        monoid.reduce(
            delta((), unbind_dims(X, x) + y()),
            {
                y: jnp.stack(
                    [unbind_dims(jnp.arange(3), x), unbind_dims(jnp.arange(3), x) + 1]
                )
            },
        ),
        x,
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDeltaSimpleRange())


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_dependent_range_mask(monoid, reductor, backend: JaxBackend):
    """A dependent range stream gets rewritten to the referent's bbox stream,
    with the original constraint folded into the body as a where-guard.

    reduce(M, {u: range(0, N, 1), v: range(0, u(), 1)}, body)
    ≡ reduce(M, {u: range(0, N, 1), v: range(0, N, 1)}, where(v() < u(), body, M.identity))
    """
    (u, v) = backend.define_vars("u", "v", ret="scalar")
    N = 5
    f = backend.define_vars(
        "f", arg_types=[backend.scalar_typ, backend.scalar_typ], ret="scalar"
    )

    body = f(u(), v())

    lhs = monoid.reduce(body, {u: range(N), v: jnp.arange(u())})
    rhs = monoid.reduce(
        jnp.where(v() < u(), body, monoid.identity), {u: range(N), v: range(N)}
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDependentRangeMask())


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_dependent_range_mask_delta_body(monoid, reductor, backend: JaxBackend):
    """When the body is a delta term, R4 folds the constraint into the delta's
    weight while leaving its index tuple untouched.

    reduce(M, {u: range(N), v: range(u())}, delta((u(), v()), w))
    ≡ reduce(M, {u: range(N), v: range(N)},
             delta((u(), v()), where(v() < u(), w, M.identity)))
    """
    (u, v) = backend.define_vars("u", "v", ret="scalar")
    N = 5
    f = backend.define_vars(
        "f", arg_types=[backend.scalar_typ, backend.scalar_typ], ret="scalar"
    )

    weight = f(u(), v())
    idx = (u(), v())

    lhs = monoid.reduce(delta(idx, weight), {u: range(N), v: jnp.arange(u())})
    rhs = monoid.reduce(
        delta(idx, jnp.where(v() < u(), weight, monoid.identity)),
        {u: range(N), v: range(N)},
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDependentRangeMask())


def test_reduce_contraction_single(backend: JaxBackend):
    i = backend.define_vars("i", ret="scalar")
    (A, B) = backend.define_vars(
        "A", "B", arg_types=(backend.scalar_typ,), ret="scalar"
    )

    lhs = Sum.reduce(Product.plus(A(i()), B(i())), {i: range(5)})
    rhs = jnp.einsum(
        "a...,a...->...",
        Sum.reduce(delta((i(),), A(i())), {i: range(5)}),
        Sum.reduce(delta((i(),), B(i())), {i: range(5)}),
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceSumProductContraction())


def test_reduce_contraction_double(backend: JaxBackend):
    i, j = backend.define_vars("i", "j", ret="scalar")
    (A, B) = backend.define_vars(
        "A", "B", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = Sum.reduce(Product.plus(A(i(), j()), B(i(), j())), {i: range(5), j: range(7)})
    rhs = jnp.einsum(
        "ab...,ab...->...",
        Sum.reduce(delta((i(), j()), A(i(), j())), {i: range(5), j: range(7)}),
        Sum.reduce(delta((i(), j()), B(i(), j())), {i: range(5), j: range(7)}),
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceSumProductContraction())


def test_reduce_matmul(backend: JaxBackend):
    key = jax.random.PRNGKey(0)
    # Define dimensions
    B, I, J, K = 2, 3, 4, 5

    # Create sample matrices
    X = random.normal(key, (B, I, J))
    Y = random.normal(key, (B, J, K))
    (b, i, j, k) = backend.define_vars("b", "i", "j", "k", ret="scalar")

    with handler(NormalizeIntp):
        norm = Sum.reduce(
            delta((b(), i(), k()), unbind_dims(X, b, i, j) * unbind_dims(Y, b, j, k)),
            {b: range(B), i: range(I), j: range(J), k: range(K)},
        )
    with handler(EvaluateIntp), handler(NormalizeIntp):
        actual = evaluate(norm)

    expected = jnp.einsum("bij,bjk->bik", X, Y)
    assert jnp.allclose(actual, expected)


EINSUM_CASES = [
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
    # in ReduceDeltaSimpleRange (no explicit jnp.diagonal step).
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


@pytest.mark.parametrize(
    "impl", [pytest.param(jnp.einsum, id="jax"), pytest.param(einsum, id="effectful")]
)
@pytest.mark.parametrize("spec,sizes", EINSUM_CASES)
@pytest.mark.benchmark(warmup=True, warmup_iterations=1)
def test_einsum_bench(benchmark, impl, spec, sizes, rng_key):
    """Time one ``(spec, impl)`` pair. Group by ``spec`` to compare ``jnp``
    against ``effectful`` for the same subscript pattern (see module docstring).
    """
    operands = _make_operands(spec, sizes, rng_key)

    @jax.jit
    def f(*operands):
        return impl(spec, *operands)

    @benchmark
    def _run():
        return f(*operands).block_until_ready()


@pytest.mark.parametrize("spec,sizes", EINSUM_CASES)
def test_einsum_matches_jnp(spec: str, sizes, rng_key):
    """``einsum`` returns the same result as ``jnp.einsum`` for every spec
    in ``EINSUM_EXAMPLES``.
    """
    operands = _make_operands(spec, sizes, rng_key)
    actual = einsum(spec, *operands)
    expected = jnp.einsum(spec, *operands)
    assert actual.shape == expected.shape, (
        f"shape mismatch for {spec!r}: got {actual.shape}, expected {expected.shape}"
    )
    assert jnp.allclose(actual, expected, atol=1e-4, rtol=1e-4), (
        f"value mismatch for {spec!r}"
    )


# see https://github.com/pyro-ppl/pyro/blob/dev/tests/ops/test_contract.py
# Let abcde be enum dims and ijk be plates.
PLATED_EINSUM_CASES = [
    ("abi,abi->", "i"),
    ("aij,bi->", "ij"),
    ("aij,bi,c->", "ij"),
    ("abi,b->", "i"),
]


@pytest.mark.parametrize("spec,plates", PLATED_EINSUM_CASES)
def test_plated_einsum(spec, plates, rng_key):
    def _to_torch(arr):
        return torch.from_dlpack(arr)

    operands = _make_operands(
        spec,
        {
            "a": 2,
            "b": 3,
            "c": 4,
            "d": 5,
            "e": 6,
            "f": 7,
            "g": 8,
            "i": 2,
            "j": 3,
            "k": 4,
        },
        rng_key,
    )
    torch_operands = (_to_torch(op) for op in operands)

    try:
        expected = pyro.ops.contract.naive_ubersum(
            spec, *torch_operands, plates=plates, backend="torch"
        )[0]
    except NotImplementedError:
        pytest.skip("Not implemented by pyro.ops.contract.einsum")

    actual = einsum(spec, *operands, plates=plates)
    assert actual.shape == expected.shape, (
        f"shape mismatch for {spec!r}: got {actual.shape}, expected {expected.shape}"
    )
    assert torch.allclose(_to_torch(actual), expected, atol=1e-4, rtol=1e-4), (
        f"value mismatch for {spec!r}"
    )
