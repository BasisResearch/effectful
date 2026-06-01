import functools
import typing

import jax
import pytest
from jax import random as random

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.monoid import (
    ArrayReduce,
    LogSumExp,
    ProductPlusJax,
    ReduceDeltaIndependent,
    ReduceDependentRangeMask,
    ReduceSumProductContraction,
    delta,
    einsum,
)
from effectful.handlers.jax.monoid import range as Range
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import (
    Max,
    Min,
    NormalizeIntp,
    Product,
    ReduceWeightedStream,
    Sum,
)
from effectful.ops.semantics import coproduct, handler
from effectful.ops.types import Interpretation
from tests._monoid_helpers import JaxBackend

MONOIDS = [
    pytest.param(Sum, jnp.sum, id="Sum"),
    pytest.param(Product, jnp.prod, id="Product"),
    pytest.param(Min, jnp.min, id="Min"),
    pytest.param(Max, jnp.max, id="Max"),
    pytest.param(LogSumExp, logsumexp, id="LogSumExp"),
]


@pytest.fixture
def backend() -> JaxBackend:
    return JaxBackend()


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_1(monoid, reductor, backend: JaxBackend):
    (x, k) = backend.define_vars("x", "k", ret="scalar")
    X = backend.define_vars("X", ret="stream")

    lhs = monoid.reduce(x(), {x: X()})
    rhs = reductor(bind_dims(unbind_dims(X(), k), k), axis=0)
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ArrayReduce())


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_2(monoid, reductor, backend: JaxBackend):
    (x, y, k1, k2) = backend.define_vars("x", "y", "k1", "k2", ret="scalar")
    (X, Y) = backend.define_vars("X", "Y", ret="stream")
    f = backend.define_vars(
        "f", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = monoid.reduce(f(x(), y()), {x: X(), y: Y()})
    rhs = reductor(
        bind_dims(
            reductor(
                bind_dims(f(unbind_dims(X(), k1), unbind_dims(Y(), k2)), k2),
                axis=0,
            ),
            k1,
        ),
        axis=0,
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ArrayReduce())


@pytest.mark.parametrize("monoid,reductor", MONOIDS)
def test_reduce_array_3(monoid, reductor, backend: JaxBackend):
    """Stream `y` is `g(x())` — depends on the bound element of X. The reducer
    must inline ``g`` along the same named dim used to unbind `x`."""
    (x, y, k1, k2) = backend.define_vars("x", "y", "k1", "k2", ret="scalar")
    X = backend.define_vars("X", ret="stream")

    f = backend.define_vars(
        "f", arg_types=[backend.scalar_typ, backend.scalar_typ], ret="scalar"
    )
    g = backend.define_vars("g", arg_types=[backend.scalar_typ], ret="stream")

    lhs = monoid.reduce(f(x(), y()), {x: X(), y: g(x())})
    rhs = reductor(
        bind_dims(
            reductor(
                bind_dims(
                    f(unbind_dims(X(), k1), unbind_dims(g(unbind_dims(X(), k1)), k2)),
                    k2,
                ),
                axis=0,
            ),
            k1,
        ),
        axis=0,
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ArrayReduce())


def test_jax_weighted_reduce(backend: JaxBackend):
    """Sum over a single stream with ``Product`` weights lowers to
    ``jnp.sum(w(X) * body(X))`` under ``NormalizeIntp`` ∘ ``ArrayReduce``.

    Verifies that the desugaring rule composes cleanly with the JAX lowering
    so existing handlers need no changes to support weighted streams.

    """
    (x, k) = backend.define_vars("x", "k", ret="scalar")
    X = backend.define_vars("X", ret="stream")
    body = backend.define_vars("body", arg_types=[backend.scalar_typ], ret="scalar")
    w = backend.define_vars("w", arg_types=[backend.scalar_typ], ret="scalar")

    ws = Product.weighted(X(), w)
    lhs = Sum.reduce(body(x()), {x: ws})
    rhs = jnp.sum(
        bind_dims(w(unbind_dims(X(), k)) * body(unbind_dims(X(), k)), k), axis=0
    )
    backend.check_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=functools.reduce(
            coproduct,
            typing.cast(
                list[Interpretation],
                [ReduceWeightedStream(), ArrayReduce(), ProductPlusJax()],
            ),
        ),
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
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDeltaIndependent())


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
    lhs = monoid.reduce(delta((y(),), f(y())), {y: Range(3)})
    rhs = bind_dims(f(unbind_dims(jnp.arange(3), k)), k)
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDeltaIndependent())


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

    lhs = monoid.reduce(delta((x(), y()), f(x(), y())), {x: Range(2), y: Range(3)})
    rhs = bind_dims(
        bind_dims(f(unbind_dims(jnp.arange(2), x), unbind_dims(jnp.arange(3), k)), k), x
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDeltaIndependent())


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

    lhs = monoid.reduce(body, {u: Range(0, N, 1), v: Range(0, u(), 1)})
    rhs = monoid.reduce(
        jnp.where(v() < u(), body, monoid.identity),
        {u: Range(0, N, 1), v: Range(0, N, 1)},
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

    lhs = monoid.reduce(delta(idx, weight), {u: Range(0, N, 1), v: Range(0, u(), 1)})
    rhs = monoid.reduce(
        delta(idx, jnp.where(v() < u(), weight, monoid.identity)),
        {u: Range(0, N, 1), v: Range(0, N, 1)},
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceDependentRangeMask())


def test_reduce_contraction_single(backend: JaxBackend):
    i = backend.define_vars("i", ret="scalar")
    I = backend.define_vars("I", ret="stream")
    (A, B) = backend.define_vars(
        "A", "B", arg_types=(backend.scalar_typ,), ret="scalar"
    )

    lhs = Sum.reduce(Product.plus(A(i()), B(i())), {i: I()})
    rhs = Product.plus(
        jnp.tensordot(
            bind_dims(A(unbind_dims(I(), i)), i),
            bind_dims(B(unbind_dims(I(), i)), i),
            axes=((0,), (0,)),
        )
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceSumProductContraction())


def test_reduce_contraction_double(backend: JaxBackend):
    i, j = backend.define_vars("i", "j", ret="scalar")
    I, J = backend.define_vars("I", "J", ret="stream")
    (A, B) = backend.define_vars(
        "A", "B", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = Sum.reduce(Product.plus(A(i(), j()), B(i(), j())), {i: I(), j: J()})
    rhs = Product.plus(
        jnp.tensordot(
            bind_dims(A(unbind_dims(I(), i), unbind_dims(J(), j)), i, j),
            bind_dims(B(unbind_dims(I(), i), unbind_dims(J(), j)), i, j),
            axes=((0, 1), (0, 1)),
        )
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceSumProductContraction())


def test_reduce_contraction_multi_factor(backend: JaxBackend):
    i, j = backend.define_vars("i", "j", ret="scalar")
    I, J = backend.define_vars("I", "J", ret="stream")
    B = backend.define_vars(
        "B", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )
    A, C = backend.define_vars("A", "C", arg_types=(backend.scalar_typ,), ret="scalar")

    lhs = Sum.reduce(Product.plus(A(i()), B(i(), j()), C(j())), {i: I(), j: J()})
    rhs = Product.plus(
        jnp.tensordot(
            bind_dims(
                jnp.tensordot(
                    bind_dims(A(unbind_dims(I(), i)), i),
                    bind_dims(B(unbind_dims(I(), i), unbind_dims(J(), j)), i),
                    axes=((0,), (0,)),
                ),
                j,
            ),
            bind_dims(C(unbind_dims(J(), j)), j),
            axes=((0,), (0,)),
        )
    )
    backend.check_rewrite(lhs=lhs, rhs=rhs, rule=ReduceSumProductContraction())


def test_reduce_contraction_shared_only(backend: JaxBackend):
    i, j, k = backend.define_vars("i", "j", "k", ret="scalar")
    I, J, K = backend.define_vars("I", "J", "K", ret="stream")
    A = backend.define_vars(
        "A",
        arg_types=(backend.scalar_typ, backend.scalar_typ, backend.scalar_typ),
        ret="scalar",
    )
    B, C = backend.define_vars(
        "B", "C", arg_types=(backend.scalar_typ, backend.scalar_typ), ret="scalar"
    )

    lhs = Sum.reduce(
        Product.plus(A(i(), j(), k()), B(i(), j()), C(j(), k())),
        {i: I(), j: J(), k: K()},
    )
    rhs = Product.plus(
        jnp.tensordot(
            bind_dims(
                jnp.tensordot(
                    bind_dims(
                        A(
                            unbind_dims(I(), i),
                            unbind_dims(J(), j),
                            unbind_dims(K(), k),
                        ),
                        i,
                    ),
                    bind_dims(B(unbind_dims(I(), i), unbind_dims(J(), j)), i),
                    axes=((0,), (0,)),
                ),
                j,
                k,
            ),
            bind_dims(C(unbind_dims(J(), j), unbind_dims(K(), k)), j, k),
            axes=((0, 1), (0, 1)),
        )
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
        actual = Sum.reduce(
            delta((b(), i(), k()), unbind_dims(X, b, i, j) * unbind_dims(Y, b, j, k)),
            {b: Range(B), i: Range(I), j: Range(J), k: Range(K)},
        )

    expected = jnp.einsum("bij,bjk->bik", X, Y)
    assert jnp.allclose(actual, expected)


# Default per-letter sizes used to materialize random operands for the einsum
# correctness tests. Distinct primes keep transposes from sneaking by.
_EINSUM_DIM_SIZES = {
    "a": 2,
    "b": 3,
    "c": 4,
    "d": 5,
    "e": 6,
    "i": 7,
    "j": 8,
    "k": 5,
}


EINSUM_EXAMPLES = [
    # vector operations
    "i->i",  # do nothing
    "i->",  # vector sum
    ",i->i",  # scalar-vector product
    "i,i->",  # inner product
    "i,j->ij",  # outer product
    "i,i->i",  # element-wise product
    # matrix operations
    "ij->ij",  # do nothing
    "ij->ji",  # matrix transpose
    "ii->",  # matrix trace
    "ii->i",  # matrix diagonal
    ",ij->ij",  # scalar-matrix product
    "ij,j->i",  # matrix-vector product
    "ij,ij->ij",  # hadamard product
    "ij,jk->ik",  # matrix-matrix product
    # composite contractions
    "ab,a->",
    "a,a,a,ab->ab",
    "ab,bc,cd->da",
    "ai->i",
    ",ai,abij->ij",
    "a,ai,bij->ij",
    "ai,abi,bci,cdi->i",
    "aij,abij,bcij->ij",
    "a,abi,bcij,cdij->ij",
]


def _make_einsum_operands(spec: str, key: jax.Array) -> list[jax.Array]:
    in_part = spec.split("->")[0] if "->" in spec else spec
    in_specs = in_part.split(",")
    keys = random.split(key, max(len(in_specs), 1))
    operands = []
    for k, s in zip(keys, in_specs, strict=True):
        shape = tuple(_EINSUM_DIM_SIZES[c] for c in s) if s else ()
        operands.append(random.normal(k, shape))
    return operands


@pytest.mark.parametrize("spec", EINSUM_EXAMPLES)
def test_einsum_matches_jnp(spec: str):
    """``einsum`` returns the same result as ``jnp.einsum`` for every spec
    in ``EINSUM_EXAMPLES``.
    """
    key = jax.random.PRNGKey(hash(spec) & 0xFFFFFFFF)
    operands = _make_einsum_operands(spec, key)
    actual = einsum(spec, *(arr.shape for arr in operands))(*operands)
    expected = jnp.einsum(spec, *operands)
    assert actual.shape == expected.shape, (
        f"shape mismatch for {spec!r}: got {actual.shape}, expected {expected.shape}"
    )
    assert jnp.allclose(actual, expected, atol=1e-4, rtol=1e-4), (
        f"value mismatch for {spec!r}"
    )
