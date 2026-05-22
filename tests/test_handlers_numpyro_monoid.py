"""Unit tests for the rewrite rules in ``effectful.handlers.numpyro.monoid``.

Tests follow the conventions in ``test_ops_monoid.py``: each rule is verified
via a symbolic ``lhs`` and the expected post-rewrite ``rhs``. We assert both
syntactic equivalence after applying the rule and semantic equivalence under
random interpretations of the free body op.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from hypothesis import HealthCheck, given, settings

import effectful.handlers.jax.monoid  # noqa: F401  # registers jax monoid handlers
import effectful.handlers.jax.numpy as ejnp
from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax.monoid import LogSumExp
from effectful.handlers.numpyro.monoid import (
    NumpyroCategorical,
    NumpyroGaussHermite,
    NumpyroLogProb,
    NumpyroSampling,
)
from effectful.ops.monoid import (
    NormalizeIntp,
    Product,
    Sum,
    WeightedStream,
    stream,
    weighted,
)
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import deffn
from effectful.ops.types import Operation
from tests._monoid_helpers import (
    JAX_BACKEND,
    Backend,
    define_vars,
    random_interpretation,
    syntactic_eq_alpha,
)


@pytest.fixture
def backend() -> Backend:
    return JAX_BACKEND


def check_numpyro_rewrite(
    lhs,
    rhs,
    *,
    rule,
    backend: Backend,
    syntactic_rule=None,
    free_vars=(),
    max_examples: int = 25,
) -> None:
    """``check_rewrite`` variant for numpyro rules.

    ``syntactic_rule`` (default ``rule``) is installed for the syntactic
    step; ``rule`` itself is installed alongside :data:`NormalizeIntp` for
    the property-based semantic step so both sides can reduce to a value.
    """
    syn = syntactic_rule if syntactic_rule is not None else rule
    with handler(syn):
        norm = evaluate(lhs)
    assert syntactic_eq_alpha(norm, rhs)

    @given(intp=random_interpretation(free_vars))
    @settings(
        max_examples=max_examples,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def _check_semantics(intp):
        with handler(coproduct(NormalizeIntp, rule)), handler(intp):
            lhs_val = evaluate(lhs)
            rhs_val = evaluate(rhs)
        assert backend.eq(lhs_val, rhs_val)

    _check_semantics()


# ---------------------------------------------------------------------------
# NumpyroLogProb — pure structural rewrite
# ---------------------------------------------------------------------------


def test_logprob_lowering(backend):
    """``NumpyroLogProb`` replaces ``weighted(d)`` with
    ``WeightedStream(stream(d.support), d.log_prob, Sum)``.
    """
    a = define_vars("a", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")
    d = dist.Normal(0.0, 1.0)

    lhs = Sum.reduce(body(a()), {a: weighted(d)})
    rhs = Sum.reduce(
        body(a()),
        {
            a: WeightedStream(
                stream=stream(d.support), weight=d.log_prob, monoid=Sum
            )
        },
    )

    check_numpyro_rewrite(
        lhs=lhs, rhs=rhs, rule=NumpyroLogProb(), backend=backend, free_vars=[body]
    )


# ---------------------------------------------------------------------------
# NumpyroGaussHermite — replace weighted(Normal) with explicit n-node sum
# ---------------------------------------------------------------------------


def _gauss_hermite_nodes_weights(loc, scale, n, log_space: bool):
    u, w_raw = np.polynomial.hermite.hermgauss(n)
    u_jax = jnp.asarray(u, dtype=jnp.float32)
    w_jax = jnp.asarray(w_raw, dtype=jnp.float32)
    nodes = loc + jnp.sqrt(2.0) * scale * u_jax
    if log_space:
        weights = jnp.log(w_jax) - 0.5 * jnp.log(jnp.pi)
    else:
        weights = w_jax / jnp.sqrt(jnp.pi)
    return nodes, weights


def test_gauss_hermite_linear(backend):
    """Under ``Sum.reduce``, ``NumpyroGaussHermite`` lowers
    ``weighted(Normal(μ, σ))`` to a Product-weighted stream of ``n_nodes``
    nodes, which then reduces (via ``ReduceWeightedStream`` and the default
    rule) to the explicit weighted sum ``Σᵢ wᵢ · body(xᵢ)``.
    """
    n = 8
    loc, scale = 0.5, 1.3
    d = dist.Normal(loc, scale)

    a = define_vars("a", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")
    nodes, weights = _gauss_hermite_nodes_weights(loc, scale, n, log_space=False)

    lhs = Sum.reduce(body(a()), {a: weighted(d)})
    rhs = Sum.plus(
        *[
            Product.plus(jax_getitem(weights, (i,)), body(jax_getitem(nodes, (i,))))
            for i in range(n)
        ]
    )

    # Full pipeline (rule + NormalizeIntp) for syntactic comparison so the
    # opaque weight closure inside the WeightedStream gets reduced away.
    full = coproduct(NormalizeIntp, NumpyroGaussHermite(n_nodes=n))
    check_numpyro_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=NumpyroGaussHermite(n_nodes=n),
        syntactic_rule=full,
        backend=backend,
        free_vars=[body],
    )


def test_gauss_hermite_logsumexp(backend):
    """Under ``LogSumExp.reduce``, weights are log-space and combined via
    ``Sum`` (log-multiplication). The lowered form is
    ``LogSumExp.plus(Sum.plus(log_wᵢ, log_body(xᵢ)) for i)``.
    """
    n = 8
    loc, scale = 0.0, 1.0
    d = dist.Normal(loc, scale)

    a = define_vars("a", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")
    nodes, log_weights = _gauss_hermite_nodes_weights(loc, scale, n, log_space=True)

    lhs = LogSumExp.reduce(body(a()), {a: weighted(d)})
    rhs = LogSumExp.plus(
        *[
            Sum.plus(jax_getitem(log_weights, (i,)), body(jax_getitem(nodes, (i,))))
            for i in range(n)
        ]
    )

    full = coproduct(NormalizeIntp, NumpyroGaussHermite(n_nodes=n))
    check_numpyro_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=NumpyroGaussHermite(n_nodes=n),
        syntactic_rule=full,
        backend=backend,
        free_vars=[body],
    )


# ---------------------------------------------------------------------------
# NumpyroCategorical — replace weighted(Categorical) with explicit K-term sum
# ---------------------------------------------------------------------------


def test_categorical_probs_linear(backend):
    """Under ``Sum.reduce``, ``NumpyroCategorical`` lowers
    ``weighted(CategoricalProbs(probs))`` to ``Σᵢ probs[i] · body(i)``.
    """
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    d = dist.CategoricalProbs(probs=probs)
    k = probs.shape[-1]

    i_op = define_vars("i_op", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")
    indices = jnp.arange(k)

    lhs = Sum.reduce(body(i_op()), {i_op: weighted(d)})
    rhs = Sum.plus(
        *[
            Product.plus(jax_getitem(probs, (i,)), body(jax_getitem(indices, (i,))))
            for i in range(k)
        ]
    )

    full = coproduct(NormalizeIntp, NumpyroCategorical())
    check_numpyro_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=NumpyroCategorical(),
        syntactic_rule=full,
        backend=backend,
        free_vars=[body],
    )


def test_categorical_logits_matches_probs(backend):
    """``CategoricalLogits(log probs)`` and ``CategoricalProbs(probs)`` must
    lower to the same value under the same body.
    """
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    d_p = dist.CategoricalProbs(probs=probs)
    d_l = dist.CategoricalLogits(logits=jnp.log(probs))

    i_op = define_vars("i_op", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")

    expr_p = Sum.reduce(body(i_op()), {i_op: weighted(d_p)})
    expr_l = Sum.reduce(body(i_op()), {i_op: weighted(d_l)})

    @given(intp=random_interpretation([body]))
    @settings(
        max_examples=25,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def _check(intp):
        with handler(coproduct(NormalizeIntp, NumpyroCategorical())), handler(intp):
            r_p = evaluate(expr_p)
            r_l = evaluate(expr_l)
        assert backend.eq(r_p, r_l)

    _check()


def test_categorical_logsumexp(backend):
    """Under ``LogSumExp.reduce`` with ``CategoricalProbs``, weights are
    ``log(probs)`` combined via ``Sum`` (log-multiplication).
    """
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    d = dist.CategoricalProbs(probs=probs)
    k = probs.shape[-1]
    log_probs = jnp.log(probs)

    i_op = define_vars("i_op", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")
    indices = jnp.arange(k)

    lhs = LogSumExp.reduce(body(i_op()), {i_op: weighted(d)})
    rhs = LogSumExp.plus(
        *[
            Sum.plus(jax_getitem(log_probs, (i,)), body(jax_getitem(indices, (i,))))
            for i in range(k)
        ]
    )

    full = coproduct(NormalizeIntp, NumpyroCategorical())
    check_numpyro_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=NumpyroCategorical(),
        syntactic_rule=full,
        backend=backend,
        free_vars=[body],
    )


# ---------------------------------------------------------------------------
# NumpyroSampling — replace weighted(d) with a sample-backed WeightedStream
# ---------------------------------------------------------------------------


def test_sampling_linear(backend):
    """``NumpyroSampling`` lowers ``weighted(d)`` to a Product-weighted
    sample stream; the rewrite is deterministic for a fixed ``rng_key``.
    """
    n = 64
    key = jax.random.key(0)
    d = dist.Normal(0.0, 1.0)
    samples = d.sample(key, sample_shape=(n,))

    a = define_vars("a", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")

    lhs = Sum.reduce(body(a()), {a: weighted(d)})
    rhs = Sum.plus(
        *[
            Product.plus(1.0 / n, body(jax_getitem(samples, (i,))))
            for i in range(n)
        ]
    )

    full = coproduct(NormalizeIntp, NumpyroSampling(rng_key=key, n_samples=n))
    check_numpyro_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=NumpyroSampling(rng_key=key, n_samples=n),
        syntactic_rule=full,
        backend=backend,
        free_vars=[body],
    )


def test_sampling_logsumexp(backend):
    """Under ``LogSumExp.reduce``, ``NumpyroSampling`` uses log-uniform weights
    ``-log(N)`` combined via ``Sum`` (log-multiplication).
    """
    n = 64
    key = jax.random.key(0)
    d = dist.Normal(0.0, 1.0)
    samples = d.sample(key, sample_shape=(n,))
    log_w = -jnp.log(n)

    a = define_vars("a", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")

    lhs = LogSumExp.reduce(body(a()), {a: weighted(d)})
    rhs = LogSumExp.plus(
        *[
            Sum.plus(log_w, body(jax_getitem(samples, (i,))))
            for i in range(n)
        ]
    )

    full = coproduct(NormalizeIntp, NumpyroSampling(rng_key=key, n_samples=n))
    check_numpyro_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=NumpyroSampling(rng_key=key, n_samples=n),
        syntactic_rule=full,
        backend=backend,
        free_vars=[body],
    )
