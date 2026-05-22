"""Unit tests for the rewrite rules in ``effectful.handlers.numpyro.monoid``.

Tests follow the conventions in ``test_ops_monoid.py``: each rule is verified
via a symbolic ``lhs`` and the expected post-rewrite ``rhs``. The numpyro
categorical rule is part of :data:`NormalizeIntp`, so plain ``check_rewrite``
suffices.
"""

import jax
import numpyro.distributions as dist
import pytest

import effectful.handlers.jax.monoid  # noqa: F401  # registers jax monoid handlers
import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numpyro.monoid  # noqa: F401  # registers numpyro monoid handlers
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.monoid import LogSumExp
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import NormalizeIntp, Sum, weighted
from tests._monoid_helpers import JAX_BACKEND, Backend, check_rewrite, define_vars


@pytest.fixture
def backend() -> Backend:
    return JAX_BACKEND


def test_categorical_probs_linear(backend):
    """Under ``Sum.reduce`` with ``CategoricalProbs``, the per-index weight
    is ``probs[i]`` (linear) and the lowered form is the named-dim
    ``jnp.sum(probs[k] * body(indices[k]))``.
    """
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    d = dist.CategoricalProbs(probs=probs)
    indices = jnp.arange(probs.shape[-1])

    i_op, k = define_vars("i_op", "k", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")

    lhs = Sum.reduce(body(i_op()), {i_op: weighted(d)})
    rhs = jnp.sum(
        bind_dims(unbind_dims(probs, k) * body(unbind_dims(indices, k)), k), axis=0
    )

    check_rewrite(
        lhs=lhs, rhs=rhs, rule=NormalizeIntp, backend=backend, free_vars=[k, body]
    )


def test_categorical_logits_linear(backend):
    """Under ``Sum.reduce`` with ``CategoricalLogits``, the per-index weight
    is ``softmax(logits)[i]`` and the lowered form matches the same
    named-dim ``jnp.sum`` shape as the probs case.
    """
    logits = jnp.array([0.5, -1.0, 2.0, 0.1])
    d = dist.CategoricalLogits(logits=logits)
    probs = jax.nn.softmax(logits, axis=-1)
    indices = jnp.arange(logits.shape[-1])

    i_op, k = define_vars("i_op", "k", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")

    lhs = Sum.reduce(body(i_op()), {i_op: weighted(d)})
    rhs = jnp.sum(
        bind_dims(unbind_dims(probs, k) * body(unbind_dims(indices, k)), k), axis=0
    )

    check_rewrite(
        lhs=lhs, rhs=rhs, rule=NormalizeIntp, backend=backend, free_vars=[k, body]
    )


def test_categorical_logsumexp(backend):
    """Under ``LogSumExp.reduce`` with ``CategoricalProbs``, weights are
    ``log(probs)`` combined via ``Sum`` (log-multiplication); the lowered
    form is ``logsumexp(log_probs[k] + body(indices[k]))`` along the
    named dim.
    """
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    d = dist.CategoricalProbs(probs=probs)
    log_probs = jnp.log(probs)
    indices = jnp.arange(probs.shape[-1])

    i_op, k = define_vars("i_op", "k", typ=backend.scalar_typ)
    body = backend.fresh_op("body", n_args=1, ret="scalar")

    lhs = LogSumExp.reduce(body(i_op()), {i_op: weighted(d)})
    rhs = logsumexp(
        bind_dims(unbind_dims(log_probs, k) + body(unbind_dims(indices, k)), k),
        axis=0,
    )

    check_rewrite(
        lhs=lhs, rhs=rhs, rule=NormalizeIntp, backend=backend, free_vars=[k, body]
    )
