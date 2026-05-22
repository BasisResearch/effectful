"""Unit tests for the rewrite rules in ``effectful.handlers.numpyro.monoid``.

Tests follow the conventions in ``test_ops_monoid.py``: each rule is verified
via a symbolic ``lhs`` and the expected post-rewrite ``rhs``. We assert both
syntactic equivalence after applying the rule and semantic equivalence under
random interpretations of the free body op.

For the categorical rule the lowered form is naturally a length-K explicit
sum, but ``ArrayReduce`` (inside ``NormalizeIntp``) further collapses that
into a single named-dim ``jnp.sum``; the RHS therefore matches that final
form, in the style of ``test_jax_weighted_reduce``.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from hypothesis import HealthCheck, given, settings

import effectful.handlers.jax.monoid  # noqa: F401  # registers jax monoid handlers
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.monoid import LogSumExp
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.handlers.numpyro.monoid import NumpyroCategorical
from effectful.ops.monoid import NormalizeIntp, Sum, weighted
from effectful.ops.semantics import coproduct, evaluate, handler
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
# NumpyroCategorical
# ---------------------------------------------------------------------------


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

    full = coproduct(NormalizeIntp, NumpyroCategorical())
    check_numpyro_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=NumpyroCategorical(),
        syntactic_rule=full,
        backend=backend,
        free_vars=[k, body],
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

    full = coproduct(NormalizeIntp, NumpyroCategorical())
    check_numpyro_rewrite(
        lhs=lhs,
        rhs=rhs,
        rule=NumpyroCategorical(),
        syntactic_rule=full,
        backend=backend,
        free_vars=[k, body],
    )
