import effectful.handlers.jax.distribution as dist
import effectful.handlers.jax.numpy as jnp
import jax
from effectful.handlers.jax import sizesof
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import deffn, defop

from weighted.fold_lang_v1 import (
    ArgMinAlg,
    GradientOptimizationFold,
    LikelihoodWeightingFold,
    LinAlg,
    dense_fold_intp,
    fold,
    log_prob,
    reals,
    sample,
)


def run_svi(data):
    latent_fairness = defop(jax.Array, name="latent_fairness")
    latent_fairness_w = defop(jax.Array, name="latent_fairness_w")
    alpha_q = defop(jax.Array, name="alpha_q")
    beta_q = defop(jax.Array, name="beta_q")

    def model_log_prob(data):
        """Return the log joint probability of the latent variables and the observed data according to the model."""
        alpha0 = jnp.array(10.0)
        beta0 = jnp.array(10.0)
        beta_prior = dist.Beta(alpha0, beta0)
        return log_prob(beta_prior, latent_fairness()) + jnp.sum(log_prob(dist.BernoulliProbs(latent_fairness()), data))

    with (
        handler(
            GradientOptimizationFold(
                steps=2000, learning_rate=0.1, progress=True, init={alpha_q: jnp.array(1.0), beta_q: jnp.array(1.0)}
            )
        ),
        handler(LikelihoodWeightingFold(samples=100)),
        handler(dense_fold_intp),
    ):
        elbo = fold(
            LinAlg,
            {(latent_fairness, latent_fairness_w): dist.Beta(alpha_q(), beta_q())},
            -(jnp.exp(latent_fairness_w()) * (model_log_prob(data) - latent_fairness_w())),
        )

        with handler({alpha_q: deffn(jnp.array(15.0)), beta_q: deffn(jnp.array(15.0))}):
            _sample = sample(jax.random.key(0), dist.Beta(alpha_q(), beta_q()), (1,))
            x = evaluate(elbo)
            assert isinstance(x, jax.Array) and len(x.shape) == 0 and len(sizesof(x)) == 0

        (_, (alpha_est, beta_est)) = fold(ArgMinAlg, {alpha_q: reals(), beta_q: reals()}, (elbo, (alpha_q(), beta_q())))

        inferred_prob = alpha_est / (alpha_est + beta_est)
    return inferred_prob


def test_svi():
    """Implementation of the SVI example from Pyro's documentation (https://pyro.ai/examples/svi_part_i.html)"""
    # Generate data from a biased coin
    key = jax.random.key(0)
    true_prob = jnp.array([0.6])
    n_samples = 10
    data = sample(key, dist.BernoulliProbs(true_prob), (n_samples,))
    inferred_prob = run_svi(data)
    true_posterior_mean = jnp.array(16 / 30)
    assert jnp.isclose(inferred_prob, true_posterior_mean, atol=1e-2)
