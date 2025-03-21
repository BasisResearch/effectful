import torch
import torch.distributions as dist
from effectful.ops.semantics import handler, evaluate
from effectful.ops.syntax import defop, deffn
from effectful.ops.types import Term
from effectful.handlers.indexed import sizesof

from weighted.fold_lang_v1 import ArgMaxAlg, ArgMinAlg, GradientOptimizationFold, LinAlg, dense_fold_intp, fold, reals


def setup_module():
    torch.distributions.Distribution.set_default_validate_args(False)


# Expectation(
#     f(x)
#     for z1 in sample(z1_dist)
#     for z2 in sample(z2_dist(z1))
#     for x in sample(x_dist(z1, z2))
# )
#
#
# # unnormalized
# Expectation(
#     weight * vars[-1]
#     for (weight, vars) in Infer(
#         (w1(z1) * w2(z1, z2) * w3(z1, z2, x), (z1, z2, x))
#         for z1 in sample(z1_dist)
#         # if factor(w1(z1)) != 0
#         for z2 in sample(z2_dist(z1))
#         # if factor(w2(z1, z2)) != 0
#         for x in sample(x_dist(z1, z2))
#         # if factor(w3(z1, z2, x)) != 0
#     )
# )


@defop
def Normal(m, s):
    if any(isinstance(v, Term) and not isinstance(v, torch.Tensor) for v in (m, s)):
        raise NotImplementedError
    return dist.Normal(m, s)


@defop
def Beta(*args):
    if any(isinstance(v, Term) and not isinstance(v, torch.Tensor) for v in args):
        raise NotImplementedError
    return dist.Beta(*args)


@defop
def Bernoulli(*args):
    if any(isinstance(v, Term) and not isinstance(v, torch.Tensor) for v in args):
        raise NotImplementedError
    return dist.Bernoulli(*args)


@defop
def sample(d: dist.Distribution, sample_shape: tuple[int]) -> torch.Tensor:
    if not (isinstance(d, dist.Distribution) and isinstance(sample_shape, tuple)):
        raise NotImplementedError
    return d.sample(sample_shape=torch.Size(sample_shape))


@defop
def rsample(d: dist.Distribution, sample_shape: tuple[int]) -> torch.Tensor:
    if not (isinstance(d, dist.Distribution) and isinstance(sample_shape, tuple)):
        raise NotImplementedError
    return d.rsample(sample_shape=torch.Size(sample_shape))


@defop
def log_prob(d: dist.Distribution, value: torch.Tensor) -> torch.Tensor:
    if not (isinstance(d, dist.Distribution) and isinstance(value, torch.Tensor)):
        raise NotImplementedError
    return d.log_prob(value)


def test_maximum_marginal_likelihood_smoke():
    data = torch.randn(10).exp()

    m_z = defop(torch.Tensor, name="m_z")
    s_z = defop(torch.Tensor, name="s_z")
    z = defop(torch.Tensor, name="z")
    s_x = defop(torch.Tensor, name="s_x")

    z_dist = Normal(m_z(), s_z())
    x_dist = Normal(torch.exp(z()), s_x())

    n_samples = 1

    with (
        handler(dense_fold_intp),
        handler(GradientOptimizationFold(steps=1, init={s_z: torch.tensor(1.0), s_x: torch.tensor(1.0)})),
    ):
        weight = -(log_prob(z_dist, z()) + torch.sum(log_prob(x_dist, data)))
        intg_weight = fold(LinAlg, {z: sample(z_dist, (n_samples,))}, weight)
        min = fold(ArgMinAlg, {m_z: reals(), s_z: reals(), s_x: reals()}, (intg_weight, (m_z(), s_z(), s_x())))


def test_integration():
    loc = 0.0
    scale = 1.0

    def f(x):
        return x**2

    x = defop(torch.Tensor, name="x")
    w = defop(torch.Tensor, name="w")
    with handler(dense_fold_intp):
        intg = fold(LinAlg, {(x, w): Normal(loc, scale)}, torch.exp(w()) * f(x()))

    assert torch.isclose(intg, torch.tensor(0.5), atol=1e-1)

def test_svi():
    """Implementation of the SVI example from Pyro's documentation (https://pyro.ai/examples/svi_part_i.html)"""
    # Generate data from a biased coin
    true_prob = torch.tensor([0.6])
    n_samples = 1000
    data = sample(Bernoulli(true_prob), (n_samples,))

    latent_fairness = defop(torch.Tensor, name="latent_fairness")
    latent_fairness_w = defop(torch.Tensor, name="latent_fairness_w")
    alpha_q = defop(torch.Tensor, name="alpha_q")
    beta_q = defop(torch.Tensor, name="beta_q")

    def model_log_prob(data):
        """Return the log joint probability of the latent variables and the observed data according to the model."""
        alpha0 = torch.tensor(10.0)
        beta0 = torch.tensor(10.0)
        beta_prior = Beta(alpha0, beta0)
        return log_prob(beta_prior, latent_fairness()) + torch.sum(log_prob(Bernoulli(latent_fairness()), data))

    with handler(GradientOptimizationFold(steps=500, lr=0.5, init={alpha_q: torch.tensor(1.), beta_q: torch.tensor(1.)})), handler(dense_fold_intp):
        elbo = fold(
            LinAlg,
            {(latent_fairness, latent_fairness_w): Beta(alpha_q(), beta_q())},
            -(torch.exp(latent_fairness_w()) * (model_log_prob(data) - latent_fairness_w())),
        )
        (_, (alpha_est, beta_est)) = fold(ArgMinAlg, {alpha_q: reals(), beta_q: reals()}, (elbo, (alpha_q(), beta_q())))

        with handler({alpha_q: deffn(torch.tensor(15.0)), beta_q: deffn(torch.tensor(15.0))}):
            x = evaluate(elbo)
            assert isinstance(x, torch.Tensor) and len(x.shape) == 0 and len(sizesof(x)) == 0

        breakpoint()
        inferred_prob = alpha_est / (alpha_est + beta_est)
        assert torch.isclose(inferred_prob, true_prob, atol=1e-1)

