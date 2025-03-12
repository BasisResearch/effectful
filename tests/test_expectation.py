import torch
import torch.distributions as dist
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop
from effectful.ops.types import Term

from weighted.fold_lang_v1 import ArgMinAlg, GradientOptimizationFold, LinAlg, dense_fold_intp, fold, reals


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
def sample(d: dist.Distribution, sample_shape: tuple[int]) -> torch.Tensor:
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
