# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import OrderedDict, namedtuple
import functools
from importlib import import_module

import pyro.distributions as dist

import torch
from torch import rand, exp, randint
from torch.testing import assert_close

import numpy as np
import pytest

from effectful.internals.sugar import gensym, torch_getitem
from effectful.indexed.ops import indices_of, name_to_sym, IndexSet, to_tensor

##################################################
# Test cases
##################################################

torch.distributions.Distribution.set_default_validate_args(False)

TEST_CASES = []


def to_indexed(tensor, batch_dims):
    return torch_getitem(
        tensor, tuple(name_to_sym(str(i))() for i in range(batch_dims))
    )


class DistTestCase:
    def __init__(self, raw_dist, raw_params, batch_shape, xfail=None):
        assert isinstance(raw_dist, str)
        self.xfail = xfail
        self.raw_dist = re.sub(r"\s+", " ", raw_dist.strip())
        self.raw_params = raw_params
        self.params = None
        self.indexed_params = None
        self.batch_shape = batch_shape
        TEST_CASES.append(self)

    def get_dist(self):
        if self.xfail is not None:
            pytest.xfail(self.xfail)

        Case = namedtuple("Case", tuple(name for name, _ in self.raw_params))

        self.params = {}
        self.indexed_params = {}

        for name, raw_param in self.raw_params:
            param = eval(raw_param)
            self.params[name] = param
            if (
                isinstance(param, torch.Tensor)
                and param.shape[: len(self.batch_shape)] == self.batch_shape
            ):
                self.indexed_params[name] = to_indexed(param, len(self.batch_shape))
            else:
                self.indexed_params[name] = param

        case = Case(**self.params)
        dist_ = eval(self.raw_dist)

        case = Case(**self.indexed_params)
        indexed_dist = eval(self.raw_dist)

        return dist_, indexed_dist

    def __str__(self):
        return self.raw_dist + " " + str(self.raw_params)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.raw_dist, self.raw_params, self.batch_shape))


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
def test_dist_indexes(case_):
    dist, indexed_dist = case_.get_dist()

    sample = dist.sample()
    indexed_sample = indexed_dist.sample()

    # Indexed samples should have the same indices as the parameters to their distribution (if those parameters are
    # indexed)
    sample_indices = indices_of(indexed_sample)
    for param in case_.indexed_params.values():
        param_indices = indices_of(param)
        assert param_indices in [IndexSet(), sample_indices]

    # Indexed samples should have the same shape as regular samples, but with the batch dimensions indexed
    indexed_sample_t = to_tensor(indexed_sample)
    assert sample.shape == indexed_sample_t.shape
    assert sample.dtype == indexed_sample_t.dtype

    lprob = dist.log_prob(sample)
    indexed_lprob = indexed_dist.log_prob(indexed_sample)

    # Indexed logprobs should have the same shape as regular logprobs, but with the batch dimensions indexed
    indexed_lprob_t = to_tensor(indexed_lprob)
    assert lprob.shape == indexed_lprob_t.shape
    assert lprob.dtype == indexed_lprob_t.dtype


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
@pytest.mark.parametrize("statistic", ["mean", "variance", "entropy"])
def test_dist_stats(case_, statistic):
    dist, indexed_dist = case_.get_dist()

    EXPECTED_FAILURES = [
        ("StudentT", ["mean", "variance"]),
        ("FisherSnedecor", ["mean", "variance"]),
        ("Binomial", ["entropy"]),
    ]
    for dist_name, methods in EXPECTED_FAILURES:
        if dist_name in case_.raw_dist and statistic in methods:
            pytest.xfail(
                f"{dist_name} mean uses masking which is not supported by indexed tensors"
            )

    try:
        actual_stat = getattr(indexed_dist, statistic)
        expected_stat = getattr(dist, statistic)

        if statistic == "entropy":
            expected_stat = expected_stat()
            actual_stat = actual_stat()
        actual_stat = to_tensor(actual_stat)
    except NotImplementedError:
        pytest.xfail(f"{statistic} not implemented")

    actual_stat, expected_stat = torch.broadcast_tensors(actual_stat, expected_stat)

    if actual_stat.isnan().all():
        pytest.xfail("expected statistic is NaN")

    assert_close(actual_stat, expected_stat, atol=1e-5, rtol=0)


for batch_shape in [(5,), (2, 3), ()]:
    # BernoulliLogits
    DistTestCase(
        "dist.Bernoulli(logits=case.logits)",
        (("logits", f"rand({batch_shape})"),),
        batch_shape,
    )

    # BernoulliProbs
    DistTestCase(
        "dist.Bernoulli(probs=case.probs)",
        (("probs", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Beta
    DistTestCase(
        "dist.Beta(case.concentration1, case.concentration0)",
        (
            ("concentration1", f"exp(rand({batch_shape}))"),
            ("concentration0", f"exp(rand({batch_shape}))"),
        ),
        batch_shape,
    )

    # Binomial
    DistTestCase(
        "dist.Binomial(total_count=case.total_count, probs=case.probs)",
        (
            ("total_count", "5"),
            ("probs", f"rand({batch_shape})"),
        ),
        batch_shape,
    )

    # CategoricalLogits
    for size in [2, 4]:
        DistTestCase(
            "dist.Categorical(logits=case.logits)",
            (("logits", f"rand({batch_shape + (size,)})"),),
            batch_shape,
        )

    # CategoricalProbs
    for size in [2, 4]:
        DistTestCase(
            "dist.Categorical(probs=case.probs)",
            (("probs", f"rand({batch_shape + (size,)})"),),
            batch_shape,
        )

    # Cauchy
    DistTestCase(
        "dist.Cauchy(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # Chi2
    DistTestCase(
        "dist.Chi2(df=case.df)",
        (("df", f"rand({batch_shape})"),),
        batch_shape,
    )

    # ContinuousBernoulli
    DistTestCase(
        "dist.ContinuousBernoulli(logits=case.logits)",
        (("logits", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Delta
    for event_shape in [(), (4,), (3, 2)]:
        DistTestCase(
            f"dist.Delta(v=case.v, log_density=case.log_density, event_dim={len(event_shape)})",
            (
                ("v", f"rand({batch_shape + event_shape})"),
                ("log_density", f"rand({batch_shape})"),
            ),
            batch_shape,
        )

    # Dirichlet
    for event_shape in [(1,), (4,)]:
        DistTestCase(
            "dist.Dirichlet(case.concentration)",
            (("concentration", f"rand({batch_shape + event_shape})"),),
            batch_shape,
        )

    # DirichletMultinomial
    for event_shape in [(1,), (4,)]:
        DistTestCase(
            "dist.DirichletMultinomial(case.concentration, case.total_count)",
            (
                ("concentration", f"rand({batch_shape + event_shape})"),
                ("total_count", "randint(10, 12, ())"),
            ),
            batch_shape,
            xfail="problem with vmap and scatter_add_",
        )

    # Exponential
    DistTestCase(
        "dist.Exponential(rate=case.rate)",
        (("rate", f"rand({batch_shape})"),),
        batch_shape,
    )

    # FisherSnedecor
    DistTestCase(
        "dist.FisherSnedecor(df1=case.df1, df2=case.df2)",
        (("df1", f"rand({batch_shape})"), ("df2", f"rand({batch_shape})")),
        batch_shape,
    )

    # Gamma
    DistTestCase(
        "dist.Gamma(case.concentration, case.rate)",
        (("concentration", f"rand({batch_shape})"), ("rate", f"rand({batch_shape})")),
        batch_shape,
    )

    # Geometric
    DistTestCase(
        "dist.Geometric(probs=case.probs)",
        (("probs", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Gumbel
    DistTestCase(
        "dist.Gumbel(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # HalfCauchy
    DistTestCase(
        "dist.HalfCauchy(scale=case.scale)",
        (("scale", f"rand({batch_shape})"),),
        batch_shape,
    )

    # HalfNormal
    DistTestCase(
        "dist.HalfNormal(scale=case.scale)",
        (("scale", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Laplace
    DistTestCase(
        "dist.Laplace(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # Logistic
    DistTestCase(
        "dist.Logistic(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # # LowRankMultivariateNormal
    # for event_shape in [(3,), (4,)]:
    #     DistTestCase(
    #         "dist.LowRankMultivariateNormal(loc=case.loc, cov_factor=case.cov_factor, cov_diag=case.cov_diag)",
    #         (
    #             ("loc", f"rand({batch_shape + event_shape})"),
    #             ("cov_factor", f"rand({batch_shape + event_shape + (2,)})"),
    #             ("cov_diag", f"rand({batch_shape + event_shape})"),
    #         ),
    #         batch_shape,
    #     )

    # Multinomial
    for event_shape in [(1,), (4,)]:
        DistTestCase(
            "dist.Multinomial(case.total_count, probs=case.probs)",
            (
                ("total_count", "5"),
                ("probs", f"rand({batch_shape + event_shape})"),
            ),
            batch_shape,
            xfail="problem with vmap and scatter_add_",
        )

    # # MultivariateNormal
    # for event_shape in [(1,), (3,)]:
    #     DistTestCase(
    #         "dist.MultivariateNormal(loc=case.loc, scale_tril=case.scale_tril)",
    #         (
    #             ("loc", f"rand({batch_shape + event_shape})"),
    #             ("scale_tril", f"random_scale_tril({batch_shape + event_shape * 2})"),
    #         ),
    #         batch_shape,
    #     )

    # NegativeBinomial
    DistTestCase(
        "dist.NegativeBinomial(total_count=case.total_count, probs=case.probs)",
        (
            ("total_count", "5"),
            ("probs", f"rand({batch_shape})"),
        ),
        batch_shape,
    )

    # Normal
    DistTestCase(
        "dist.Normal(case.loc, case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # OneHotCategorical
    for size in [2, 4]:
        DistTestCase(
            "dist.OneHotCategorical(probs=case.probs)",
            (("probs", f"rand({batch_shape + (size,)})"),),
            batch_shape,  # funsor.Bint[size],
        )

    # Pareto
    DistTestCase(
        "dist.Pareto(scale=case.scale, alpha=case.alpha)",
        (("scale", f"rand({batch_shape})"), ("alpha", f"rand({batch_shape})")),
        batch_shape,
    )

    # Poisson
    DistTestCase(
        "dist.Poisson(rate=case.rate)",
        (("rate", f"rand({batch_shape})"),),
        batch_shape,
    )

    # RelaxedBernoulli
    DistTestCase(
        "dist.RelaxedBernoulli(temperature=case.temperature, logits=case.logits)",
        (("temperature", f"rand({batch_shape})"), ("logits", f"rand({batch_shape})")),
        batch_shape,
    )

    # StudentT
    DistTestCase(
        "dist.StudentT(df=case.df, loc=case.loc, scale=case.scale)",
        (
            ("df", f"rand({batch_shape})"),
            ("loc", f"rand({batch_shape})"),
            ("scale", f"rand({batch_shape})"),
        ),
        batch_shape,
    )

    # Uniform
    DistTestCase(
        "dist.Uniform(low=case.low, high=case.high)",
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # VonMises
    DistTestCase(
        "dist.VonMises(case.loc, case.concentration)",
        (("loc", f"rand({batch_shape})"), ("concentration", f"rand({batch_shape})")),
        batch_shape,
        xfail="problem with vmap and data-dependent control flow in rejection sampling",
    )

    # Weibull
    DistTestCase(
        "dist.Weibull(scale=case.scale, concentration=case.concentration)",
        (
            ("scale", f"exp(rand({batch_shape}))"),
            ("concentration", f"exp(rand({batch_shape}))"),
        ),
        batch_shape,
    )

    # TransformedDistributions
    # ExpTransform
    DistTestCase(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.ExpTransform()])
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # InverseTransform (log)
    DistTestCase(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.ExpTransform().inv])
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # TanhTransform
    DistTestCase(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.TanhTransform(),])
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # AtanhTransform
    DistTestCase(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.TanhTransform().inv])
        """,
        (
            ("low", f"0.5*rand({batch_shape})"),
            ("high", f"0.5 + 0.5*rand({batch_shape})"),
        ),
        batch_shape,
    )

    # multiple transforms
    DistTestCase(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.TanhTransform(),
             dist.transforms.ExpTransform()])
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # ComposeTransform
    DistTestCase(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            dist.transforms.ComposeTransform([
                dist.transforms.TanhTransform(),
                dist.transforms.ExpTransform()]))
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # PowerTransform
    DistTestCase(
        """
        dist.TransformedDistribution(
            dist.Exponential(rate=case.rate),
            dist.transforms.PowerTransform(0.5))
        """,
        (("rate", f"rand({batch_shape})"),),
        batch_shape,
    )

    # HaarTransform
    DistTestCase(
        """
        dist.TransformedDistribution(
            dist.Normal(loc=case.loc, scale=1.).to_event(1),
            dist.transforms.HaarTransform(dim=-1))
        """,
        (("loc", f"rand({batch_shape} + (3,))"),),
        batch_shape,
    )

    # Independent
    for indep_shape in [(3,), (2, 3)]:
        # Beta.to_event
        DistTestCase(
            f"dist.Beta(case.concentration1, case.concentration0).to_event({len(indep_shape)})",
            (
                ("concentration1", f"exp(rand({batch_shape + indep_shape}))"),
                ("concentration0", f"exp(rand({batch_shape + indep_shape}))"),
            ),
            batch_shape,
        )

        # Dirichlet.to_event
        for event_shape in [(2,), (4,)]:
            DistTestCase(
                f"dist.Dirichlet(case.concentration).to_event({len(indep_shape)})",
                (
                    (
                        "concentration",
                        f"rand({batch_shape + indep_shape + event_shape})",
                    ),
                ),
                batch_shape,
            )

        # TransformedDistribution.to_event
        DistTestCase(
            f"""
            dist.Independent(
                dist.TransformedDistribution(
                    dist.Uniform(low=case.low, high=case.high),
                    dist.transforms.ComposeTransform([
                        dist.transforms.TanhTransform(),
                        dist.transforms.ExpTransform()])),
                {len(indep_shape)})
            """,
            (
                ("low", f"rand({batch_shape + indep_shape})"),
                ("high", f"2. + rand({batch_shape + indep_shape})"),
            ),
            batch_shape,
        )
