import re
from collections import namedtuple
from collections.abc import Sequence

import pyro.distributions as dist
import pytest
import torch

#################################################
# these imports are used by generated code
from torch import exp, rand, randint  # noqa: F401

#################################################
from torch.testing import assert_close

import effectful.handlers.pyro  # noqa: F401
from effectful.handlers.indexed import name_to_sym
from effectful.handlers.torch import bind_dims, sizesof, unbind_dims
from effectful.ops.syntax import defop

##################################################
# Test cases
# Based on https://github.com/pyro-ppl/funsor/blob/master/test/test_distribution_generic.py
##################################################


def setup_module():
    torch.manual_seed(0)
    torch.distributions.Distribution.set_default_validate_args(False)


TEST_CASES = []


def random_scale_tril(*args):
    if isinstance(args[0], tuple):
        assert len(args) == 1
        shape = args[0]
    else:
        shape = args

    data = torch.randn(shape)
    return dist.transforms.transform_to(dist.constraints.lower_cholesky)(data)


def from_indexed(tensor, batch_dims):
    tensor_sizes = sizesof(tensor)
    indices = [name_to_sym(str(i)) for i in range(batch_dims)]
    indices = [i for i in indices if i in tensor_sizes]
    return bind_dims(tensor, *indices)


class DistTestCase:
    raw_dist: str
    params: dict[str, torch.Tensor]
    indexed_params: dict[str, torch.Tensor]
    batch_shape: tuple[int, ...]
    xfail: str | None
    kind: str

    def __init__(
        self,
        raw_dist: str,
        params: dict[str, torch.Tensor],
        indexed_params: dict[str, torch.Tensor],
        batch_shape: tuple[int, ...],
        xfail: str | None,
        kind: str,
    ):
        self.raw_dist = re.sub(r"\s+", " ", raw_dist.strip())
        self.params = params
        self.indexed_params = indexed_params
        self.batch_shape = batch_shape
        self.xfail = xfail
        self.kind = kind

    def get_dist(self):
        """Return positional and indexed distributions."""
        if self.xfail is not None:
            pytest.xfail(self.xfail)

        Case = namedtuple("Case", tuple(name for name, _ in self.params.items()))
        case = Case(**self.params)
        dist_ = eval(self.raw_dist)

        # case is used by generated code in self.raw_dist
        case = Case(**self.indexed_params)  # noqa: F841
        indexed_dist = eval(self.raw_dist)

        return dist_, indexed_dist

    def __eq__(self, other):
        if isinstance(other, DistTestCase):
            return (
                self.raw_dist == other.raw_dist
                and self.batch_shape == other.batch_shape
                and self.kind == other.kind
            )

    def __hash__(self):
        return hash((self.raw_dist, self.batch_shape, self.kind))

    def __repr__(self):
        return f"{self.raw_dist} {self.batch_shape} {self.kind}"


def full_indexed_test_case(
    raw_dist: str,
    params: dict[str, torch.Tensor],
    batch_shape: tuple[int, ...],
    xfail: str | None = None,
):
    indexed_params = {}
    for name, param in params.items():
        if (
            isinstance(param, torch.Tensor)
            and param.shape[: len(batch_shape)] == batch_shape
        ):
            indexes = tuple(name_to_sym(str(i))() for i in range(len(batch_shape)))
            indexed_params[name] = param[indexes]
        else:
            indexed_params[name] = param

    return DistTestCase(raw_dist, params, indexed_params, batch_shape, xfail, "full")


def partial_indexed_test_case(
    raw_dist: str,
    params: dict[str, torch.Tensor],
    batch_shape: tuple[int, ...],
    xfail: str | None = None,
):
    """Produces parameters with a subset of batch dimensions indexed.

    For example, if batch_shape is (2, 3) and params is
    {"loc": torch.randn(2, 3, 4), "scale": torch.randn(2, 3, 4)},
    this function will return a test case with indexed parameters
    {"loc": torch.randn(2, 3, 4)[i0(), 0, i2()], "scale": torch.randn(2, 3, 4)[0, i1(), i2()]}.


    """
    non_indexed_params = {
        k: v
        for (k, v) in params.items()
        if not (
            isinstance(v, torch.Tensor) and v.shape[: len(batch_shape)] == batch_shape
        )
    }
    broadcast_params = params.copy()
    indexed_params = {}

    indexed_param_names = set(name for name in params if name not in non_indexed_params)
    for i, name in enumerate(indexed_param_names):
        param = params[name]

        if (
            isinstance(param, torch.Tensor)
            and param.shape[: len(batch_shape)] == batch_shape
        ):
            indexes = []
            for j in range(len(batch_shape)):
                if i == j or j >= len(indexed_param_names):
                    index = name_to_sym(str(j))()
                else:
                    index = torch.tensor(0)
                    broadcast_params[name] = torch.unsqueeze(
                        torch.select(broadcast_params[name], j, 0), j
                    )
                indexes.append(index)
            indexed_params[name] = param[tuple(indexes)]
        else:
            indexed_params[name] = param

    indexed_params.update(non_indexed_params)
    return DistTestCase(
        raw_dist, broadcast_params, indexed_params, batch_shape, xfail, "partial"
    )


def add_dist_test_case(
    raw_dist: str,
    raw_params: Sequence[tuple[str, str]],
    batch_shape: tuple[int, ...],
    xfail: str | None = None,
):
    params = {name: eval(raw_param) for name, raw_param in raw_params}
    TEST_CASES.append(full_indexed_test_case(raw_dist, params, batch_shape, xfail))

    # This case is trivial if there are not multiple batch dimensions and
    # multiple parameters
    if len(batch_shape) > 1 and len(params) > 1:
        TEST_CASES.append(
            partial_indexed_test_case(raw_dist, params, batch_shape, xfail)
        )


for batch_shape in [(5,), (2, 3, 4), ()]:
    # BernoulliLogits
    add_dist_test_case(
        "dist.Bernoulli(logits=case.logits)",
        (("logits", f"rand({batch_shape})"),),
        batch_shape,
    )

    # BernoulliProbs
    add_dist_test_case(
        "dist.Bernoulli(probs=case.probs)",
        (("probs", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Beta
    add_dist_test_case(
        "dist.Beta(case.concentration1, case.concentration0)",
        (
            ("concentration1", f"exp(rand({batch_shape}))"),
            ("concentration0", f"exp(rand({batch_shape}))"),
        ),
        batch_shape,
    )

    # Binomial
    add_dist_test_case(
        "dist.Binomial(total_count=case.total_count, probs=case.probs)",
        (
            ("total_count", "5"),
            ("probs", f"rand({batch_shape})"),
        ),
        batch_shape,
    )

    # CategoricalLogits
    for size in [2, 4]:
        add_dist_test_case(
            "dist.Categorical(logits=case.logits)",
            (("logits", f"rand({batch_shape + (size,)})"),),
            batch_shape,
        )

    # CategoricalProbs
    for size in [2, 4]:
        add_dist_test_case(
            "dist.Categorical(probs=case.probs)",
            (("probs", f"rand({batch_shape + (size,)})"),),
            batch_shape,
        )

    # Cauchy
    add_dist_test_case(
        "dist.Cauchy(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # Chi2
    add_dist_test_case(
        "dist.Chi2(df=case.df)",
        (("df", f"rand({batch_shape})"),),
        batch_shape,
    )

    # ContinuousBernoulli
    add_dist_test_case(
        "dist.ContinuousBernoulli(logits=case.logits)",
        (("logits", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Delta
    for event_shape in [(), (4,), (3, 2)]:
        add_dist_test_case(
            f"dist.Delta(v=case.v, log_density=case.log_density, event_dim={len(event_shape)})",
            (
                ("v", f"rand({batch_shape + event_shape})"),
                ("log_density", f"rand({batch_shape})"),
            ),
            batch_shape,
        )

    # Dirichlet
    for event_shape in [(1,), (4,)]:
        add_dist_test_case(
            "dist.Dirichlet(case.concentration)",
            (("concentration", f"rand({batch_shape + event_shape})"),),
            batch_shape,
        )

    # DirichletMultinomial
    for event_shape in [(1,), (4,)]:
        add_dist_test_case(
            "dist.DirichletMultinomial(case.concentration, case.total_count)",
            (
                ("concentration", f"rand({batch_shape + event_shape})"),
                ("total_count", "randint(10, 12, ())"),
            ),
            batch_shape,
            xfail="problem with vmap and scatter_add_",
        )

    # Exponential
    add_dist_test_case(
        "dist.Exponential(rate=case.rate)",
        (("rate", f"rand({batch_shape})"),),
        batch_shape,
    )

    # FisherSnedecor
    add_dist_test_case(
        "dist.FisherSnedecor(df1=case.df1, df2=case.df2)",
        (("df1", f"rand({batch_shape})"), ("df2", f"rand({batch_shape})")),
        batch_shape,
    )

    # Gamma
    add_dist_test_case(
        "dist.Gamma(case.concentration, case.rate)",
        (("concentration", f"rand({batch_shape})"), ("rate", f"rand({batch_shape})")),
        batch_shape,
    )

    # Geometric
    add_dist_test_case(
        "dist.Geometric(probs=case.probs)",
        (("probs", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Gumbel
    add_dist_test_case(
        "dist.Gumbel(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # HalfCauchy
    add_dist_test_case(
        "dist.HalfCauchy(scale=case.scale)",
        (("scale", f"rand({batch_shape})"),),
        batch_shape,
    )

    # HalfNormal
    add_dist_test_case(
        "dist.HalfNormal(scale=case.scale)",
        (("scale", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Laplace
    add_dist_test_case(
        "dist.Laplace(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # Logistic
    add_dist_test_case(
        "dist.Logistic(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # # LowRankMultivariateNormal
    for event_shape in [(3,), (4,)]:
        add_dist_test_case(
            "dist.LowRankMultivariateNormal(loc=case.loc, cov_factor=case.cov_factor, cov_diag=case.cov_diag)",
            (
                ("loc", f"rand({batch_shape + event_shape})"),
                ("cov_factor", f"rand({batch_shape + event_shape + (2,)})"),
                ("cov_diag", f"rand({batch_shape + event_shape})"),
            ),
            batch_shape,
            xfail="Requires support for setitem",
        )

    # multinomial
    for event_shape in [(1,), (4,)]:
        add_dist_test_case(
            "dist.Multinomial(case.total_count, probs=case.probs)",
            (
                ("total_count", "5"),
                ("probs", f"rand({batch_shape + event_shape})"),
            ),
            batch_shape,
            xfail="problem with vmap and scatter_add_",
        )

    # # MultivariateNormal
    for event_shape in [(1,), (3,)]:
        add_dist_test_case(
            "dist.MultivariateNormal(loc=case.loc, scale_tril=case.scale_tril)",
            (
                ("loc", f"rand({batch_shape + event_shape})"),
                ("scale_tril", f"random_scale_tril({batch_shape + event_shape * 2})"),
            ),
            batch_shape,
        )

    # NegativeBinomial
    add_dist_test_case(
        "dist.NegativeBinomial(total_count=case.total_count, probs=case.probs)",
        (
            ("total_count", "5"),
            ("probs", f"rand({batch_shape})"),
        ),
        batch_shape,
    )

    # Normal
    add_dist_test_case(
        "dist.Normal(case.loc, case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # OneHotCategorical
    for size in [2, 4]:
        add_dist_test_case(
            "dist.OneHotCategorical(probs=case.probs)",
            (("probs", f"rand({batch_shape + (size,)})"),),
            batch_shape,  # funsor.Bint[size],
        )

    # Pareto
    add_dist_test_case(
        "dist.Pareto(scale=case.scale, alpha=case.alpha)",
        (("scale", f"rand({batch_shape})"), ("alpha", f"rand({batch_shape})")),
        batch_shape,
    )

    # Poisson
    add_dist_test_case(
        "dist.Poisson(rate=case.rate)",
        (("rate", f"rand({batch_shape})"),),
        batch_shape,
    )

    # RelaxedBernoulli
    add_dist_test_case(
        "dist.RelaxedBernoulli(temperature=case.temperature, logits=case.logits)",
        (("temperature", f"rand({batch_shape})"), ("logits", f"rand({batch_shape})")),
        batch_shape,
    )

    # StudentT
    add_dist_test_case(
        "dist.StudentT(df=case.df, loc=case.loc, scale=case.scale)",
        (
            ("df", f"rand({batch_shape})"),
            ("loc", f"rand({batch_shape})"),
            ("scale", f"rand({batch_shape})"),
        ),
        batch_shape,
    )

    # Uniform
    add_dist_test_case(
        "dist.Uniform(low=case.low, high=case.high)",
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # VonMises
    add_dist_test_case(
        "dist.VonMises(case.loc, case.concentration)",
        (("loc", f"rand({batch_shape})"), ("concentration", f"rand({batch_shape})")),
        batch_shape,
        xfail="problem with vmap and data-dependent control flow in rejection sampling",
    )

    # Weibull
    add_dist_test_case(
        "dist.Weibull(scale=case.scale, concentration=case.concentration)",
        (
            ("scale", f"exp(rand({batch_shape}))"),
            ("concentration", f"exp(rand({batch_shape}))"),
        ),
        batch_shape,
    )

    # TransformedDistributions
    # ExpTransform
    add_dist_test_case(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.ExpTransform()])
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # InverseTransform (log)
    add_dist_test_case(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.ExpTransform().inv])
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # TanhTransform
    add_dist_test_case(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.TanhTransform(),])
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # AtanhTransform
    add_dist_test_case(
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
    add_dist_test_case(
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
    add_dist_test_case(
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
    add_dist_test_case(
        """
        dist.TransformedDistribution(
            dist.Exponential(rate=case.rate),
            dist.transforms.PowerTransform(0.5))
        """,
        (("rate", f"rand({batch_shape})"),),
        batch_shape,
    )

    # HaarTransform
    add_dist_test_case(
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
        add_dist_test_case(
            f"dist.Beta(case.concentration1, case.concentration0).to_event({len(indep_shape)})",
            (
                ("concentration1", f"exp(rand({batch_shape + indep_shape}))"),
                ("concentration0", f"exp(rand({batch_shape + indep_shape}))"),
            ),
            batch_shape,
        )

        # Dirichlet.to_event
        for event_shape in [(2,), (4,)]:
            add_dist_test_case(
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
        add_dist_test_case(
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


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
def test_dist_to_positional(case_):
    _, indexed_dist = case_.get_dist()

    try:
        sizes = sizesof(indexed_dist)
        pos_dist = bind_dims(indexed_dist, *sizes.keys())
        pos_sample = pos_dist.sample()
        assert sizesof(pos_sample) == {}
        indexed_sample = indexed_dist.sample()

        # positional samples should be broadcastable with indexed samples, but
        # they may not have the same shape
        torch.distributions.utils.broadcast_all(pos_sample, indexed_sample)
    except ValueError as e:
        if (
            "No embedding provided for distribution of type TransformedDistribution"
            in str(e)
        ):
            pytest.xfail("TransformedDistribution not supported")
        else:
            raise e


@pytest.mark.parametrize("case_", [c for c in TEST_CASES if c.kind == "full"], ids=str)
def test_dist_to_named(case_):
    try:
        dist, _ = case_.get_dist()
        indexes = [name_to_sym(str(i)) for i in range(len(case_.batch_shape))]
        indexed_dist = unbind_dims(dist, *indexes)

        indexed_sample = indexed_dist.sample()
        assert set(sizesof(indexed_sample)) == set(indexes)
    except ValueError as e:
        if (
            "No embedding provided for distribution of type TransformedDistribution"
            in str(e)
        ):
            pytest.xfail("TransformedDistribution not supported")
        else:
            raise e


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
@pytest.mark.parametrize("sample_shape", [(), (3, 2)])
@pytest.mark.parametrize("indexed_sample_shape", [(), (3, 2)])
@pytest.mark.parametrize("extra_batch_shape", [(), (3, 2)])
def test_dist_expand(case_, sample_shape, indexed_sample_shape, extra_batch_shape):
    _, indexed_dist = case_.get_dist()

    expanded = indexed_dist.expand(extra_batch_shape + indexed_dist.batch_shape)
    sample = expanded.sample(indexed_sample_shape + sample_shape)
    indexed_sample = sample[
        tuple(defop(torch.Tensor)() for _ in range(len(indexed_sample_shape)))
    ]

    assert (
        indexed_sample.shape
        == sample_shape
        + extra_batch_shape
        + indexed_dist.batch_shape
        + indexed_dist.event_shape
    )

    assert expanded.log_prob(indexed_sample).shape == extra_batch_shape + sample_shape


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
def test_dist_indexes(case_):
    """Test that indexed samples and logprobs have the correct shape and indices."""
    dist, indexed_dist = case_.get_dist()

    sample = dist.sample()
    indexed_sample = indexed_dist.sample()

    # Samples should not have any indices that their parameters don't have
    assert set(sizesof(indexed_sample)) <= set().union(
        *[set(sizesof(p)) for p in case_.indexed_params.values()]
    )

    # Indexed samples should have the same shape as regular samples, modulo
    # possible extra unit dimensions
    indexed_sample_t = from_indexed(indexed_sample, len(case_.batch_shape))
    assert sample.squeeze().shape == indexed_sample_t.squeeze().shape
    assert sample.dtype == indexed_sample_t.dtype

    lprob = dist.log_prob(sample)
    indexed_lprob = indexed_dist.log_prob(indexed_sample)

    # Indexed logprobs should have the same shape as regular logprobs, but with
    # the batch dimensions indexed
    indexed_lprob_t = from_indexed(indexed_lprob, len(case_.batch_shape))
    assert lprob.shape == indexed_lprob_t.shape
    assert lprob.dtype == indexed_lprob_t.dtype


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
@pytest.mark.parametrize("sample_shape", [(), (2,), (3, 2)])
@pytest.mark.parametrize("use_rsample", [False, True])
def test_dist_randomness(case_, sample_shape, use_rsample):
    """Test that indexed samples differ across the batch dimensions."""
    pos_dist, indexed_dist = case_.get_dist()

    # Skip discrete distributions (and Poisson, which is discrete but has no enumerate support)
    if (
        pos_dist.has_enumerate_support
        or "Poisson" in case_.raw_dist
        or "Geometric" in case_.raw_dist
    ):
        pytest.xfail("Discrete distributions not supported")

    if use_rsample:
        try:
            indexed_sample = indexed_dist.rsample(sample_shape)
            pos_sample = pos_dist.rsample(sample_shape)
        except NotImplementedError:
            pytest.xfail("Distributions without rsample not supported")
    else:
        indexed_sample = indexed_dist.sample(sample_shape)
        pos_sample = pos_dist.sample(sample_shape)

    indexed_sample_t = from_indexed(indexed_sample, len(case_.batch_shape))

    new_shape = (-1, *pos_sample.shape[len(case_.batch_shape) :])
    flat_sample = pos_sample.reshape(new_shape)
    flat_indexed_sample = indexed_sample_t.reshape(new_shape)

    # with high probability, samples should differ across batch dimensions
    if flat_sample.unique(dim=0).shape[0] > 1:
        assert flat_indexed_sample.unique(dim=0).shape[0] > 1


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
@pytest.mark.parametrize("statistic", ["mean", "variance", "entropy"])
def test_dist_stats(case_, statistic):
    """Test that indexed distributions have the same statistics as their unindexed counterparts."""
    dist, indexed_dist = case_.get_dist()

    EXPECTED_FAILURES = [
        ("StudentT", ["mean", "variance"]),
        ("FisherSnedecor", ["mean", "variance"]),
        ("Binomial", ["entropy"]),
        (
            "Geometric",
            [
                "entropy"  # flaky, but no failure
            ],
        ),
    ]
    for dist_name, methods in EXPECTED_FAILURES:
        if dist_name in case_.raw_dist and statistic in methods:
            pytest.xfail(f"{dist_name} {statistic} is an expected failure")

    try:
        actual_stat = getattr(indexed_dist, statistic)
        expected_stat = getattr(dist, statistic)

        if statistic == "entropy":
            expected_stat = expected_stat()
            actual_stat = actual_stat()
    except NotImplementedError:
        pytest.xfail(f"{statistic} not implemented")

    if expected_stat.isnan().all():
        assert bind_dims(actual_stat).isnan().all()
    else:
        # Stats may not be indexed in all batch dimensions, but they should be
        # extensionally equal to the indexed expected stat
        indexes = [name_to_sym(str(i)) for i in range(len(case_.batch_shape))]
        expected_stat_i = expected_stat[tuple(n() for n in indexes)]
        expected_stat_i, actual_stat_i = torch.broadcast_tensors(
            expected_stat_i, actual_stat
        )
        assert_close(
            bind_dims(expected_stat_i, *indexes), bind_dims(actual_stat_i, *indexes)
        )
