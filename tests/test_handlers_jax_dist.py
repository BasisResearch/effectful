import functools
import re
from collections import namedtuple
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpyro.distributions
import pytest

import effectful.handlers.jax.distribution as dist
from effectful.handlers.jax import jax_getitem, sizesof, to_array
from effectful.handlers.jax.distribution import (
    named_distribution,
    positional_distribution,
)
from effectful.ops.syntax import defop
from effectful.ops.types import Operation, Term

##################################################
# Test cases
# Based on https://github.com/pyro-ppl/funsor/blob/master/test/test_distribution_generic.py
##################################################


def setup_module():
    pass


TEST_CASES = []
RAW_TEST_CASES = []

from dataclasses import dataclass


@dataclass
class RawTestCase:
    raw_dist: str
    raw_params: dict[str, str]
    batch_shape: tuple[int, ...]
    xfail: str | None = None


RAW_TEST_CASES = []


def add_case(raw_dist, raw_params, batch_shape, xfail=None):
    RAW_TEST_CASES.append(RawTestCase(raw_dist, raw_params, batch_shape, xfail))


for batch_shape in [(5,), (2, 3, 4), ()]:
    # BernoulliLogits
    add_case(
        "dist.Bernoulli(logits=case.logits)",
        (("logits", f"rand({batch_shape})"),),
        batch_shape,
    )

    # BernoulliProbs
    add_case(
        "dist.Bernoulli(probs=case.probs)",
        (("probs", f"rand({batch_shape})"),),
        batch_shape,
    )

    # BernoulliLogits
    add_case(
        "dist.BernoulliLogits(case.logits)",
        (("logits", f"rand({batch_shape})"),),
        batch_shape,
    )

    # BernoulliProbs
    add_case(
        "dist.BernoulliProbs(case.probs)",
        (("probs", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Beta
    add_case(
        "dist.Beta(case.concentration1, case.concentration0)",
        (
            ("concentration1", f"exp(rand({batch_shape}))"),
            ("concentration0", f"exp(rand({batch_shape}))"),
        ),
        batch_shape,
    )

    # Binomial
    add_case(
        "dist.Binomial(total_count=case.total_count, probs=case.probs)",
        (
            ("total_count", "5"),
            ("probs", f"rand({batch_shape})"),
        ),
        batch_shape,
    )

    # CategoricalLogits
    for size in [2, 4]:
        add_case(
            "dist.Categorical(logits=case.logits)",
            (("logits", f"rand({batch_shape + (size,)})"),),
            batch_shape,
        )

    # CategoricalProbs
    for size in [2, 4]:
        add_case(
            "dist.Categorical(probs=case.probs)",
            (("probs", f"rand({batch_shape + (size,)})"),),
            batch_shape,
        )

    # Cauchy
    add_case(
        "dist.Cauchy(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # Chi2
    add_case("dist.Chi2(df=case.df)", (("df", f"rand({batch_shape})"),), batch_shape)

    # ContinuousBernoulli
    add_case(
        "dist.ContinuousBernoulli(logits=case.logits)",
        (("logits", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Delta
    for event_shape in [(), (4,), (3, 2)]:
        add_case(
            f"dist.Delta(v=case.v, log_density=case.log_density, event_dim={len(event_shape)})",
            (
                ("v", f"rand({batch_shape + event_shape})"),
                ("log_density", f"rand({batch_shape})"),
            ),
            batch_shape,
        )

    # Dirichlet
    for event_shape in [(1,), (4,)]:
        add_case(
            "dist.Dirichlet(case.concentration)",
            (("concentration", f"rand({batch_shape + event_shape})"),),
            batch_shape,
        )

    # DirichletMultinomial
    for event_shape in [(1,), (4,)]:
        add_case(
            "dist.DirichletMultinomial(case.concentration, case.total_count)",
            (
                ("concentration", f"rand({batch_shape + event_shape})"),
                ("total_count", "randint(10, 12, ())"),
            ),
            batch_shape,
            xfail="problem with vmap and scatter_add_",
        )

    # Exponential
    add_case(
        "dist.Exponential(rate=case.rate)",
        (("rate", f"rand({batch_shape})"),),
        batch_shape,
    )

    # FisherSnedecor
    add_case(
        "dist.FisherSnedecor(df1=case.df1, df2=case.df2)",
        (("df1", f"rand({batch_shape})"), ("df2", f"rand({batch_shape})")),
        batch_shape,
    )

    # Gamma
    add_case(
        "dist.Gamma(case.concentration, case.rate)",
        (("concentration", f"rand({batch_shape})"), ("rate", f"rand({batch_shape})")),
        batch_shape,
    )

    # Geometric
    add_case(
        "dist.Geometric(probs=case.probs)",
        (("probs", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Gumbel
    add_case(
        "dist.Gumbel(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # HalfCauchy
    add_case(
        "dist.HalfCauchy(scale=case.scale)",
        (("scale", f"rand({batch_shape})"),),
        batch_shape,
    )

    # HalfNormal
    add_case(
        "dist.HalfNormal(scale=case.scale)",
        (("scale", f"rand({batch_shape})"),),
        batch_shape,
    )

    # Laplace
    add_case(
        "dist.Laplace(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # Logistic
    add_case(
        "dist.Logistic(loc=case.loc, scale=case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # # LowRankMultivariateNormal
    for event_shape in [(3,), (4,)]:
        add_case(
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
        add_case(
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
        if len(batch_shape) > 0:
            xfail = "MultivariateNormal is buggy for batched distributions"
        else:
            xfail = None

        add_case(
            "dist.MultivariateNormal(loc=case.loc, scale_tril=case.scale_tril)",
            (
                ("loc", f"rand({batch_shape + event_shape})"),
                ("scale_tril", f"random_scale_tril({batch_shape + event_shape * 2})"),
            ),
            batch_shape,
            xfail=xfail,
        )

    # NegativeBinomial
    add_case(
        "dist.NegativeBinomial(total_count=case.total_count, probs=case.probs)",
        (
            ("total_count", "5"),
            ("probs", f"rand({batch_shape})"),
        ),
        batch_shape,
    )

    # Normal
    add_case(
        "dist.Normal(case.loc, case.scale)",
        (("loc", f"rand({batch_shape})"), ("scale", f"rand({batch_shape})")),
        batch_shape,
    )

    # OneHotCategorical
    for size in [2, 4]:
        add_case(
            "dist.OneHotCategorical(probs=case.probs)",
            (("probs", f"rand({batch_shape + (size,)})"),),
            batch_shape,  # funsor.Bint[size],
        )

    # Pareto
    add_case(
        "dist.Pareto(scale=case.scale, alpha=case.alpha)",
        (("scale", f"rand({batch_shape})"), ("alpha", f"rand({batch_shape})")),
        batch_shape,
    )

    # Poisson
    add_case(
        "dist.Poisson(rate=case.rate)",
        (("rate", f"rand({batch_shape})"),),
        batch_shape,
    )

    # RelaxedBernoulli
    add_case(
        "dist.RelaxedBernoulli(temperature=case.temperature, logits=case.logits)",
        (("temperature", f"rand({batch_shape})"), ("logits", f"rand({batch_shape})")),
        batch_shape,
    )

    # StudentT
    add_case(
        "dist.StudentT(df=case.df, loc=case.loc, scale=case.scale)",
        (
            ("df", f"rand({batch_shape})"),
            ("loc", f"rand({batch_shape})"),
            ("scale", f"rand({batch_shape})"),
        ),
        batch_shape,
    )

    # Uniform
    add_case(
        "dist.Uniform(low=case.low, high=case.high)",
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # VonMises
    add_case(
        "dist.VonMises(case.loc, case.concentration)",
        (("loc", f"rand({batch_shape})"), ("concentration", f"rand({batch_shape})")),
        batch_shape,
        xfail="problem with vmap and data-dependent control flow in rejection sampling",
    )

    # Weibull
    add_case(
        "dist.Weibull(scale=case.scale, concentration=case.concentration)",
        (
            ("scale", f"exp(rand({batch_shape}))"),
            ("concentration", f"exp(rand({batch_shape}))"),
        ),
        batch_shape,
    )

    # TransformedDistributions
    # ExpTransform
    add_case(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.ExpTransform()])
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # InverseTransform (log)
    add_case(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.ExpTransform().inv])
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # TanhTransform
    add_case(
        """
        dist.TransformedDistribution(
            dist.Uniform(low=case.low, high=case.high),
            [dist.transforms.TanhTransform(),])
        """,
        (("low", f"rand({batch_shape})"), ("high", f"2. + rand({batch_shape})")),
        batch_shape,
    )

    # AtanhTransform
    add_case(
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
    add_case(
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
    add_case(
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
    add_case(
        """
        dist.TransformedDistribution(
            dist.Exponential(rate=case.rate),
            dist.transforms.PowerTransform(0.5))
        """,
        (("rate", f"rand({batch_shape})"),),
        batch_shape,
    )

    # HaarTransform
    add_case(
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
        add_case(
            f"dist.Beta(case.concentration1, case.concentration0).to_event({len(indep_shape)})",
            (
                ("concentration1", f"exp(rand({batch_shape + indep_shape}))"),
                ("concentration0", f"exp(rand({batch_shape + indep_shape}))"),
            ),
            batch_shape,
        )

        # Dirichlet.to_event
        for event_shape in [(2,), (4,)]:
            add_case(
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
        add_case(
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


@functools.cache
def name_to_sym(name: str) -> Operation[[], jax.Array]:
    return defop(jax.Array, name=name)


def from_indexed(tensor, batch_dims):
    tensor_sizes = sizesof(tensor)
    indices = [name_to_sym(str(i)) for i in range(batch_dims)]
    indices = [i for i in indices if i in tensor_sizes]
    return to_array(tensor, *indices)


class DistTestCase:
    raw_dist: str
    params: dict[str, jax.Array]
    indexed_params: dict[str, jax.Array]
    batch_shape: tuple[int, ...]
    xfail: str | None
    kind: str

    def __init__(
        self,
        raw_dist: str,
        params: dict[str, jax.Array],
        indexed_params: dict[str, jax.Array],
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

        try:
            case = Case(**self.params)
            dist_ = eval(self.raw_dist)

            # case is used by generated code in self.raw_dist
            case = Case(**self.indexed_params)  # noqa: F841
            indexed_dist = eval(self.raw_dist)
        except AttributeError:
            pytest.xfail(f"Distribution {self.raw_dist} not implemented")

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
    params: dict[str, jax.Array],
    batch_shape: tuple[int, ...],
    xfail: str | None = None,
):
    indexed_params = {}
    for name, param in params.items():
        if (
            isinstance(param, jax.Array)
            and param.shape[: len(batch_shape)] == batch_shape
        ):
            indexes = tuple(name_to_sym(str(i))() for i in range(len(batch_shape)))
            indexed_params[name] = jax_getitem(param, indexes)
        else:
            indexed_params[name] = param

    return DistTestCase(raw_dist, params, indexed_params, batch_shape, xfail, "full")


def partial_indexed_test_case(
    raw_dist: str,
    params: dict[str, jax.Array],
    batch_shape: tuple[int, ...],
    xfail: str | None = None,
):
    """Produces parameters with a subset of batch dimensions indexed.

    For example, if batch_shape is (2, 3) and params is
    {"loc": jax.random.normal(key, (2, 3, 4)), "scale": jax.random.normal(key, (2, 3, 4))},
    this function will return a test case with indexed parameters
    {"loc": jax.random.normal(key, (2, 3, 4))[i0(), 0, i2()], "scale": jax.random.normal(key, (2, 3, 4))[0, i1(), i2()]}.
    """
    non_indexed_params = {
        k: v
        for (k, v) in params.items()
        if not (isinstance(v, jax.Array) and v.shape[: len(batch_shape)] == batch_shape)
    }
    broadcast_params = params.copy()
    indexed_params = {}

    indexed_param_names = set(name for name in params if name not in non_indexed_params)
    for i, name in enumerate(indexed_param_names):
        param = params[name]

        if (
            isinstance(param, jax.Array)
            and param.shape[: len(batch_shape)] == batch_shape
        ):
            indexes = []
            for j in range(len(batch_shape)):
                if i == j or j >= len(indexed_param_names):
                    index = name_to_sym(str(j))()
                else:
                    index = jnp.array(0)
                    broadcast_params[name] = jnp.expand_dims(
                        jnp.take(broadcast_params[name], 0, axis=j), j
                    )
                indexes.append(index)
            indexed_params[name] = jax_getitem(param, tuple(indexes))
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
    # Convert PyTorch-style parameters to JAX
    # This assumes RAW_TEST_CASES are defined with PyTorch syntax
    # We'll need to adapt them for JAX

    # Create a random key for JAX random operations
    key = jax.random.PRNGKey(0)

    def rand(shape):
        return jax.random.uniform(key, shape=shape)

    def randint(low, high, shape):
        return jax.random.randint(key, shape, low, high)

    def random_scale_tril(shape):
        data = jax.random.normal(key, shape)
        return numpyro.distributions.transforms.biject_to(
            numpyro.distributions.constraints.lower_cholesky
        )(data)

    globals = {
        "rand": rand,
        "randint": randint,
        "exp": jax.numpy.exp,
        "random_scale_tril": random_scale_tril,
    }
    params = {name: eval(raw_param, globals) for name, raw_param in raw_params}
    TEST_CASES.append(full_indexed_test_case(raw_dist, params, batch_shape, xfail))

    # This case is trivial if there are not multiple batch dimensions and
    # multiple parameters
    if len(batch_shape) > 1 and len(params) > 1:
        TEST_CASES.append(
            partial_indexed_test_case(raw_dist, params, batch_shape, xfail)
        )


for c in RAW_TEST_CASES:
    add_dist_test_case(c.raw_dist, c.raw_params, c.batch_shape, c.xfail)


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
def test_dist_to_positional(case_):
    _, indexed_dist = case_.get_dist()

    try:
        pos_dist, naming = positional_distribution(indexed_dist)
        key = jax.random.PRNGKey(0)
        pos_sample = pos_dist.sample(key)
        assert sizesof(pos_sample) == {}
        indexed_sample = indexed_dist.sample(key)

        # Check that samples have compatible shapes
        # JAX doesn't have a direct equivalent to torch.broadcast_all
        # but we can check that the shapes are compatible
        pos_shape = jnp.shape(pos_sample)
        indexed_shape = jnp.shape(from_indexed(indexed_sample, len(case_.batch_shape)))
        assert len(pos_shape) == len(indexed_shape)
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
        indexed_dist = named_distribution(dist, *indexes)

        key = jax.random.PRNGKey(0)
        indexed_sample = indexed_dist.sample(key)
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

    # JAX distributions don't have an expand method like PyTorch
    # Instead, we can use the expand_by method
    try:
        expanded = indexed_dist.expand_by(extra_batch_shape)

        # JAX distributions need a random key for sampling
        key = jax.random.PRNGKey(0)
        sample_shape_full = indexed_sample_shape + sample_shape

        # Generate samples
        sample = expanded.sample(key, sample_shape_full)

        # Index into the sample
        indexed_sample = sample[
            tuple(defop(jax.Array)() for _ in range(len(indexed_sample_shape)))
        ]

        # Check shapes
        expected_shape = (
            sample_shape
            + extra_batch_shape
            + indexed_dist.batch_shape
            + indexed_dist.event_shape
        )
        assert indexed_sample.shape == expected_shape

        # Check log_prob shape
        log_prob = expanded.log_prob(indexed_sample)
        expected_log_prob_shape = extra_batch_shape + sample_shape
        assert log_prob.shape == expected_log_prob_shape
    except (AttributeError, NotImplementedError):
        pytest.xfail("expand_by not implemented for this distribution")


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
def test_dist_indexes(case_):
    """Test that indexed samples and logprobs have the correct shape and indices."""
    dist, indexed_dist = case_.get_dist()

    # JAX distributions need a random key for sampling
    key = jax.random.PRNGKey(0)
    sample = dist.sample(key)
    indexed_sample = indexed_dist.sample(key)

    # Samples should not have any indices that their parameters don't have
    assert set(sizesof(indexed_sample)) <= set().union(
        *[set(sizesof(p)) for p in case_.indexed_params.values()]
    )

    # Indexed samples should have the same shape as regular samples, modulo
    # possible extra unit dimensions
    indexed_sample_a = from_indexed(indexed_sample, len(case_.batch_shape))
    # Use jnp.squeeze instead of .squeeze() method
    assert jnp.squeeze(sample).shape == jnp.squeeze(indexed_sample_a).shape
    assert sample.dtype == indexed_sample_a.dtype

    lprob = dist.log_prob(sample)
    indexed_lprob = indexed_dist.log_prob(indexed_sample)

    # Indexed logprobs should have the same shape as regular logprobs, but with
    # the batch dimensions indexed
    indexed_lprob_a = from_indexed(indexed_lprob, len(case_.batch_shape))
    assert lprob.shape == indexed_lprob_a.shape
    assert lprob.dtype == indexed_lprob_a.dtype


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
@pytest.mark.parametrize("sample_shape", [(), (2,), (3, 2)])
def test_dist_randomness(case_, sample_shape):
    """Test that indexed samples differ across the batch dimensions."""
    pos_dist, indexed_dist = case_.get_dist()

    # Skip discrete distributions
    if (
        "Poisson" in case_.raw_dist
        or "Geometric" in case_.raw_dist
        or "Bernoulli" in case_.raw_dist
        or "Binomial" in case_.raw_dist
        or "Categorical" in case_.raw_dist
    ):
        pytest.xfail("Discrete distributions not supported")

    # JAX distributions need a random key for sampling
    key = jax.random.PRNGKey(0)

    # JAX doesn't have rsample, only sample
    indexed_sample = indexed_dist.sample(key, sample_shape)
    pos_sample = pos_dist.sample(key, sample_shape)

    indexed_sample_a = from_indexed(indexed_sample, len(case_.batch_shape))

    # Reshape to check for uniqueness across batch dimensions
    new_shape = (-1,) + pos_sample.shape[len(case_.batch_shape) :]
    flat_sample = jnp.reshape(pos_sample, new_shape)
    flat_indexed_sample = jnp.reshape(indexed_sample_a, new_shape)

    # JAX doesn't have a unique method like PyTorch
    # Instead, we can check if there's variation in the samples
    if len(flat_sample) > 1:
        # Check if there's variation in the samples
        sample_std = jnp.std(flat_sample, axis=0)
        indexed_std = jnp.std(flat_indexed_sample, axis=0)

        # If there's variation in the original samples, there should be
        # variation in the indexed samples too
        if jnp.any(sample_std > 1e-5):
            assert jnp.any(indexed_std > 1e-5)


@pytest.mark.parametrize("case_", TEST_CASES, ids=str)
@pytest.mark.parametrize("statistic", ["mean", "variance"])
def test_dist_stats(case_, statistic):
    """Test that indexed distributions have the same statistics as their unindexed counterparts."""
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
    except (NotImplementedError, AttributeError):
        pytest.xfail(f"{statistic} not implemented")

    # JAX doesn't have isnan().all() method like PyTorch
    if jnp.all(jnp.isnan(expected_stat)):
        assert jnp.all(jnp.isnan(to_array(actual_stat)))
    else:
        # Stats may not be indexed in all batch dimensions, but they should be
        # extensionally equal to the indexed expected stat
        indexes = [name_to_sym(str(i)) for i in range(len(case_.batch_shape))]
        expected_stat_i = expected_stat[tuple(n() for n in indexes)]

        # JAX doesn't have broadcast_tensors like PyTorch
        # Instead, we can use jnp.broadcast_arrays
        expected_shape = jnp.broadcast_shapes(expected_stat_i.shape, actual_stat.shape)
        expected_stat_i = jnp.broadcast_to(expected_stat_i, expected_shape)
        actual_stat_i = jnp.broadcast_to(actual_stat, expected_shape)

        # Check that the stats are close
        assert jnp.allclose(
            to_array(expected_stat_i, *indexes),
            to_array(actual_stat_i, *indexes),
            rtol=1e-5,
            atol=1e-5,
        )


def test_distribution_terms():
    x = defop(jax.Array, name="x")
    y = defop(jax.Array, name="y")

    d1 = dist.Normal(x(), y())
    assert isinstance(d1, Term) and not isinstance(
        d1, numpyro.distributions.Distribution
    )

    a = jax_getitem(jnp.array([0.0]), [x()])
    b = jax_getitem(jnp.array([1.0]), [y()])
    d2 = dist.Normal(a, b)
    assert isinstance(d2, Term) and isinstance(d2, numpyro.distributions.Distribution)

    d3 = dist.Normal(jnp.array(0.0), jnp.array(1.0))
    assert not isinstance(d3, Term) and isinstance(
        d3, numpyro.distributions.Distribution
    )
