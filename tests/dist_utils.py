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
        add_case(
            "dist.MultivariateNormal(loc=case.loc, scale_tril=case.scale_tril)",
            (
                ("loc", f"rand({batch_shape + event_shape})"),
                ("scale_tril", f"random_scale_tril({batch_shape + event_shape * 2})"),
            ),
            batch_shape,
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
