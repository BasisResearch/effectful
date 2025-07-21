import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numpyro as dist
import jax
import pytest
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import deffn, defop
from jax.numpy import isclose

from weighted.handlers.jax import (
    GradientOptimizationFold,
    LikelihoodWeightingFold,
    log_prob,
    reals,
    sample,
)
from weighted.handlers.jax import interpretation as jax_intp
from weighted.ops.fold import BaselineFold
from weighted.ops.sugar import ArgMin, Sum

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


def test_sampling():
    n_samples = 1
    key = jax.random.key(0)

    s1 = sample(key, dist.Normal(0.0, 1.0), (n_samples,))
    assert isinstance(s1, jax.Array)

    loc, scale = defop(jax.Array, name="loc"), defop(jax.Array, name="scale")
    with handler({loc: deffn(0.0), scale: deffn(1.0)}):
        s2 = sample(key, dist.Normal(loc(), scale()), (n_samples,))
    assert isinstance(s2, jax.Array)

    s3_term = sample(key, dist.Normal(loc(), scale()), (n_samples,))
    assert not isinstance(s3_term, jax.Array)
    with handler({loc: deffn(0.0), scale: deffn(1.0)}):
        s3 = evaluate(s3_term)
    assert isinstance(s3, jax.Array)


def test_maximum_marginal_likelihood_smoke():
    data = jnp.exp(jax.random.normal(jax.random.key(0), (10,)))

    loc_z = defop(jax.Array, name="loc_z")
    scale_z = defop(jax.Array, name="scale_z")
    z = defop(jax.Array, name="z")
    scale_x = defop(jax.Array, name="scale_x")

    z_dist = dist.Normal(loc_z(), scale_z())
    x_dist = dist.Normal(jnp.exp(z()), scale_x())

    n_samples = 1

    with (
        handler(jax_intp),
        handler(
            GradientOptimizationFold(
                steps=1,
                learning_rate=0.1,
                init={scale_z: jnp.array(1.0), scale_x: jnp.array(1.0)},
            )
        ),
    ):
        weight = -(log_prob(z_dist, z()) + jnp.sum(log_prob(x_dist, data)))
        intg_weight = Sum({z: sample(jax.random.key(0), z_dist, (n_samples,))}, weight)
        _min = ArgMin(
            {loc_z: reals(), scale_z: reals(), scale_x: reals()},
            (intg_weight, (loc_z(), scale_z(), scale_x())),
        )


def run_expectation():
    loc = 0.0
    scale = 1.0

    def f(x):
        return x**2

    x = defop(jax.Array, name="x")
    w = defop(jax.Array, name="w")
    with handler(jax_intp), handler(LikelihoodWeightingFold(samples=1000)):
        return Sum({(x, w): dist.Normal(loc, scale)}, jnp.exp(w()) * f(x()))


def test_interpretation_body():
    with handler(BaselineFold()):
        w, x, y, z = (
            defop(int, name="w"),
            defop(int, name="x"),
            defop(int, name="y"),
            defop(int, name="z"),
        )

        intp = Sum(
            {x: [1, 2, 3], y: (x() + 1,)}, {z: deffn(x() + y()), w: deffn(x() * y())}
        )
        assert evaluate(z(), intp=intp) == sum(x + y for x in [1, 2, 3] for y in [x + 1])
        assert evaluate(w(), intp=intp) == sum(x * y for x in [1, 2, 3] for y in [x + 1])

        intp = Sum(
            {x: [1, 2, 3], y: (x() + 1,)}, {z: lambda: x() + y(), w: lambda: z() + 2}
        )
        assert evaluate(w(), intp=intp) == sum(x + (x + 1) + 2 for x in [1, 2, 3])

        # Edge case: Complex stream dependency graph
        a, b, c, d, e, f = (
            defop(int, name="a"),
            defop(int, name="b"),
            defop(int, name="c"),
            defop(int, name="d"),
            defop(int, name="e"),
            defop(int, name="f"),
        )
        intp_stream_deps = Sum(
            {
                a: [1, 2],  # a independent
                b: [10, 20],  # b independent
                c: [a() * 2],  # c depends on a
                d: [b() + 3],  # d depends on b
                e: [a() + d()],  # e depends on a and d
                f: [c() + d()],  # f depends on c and d
            },
            {z: f},
        )
        expected_stream_deps = sum(
            (a_val * 2) + (b_val + 3)
            for a_val in [1, 2]
            for b_val in [10, 20]
            for c_val in [a_val * 2]
            for d_val in [b_val + 3]
            for e_val in [a_val + d_val]
            for f_val in [c_val + d_val]
        )
        assert evaluate(z(), intp=intp_stream_deps) == expected_stream_deps

        # Edge case: Circular dependencies in streams should raise exception
        x1, x2 = defop(int, name="x1"), defop(int, name="x2")
        with pytest.raises(Exception):  # noqa: B017
            intp_circular = Sum(
                {
                    x1: [x2()],  # x1 depends on x2
                    x2: [x1() + 1],  # x2 depends on x1 - circular!
                },
                {z: deffn(x1())},
            )
            evaluate(z(), intp=intp_circular)


def test_integration(benchmark):
    intg = run_expectation()
    assert isinstance(intg, jax.Array)
    assert isclose(intg, jnp.array(0.5), atol=1e-1)

    intg = jax.jit(run_expectation)()
    assert isinstance(intg, jax.Array)
    assert isclose(intg, jnp.array(0.5), atol=1e-1)


def test_integration_benchmark(benchmark):
    benchmark(run_expectation)


def test_integration_jit_benchmark(benchmark):
    benchmark(jax.jit(run_expectation))
