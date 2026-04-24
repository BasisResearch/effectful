import jax
import pytest
from jax import random
from jax.numpy import allclose
from pytest import mark, param

import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numpyro as dist
from effectful.handlers.jax import jax_getitem, unbind_dims
from effectful.handlers.jax.monoid import CartesianProd, Max, Product, Sum
from effectful.handlers.weighted.optimization import ReducePropagateUnusedStreams
from effectful.handlers.weighted.optimization.cartesian_product import (
    ReduceDistributeCartesianProduct,
)
from effectful.handlers.weighted.optimization.distribution import (
    NormalVerticalFusion,
    SampleAddNormalFusion,
    SampleMulConstantFusion,
)
from effectful.handlers.weighted.optimization.distribution import (
    interpretation as simplify_normals_intp,
)
from effectful.handlers.weighted.optimization.polyhedral import ReduceLinearIndexer
from effectful.handlers.weighted.optimization.reorder import (
    ReduceDistributeTerm,
    ReduceNoStreams,
)
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import deffn, defop, syntactic_eq
from tests.utils import REDUCE_TRANSFORMS

parameterize_transform_intp = mark.parametrize(
    "transform_intp", [param(x, id=type(x).__name__) for x in REDUCE_TRANSFORMS]
)


@pytest.mark.skip(reason="evaluation order")
@parameterize_transform_intp
def test_factorize(base_intp, transform_intp):
    dim_size = 4
    key = random.PRNGKey(42)

    ix_names = ("i", "j", "k", "l", "m")
    arr_names = ("x", "y", "z")
    ops = {k: defop(jax.Array, name=k) for k in arr_names + ix_names}
    x_bound = unbind_dims(ops["x"](), ops["i"], ops["j"])
    y_bound = unbind_dims(ops["y"](), ops["k"], ops["l"])
    z_bound = unbind_dims(ops["z"](), ops["l"], ops["m"])

    streams = {ops[k]: jnp.arange(dim_size) for k in ix_names}
    reduce_expr = Sum.reduce(streams, x_bound * y_bound * z_bound)
    with handler(transform_intp):
        reduce_expr = evaluate(reduce_expr)

    keys = random.split(key, len(arr_names))
    arrs = {
        k: random.normal(key, (dim_size, dim_size))
        for k, key in zip(arr_names, keys, strict=False)
    }
    arr_intp = {ops[k]: deffn(v) for k, v in arrs.items()}
    with handler(base_intp), handler(arr_intp):
        result = evaluate(reduce_expr)

    expected = jnp.einsum("ij,kl,lm->", arrs["x"], arrs["y"], arrs["z"])
    assert allclose(result, expected)


@parameterize_transform_intp
def test_fuse_split(base_intp, transform_intp):
    dim_size = 4
    key = random.PRNGKey(42)
    x_ix = defop(jax.Array, name="x_ix")
    y_ix = defop(jax.Array, name="y_ix")
    x_op = defop(jax.Array, name="x")
    y_op = defop(jax.Array, name="y")

    k1, k2 = random.split(key, 2)
    x_arr = random.normal(k1, (dim_size,))
    y_arr = random.normal(k2, (dim_size,))

    x_stream = {x_ix: jnp.arange(dim_size)}
    y_stream = {y_ix: jnp.arange(dim_size)}
    arr_intp = {x_op: lambda: x_arr, y_op: lambda: y_arr}

    reduce_intp = coproduct(base_intp, transform_intp)
    expected = (x_arr.sum() + y_arr.sum()) * dim_size

    with handler(reduce_intp), handler(arr_intp):
        x = jax_getitem(x_op(), (x_ix(),))
        y = jax_getitem(y_op(), (y_ix(),))
        result1 = Sum.reduce(x_stream, Sum.reduce(y_stream, x + y))
        result2 = Sum.reduce(x_stream | y_stream, x + y)

    assert allclose(result1, expected)
    assert allclose(result2, expected)


def test_unused_streams_optim():
    i, j, k = defop(jax.Array), defop(jax.Array), defop(jax.Array)
    intp = ReducePropagateUnusedStreams()

    with handler(intp):
        expr = Sum.reduce({i: jnp.arange(3), j: jnp.arange(3)}, k())
    assert expr.op is jnp.multiply

    with handler(intp):
        expr = Max.reduce({i: jnp.arange(3), j: jnp.arange(3)}, k())
    assert expr.op is k


def test_normal_vertical_fusion():
    key = random.PRNGKey(42)

    mu = defop(jax.Array, name="mu")()
    sigma1 = defop(jax.Array, name="s1")()
    sigma2 = defop(jax.Array, name="s2")()

    d1 = dist.Normal(mu, sigma1)
    d1_samples = d1.sample(key)
    with handler(NormalVerticalFusion()):
        d2_opt = dist.Normal(d1_samples, sigma2)

    expected = dist.Normal(mu, jnp.sqrt(sigma2**2 + sigma1**2))
    assert syntactic_eq(d2_opt, expected)


def test_normal_constant_mul():
    key = random.PRNGKey(42)
    mu = defop(jax.Array, name="mu1")
    sigma = defop(jax.Array, name="s1")
    d1 = dist.Normal(mu(), sigma())

    with handler(SampleMulConstantFusion()):
        expr = 2.0 * d1.sample(key, (1,))

    expected = dist.Normal(2.0 * mu(), jnp.abs(2.0) * sigma()).sample(key, (1,))
    assert syntactic_eq(expr, expected)


def test_add_normal_distributions():
    key = random.PRNGKey(42)

    mu1 = defop(jax.Array, name="mu1")()
    sigma1 = defop(jax.Array, name="s1")()
    mu2 = defop(jax.Array, name="mu2")()
    sigma2 = defop(jax.Array, name="s2")()

    d1 = dist.Normal(mu1, sigma1)
    d2 = dist.Normal(mu2, sigma2)

    with handler(SampleAddNormalFusion()):
        expr = d1.sample(key) + d2.sample(key)

    d3 = dist.Normal(mu1 + mu2, jnp.sqrt(sigma1**2 + sigma2**2))
    expected = d3.sample(key)

    assert syntactic_eq(expr, expected)


def test_distributional_equivalence_normal_transforms():
    nb_samples = 10000
    keys = random.split(random.PRNGKey(1), 6)
    mu1 = defop(jax.Array, name="mu1")
    sigma1 = defop(jax.Array, name="s1")
    mu2 = defop(jax.Array, name="mu2")
    sigma2 = defop(jax.Array, name="s2")
    sigma3 = defop(jax.Array, name="sigma3")
    d1 = dist.Normal(mu1(), sigma1())
    d2 = dist.Normal(mu2(), sigma2())

    samples_d1 = d1.sample(keys[0], (nb_samples,))
    samples_d2 = d2.sample(keys[1], (nb_samples,))
    c = random.uniform(keys[2]) * 10
    expected_d = dist.Normal(c * samples_d1 + samples_d2, sigma3())
    with handler(simplify_normals_intp):
        actual_d = evaluate(expected_d)

    params = (mu1, sigma1, mu2, sigma2, sigma3)
    intp = {
        k: deffn(random.uniform(key) * 10) for k, key in zip(params, keys, strict=False)
    }
    with handler(intp):
        expected_d = evaluate(expected_d)
        actual_d = evaluate(actual_d)
    expected_samples = expected_d.sample(keys[4], (nb_samples,))
    actual_samples = actual_d.sample(keys[5], (nb_samples,))

    assert allclose(jnp.std(expected_samples), jnp.std(actual_samples), atol=1)
    assert allclose(jnp.mean(expected_samples), jnp.mean(actual_samples), atol=1)


def test_cartesian_product_distribution():
    key = random.PRNGKey(42)

    x_size, i_size = 3, 2
    i = defop(jax.Array, name="i")
    x = defop(jax.Array, name="x")
    i_stream = {i: jnp.arange(i_size)}
    x_stream = {x: CartesianProd.reduce(i_stream, jnp.arange(x_size))}

    arr = random.uniform(key, shape=(x_size, i_size))
    expr = lambda: Sum.reduce(
        x_stream, Product.reduce(i_stream, jax_getitem(arr, (x()[i()], i())))
    )

    expected = expr()

    # check if the result is the same
    with handler(ReduceDistributeCartesianProduct()), handler(ReduceNoStreams()):
        result = expr()
    assert allclose(result, expected)


def test_reduce_distribute_term():
    i = defop(jax.Array, name="i")
    j = defop(jax.Array, name="j")
    k = defop(jax.Array, name="k")

    I = defop(jax.Array, name="I")
    J = defop(jax.Array, name="J")

    k_stream = jax_getitem(jnp.arange(12).reshape((3, 4)), (j(),))
    streams = {i: I(), j: J(), k: k_stream}

    a = defop(jax.Array, name="a")
    b = defop(jax.Array, name="b")

    term1 = jax_getitem(a(), (i(),))
    term2 = jax_getitem(b(), (i(), k()))

    with handler(ReduceDistributeTerm()):
        # Sum.reduce({i: i_stream}, term_1 * Sum.reduce({j: j_stream, k: k_stream}, term_2))
        optimized_expr = Sum.reduce(streams, term1 * term2)

    # Check if the optimization triggered
    assert str(optimized_expr.op) == "reduce"
    assert optimized_expr.args[1].op is jnp.multiply
    assert str(optimized_expr.args[1].args[1].op) == "reduce"

    k1, k2 = random.split(random.PRNGKey(42))
    arr_intp = {
        I: deffn(jnp.arange(2)),
        J: deffn(jnp.arange(3)),
        a: deffn(random.normal(k1, (2,))),
        b: deffn(random.normal(k2, (2, 4))),
    }

    with handler(arr_intp):
        expected = Sum.reduce(streams, term1 * term2)
        result = evaluate(optimized_expr)
    assert allclose(result, expected)


def test_polyhedral():
    # 2D triangle
    # { [i, j] : i + 10 <= j < 2 * i + 10 and 1 <= i < 10 + 1}
    i = defop(jax.Array, name="i")
    j = defop(jax.Array, name="j")

    streams = {
        i: jnp.arange(1, 10),
        j: jnp.arange(i() + 10, 2 * i() + 10),
    }

    with handler(ReduceLinearIndexer()):
        result = Sum.reduce(streams, i() / j())

    expected = Sum.reduce(streams, i() / j())
    assert allclose(result, expected)
