import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numbers  # noqa: F401
import effectful.handlers.numpyro as dist
import jax
from effectful.handlers.jax import jax_getitem, unbind_dims
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import deffn, defop, syntactic_eq
from jax import random
from jax.numpy import allclose
from pytest import mark, param

from tests.utils import FOLD_TRANSFORMS
from weighted.handlers.jax import (
    DenseTensorFold,
    syntactic_eq_jax,
)
from weighted.handlers.optimization import FoldPropagateUnusedStreams
from weighted.handlers.optimization.cartesian_product import (
    FoldDistributeCartesianProduct,
    unify_streams,
)
from weighted.handlers.optimization.distribution import (
    NormalVerticalFusion,
    SampleAddNormalFusion,
    SampleMulConstantFusion,
)
from weighted.handlers.optimization.distribution import (
    interpretation as simplify_normals_intp,
)
from weighted.handlers.optimization.polyhedral import FoldLinearIndexer
from weighted.handlers.optimization.quadrature import GaussHermiteQuadrature
from weighted.handlers.optimization.reorder import FoldDistributeTerm, FoldNoStreams
from weighted.ops.distribution import log_prob, sample
from weighted.ops.fold import BaselineFold, fold
from weighted.ops.jax import reals
from weighted.ops.monoid import mul
from weighted.ops.sugar import CartesianProd, Max, Prod, Sum

parameterize_base_intp = mark.parametrize(
    "base_intp",
    [param(BaselineFold(), id="baseline"), param(DenseTensorFold(), id="jax")],
)

parameterize_transform_intp = mark.parametrize(
    "transform_intp", [param(x, id=type(x).__name__) for x in FOLD_TRANSFORMS]
)


@parameterize_base_intp
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
    fold_expr = Sum(streams, x_bound * y_bound * z_bound)
    with handler(transform_intp):
        fold_expr = evaluate(fold_expr)

    keys = random.split(key, len(arr_names))
    arrs = {
        k: random.normal(key, (dim_size, dim_size))
        for k, key in zip(arr_names, keys, strict=False)
    }
    arr_intp = {ops[k]: deffn(v) for k, v in arrs.items()}
    with handler(base_intp), handler(arr_intp):
        result = evaluate(fold_expr)

    expected = jnp.einsum("ij,kl,lm->", arrs["x"], arrs["y"], arrs["z"])
    assert allclose(result, expected)


@parameterize_base_intp
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

    fold_intp = coproduct(base_intp, transform_intp)
    expected = (x_arr.sum() + y_arr.sum()) * dim_size

    with handler(fold_intp), handler(arr_intp):
        x = jax_getitem(x_op(), (x_ix(),))
        y = jax_getitem(y_op(), (y_ix(),))
        result1 = Sum(x_stream, Sum(y_stream, x + y))
        result2 = Sum(x_stream | y_stream, x + y)

    assert allclose(result1, expected)
    assert allclose(result2, expected)


def test_unused_streams_optim():
    i, j = defop(jax.Array), defop(jax.Array)
    intp = FoldPropagateUnusedStreams()

    with handler(intp):
        expr = Sum({i: jnp.arange(3), j: jnp.arange(3)}, i())

    assert expr.op is mul
    fold_expr, const = expr.args
    assert fold_expr.op is fold
    assert len(fold_expr.args[1]) == 1
    assert const == 3

    with handler(intp):
        expr = Max({i: jnp.arange(3), j: jnp.arange(3)}, i())

    assert expr.op is fold
    assert len(expr.args[1]) == 1


@parameterize_base_intp
def test_quadrature(base_intp):
    x = defop(jax.Array, name="x")
    polynomial = 2 * x() + x() ** 2
    mu, sigma = 2.5, 5.0
    d = dist.Normal(mu, sigma)

    body = jnp.exp(log_prob(d, x())) * polynomial
    expr = Sum({x: reals()}, body)
    # 3 points are sufficient as polynomial is quadratic
    with handler(GaussHermiteQuadrature(3)):
        expr = evaluate(expr)

    with handler(base_intp):
        expr = evaluate(expr)
    expected = 2 * mu + mu**2 + sigma**2
    assert allclose(expr, expected)


@parameterize_base_intp
def test_bivariate_quadrature(base_intp):
    x = defop(jax.Array, name="x")
    y = defop(jax.Array, name="y")
    mu_x, sigma_x = 5.0, 3.0
    mu_y, sigma_y = 1.0, 2.0
    d_x = dist.Normal(mu_x, sigma_x)
    d_y = dist.Normal(mu_y, sigma_y)
    prob_x = jnp.exp(log_prob(d_x, x()))
    prob_y = jnp.exp(log_prob(d_y, y()))

    polynomial = 2 * x() + y() ** 2
    body = prob_x * prob_y * polynomial
    expr = Sum({x: reals(), y: reals()}, body)
    # 3 points are sufficient as polynomial is quadratic
    with handler(GaussHermiteQuadrature(3)):
        expr = evaluate(expr)

    with handler(base_intp):
        expr = evaluate(expr)
    expected = 2 * mu_x + mu_y**2 + sigma_y**2
    assert allclose(expr, expected)


def test_normal_vertical_fusion():
    key = random.PRNGKey(42)
    mu = defop(jax.Array, name="mu")
    sigma1 = defop(jax.Array, name="s1")
    sigma2 = defop(jax.Array, name="s2")

    d1 = dist.Normal(mu(), sigma1())
    d1_samples = sample(key, d1, (1,))
    with handler(NormalVerticalFusion()):
        d2_opt = dist.Normal(d1_samples, sigma2())

    expected = dist.Normal(mu(), jnp.sqrt(sigma2() ** 2 + sigma1() ** 2))
    assert syntactic_eq(d2_opt, expected)


def test_normal_constant_mul():
    key = random.PRNGKey(42)
    mu = defop(jax.Array, name="mu1")
    sigma = defop(jax.Array, name="s1")
    d1 = dist.Normal(mu(), sigma())

    with handler(SampleMulConstantFusion()):
        expr = 2.0 * sample(key, d1, (1,))

    expected = sample(key, dist.Normal(2.0 * mu(), jnp.abs(2.0) * sigma()), (1,))
    assert syntactic_eq_jax(expr, expected)


def test_add_normal_distributions():
    key = random.PRNGKey(42)
    mu1 = defop(jax.Array, name="mu1")
    sigma1 = defop(jax.Array, name="s1")
    mu2 = defop(jax.Array, name="mu2")
    sigma2 = defop(jax.Array, name="s2")
    d1 = dist.Normal(mu1(), sigma1())
    d2 = dist.Normal(mu2(), sigma2())

    with handler(SampleAddNormalFusion()):
        expr = sample(key, d1, (1,)) + sample(key, d2, (1,))

    d3 = dist.Normal(mu1() + mu2(), jnp.sqrt(sigma1() ** 2 + sigma2() ** 2))
    expected = sample(key, d3, (1,))
    assert syntactic_eq_jax(expr, expected)


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

    samples_d1 = sample(keys[0], d1, (nb_samples,))
    samples_d2 = sample(keys[1], d2, (nb_samples,))
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
    expected_samples = sample(keys[4], expected_d, (nb_samples,))
    actual_samples = sample(keys[5], actual_d, (nb_samples,))

    assert allclose(jnp.std(expected_samples), jnp.std(actual_samples), atol=1)
    assert allclose(jnp.mean(expected_samples), jnp.mean(actual_samples), atol=1)


def test_cartesian_product_distribution():
    key = random.PRNGKey(42)

    x_size, i_size = 3, 2
    i = defop(jax.Array, name="i")
    x = defop(jax.Array, name="x")
    i_stream = {i: jnp.arange(i_size)}
    x_stream = {x: CartesianProd(i_stream, jnp.arange(x_size))}

    arr = random.uniform(key, shape=(x_size, i_size))
    expr = Sum(x_stream, Prod(i_stream, jax_getitem(arr, (x()[i()], i()))))

    with handler(BaselineFold()):
        expected = evaluate(expr)

    with handler(FoldDistributeCartesianProduct()), handler(FoldNoStreams()):
        expr_eval = evaluate(expr)
        # check if optimization is applied
        assert str(expr_eval.args[0]) == "Prod"

    # check if the result is the same
    with handler(DenseTensorFold()):
        result = evaluate(expr_eval)
    assert allclose(result, expected)


def test_unify_streams():
    i = defop(jax.Array, name="i")
    j = defop(jax.Array, name="j")
    i_stream = jnp.arange(3)
    j_stream = jax_getitem(jnp.ones((3, 4)), (i(),))
    expr = Sum({i: i_stream}, i())

    # different sized streams, don't unify
    expr2 = Sum({i: jnp.arange(4)}, i())
    assert unify_streams(expr.args[1], expr2.args[1]) is None

    # different key in stream, don't unify
    expr2 = Sum({j: i_stream}, i())
    assert unify_streams(expr.args[1], expr2.args[1]) is None

    # same streams, unify
    expr2 = Sum({i: i_stream}, i())
    unifier = unify_streams(expr.args[1], expr2.args[1])
    fresh_i = tuple(expr.args[1].keys())[0]
    fresh_i2 = tuple(expr2.args[1].keys())[0]
    assert syntactic_eq(unifier, {fresh_i: fresh_i2})

    # same dependent streams, unify
    expr = Sum({i: i_stream, j: j_stream}, j())
    expr2 = Sum({i: i_stream, j: j_stream}, j())
    unifier = unify_streams(expr.args[1], expr2.args[1])
    assert unifier is not None

    # dependent streams with different structure, don't unify
    expr2 = Sum({i: i_stream, j: jax_getitem(jnp.ones((4, 4)), (i(),))}, j())
    assert unify_streams(expr.args[1], expr2.args[1]) is None

    # different variable names in dependency, don't unify
    k = defop(jax.Array, name="k")
    expr2 = Sum({k: jnp.arange(3), j: jax_getitem(jnp.ones((3, 4)), (k(),))}, j())
    assert unify_streams(expr.args[1], expr2.args[1]) is None


def test_fold_distribute_term():
    i = defop(jax.Array, name="i")
    j = defop(jax.Array, name="j")
    k = defop(jax.Array, name="k")

    i_stream = jnp.arange(2)
    j_stream = jnp.arange(3)
    k_stream = jax_getitem(jnp.arange(12).reshape((3, 4)), (j(),))
    streams = {i: i_stream, j: j_stream, k: k_stream}

    a = defop(jax.Array, name="a")
    b = defop(jax.Array, name="b")

    term1 = jax_getitem(a(), (i(),))
    term2 = jax_getitem(b(), (i(), k()))

    expr = Sum(streams, term1 * term2)
    with handler(FoldDistributeTerm()):
        # Sum({i: i_stream}, term_1 * Sum({j: j_stream, k: k_stream}, term_2))
        optimized_expr = evaluate(expr)

    # Check if the optimization triggered
    assert len(optimized_expr.args[1]) == 1

    k1, k2 = random.split(random.PRNGKey(42))
    arr_intp = {a: deffn(random.normal(k1, (2,))), b: deffn(random.normal(k2, (2, 4)))}

    with handler(BaselineFold()), handler(arr_intp):
        expected = evaluate(expr)
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

    with handler(DenseTensorFold()), handler(FoldLinearIndexer()):
        result = Sum(streams, i() / j())

    with handler(BaselineFold()):
        expected = Sum(streams, i() / j())
    assert allclose(result, expected)
