import effectful.handlers.jax.numpy as jnp
import jax
from effectful.handlers.jax import jax_getitem, unbind_dims
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import deffn, defop
from jax import random
from jax.numpy import allclose
from pytest import mark, param

from tests.utils import FOLD_TRANSFORMS
from weighted.handlers.jax import DenseTensorFold
from weighted.handlers.optimization import FoldPropagateUnusedStreams
from weighted.ops.fold import BaselineFold, fold
from weighted.ops.semiring import mul
from weighted.ops.sugar import Max, Sum

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

    x = jax_getitem(x_op(), (x_ix(),))
    y = jax_getitem(y_op(), (y_ix(),))

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
