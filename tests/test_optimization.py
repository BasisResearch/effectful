import effectful.handlers.jax.numpy as jnp
import jax
import pytest
from effectful.handlers.jax import jax_getitem, unbind_dims
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import deffn, defop
from jax import random
from jax.numpy import allclose

from tests.utils import get_fold_params
from weighted.ops.sugar import Sum

parameterize_base_intp = pytest.mark.parametrize(
    "base_intp", get_fold_params("jax_intp", "baseline_intp")
)

parameterize_transform_intp = pytest.mark.parametrize(
    "transform_intp", get_fold_params("factorize_fold", "split_fold", "fuse_fold")
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
        result = Sum(x_stream, Sum(y_stream, x + y))
    assert allclose(result, expected)

    with handler(fold_intp), handler(arr_intp):
        result = Sum(x_stream | y_stream, x + y)
    assert allclose(result, expected)
