import effectful.handlers.jax.numpy as jnp
import jax
import pytest
from effectful.handlers.jax import unbind_dims
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import deffn, defop

from weighted.handlers.jax import (
    DenseTensorFold,
)
from weighted.handlers.optimization import FoldFactorization
from weighted.ops.fold import BaselineFold
from weighted.ops.sugar import Sum

baseline_intp = coproduct(BaselineFold(), FoldFactorization())

jax_intp = DenseTensorFold()

parameterize_base_intp = pytest.mark.parametrize(
    "base_intp",
    [
        pytest.param(jax_intp, id="jax"),
        pytest.param(baseline_intp, id="baseline"),
    ],
)

parameterize_transform_intp = pytest.mark.parametrize(
    "transform_intp",
    [
        pytest.param(FoldFactorization(), id="fold-factorization"),
        pytest.param({}, id="nop"),
    ],
)


@parameterize_base_intp
@parameterize_transform_intp
def test_factorize(base_intp, transform_intp):
    dim_size = 4
    key = jax.random.PRNGKey(42)

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

    keys = jax.random.split(key, len(arr_names))
    arrs = {
        k: jax.random.normal(key, (dim_size, dim_size))
        for k, key in zip(arr_names, keys, strict=False)
    }
    arr_intp = {ops[k]: deffn(v) for k, v in arrs.items()}
    with handler(base_intp), handler(arr_intp):
        result = evaluate(fold_expr)

    expected = jnp.einsum("ij,kl,lm->", arrs["x"], arrs["y"], arrs["z"])
    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, expected)
