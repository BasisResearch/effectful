import effectful.handlers.jax.numpy as jnp
import jax
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop

from weighted.fold_lang_v1 import (
    ArgMaxAlg,
    FlipOptimizationFold,
    GradientOptimizationFold,
    MaxAlg,
    dense_fold_intp,
    fold,
    reals,
)


def test_flip_optimization_max():
    """Test the FlipOptimizationFold handler for converting Max to Min problems."""
    x = defop(jax.Array, name="x")

    # Test with MaxAlg
    with handler(dense_fold_intp), handler(FlipOptimizationFold()):
        # Simple maximization problem
        f_max = fold(MaxAlg, {x: jnp.arange(-10, 10)}, -(x() ** 2) + 3)
        assert f_max[()] == 3  # max of -x² is 0 at x=0


def test_flip_optimization_max_real():
    x = defop(jax.Array, name="x")

    # Test with MaxAlg
    with (
        handler(dense_fold_intp),
        handler(GradientOptimizationFold(learning_rate=0.1)),
        handler(FlipOptimizationFold()),
    ):
        # Simple maximization problem
        f_max = fold(MaxAlg, {x: reals()}, -(x() ** 2) + 3)
        assert f_max[()] == 3  # max of -x² is 0 at x=0


def test_flip_optimization_argmax():
    # Test with ArgMaxAlg
    x, y = defop(jax.Array, name="x"), defop(jax.Array, name="y")

    with handler(dense_fold_intp), handler(FlipOptimizationFold()):
        # Simple argmax problem
        f_argmax = fold(ArgMaxAlg, {x: jnp.arange(-10, 10)}, (-(x() ** 2), x()))
        assert f_argmax[0] == 0  # argmax value is 0
        assert f_argmax[1] == 0  # argmax of -x² is at x=0

        # More complex argmax
        f_argmax_complex = fold(
            ArgMaxAlg,
            {x: jnp.arange(-5, 6), y: jnp.arange(-5, 6)},
            (-((x() - 1) ** 2) - (y() + 2) ** 2, (x(), y())),
        )
        assert f_argmax_complex[0] == 0  # argmax value is 0
        assert f_argmax_complex[1][0] == 1  # x coordinate is 1
        assert f_argmax_complex[1][1] == -2  # y coordinate is -2
