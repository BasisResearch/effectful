import jax
import pytest

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax.monoid import Max
from effectful.handlers.weighted.jax import GradientOptimizationReduce
from effectful.handlers.weighted.jax import interpretation as jax_intp
from effectful.handlers.weighted.optimization import FlipOptimizationReduce
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop
from effectful.ops.weighted.jax import reals


def test_flip_optimization_max() -> None:
    """Test the FlipOptimizationReduce handler for converting Max to Min problems."""
    x = defop(jax.Array, name="x")

    # Test with MaxAlg
    with handler(jax_intp), handler(FlipOptimizationReduce()):
        # Simple maximization problem
        f_max = Max.reduce({x: jnp.arange(-10, 10)}, -(x() ** 2) + 3)
        assert f_max[()] == 3  # max of -x² is 0 at x=0


def test_flip_optimization_max_real() -> None:
    x = defop(jax.Array, name="x")

    # Test with MaxAlg
    with (
        handler(jax_intp),
        handler(GradientOptimizationReduce(learning_rate=0.1)),
        handler(FlipOptimizationReduce()),
    ):
        # Simple maximization problem
        f_max = Max.reduce({x: reals()}, -(x() ** 2) + 3)
        assert f_max[()] == 3  # max of -x² is 0 at x=0


@pytest.mark.skip(reason="argmin refactor")
def test_flip_optimization_argmax() -> None:
    # Test with ArgMaxAlg
    x, y = defop(jax.Array, name="x"), defop(jax.Array, name="y")

    with handler(jax_intp), handler(FlipOptimizationReduce()):
        # Simple argmax problem
        f_argmax = ArgMax.reduce({x: jnp.arange(-10, 10)}, (-(x() ** 2), x()))
        assert f_argmax[0] == 0  # argmax value is 0
        assert f_argmax[1] == 0  # argmax of -x² is at x=0

        # More complex argmax
        f_argmax_complex = ArgMax.reduce(
            {x: jnp.arange(-5, 6), y: jnp.arange(-5, 6)},
            (-((x() - 1) ** 2) - (y() + 2) ** 2, (x(), y())),
        )
        assert f_argmax_complex[0] == 0  # argmax value is 0
        assert f_argmax_complex[1][0] == 1  # x coordinate is 1
        assert f_argmax_complex[1][1] == -2  # y coordinate is -2
