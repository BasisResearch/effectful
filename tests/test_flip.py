import torch
from effectful.handlers.torch import to_tensor
from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Operation, Term

from weighted.fold_lang_v1 import (
    ArgMaxAlg,
    ArgMinAlg,
    D,
    DenseTensorArgFold,
    DenseTensorFold,
    FlipOptimizationFold,
    GradientOptimizationFold,
    LinAlg,
    MaxAlg,
    MinAlg,
    ProductFold,
    dense_fold_intp,
    fold,
    reals,
    semi_ring_product,
    unfold,
)


def test_flip_optimization_max():
    """Test the FlipOptimizationFold handler for converting Max to Min problems."""
    x, y = defop(torch.Tensor, name="x"), defop(torch.Tensor, name="y")

    # Test with MaxAlg
    with handler(dense_fold_intp):
        # Simple maximization problem
        f_max = fold(MaxAlg, {x: torch.arange(-10, 10)}, -(x() ** 2) + 3)
        assert f_max[()] == 3  # max of -x² is 0 at x=0


def test_flip_optimization_max_real():
    x, y = defop(torch.Tensor, name="x"), defop(torch.Tensor, name="y")

    # Test with MaxAlg
    with handler(dense_fold_intp), handler(GradientOptimizationFold(lr=0.1)):
        # Simple maximization problem
        f_max = fold(MaxAlg, {x: reals()}, -(x() ** 2) + 3)
        assert f_max[()] == 3  # max of -x² is 0 at x=0


def test_flip_optimization_argmax():
    # Test with ArgMaxAlg
    x, y = defop(torch.Tensor, name="x"), defop(torch.Tensor, name="y")

    with handler(dense_fold_intp):
        # Simple argmax problem
        f_argmax = fold(ArgMaxAlg, {x: torch.arange(-10, 10)}, (-(x() ** 2), x()))
        assert f_argmax[0] == 0  # argmax value is 0
        assert f_argmax[1] == 0  # argmax of -x² is at x=0

        # More complex argmax
        f_argmax_complex = fold(
            ArgMaxAlg,
            {x: torch.arange(-5, 6), y: torch.arange(-5, 6)},
            (-((x() - 1) ** 2) - (y() + 2) ** 2, (x(), y())),
        )
        assert f_argmax_complex[0] == 0  # argmax value is 0
        assert f_argmax_complex[1][0] == 1  # x coordinate is 1
        assert f_argmax_complex[1][1] == -2  # y coordinate is -2
