import effectful.handlers.numbers
import torch
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop

from weighted.fold_lang_v1 import ArgMinAlg, D, DenseTensorFold, GradientOptimizationFold, LinAlg, MinAlg, fold, reals


def test_opt():
    def loss(theta):
        return (theta() - 5.0) ** 2

    theta = defop(float, name="theta")

    with handler(GradientOptimizationFold(lr=0.1)):
        min_loss = fold(MinAlg, {theta: reals()}, {(): loss(theta)})

    # assert theta is close to 5.
    assert min_loss[()] < 1e-3


def test_batched_matmul():
    expected_w = 3.0
    expected_b = 5.0

    x = torch.arange(0, 10, 0.1)
    y = expected_w * x + expected_b + torch.randn(x.size())

    K = x.shape[0]

    # Define index operations
    b, w, k = defop(torch.Tensor, name="b"), defop(torch.Tensor, name="w"), defop(torch.Tensor, name="k")

    with handler(GradientOptimizationFold(lr=0.1, steps=100)), handler(DenseTensorFold()):
        loss = fold(LinAlg, {k: range(K)}, {(): (w() * x[k()] + b() - y[k()]) ** 2})
        result = fold(ArgMinAlg, {w: reals(), b: reals()}, {(): (loss, (w(), b()))})
        breakpoint()
