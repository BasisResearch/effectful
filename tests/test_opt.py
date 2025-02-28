import effectful.handlers.numbers
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop

from weighted.fold_lang_v1 import GradientOptimizationFold, MinAlg, fold, reals, unfold


def test_opt():
    def loss(theta):
        return (theta() - 5.0) ** 2

    theta = defop(float, name="theta")

    with handler(GradientOptimizationFold(lr=0.1)):
        min_loss = fold(MinAlg, {theta: reals()}, {0: loss(theta)})

    # assert theta is close to 5.
    assert min_loss[(0,)] < 1e-3
