import effectful.handlers.jax.numpy as jnp
import jax
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop

from weighted.fold_lang_v1 import (
    ArgMinAlg,
    GradientOptimizationFold,
    LinAlg,
    MinAlg,
    dense_fold_intp,
    fold,
    reals,
)


def assert_no_base_case(*args, **kwargs):
    assert False, "vectorized fold missed a case"


def test_opt():
    def loss(theta):
        return (theta() - 5.0) ** 2

    theta = defop(jax.Array, name="theta")

    with handler(GradientOptimizationFold(learning_rate=0.1)), handler(dense_fold_intp):
        min_loss = fold(MinAlg, {theta: reals()}, loss(theta))

    # assert theta is close to 5.
    assert min_loss[()] < 1e-3


def test_batched_matmul():
    """Fit a line to noisy data using gradient descent."""
    key = jax.random.PRNGKey(0)
    expected_w = 3.0
    expected_b = 5.0

    x = jnp.arange(0, 10, 0.1)
    y = expected_w * x + expected_b + jax.random.normal(key, shape=x.shape)

    K = x.shape[0]

    # Define index operations
    b, w, k = defop(jax.Array, name="b"), defop(jax.Array, name="w"), defop(jax.Array, name="k")

    with handler(GradientOptimizationFold(learning_rate=0.1, steps=100)), handler(dense_fold_intp):
        loss = fold(LinAlg, {k: jnp.arange(K)}, (w() * x[k()] + b() - y[k()]) ** 2)
        (_, (predicted_w, predicted_b)) = fold(ArgMinAlg, {w: reals(), b: reals()}, (loss, (w(), b())))
        assert 2 < predicted_w < 4 and 3 < predicted_b < 6
