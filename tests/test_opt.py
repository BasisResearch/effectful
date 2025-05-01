import effectful.handlers.jax.numpy as jnp
import jax
from effectful.handlers.jax import unbind_dims
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop

from weighted.handlers.jax import GradientOptimizationFold, reals
from weighted.handlers.jax import interpretation as jax_intp
from weighted.ops.sugar import ArgMin, Min, Sum


def assert_no_base_case(*args, **kwargs):
    assert False, "vectorized fold missed a case"


def test_opt():
    def loss(theta):
        return (theta() - 5.0) ** 2

    theta = defop(jax.Array, name="theta")

    with handler(GradientOptimizationFold(learning_rate=0.1)), handler(jax_intp):
        min_loss = Min({theta: reals()}, loss(theta))

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
    b, w, k = (
        defop(jax.Array, name="b"),
        defop(jax.Array, name="w"),
        defop(jax.Array, name="k"),
    )

    with (
        handler(GradientOptimizationFold(learning_rate=0.1, steps=100)),
        handler(jax_intp),
    ):
        loss = Sum(
            {k: jnp.arange(K)}, (w() * unbind_dims(x, k) + b() - unbind_dims(y, k)) ** 2
        )
        (_, (predicted_w, predicted_b)) = ArgMin(
            {w: reals(), b: reals()}, (loss, (w(), b()))
        )
        assert 2 < predicted_w < 4 and 3 < predicted_b < 6
