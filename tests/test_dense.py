import effectful.handlers.jax.numpy as jnp
import jax
from effectful.handlers.jax import unbind_dims
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop
from jax import random as random

from weighted.handlers.jax import GradientOptimizationFold, reals
from weighted.handlers.jax import interpretation as jax_intp
from weighted.ops.fold import D, fold
from weighted.ops.sugar import ArgMin, Min, Sum


def infer_shape(sparse):
    shape = [0] * len(next(iter(sparse.keys())))
    for index in sparse.keys():
        shape = [max(a, b + 1) for a, b in zip(shape, index)]
    return shape


def sparse_to_tensor(sparse):
    shape = infer_shape(sparse)
    tensor = jnp.zeros(shape)
    for index, value in sparse.items():
        tensor[index] = value
    return tensor


def test_batched_matmul():
    key = jax.random.PRNGKey(0)
    # Define dimensions
    B, I, J, K = 2, 3, 4, 5

    # Create sample matrices
    X = random.normal(key, (B, I, J))
    Y = random.normal(key, (B, J, K))

    # Define index operations
    b, i, j, k = (
        defop(jax.Array, name="b"),
        defop(jax.Array, name="i"),
        defop(jax.Array, name="j"),
        defop(jax.Array, name="k"),
    )

    with handler(jax_intp):
        actual = Sum(
            {b: jnp.arange(B), i: jnp.arange(I), j: jnp.arange(J), k: jnp.arange(K)},
            D(((b(), i(), k()), unbind_dims(X, b, i, j) * unbind_dims(Y, b, j, k))),
        )

    expected = jnp.einsum("bij,bjk->bik", X, Y)
    assert jnp.allclose(actual, expected)


def test_linalg_folds():
    key = jax.random.PRNGKey(0)
    x, y, z = (
        defop(jax.Array, name="x"),
        defop(jax.Array, name="y"),
        defop(jax.Array, name="z"),
    )
    A = random.normal(key, (3, 3))
    B = random.normal(key, (3,))

    with handler(jax_intp):
        f1 = Sum({x: jnp.arange(3)}, x())
        assert isinstance(f1, jax.Array)
        assert f1[()] == 3

        f2 = Sum({x: jnp.arange(3), y: jnp.arange(3)}, x() + y())
        assert isinstance(f2, jax.Array)
        assert f2[()] == 18

        f3 = Sum({x: jnp.arange(3), y: jnp.arange(3)}, x())
        assert isinstance(f3, jax.Array)
        assert f3[()] == 9

        f4 = Sum(
            {x: jnp.arange(3), y: jnp.arange(3)},
            D(((x(),), unbind_dims(A, x, y) * unbind_dims(B, y))),
        )
        assert isinstance(f4, jax.Array)
        assert jnp.allclose(f4, jnp.einsum("ij,j->i", A, B))

        with handler({z: lambda: jnp.array(2)}):
            f5 = Sum({x: jnp.arange(3), y: jnp.arange(3)}, z() + x())
        assert isinstance(f5, jax.Array)
        assert f5[()] == 3 * (2 + 0 + 2 + 1 + 2 + 2)

        f6 = Sum({x: jnp.arange(3), y: jnp.arange(3)}, jnp.array(2))
        assert isinstance(f6, jax.Array)
        assert f6[()] == 18

        f7 = Sum({x: jnp.arange(3), y: jnp.arange(3)}, 2 * x())
        assert isinstance(f7, jax.Array)
        assert f7[()] == 2 * 9


def run_min_folds():
    x, y, z = (
        defop(jax.Array, name="x"),
        defop(jax.Array, name="y"),
        defop(jax.Array, name="z"),
    )

    # Basic tests
    f1 = Min({x: jnp.arange(100)}, -x())
    assert isinstance(f1, jax.Array)
    assert f1[()] == -99

    f2 = Min({x: jnp.arange(-10, 10)}, x() ** 2)
    assert isinstance(f2, jax.Array)
    assert f2[()] == 0

    f2_arg = ArgMin({x: jnp.arange(-10, 10)}, (x() ** 2, x()))
    assert isinstance(f2_arg[0], jax.Array)
    assert isinstance(f2_arg[1], jax.Array)
    assert f2_arg == (jnp.array(0), jnp.array(0))

    f2_arg_pair = ArgMin(
        {x: jnp.arange(1, 5), y: jnp.arange(4, 8)}, (x() + y(), (x(), y()))
    )
    assert isinstance(f2_arg_pair[0], jax.Array)
    assert isinstance(f2_arg_pair[1][0], jax.Array)
    assert isinstance(f2_arg_pair[1][1], jax.Array)
    assert f2_arg_pair == (jnp.array(5), (jnp.array(1), jnp.array(4)))

    # Edge case: single element range
    f_single = Min({x: jnp.arange(1, 2)}, x() ** 2)
    assert isinstance(f_single, jax.Array)
    assert f_single[()] == 1

    # Edge case: tied minimum values
    f_tied = ArgMin({x: jnp.arange(-3, 4)}, (abs(x()), x()))
    assert isinstance(f_tied[0], jax.Array)
    assert isinstance(f_tied[1], jax.Array)
    assert f_tied == (jnp.array(0), jnp.array(0))

    # Edge case: large numbers
    f_large = Min({x: jnp.arange(10**6, 10**6 + 10)}, x() - 10**6)
    assert isinstance(f_large, jax.Array)
    assert f_large[()] == 0

    # Edge case: custom function with multiple variables
    def custom_func(a, b, c=1):
        return a**2 + b**2 - c

    f_custom = Min(
        {x: jnp.arange(-5, 6), y: jnp.arange(-5, 6)}, custom_func(x(), y(), 10)
    )
    assert isinstance(f_custom, jax.Array)
    assert f_custom[()] == -10  # Minimum is at x=0, y=0: 0²+0²-10 = -10

    # Edge case: complex expression with three variables
    f_complex = ArgMin(
        {x: jnp.arange(-3, 4), y: jnp.arange(-3, 4), z: jnp.arange(-3, 4)},
        ((x() - 1) ** 2 + (y() + 2) ** 2 + (z() - 3) ** 2, (x(), y(), z())),
    )
    assert isinstance(f_complex[0], jax.Array)
    assert isinstance(f_complex[1][0], jax.Array)
    assert isinstance(f_complex[1][1], jax.Array)
    assert isinstance(f_complex[1][2], jax.Array)
    assert f_complex == (jnp.array(0), (jnp.array(1), jnp.array(-2), jnp.array(3)))


def assert_no_base_case(*args, **kwargs):
    assert False, f"vectorized fold missed a case: {args}, {kwargs}"


def test_minalg_vectorized():
    with handler({fold: assert_no_base_case}), handler(jax_intp):
        run_min_folds()


def test_gradient_optimization_init():
    """Test that GradientOptimizationFold uses initialization values correctly."""
    x, y = defop(jax.Array, name="x"), defop(jax.Array, name="y")

    # Define a simple quadratic function with minimum at (2, -3)
    def quadratic(x_val, y_val):
        return (x_val - 2) ** 2 + (y_val + 3) ** 2

    # Test with default initialization (zeros)
    with (
        handler(GradientOptimizationFold(steps=100, learning_rate=0.1)),
        handler(jax_intp),
    ):
        result = Min({x: reals(), y: reals()}, quadratic(x(), y()))
        # Should be close to the minimum value (0)
        assert isinstance(result, jax.Array)
        assert result < 0.1

        # Test with ArgMinAlg to get both value and argmin
        result_arg = ArgMin({x: reals(), y: reals()}, (quadratic(x(), y()), (x(), y())))
        # Value should be close to minimum
        assert all(isinstance(result, jax.Array) for a in result_arg)
        assert result_arg[0] < 0.1
        # Arguments should be close to (2, -3)
        assert jnp.isclose(result_arg[1][0], 2, atol=0.1)
        assert jnp.isclose(result_arg[1][1], -3, atol=0.1)

    # Test with custom initialization
    # Starting closer to the minimum should converge faster
    with (
        handler(
            GradientOptimizationFold(steps=20, learning_rate=0.1, init={x: 1.5, y: -2.5})
        ),
        handler(jax_intp),
    ):
        result = Min({x: reals(), y: reals()}, quadratic(x(), y()))
        # Should be very close to the minimum with fewer steps
        assert result < 0.01
