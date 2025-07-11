from functools import reduce
from graphlib import TopologicalSorter

import chex
import effectful.handlers.jax.numpy as jnp
import jax
import pytest
from effectful.handlers.jax import bind_dims, jax_getitem, sizesof, unbind_dims
from effectful.handlers.jax._handlers import is_eager_array
from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.syntax import defop
from effectful.ops.types import Term
from jax import random as random

from weighted.handlers.jax import (
    D,
    DenseTensorFold,
    GradientOptimizationFold,
    PytreeMapFold,
    ScanFold,
    reals,
)
from weighted.handlers.optimization import FoldEliminateDterm, FoldIndexDistributivity
from weighted.ops.fold import BaselineFold, fold
from weighted.ops.sugar import ArgMin, LogSum, Max, Min, Sum

baseline_intp = reduce(
    coproduct,  # type: ignore
    [BaselineFold(), FoldEliminateDterm(), FoldIndexDistributivity()],
)

jax_intp = reduce(
    coproduct,  # type: ignore
    [DenseTensorFold(), FoldEliminateDterm(), FoldIndexDistributivity()],
)

parameterize_intp = pytest.mark.parametrize(
    "intp",
    [
        pytest.param(jax_intp, id="jax_no_d_term"),
        pytest.param(DenseTensorFold(), id="jax_vanilla"),
        pytest.param(baseline_intp, id="baseline_no_d_term"),
    ],
)
parameterize_ops = pytest.mark.parametrize(
    "weighted_op,python_op",
    [
        pytest.param(Sum, sum, id="sum"),
        pytest.param(Min, min, id="min"),
        pytest.param(Max, max, id="max"),
    ],
)


def infer_shape(sparse):
    shape = [0] * len(next(iter(sparse.keys())))
    for index in sparse:
        shape = [max(a, b + 1) for a, b in zip(shape, index, strict=True)]
    return shape


def sparse_to_tensor(sparse):
    shape = infer_shape(sparse)
    tensor = jnp.zeros(shape)
    for index, value in sparse.items():
        tensor[index] = value
    return tensor


@parameterize_intp
def test_batched_matmul(intp):
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

    with handler(intp):
        actual = Sum(
            {b: jnp.arange(B), i: jnp.arange(I), j: jnp.arange(J), k: jnp.arange(K)},
            D(((b(), i(), k()), unbind_dims(X, b, i, j) * unbind_dims(Y, b, j, k))),
        )

    expected = jnp.einsum("bij,bjk->bik", X, Y)
    assert isinstance(actual, jax.Array)
    assert jnp.allclose(actual, expected)


@parameterize_intp
def test_log_matmul(intp):
    key = jax.random.PRNGKey(0)
    I, J, K = 2, 3, 4

    X = random.uniform(key, (I, J))
    Y = random.uniform(key, (J, K))

    i, j, k = (
        defop(jax.Array, name="i"),
        defop(jax.Array, name="j"),
        defop(jax.Array, name="k"),
    )

    x = unbind_dims(jnp.log(X), i, j)
    y = unbind_dims(jnp.log(Y), j, k)

    with handler(intp):
        actual = LogSum(
            {i: jnp.arange(I), j: jnp.arange(J), k: jnp.arange(K)},
            D(((i(), k()), x + y)),
        )
    actual = jnp.exp(actual)

    expected = jnp.einsum("ij,jk->ik", X, Y)
    assert isinstance(actual, jax.Array)
    assert jnp.allclose(actual, expected)


@parameterize_intp
def test_linalg_folds(intp):
    x, y, z = (
        defop(jax.Array, name="x"),
        defop(jax.Array, name="y"),
        defop(jax.Array, name="z"),
    )

    with handler(intp):
        # f1 = Sum({x: jnp.arange(3)}, x())
        # assert isinstance(f1, jax.Array)
        # assert f1[()] == 3

        # f2 = Sum({x: jnp.arange(3), y: jnp.arange(3)}, x() + y())
        # assert isinstance(f2, jax.Array)
        # assert f2[()] == 18

        f3 = Sum({x: jnp.arange(3), y: jnp.arange(3)}, x())
        assert isinstance(f3, jax.Array)
        assert f3[()] == 9

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


@parameterize_intp
def test_linalg_folds_named(intp):
    key = jax.random.PRNGKey(0)
    (x, y) = (defop(jax.Array, name="x"), defop(jax.Array, name="y"))
    A = random.normal(key, (3, 3))
    B = random.normal(key, (3,))

    with handler(intp):
        f4 = Sum(
            {x: jnp.arange(3), y: jnp.arange(3)},
            D(((x(),), unbind_dims(A, x, y) * unbind_dims(B, y))),
        )
        assert isinstance(f4, jax.Array)
        assert jnp.allclose(f4, jnp.einsum("ij,j->i", A, B))


@parameterize_intp
def test_basic_min_folds(intp):
    """Test basic Min fold operations."""
    x = defop(jax.Array, name="x")

    with handler(intp):
        # Basic test with negative index
        f1 = Min({x: jnp.arange(100)}, -x())
        assert isinstance(f1, jax.Array)
        assert f1[()] == -99

        # Basic test with squared function (minimum at x=0)
        f2 = Min({x: jnp.arange(-10, 10)}, x() ** 2)
        assert isinstance(f2, jax.Array)
        assert f2[()] == 0

        # Edge case: single element range
        f_single = Min({x: jnp.arange(1, 2)}, x() ** 2)
        assert isinstance(f_single, jax.Array)
        assert f_single[()] == 1

        # Edge case: large numbers
        f_large = Min({x: jnp.arange(10**6, 10**6 + 10)}, x() - 10**6)
        assert isinstance(f_large, jax.Array)
        assert f_large[()] == 0


@parameterize_intp
def test_arg_min_folds(intp):
    """Test ArgMin fold operations."""
    x = defop(jax.Array, name="x")

    with handler(intp):
        # Basic ArgMin test
        f2_arg = ArgMin({x: jnp.arange(-10, 10)}, (x() ** 2, x()))
        assert isinstance(f2_arg[0], jax.Array)
        assert isinstance(f2_arg[1], jax.Array)
        assert f2_arg == (jnp.array(0), jnp.array(0))

        # Edge case: tied minimum values
        f_tied = ArgMin({x: jnp.arange(-3, 4)}, (abs(x()), x()))
        assert isinstance(f_tied[0], jax.Array)
        assert isinstance(f_tied[1], jax.Array)
        assert f_tied == (jnp.array(0), jnp.array(0))


@parameterize_intp
def test_multi_variable_min_folds(intp):
    """Test Min fold operations with multiple variables."""
    x, y = defop(jax.Array, name="x"), defop(jax.Array, name="y")

    with handler(intp):
        # Test with multiple variables
        def custom_func(a, b, c=1):
            return a**2 + b**2 - c

        f_custom = Min(
            {x: jnp.arange(-5, 6), y: jnp.arange(-5, 6)}, custom_func(x(), y(), 10)
        )
        assert isinstance(f_custom, jax.Array)
        assert f_custom[()] == -10  # Minimum is at x=0, y=0: 0²+0²-10 = -10


@parameterize_intp
def test_multi_variable_arg_min_folds(intp):
    """Test ArgMin fold operations with multiple variables."""
    x, y = defop(jax.Array, name="x"), defop(jax.Array, name="y")

    with handler(intp):
        # Test ArgMin with multiple variables
        f2_arg_pair = ArgMin(
            {x: jnp.arange(1, 5), y: jnp.arange(4, 8)}, (x() + y(), (x(), y()))
        )
        assert isinstance(f2_arg_pair[0], jax.Array)
        assert isinstance(f2_arg_pair[1][0], jax.Array)
        assert isinstance(f2_arg_pair[1][1], jax.Array)
        assert f2_arg_pair == (jnp.array(5), (jnp.array(1), jnp.array(4)))


@parameterize_intp
def test_complex_arg_min_folds(intp):
    """Test complex ArgMin fold operations with three variables."""
    x, y, z = (
        defop(jax.Array, name="x"),
        defop(jax.Array, name="y"),
        defop(jax.Array, name="z"),
    )

    with handler(intp):
        # Complex expression with three variables
        f_complex = ArgMin(
            {x: jnp.arange(-3, 4), y: jnp.arange(-3, 4), z: jnp.arange(-3, 4)},
            ((x() - 1) ** 2 + (y() + 2) ** 2 + (z() - 3) ** 2, (x(), y(), z())),
        )
        assert isinstance(f_complex[0], jax.Array)
        assert isinstance(f_complex[1][0], jax.Array)
        assert isinstance(f_complex[1][1], jax.Array)
        assert isinstance(f_complex[1][2], jax.Array)
        assert f_complex == (jnp.array(0), (jnp.array(1), jnp.array(-2), jnp.array(3)))


@parameterize_intp
def test_nested_folds(intp):
    """Test nested folds over jnp.arange to verify they work correctly."""
    a, b, c = (
        defop(jax.Array, name="a"),
        defop(jax.Array, name="b"),
        defop(jax.Array, name="c"),
    )

    # Define ranges for the nested folds
    a_range = jnp.arange(3)  # 0, 1, 2
    b_range = jnp.arange(2)  # 0, 1
    c_range = jnp.arange(4)  # 0, 1, 2, 3

    with handler(intp):
        # Test a simple nested fold that computes the sum of all combinations
        # This is equivalent to: sum(a + b + c for a in range(3) for b in range(2) for c in range(4))
        nested_result = Sum(
            {a: a_range}, Sum({b: b_range}, Sum({c: c_range}, a() + b() + c()))
        )

        # Calculate the expected result manually
        expected = sum(
            a_val + b_val + c_val
            for a_val in range(3)
            for b_val in range(2)
            for c_val in range(4)
        )

        assert isinstance(nested_result, jax.Array)
        assert nested_result[()] == expected

        # Test a more complex nested fold with a different operation at each level
        # Outer sum, middle product, inner min
        complex_nested = Sum(
            {a: a_range}, Min({b: b_range}, Sum({c: c_range}, a() * b() + c()))
        )

        # Calculate expected result manually
        expected_complex = sum(
            min(sum(a_val * b_val + c_val for c_val in range(4)) for b_val in range(2))
            for a_val in range(3)
        )

        assert isinstance(complex_nested, jax.Array)
        assert jnp.isclose(complex_nested[()], expected_complex)

        # Test a nested fold with the same operation (Sum) at each level
        # This should be optimizable by FoldFusion
        fusion_candidate = Sum(
            {a: a_range}, Sum({b: b_range}, Sum({c: c_range}, a() * b() * c()))
        )

        # Direct computation using a single fold (what FoldFusion would produce)
        direct_computation = Sum({a: a_range, b: b_range, c: c_range}, a() * b() * c())

        # Both should give the same result
        assert jnp.isclose(fusion_candidate[()], direct_computation[()])


@parameterize_intp
def test_partial_eval(intp):
    """Fold over arrays with named dimensions."""
    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")

    indexed_array = unbind_dims(jnp.ones((5, 4)), i)

    with handler(intp):
        r1 = Sum({j: indexed_array}, j())

    assert isinstance(r1, Term)
    assert i in sizesof(r1) and j not in sizesof(r1)
    assert is_eager_array(r1)

    with handler(intp):
        r2 = Sum({j: indexed_array}, j() + 1)

    assert isinstance(r2, Term)
    assert i in sizesof(r2) and j not in sizesof(r2)
    assert is_eager_array(r2)


@parameterize_intp
@parameterize_ops
def test_dependent_folds(intp, weighted_op, python_op):
    """Test folds where indices depend on other indices within the same fold."""
    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")

    with handler(intp):
        # Test fold where j depends on i
        # This is equivalent to: sum(i + j for i in range(5) for j in range(i+1))
        inner_1 = weighted_op({j: jax_getitem(jnp.ones((5, 4)), [i()])}, j())
        dependent_result_1 = weighted_op({i: jnp.arange(5)}, inner_1)

        dependent_result_2 = weighted_op(
            {i: jnp.arange(5), j: jax_getitem(jnp.ones((5, 4)), [i()])}, j()
        )

        expected = python_op(j for i in jnp.arange(5) for j in jnp.ones((5, 4))[i])

        for d in [dependent_result_1, dependent_result_2]:
            assert isinstance(d, jax.Array)
            assert d[()] == expected


def test_dependent_folds_unused():
    """Test a dependent fold with an unused stream whose length depends on another stream."""
    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")

    with handler(jax_intp):
        actual = Sum({i: jnp.arange(5), j: jnp.repeat(i(), 4)}, i())

        expected = sum(range(5)) * 4

        assert isinstance(actual, jax.Array)
        assert actual[()] == expected


@parameterize_intp
def test_dependent_partial_folds(intp):
    """Test folds of parts of the indices that depend on other indices within the same fold."""
    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")
    I, J = 3, 4

    with handler(intp):
        j_array = jnp.arange(I * J).reshape((I, J))
        j_dependent = jax_getitem(j_array, (i(),))
        i_array = jnp.flip(jnp.arange(I))

        # only reduce the j dimension, returning an array of shape (I,)
        expected = jnp.array([38, 22, 6])
        result = Sum({i: i_array, j: j_dependent}, D(((i(),), j())))
        assert jnp.allclose(result, expected)

        # only reduce the i dimension, returning an array of shape (J,)
        expected = jnp.array([12, 15, 18, 21])
        result = Sum({i: i_array, j: j_dependent}, D(((j(),), j())))
        assert jnp.allclose(result, expected)


@parameterize_intp
def test_doubly_dependent_partial_folds(intp):
    i, j, k = (
        defop(jax.Array, name="i"),
        defop(jax.Array, name="j"),
        defop(jax.Array, name="k"),
    )
    I, J, K = 2, 3, 4

    with handler(intp):
        k_array = jnp.flip(jnp.arange(K * J)).reshape((K, J))
        k_dependent = jax_getitem(k_array, (j(),))
        j_array = jnp.arange(I * J).reshape((I, J))
        j_dependent = jax_getitem(j_array, (i(),))
        i_array = jnp.flip(jnp.arange(I))

        result = Sum({i: i_array, j: j_dependent, k: k_dependent}, D(((i(),), k())))
        assert isinstance(result, jax.Array)
        assert jnp.allclose(result, jnp.array([9, 63]))

        result = Sum({i: i_array, j: j_dependent, k: k_dependent}, D(((j(),), k())))
        assert isinstance(result, jax.Array)
        assert jnp.allclose(result, jnp.array([33, 24, 15]))


def longest_dependency_chain(adj):
    """Returns the longest dependency chain in a directed graph represented as
    an adjacency dictionary.

    """
    ts = TopologicalSorter(adj)
    ts.prepare()
    i = 0
    while ts.is_active():
        for n in ts.get_ready():
            ts.done(n)
        i += 1
    return i


def dependency_graph_of_streams(streams):
    """Returns the data dependency graph of a streams dictionary."""
    stream_vars = set(streams.keys())
    return {var: fvsof(val) & stream_vars for (var, val) in streams.items()}


def test_fold_chain():
    def f(x):
        return jnp.expand_dims(x + 1, 0)

    n_iters = 5
    vs = [defop(jax.Array, name=f"x{i}") for i in range(n_iters)]
    streams = {vs[0]: jnp.array([[1, 2, 3]])} | {
        vs[i]: f(vs[i - 1]()) for i in range(1, n_iters)
    }

    deps = dependency_graph_of_streams(streams)
    assert longest_dependency_chain(deps) == n_iters

    with handler(ScanFold()):
        new_fold = Sum(streams, vs[-1]())

    assert isinstance(new_fold, Term) and new_fold.op is fold
    new_streams = new_fold.args[1]

    # ScanFold should break the dependency graph in the streams
    deps = dependency_graph_of_streams(new_streams)
    assert longest_dependency_chain(deps) < n_iters

    with handler(jax_intp):
        result = evaluate(new_fold)

    assert jnp.allclose(result, jnp.array([5, 6, 7]))


def test_fold_chain_named():
    def f(x):
        return jnp.expand_dims(x + 1, 0)

    n_iters = 5
    i = defop(jax.Array, name="i")
    init = jnp.expand_dims(jax_getitem(jnp.array([1, 2, 3]), [i()]), 0)

    vs = [defop(jax.Array, name=f"x{i}") for i in range(n_iters)]
    streams = {vs[0]: init} | {vs[i]: f(vs[i - 1]()) for i in range(1, n_iters)}

    deps = dependency_graph_of_streams(streams)
    assert longest_dependency_chain(deps) == n_iters

    with handler(ScanFold()):
        new_fold = Sum(streams, vs[-1]())

    assert isinstance(new_fold, Term) and new_fold.op is fold
    new_streams = new_fold.args[1]

    # ScanFold should break the dependency graph in the streams
    deps = dependency_graph_of_streams(new_streams)
    assert longest_dependency_chain(deps) < n_iters

    with handler(jax_intp):
        result = evaluate(new_fold)

    expected = jax_getitem(jnp.array([5, 6, 7]), [i()])
    assert jnp.all(bind_dims(jnp.allclose(result, expected), i))


@chex.dataclass
class Point:
    x: jax.Array
    y: jax.Array


@parameterize_ops
@pytest.mark.parametrize(
    "pytree_constr",
    [lambda x, y: (x, y), lambda x, y: Point(x=x, y=y), lambda x, y: [x, y]],
)
def test_pytree_fold(weighted_op, python_op, pytree_constr):
    """A fold where the body is a pytree and the fold indices are arrays should
    produce a pytree of arrays.

    """
    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")

    with handler(jax_intp), handler(PytreeMapFold()):
        actual = weighted_op(
            {i: jnp.arange(5), j: jnp.arange(7)}, pytree_constr(i() + j(), i() * j())
        )
        expected = pytree_constr(
            python_op(i + j for i in jnp.arange(5) for j in jnp.arange(7)),
            python_op(i * j for i in jnp.arange(5) for j in jnp.arange(7)),
        )
        assert expected == actual

        actual = weighted_op(
            {i: jnp.arange(5), j: jnp.arange(7)},
            D((i(), pytree_constr(i() + j(), i() * j()))),
        )
        expected = pytree_constr(
            jnp.array([python_op(i + j for j in jnp.arange(7)) for i in jnp.arange(5)]),
            jnp.array([python_op(i * j for j in jnp.arange(7)) for i in jnp.arange(5)]),
        )
        assert jax.tree.all(jax.tree.map(lambda a, e: (a == e).all(), actual, expected))


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
