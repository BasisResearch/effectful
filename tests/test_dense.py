import torch
from effectful.handlers.torch import Indexable, to_tensor
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Operation, Term

from weighted.fold_lang_v1 import ArgMinAlg, D, DenseTensorFold, GradientOptimizationFold, LinAlg, MinAlg, fold, unfold


def infer_shape(sparse):
    shape = [0] * len(next(iter(sparse.keys())))
    for index in sparse.keys():
        shape = [max(a, b + 1) for a, b in zip(shape, index)]
    return shape


def sparse_to_tensor(sparse):
    shape = infer_shape(sparse)
    tensor = torch.zeros(shape)
    for index, value in sparse.items():
        tensor[index] = value
    return tensor


def test_unfold_size():
    I, J = 2, 1

    i, j = defop(int, name="i"), defop(int, name="j")

    indices = {i: range(I), j: range(J)}
    unfold_body = ((i(), j()), (i(), j()))
    streams = list(unfold(indices, unfold_body))

    assert len(streams) == I * J


def test_batched_matmul():
    # Define dimensions
    B, I, J, K = 2, 3, 4, 5

    # Create sample matrices
    A = torch.randn(B, I, J)
    B_mat = torch.randn(B, J, K)

    # Define index operations
    b, i, j, k = (
        defop(int, name="b"),
        defop(int, name="i"),
        defop(int, name="j"),
        defop(int, name="k"),
    )

    def run_fold():
        return fold(
            LinAlg,
            {b: range(B), i: range(I), j: range(J), k: range(K)},
            D(((b(), i(), k()), Indexable(A)[b(), i(), j()] * Indexable(B_mat)[b(), j(), k()])),
        )

    result = run_fold()
    result_tensor = sparse_to_tensor(result)

    with handler(DenseTensorFold()):
        vectorized_result_tensor = run_fold()

    # Compare with pytorch
    expected = torch.einsum("bij,bjk->bik", A, B_mat)
    assert torch.allclose(result_tensor, expected, atol=1e-4)
    assert torch.allclose(vectorized_result_tensor, expected)


def test_enum_opt():
    x, y, z = defop(int, name="x"), defop(int, name="y"), defop(int, name="z")

    def run_folds():
        # Basic tests
        f1 = fold(MinAlg, {x: range(100)}, {(): -x()})
        assert f1[()] == -99

        f2 = fold(MinAlg, {x: range(-10, 10)}, {(): x() ** 2})
        assert f2[()] == 0

        f2 = fold(ArgMinAlg, {x: range(-10, 10)}, {(): (x() ** 2, x())})
        assert f2[()] == (0, 0)

        f2 = fold(ArgMinAlg, {x: range(1, 5), y: range(4, 8)}, {(): (x() + y(), (x(), y()))})
        assert f2[()] == (5, (1, 4))

        # Edge case: empty range (should use the semiring's one value)
        f_empty = fold(MinAlg, {x: range(0)}, {(): x() ** 2})
        assert f_empty[()] == float('inf')  # MinAlg's one value is infinity

        # Edge case: single element range
        f_single = fold(MinAlg, {x: range(1, 2)}, {(): x() ** 2})
        assert f_single[()] == 1

        # Edge case: tied minimum values
        f_tied = fold(ArgMinAlg, {x: range(-3, 4)}, {(): (abs(x()), x())})
        assert f_tied[()][0] == 0  # The minimum value is 0
        assert f_tied[()][1] == 0  # The argmin is 0

        # Edge case: large numbers
        f_large = fold(MinAlg, {x: range(10**6, 10**6 + 10)}, {(): x() - 10**6})
        assert f_large[()] == 0

        # Edge case: custom function with multiple variables
        def custom_func(a, b, c=1):
            return a**2 + b**2 - c

        f_custom = fold(
            MinAlg, 
            {x: range(-5, 6), y: range(-5, 6)}, 
            {(): custom_func(x(), y(), 10)}
        )
        assert f_custom[()] == -10  # Minimum is at x=0, y=0: 0²+0²-10 = -10

        # Edge case: complex expression with three variables
        f_complex = fold(
            ArgMinAlg,
            {x: range(-3, 4), y: range(-3, 4), z: range(-3, 4)},
            {(): ((x() - 1)**2 + (y() + 2)**2 + (z() - 3)**2, (x(), y(), z()))}
        )
        assert f_complex[()][0] == 0  # Minimum value is 0
        assert f_complex[()][1] == (1, -2, 3)  # Argmin is at (1, -2, 3)

    run_folds()
    with handler(DenseTensorFold()):
        run_folds()
