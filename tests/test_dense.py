import torch
from effectful.handlers.torch import to_tensor
from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Operation, Term

from weighted.fold_lang_v1 import (
    ArgMinAlg,
    D,
    DenseTensorArgFold,
    DenseTensorFold,
    GradientOptimizationFold,
    LinAlg,
    MinAlg,
    fold,
    unfold,
)


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


def test_batched_matmul():
    # Define dimensions
    B, I, J, K = 2, 3, 4, 5

    # Create sample matrices
    A = torch.randn(B, I, J)
    B_mat = torch.randn(B, J, K)

    # Define index operations
    b, i, j, k = (
        defop(torch.Tensor, name="b"),
        defop(torch.Tensor, name="i"),
        defop(torch.Tensor, name="j"),
        defop(torch.Tensor, name="k"),
    )

    def run_fold():
        return fold(
            LinAlg,
            {b: torch.arange(B), i: torch.arange(I), j: torch.arange(J), k: torch.arange(K)},
            D(((b(), i(), k()), A[b(), i(), j()] * B_mat[b(), j(), k()])),
        )

    result = run_fold()
    result_tensor = sparse_to_tensor(result)

    with handler(DenseTensorFold()):
        vectorized_result_tensor = run_fold()

    # Compare with pytorch
    expected = torch.einsum("bij,bjk->bik", A, B_mat)
    assert torch.allclose(result_tensor, expected, atol=1e-4)
    assert torch.allclose(vectorized_result_tensor, expected)


def run_min_folds():
    x, y, z = defop(torch.Tensor, name="x"), defop(torch.Tensor, name="y"), defop(torch.Tensor, name="z")

    # Basic tests
    f1 = fold(MinAlg, {x: torch.arange(100)}, {(): -x()})
    assert f1[()] == -99

    f2 = fold(MinAlg, {x: torch.arange(-10, 10)}, {(): x() ** 2})
    assert f2[()] == 0

    f2 = fold(ArgMinAlg, {x: torch.arange(-10, 10)}, {(): (x() ** 2, x())})
    assert f2 == (torch.tensor(0), torch.tensor(0))

    f2 = fold(ArgMinAlg, {x: torch.arange(1, 5), y: torch.arange(4, 8)}, {(): (x() + y(), (x(), y()))})
    assert f2 == (torch.tensor(5), (torch.tensor(1), torch.tensor(4)))

    # Edge case: single element range
    f_single = fold(MinAlg, {x: torch.arange(1, 2)}, {(): x() ** 2})
    assert f_single[()] == 1

    # Edge case: tied minimum values
    f_tied = fold(ArgMinAlg, {x: torch.arange(-3, 4)}, {(): (abs(x()), x())})
    assert f_tied == (torch.tensor(0), torch.tensor(0))

    # Edge case: large numbers
    f_large = fold(MinAlg, {x: torch.arange(10**6, 10**6 + 10)}, {(): x() - 10**6})
    assert f_large[()] == 0

    # Edge case: custom function with multiple variables
    def custom_func(a, b, c=1):
        return a**2 + b**2 - c

    f_custom = fold(MinAlg, {x: torch.arange(-5, 6), y: torch.arange(-5, 6)}, {(): custom_func(x(), y(), 10)})
    assert f_custom[()] == -10  # Minimum is at x=0, y=0: 0²+0²-10 = -10

    # Edge case: complex expression with three variables
    f_complex = fold(
        ArgMinAlg,
        {x: torch.arange(-3, 4), y: torch.arange(-3, 4), z: torch.arange(-3, 4)},
        {(): ((x() - 1) ** 2 + (y() + 2) ** 2 + (z() - 3) ** 2, (x(), y(), z()))},
    )
    assert f_complex == (torch.tensor(0), (torch.tensor(1), torch.tensor(-2), torch.tensor(3)))


def assert_no_base_case(*args, **kwargs):
    assert False, "vectorized fold missed a case"


def test_minalg():
    run_min_folds()


def test_minalg_vectorized():
    with handler({fold: assert_no_base_case}), handler(coproduct(DenseTensorArgFold(), DenseTensorFold())):
        run_min_folds()
