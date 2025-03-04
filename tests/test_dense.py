import torch
from effectful.handlers.torch import Indexable, to_tensor
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Operation, Term

from weighted.fold_lang_v1 import D, DenseTensorFold, GradientOptimizationFold, LinAlg, MinAlg, fold, unfold


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
    x = defop(int, name="x")

    with handler(DenseTensorFold()):
        f1 = fold(MinAlg, {x: range(100)}, D(((), -x())))
        assert f1.item() == -99

        f2 = fold(MinAlg, {x: range(-10, 10)}, D(((), x() ** 2)))
        assert f2.item() == 0
