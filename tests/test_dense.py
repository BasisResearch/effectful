import torch
from effectful.handlers.torch import Indexable
from effectful.ops.syntax import defop
from weighted.fold_lang_v1 import LinAlg, fold, unfold


def test_matmul():
    # Define dimensions
    B, I, J, K = 2, 3, 4, 5

    # Create sample matrices
    A = torch.randn(B, I, J)
    B_mat = torch.randn(B, J, K)

    # Define index operations
    b, i, j, k = defop(int), defop(int), defop(int), defop(int)

    # Define the computation using fold and unfold
    indices = {
        b: lambda: range(B),
        i: lambda: range(I),
        j: lambda: range(J),
        k: lambda: range(K),
    }

    def body(x):
        return {(b, i, k): x}

    unfold_body = Indexable(A)[b(), i(), j()] * Indexable(B_mat)[b(), j(), k()]
    result = fold(LinAlg, unfold(indices, unfold_body), body)

    # Compare with pytorch
    expected = torch.einsum("bij,bjk->bik", A, B_mat)
    assert torch.allclose(torch.tensor(result), expected)
