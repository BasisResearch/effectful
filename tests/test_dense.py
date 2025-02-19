import torch
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

    result = fold(
        LinAlg,
        unfold(indices, A[b, i, j] * B_mat[b, j, k]),
        lambda x: {(b(), i(), k()): x},
    )

    # Compare with pytorch
    expected = torch.einsum("bij,bjk->bik", A, B_mat)
    assert torch.allclose(torch.tensor(result), expected)
