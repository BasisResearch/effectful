import torch
from effectful.handlers.torch import Indexable
from effectful.ops.semantics import fvsof
from effectful.ops.syntax import defop
from effectful.ops.types import Term
from weighted.fold_lang_v1 import LinAlg, fold, unfold


def test_fold_simple():
    # Test summing squares with a guard
    numbers = [1, 2, 3, 4, 5]
    result = fold(
        LinAlg,
        numbers,
        body=lambda x: x**2,  # square each number
        guard=lambda x: x % 2 == 1,  # only odd numbers
    )
    # Should sum squares of [1, 3, 5]
    assert result == 35  # 1² + 3² + 5² = 1 + 9 + 25 = 35

    # Test dictionary accumulation
    result = fold(LinAlg, range(3), body=lambda x: {x: x * 10})
    assert result == {0: 0, 1: 10, 2: 20}


def test_unfold_simple():
    # Define simple index operations
    x, y = defop(int), defop(int)

    # Define ranges for indices
    indices = {x: lambda: range(2), y: lambda: range(3)}

    # Test unfolding x + y
    result = list(unfold(indices, x() + y()))
    expected = [0, 1, 2, 1, 2, 3]  # All possible x + y combinations
    assert result == expected

    # Test unfolding x * y
    result = list(unfold(indices, x() * y()))
    expected = [0, 0, 0, 0, 1, 2]  # All possible x * y combinations
    assert result == expected


@defop
def D(*args):
    if any(len(fvsof(k)) > 0 for (k, _) in args):
        raise NotImplementedError
    return dict(args)


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

    def fold_body(x):
        return x

    unfold_body = D(
        (
            (b(), i(), k()),
            Indexable(A)[b(), i(), j()] * Indexable(B_mat)[b(), j(), k()],
        ),
    )
    streams = list(unfold(indices, unfold_body))
    result = fold(LinAlg, streams, fold_body)

    # Compare with pytorch
    expected = torch.einsum("bij,bjk->bik", A, B_mat)
    assert torch.allclose(torch.tensor(result), expected)


def test_fold_dicts():
    # Test folding a sequence of dictionaries with tuple keys
    dicts = [
        {(1, 0): 1, (2, 0): 2},
        {(2, 0): 3, (3, 0): 4},
        {(1, 0): 5, (3, 0): 6}
    ]

    # Simple fold that accumulates values
    result = fold(LinAlg, dicts, lambda x: x)
    assert result == {(1, 0): 6, (2, 0): 5, (3, 0): 10}

    # Fold with transformation
    result = fold(
        LinAlg,
        dicts,
        body=lambda d: {k: v * 2 for k, v in d.items()},  # double each value
        guard=lambda d: (2, 0) in d,  # only include dicts with (2,0) key
    )
    assert result == {(1, 0): 2, (2, 0): 10, (3, 0): 8}
