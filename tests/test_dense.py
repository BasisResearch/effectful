import torch
from effectful.handlers.torch import Indexable, to_tensor
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Operation, Term

from weighted.fold_lang_v1 import D, LinAlg, fold, unfold


def test_fold_simple():
    x = defop(int, name="x")
    # Test summing squares with a guard
    numbers = {x: [1, 2, 3, 4, 5]}
    result = fold(
        LinAlg,
        numbers,
        body=x() ** 2,  # square each number
        guard=x() % 2 == 1,  # only odd numbers
    )
    # Should sum squares of [1, 3, 5]
    assert result == 35  # 1² + 3² + 5² = 1 + 9 + 25 = 35

    # # Test dictionary accumulation
    # result = fold(LinAlg, range(3), body=lambda x: {x: x * 10})
    # assert result == {0: 0, 1: 10, 2: 20}


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


def test_fold_dicts():
    # Test folding a sequence of dictionaries with tuple keys
    dicts = [{(1, 0): 1, (2, 0): 2}, {(2, 0): 3, (3, 0): 4}, {(1, 0): 5, (3, 0): 6}]

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


class DenseLinAlg(ObjectInterpretation):
    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        if not (semiring is LinAlg and all(typeof(k()) is int for k in streams.keys())):
            return fwd()

        match body:
            case Term(op, args, {}) if op is D:
                if not all(isinstance(args, tuple) and len(args) == 2 for args in args):
                    return fwd()
                if len(args) <= 0:
                    return torch.tensor([])
                if len(args) > 1:
                    # todo: handle multiple output indices
                    return fwd()
                indices, value = args[0]

        # Check that the output is indexed in a subset of the input indices, and
        # that there are no index transformations
        if not all(isinstance(i, Term) and i.op in streams for i in indices):
            return fwd()
        indices = {i.op for i in indices}

        indexed_streams = {}
        reduction_indexes = []
        for k, v in streams.items():
            if k in indices:
                indexed_streams[k] = deffn(Indexable(torch.tensor(v))[k()])
            else:
                i = defop(int)
                reduction_indexes.append(i)
                indexed_streams[k] = deffn(Indexable(torch.tensor(v))[i()])

        breakpoint()

        with handler(indexed_streams):
            result = evaluate(value)

        result = to_tensor(result, reduction_indexes)
        for _ in range(len(reduction_indexes)):
            result = torch.sum(result, dim=0)
        return result


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

    with handler(DenseLinAlg()):
        vectorized_result = run_fold()

    breakpoint()

    # Compare with pytorch
    result_tensor = sparse_to_tensor(result)
    expected = torch.einsum("bij,bjk->bik", A, B_mat)
    assert torch.allclose(result_tensor, expected)
