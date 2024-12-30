import logging
from typing import TypeVar

import pytest
import torch
from typing_extensions import ParamSpec

from effectful.handlers.torch import torch_getitem
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import as_term, defop, defun
from effectful.ops.types import Term

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def test_tpe_1():
    i, j = defop(int), defop(int)
    xval, y1_val, y2_val = torch.rand(2, 3), torch.rand(2), torch.rand(3)
    expected = torch.add(torch.add(xval, y1_val[..., None]), y2_val[None])

    x_ij = torch_getitem(xval, (i(), j()))
    x_plus_y1_ij = torch.add(x_ij, torch_getitem(y1_val, (i(),)))
    actual = torch.add(x_plus_y1_ij, torch_getitem(y2_val, (j(),)))

    assert actual.op == torch_getitem
    assert isinstance(actual.args[0], torch.Tensor)
    assert set(a.op for a in actual.args[1]) == {i, j}
    assert actual.shape == ()
    assert actual.numel() == 1
    assert actual.dim() == actual.ndim == 0

    f_actual = defun(actual, i, j)
    for ii in range(2):
        for jj in range(3):
            assert f_actual(torch.tensor(ii), torch.tensor(jj)) == expected[ii, jj]


def test_tpe_2():
    xval, ival = torch.rand(2, 3), torch.arange(2)
    expected = torch.sum(xval[ival, :], dim=0)

    j = defop(int)
    x_j = torch_getitem(
        xval,
        (
            ival,
            j(),
        ),
    )
    assert x_j.shape == (2,)
    assert x_j.size(0) == x_j.shape[0]
    actual = torch.sum(x_j, dim=0)

    assert actual.op == torch_getitem
    assert isinstance(actual.args[0], torch.Tensor)
    assert set(a.op for a in actual.args[1]) == {j}
    assert actual.shape == ()
    assert actual.numel() == 1

    f_actual = defun(actual, j)
    for jj in range(3):
        assert f_actual(torch.tensor(jj)) == expected[jj]


def test_tpe_3():
    xval, ival = torch.rand(4, 2, 3), torch.arange(2)
    expected = torch.sum(xval, dim=1)

    j, k = defop(int), defop(int)
    x_j = torch_getitem(
        xval,
        (
            k(),
            ival,
            j(),
        ),
    )
    actual = torch.sum(x_j, dim=0)

    assert actual.op == torch_getitem
    assert isinstance(actual.args[0], torch.Tensor)
    assert set(a.op for a in actual.args[1]) == {j, k}
    assert actual.shape == ()
    assert actual.numel() == 1

    f_actual = defun(actual, j, k)
    for jj in range(3):
        for kk in range(4):
            assert f_actual(torch.tensor(jj), torch.tensor(kk)) == expected[kk, jj]


def test_tpe_4():
    xval, ival = torch.rand(4, 2, 3), torch.arange(2)
    expected = torch.sum(xval, dim=1)

    @as_term
    def f_actual(x: torch.Tensor, j: int, k: int) -> torch.Tensor:  # type: ignore
        return torch.sum(x[k, ival, j], dim=0)

    for jj in range(3):
        for kk in range(4):
            assert (
                f_actual(xval, torch.tensor(jj), torch.tensor(kk)) == expected[kk, jj]
            )


def test_tpe_known_index():
    """Constant indexes are partially evaluated away."""
    i, j = defop(int, name="i"), defop(int, name="j")

    cases = [
        torch_getitem(torch.ones(2, 3), (i(), 1)),
        torch_getitem(torch.ones(2, 3), (0, i())),
        torch_getitem(torch.ones(2, 3, 4), (0, i(), 1)),
        torch_getitem(torch.ones(2, 3, 4), (0, i(), j())),
        torch_getitem(torch.ones(2, 3, 4), (i(), j(), 3)),
    ]

    for case_ in cases:
        assert all(isinstance(a, Term) for a in case_.args[1])
        assert not any(isinstance(a, int) for a in case_.args[1])


def test_tpe_constant_eval():
    """Constant indexes are partially evaluated away."""
    height, width = defop(int, name="height"), defop(int, name="width")
    t = torch.tensor([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
    A = torch_getitem(t, (height(), width()))

    layer = defop(int, name="layer")
    with handler({height: lambda: layer() // 3, width: lambda: layer() % 3}):
        A_layer = evaluate(A)
    with handler({layer: lambda: 2}):
        A_final = evaluate(A_layer)

    assert not isinstance(A_final, Term)


def test_tpe_stack():
    xval, yval = torch.rand(10, 5), torch.rand(10, 5)

    i = defop(int)
    j = defop(int)
    x_ij = torch_getitem(
        xval,
        (i(), j()),
    )
    y_ij = torch_getitem(
        yval,
        (i(), j()),
    )
    actual = torch.stack((x_ij, y_ij))
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (2,)
    f_actual = defun(actual, i, j)

    for ii in range(10):
        for jj in range(5):
            actual = f_actual(ii, jj)
            expected = torch.stack(
                (defun(x_ij, i, j)(ii, jj), defun(y_ij, i, j)(ii, jj))
            )
            assert torch.equal(actual, expected)


INDEXING_CASES = [
    # Simple integer indexing
    (torch.randn(4, 5, 6), (0,)),
    # Simple slice indexing
    (torch.randn(4, 5, 6), (slice(1, 3),)),
    # Advanced indexing with tensors
    (torch.randn(4, 5, 6), (torch.tensor([0, 2]),)),
    (torch.randn(4, 5, 6), (torch.tensor([0, 2]), slice(None), torch.tensor([0, 2]))),
    # Mixed indexing
    (torch.randn(4, 5, 6), (slice(None), torch.tensor([1, 3]), 2)),
    # Indexing with None (newaxis)
    (torch.randn(4, 5, 6), (None, slice(None), None, slice(1, 3))),
    # Indexing with Ellipsis
    (torch.randn(4, 5, 6, 7), (Ellipsis, torch.tensor([1, 3]))),
    # Integer and tensor indexing
    (torch.randn(4, 5, 6), (2, torch.tensor([1, 3, 4]))),
    # Indexing with negative indices
    (torch.randn(4, 5, 6), (-1,)),
    # Indexing with step in slice (currently supports only slice(None))
    # (torch.randn(4, 5, 6), (slice(None, None, 2),)),
    # Indexing with empty tensor
    (torch.randn(4, 5, 6), (torch.tensor([], dtype=torch.long),)),
    # Complex mixed indexing
    (torch.randn(4, 5, 6), (slice(None), torch.tensor([0, 2]), None, Ellipsis)),
    # Indexing with multiple None
    (torch.randn(4, 5, 6), (None, None, 1, slice(None), None)),
    # Additional complex cases
    (
        torch.randn(4, 5, 6),
        (torch.tensor([[0, 1], [2, 3]]), torch.tensor([[1, 2], [3, 4]]), slice(None)),
    ),
    (torch.randn(4, 5, 6), (Ellipsis, None, torch.tensor([0, 2]))),
    (torch.randn(4, 5, 6), (torch.arange(4)[..., None, None],)),
    (torch.randn(4, 5, 6), (torch.arange(4)[..., None, None], None, slice(None))),
    (torch.randn(4, 5, 6), (None, torch.arange(4)[..., None, None], None, slice(None))),
    (
        torch.randn(4, 5, 6),
        (torch.arange(4)[..., None, None], torch.arange(5)[..., None]),
    ),
    (
        torch.randn(4, 5, 6),
        (torch.arange(4)[..., None, None], torch.arange(5)[..., None], None, 1),
    ),
    (
        torch.randn(4, 5, 6),
        (
            torch.arange(4)[..., None, None],
            torch.arange(5)[..., None],
            None,
            slice(None),
        ),
    ),
    (
        torch.randn(3, 4, 5, 6),
        (
            Ellipsis,
            torch.arange(4)[..., None, None],
            torch.arange(5)[..., None],
            slice(None),
        ),
    ),
]


@pytest.mark.parametrize("tensor, idx", INDEXING_CASES)
def test_getitem_ellipsis_and_none(tensor, idx):
    from effectful.handlers.torch import _getitem_ellipsis_and_none

    expected = tensor[idx]
    t, i = _getitem_ellipsis_and_none(tensor, idx)

    if any(k is Ellipsis or k is None for k in idx):
        assert t.shape != tensor.shape or idx != i
    assert not any(k is Ellipsis or k is None for k in i)

    result = t[i]
    assert (
        result.shape == expected.shape
    ), f"Shape mismatch for idx: {idx}. Expected: {expected.shape}, Got: {result.shape}"
    assert torch.allclose(result, expected, equal_nan=True), f"Failed for idx: {idx}"


@pytest.mark.parametrize("tensor, idx", INDEXING_CASES)
def test_custom_getitem(tensor, idx):
    expected = tensor[idx]
    result = torch_getitem(tensor, idx)
    assert (
        result.shape == expected.shape
    ), f"Shape mismatch for idx: {idx}. Expected: {expected.shape}, Got: {result.shape}"
    assert torch.allclose(result, expected, equal_nan=True), f"Failed for idx: {idx}"


def test_vmap_custom_getitem():
    tensor = torch.randn(4, 5, 6)
    idx = (torch.tensor([0, 2]), slice(None), torch.tensor([0, 2]))
    result = torch.vmap(lambda i, k: torch_getitem(tensor, (i, slice(None), k)))(
        idx[0], idx[2]
    )
    assert isinstance(result, torch.Tensor)
    for i in range(2):
        idx_i = tuple(
            idxe[i] if isinstance(idxe, torch.Tensor) else idxe for idxe in idx
        )
        assert torch.allclose(result[i], tensor[idx_i])
