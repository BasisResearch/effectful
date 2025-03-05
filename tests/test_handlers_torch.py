import logging
from typing import TypeVar

import pytest
import torch
from typing_extensions import ParamSpec

from effectful.handlers.torch import (
    Indexable,
    grad,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    sizesof,
    to_tensor,
    torch_getitem,
    vjp,
    vmap,
)
from effectful.ops.semantics import evaluate, fvsof, handler
from effectful.ops.syntax import deffn, defop, defterm
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

    f_actual = deffn(actual, i, j)
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

    f_actual = deffn(actual, j)
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

    f_actual = deffn(actual, j, k)
    for jj in range(3):
        for kk in range(4):
            assert f_actual(torch.tensor(jj), torch.tensor(kk)) == expected[kk, jj]


def test_tpe_4():
    xval, ival = torch.rand(4, 2, 3), torch.arange(2)
    expected = torch.sum(xval, dim=1)

    @defterm
    def f_actual(x: torch.Tensor, j: int, k: int) -> torch.Tensor:
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
    f_actual = deffn(actual, i, j)

    for ii in range(10):
        for jj in range(5):
            actual = f_actual(ii, jj)
            expected = torch.stack(
                (deffn(x_ij, i, j)(ii, jj), deffn(y_ij, i, j)(ii, jj))
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
    assert result.shape == expected.shape, (
        f"Shape mismatch for idx: {idx}. Expected: {expected.shape}, Got: {result.shape}"
    )
    assert torch.allclose(result, expected, equal_nan=True), f"Failed for idx: {idx}"


@pytest.mark.parametrize("tensor, idx", INDEXING_CASES)
def test_custom_getitem(tensor, idx):
    expected = tensor[idx]
    result = torch_getitem(tensor, idx)
    assert result.shape == expected.shape, (
        f"Shape mismatch for idx: {idx}. Expected: {expected.shape}, Got: {result.shape}"
    )
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


def test_grad_1():
    def sin(x):
        return torch.sin(x)

    grad_sin = grad(sin)
    i = defop(int, name="i")
    x = Indexable(torch.randn([10]))[i()]
    cos_x_actual = grad_sin(x)

    assert isinstance(cos_x_actual, Term)
    assert sizesof(cos_x_actual) == {i: 10}

    cos_x_expected = x.cos()

    assert torch.allclose(to_tensor(cos_x_actual, [i]), to_tensor(cos_x_expected, [i]))

    # Second-order gradients
    neg_sin_x_actual = grad(grad(lambda x: torch.sin(x)))(x)
    neg_sin_x_expected = -x.sin()

    assert torch.allclose(
        to_tensor(neg_sin_x_actual, [i]), to_tensor(neg_sin_x_expected, [i])
    )


def test_jacfwd_1():
    i = defop(int, name="i")
    x = Indexable(torch.randn(11, 5))[i()]
    jacobian = jacfwd(torch.sin)(x)
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian, [i]), to_tensor(expected, [i]))


def test_jacfwd_nested_1():
    i = defop(int, name="i")
    a = defop(int, name="a")
    y = Indexable(torch.randn(7, 5))[a()]
    x = Indexable(torch.randn(11, 5))[i()]

    def sin(x):
        return torch.sin(x) + y

    jacobian = jacfwd(sin)(x)
    expected = torch.diag(torch.cos(x) + 0 * y)

    assert torch.allclose(to_tensor(jacobian, [i, a]), to_tensor(expected, [i, a]))


def test_jacfwd_nested_2():
    i = defop(int, name="i")
    a = defop(int, name="a")
    y = Indexable(torch.randn(7, 5))[a()]
    x = Indexable(torch.randn(11, 5))[i()]

    def sin(x):
        return [torch.sin(x), y]

    jacobian = jacfwd(sin)(x)[0]
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian, [i]), to_tensor(expected, [i]))


def test_jacrev_1():
    i = defop(int, name="i")
    x = Indexable(torch.randn(11, 5))[i()]
    jacobian = jacrev(torch.sin)(x)
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian, [i]), to_tensor(expected, [i]))


def test_hessian_1():
    def f(x):
        return x.sin().sum()

    i = defop(int, name="i")
    x = Indexable(torch.randn(11, 5))[i()]
    hess = hessian(f)(x)  # equivalent to jacfwd(jacrev(f))(x)
    assert torch.allclose(to_tensor(hess, [i]), to_tensor(torch.diag(-x.sin()), [i]))


def test_jvp_1():
    i = defop(int, name="i")
    x = Indexable(torch.randn([10]))[i()]

    def f(x):
        return x * torch.tensor([1.0, 2.0, 3])

    value, grad = jvp(f, (x,), (torch.tensor(1.0),))

    assert torch.allclose(to_tensor(value, [i]), to_tensor(f(x), [i]))
    assert torch.allclose(to_tensor(grad, [i]), torch.tensor([1.0, 2, 3]))


def test_jvp_nested():
    i = defop(int, name="i")
    j = defop(int, name="j")
    x = Indexable(torch.randn([10]))[i()]
    a = Indexable(torch.ones([7]))[j()]

    def f(x):
        return a + x * torch.tensor([1.0, 2.0, 3])

    value, grad = jvp(f, (x,), (torch.tensor(1.0),))

    assert torch.allclose(to_tensor(value, [i, j]), to_tensor(f(x), [i, j]))
    assert torch.allclose(to_tensor(grad, [i, j]), torch.tensor([1.0, 2, 3]))


def test_vjp_1():
    i = defop(int, name="i")
    x = Indexable(torch.randn([10, 5]))[i()]
    y = Indexable(torch.ones([10, 5]))[i()]
    z = Indexable(torch.ones([10, 5]))[i()]

    def f(x):
        return (x.sin(), x.cos())

    (_, vjpfunc) = vjp(f, x)
    vjps = vjpfunc((y, z))
    assert torch.allclose(to_tensor(vjps[0], [i]), to_tensor(x.cos() + -x.sin(), [i]))


def test_vjp_nested():
    i = defop(int, name="i")
    a = defop(int, name="a")
    x = Indexable(torch.randn([10, 5]))[i()]
    z = Indexable(torch.ones([7, 5]))[a()]
    y = Indexable(torch.ones([10, 7, 5]))[i(), a()]

    def f(x):
        return x * z

    (result, vjpfunc) = vjp(f, x)
    vjps = vjpfunc(y)
    assert torch.allclose(to_tensor(vjps[0], [i]), torch.tensor(7.0))


def test_vmap_1():
    i = defop(int, name="i")
    x = torch.randn([10, 5])
    x_i = Indexable(x)[i()]

    def f(x):
        return x + 1

    actual = vmap(f)(x_i)
    expected = x + 1
    assert torch.allclose(to_tensor(actual, [i]), expected)


def test_vmap_nested():
    i = defop(int, name="i")
    j = defop(int, name="j")
    x = torch.randn([10, 5, 4])
    x_i = Indexable(x)[i()]
    y = torch.randn([7])
    y_j = Indexable(y)[j()]

    def f(x):
        return y_j + x

    actual = vmap(f)(x_i)
    actual_t = to_tensor(actual, [i, j])

    for ii in range(10):
        for jj in range(7):
            assert (actual_t[ii, jj] == x[ii] + y[jj]).all()


def test_vmap_and_grad():
    sin = torch.sin
    grad_sin = grad(sin)

    i = defop(int, name="i")
    x = Indexable(torch.randn([10, 7]))[i()]

    # implicit vmap over i and explicit vmap over first positional dim
    actual = vmap(grad_sin)(x)
    assert actual.shape == torch.Size([7])

    actual_t = to_tensor(actual, [i])
    x_t = to_tensor(x, [i])
    for ii in range(10):
        for jj in range(7):
            assert torch.allclose(actual_t[ii, jj], x_t[ii, jj].cos())


def test_index_incompatible():
    """Check that using the same index in two incompatible dimensions raises an error."""
    i = defop(int)
    with pytest.raises(ValueError):
        torch_getitem(torch.randn(2, 3), (i(), i()))

    torch_getitem(torch.randn(2, 2), (i(), i()))


def test_to_tensor():
    """Test to_tensor's handling of free variables and tensor shapes"""
    i, j, k = defop(int, name="i"), defop(int, name="j"), defop(int, name="k")
    t = torch.randn([2, 3, 4])

    # Test case 1: Converting all named dimensions to positional
    t_ijk = Indexable(t)[i(), j(), k()]
    assert fvsof(t_ijk) == {i, j, k}

    t1 = to_tensor(t_ijk, [i, j, k])
    assert fvsof(t1) == set()  # No free variables remain
    assert t1.shape == torch.Size([2, 3, 4])

    # Test case 2: Different ordering of dimensions
    t2 = to_tensor(t_ijk, [k, j, i])
    assert fvsof(t2) == set()
    assert t2.shape == torch.Size([4, 3, 2])

    # Test case 3: Keeping some dimensions as free variables
    t3 = to_tensor(t_ijk, [i])  # Convert only i to positional
    assert fvsof(t3) == {j, k}  # j and k remain free
    assert isinstance(t3, Term)
    assert t3.shape == torch.Size([2])

    t4 = to_tensor(t_ijk, [i, j])  # Convert i and j to positional
    assert fvsof(t4) == {k}  # only k remains free
    assert isinstance(t4, Term)
    assert t4.shape == torch.Size([2, 3])

    # Test case 4: Empty order list keeps all variables free
    t5 = to_tensor(t_ijk, [])
    assert fvsof(t5) == {i, j, k}  # All variables remain free
    assert isinstance(t5, Term)
    assert t5.shape == torch.Size([])

    # Test case 5: Verify permuted tensors maintain correct relationships
    t_kji = Indexable(t.permute(2, 1, 0))[k(), j(), i()]
    t6 = to_tensor(t_kji, [i, j, k])
    t7 = to_tensor(t_ijk, [i, j, k])
    assert torch.allclose(t6, t7)

    # Test case 6: Mixed operations with free variables
    x = torch.sin(t_ijk)  # Apply operation to indexed tensor
    x1 = to_tensor(x, [i, j])  # Convert some dimensions
    assert fvsof(x1) == {k}  # k remains free
    assert isinstance(x1, Term)
    assert x1.shape == torch.Size([2, 3])

    # Test case 7: Multiple tensors sharing variables
    t2_ijk = Indexable(torch.randn([2, 3, 4]))[i(), j(), k()]
    sum_t = t_ijk + t2_ijk
    sum1 = to_tensor(sum_t, [i, j])
    assert fvsof(sum1) == {k}  # k remains free
    assert isinstance(sum1, Term)
    assert sum1.shape == torch.Size([2, 3])

    # Test case 8: Tensor term with non-sized free variables
    w = defop(torch.Tensor, name="w")
    t_ijk = Indexable(t)[i(), j(), k()] + w()
    t8 = to_tensor(t_ijk, [i, j, k])
    assert fvsof(t8) == {w}

    # Test case 9: Eliminate remaining free variables in result
    with handler({w: lambda: torch.tensor(1.0)}):
        t9 = evaluate(t8)
    assert fvsof(t9) == set([])


def test_tensor_term_operators():
    x, y = defop(torch.Tensor), defop(torch.Tensor)

    # Test basic arithmetic operators
    binary = [
        # Arithmetic
        (x() + y(), "add"),
        (x() - y(), "sub"),
        (x() * y(), "mul"),
        (x() / y(), "div"),
        (x() // y(), "floordiv"),
        (x() % y(), "mod"),
        (x() @ y(), "matmul"),
        (x() ** y(), "pow"),
        # Bitwise
        (x() & y(), "and"),
        (x() | y(), "or"),
        (x() ^ y(), "xor"),
        # Comparison
        (x() > y(), "gt"),
        (x() >= y(), "ge"),
        (x() < y(), "lt"),
        (x() <= y(), "le"),
        (x() == y(), "eq"),
        (x() != y(), "ne"),
        # Additional math operations
        (torch.maximum(x(), y()), "maximum"),
        (torch.minimum(x(), y()), "minimum"),
        (torch.fmod(x(), y()), "fmod"),
        (torch.remainder(x(), y()), "remainder"),
    ]

    # Test unary operators
    unary = [
        (-x(), "neg"),
        (~x(), "not"),
        (abs(x()), "abs"),
        (torch.exp(x()), "exp"),
        (torch.log(x()), "log"),
        (torch.sqrt(x()), "sqrt"),
        (torch.sin(x()), "sin"),
        (torch.cos(x()), "cos"),
        (torch.tan(x()), "tan"),
        (torch.sigmoid(x()), "sigmoid"),
        (torch.tanh(x()), "tanh"),
        (torch.relu(x()), "relu"),
        (torch.ceil(x()), "ceil"),
        (torch.floor(x()), "floor"),
        (torch.round(x()), "round"),
        (torch.sign(x()), "sign"),
    ]

    for t, op_name in binary + unary:
        assert isinstance(t, Term), f"Failed for operation: {op_name}"

    for t, op_name in binary:
        assert {x, y} <= fvsof(t), f"Missing variables for operation: {op_name}"

    for t, op_name in unary:
        assert {x} <= fvsof(t), f"Missing variables for operation: {op_name}"


def test_indexed_tensor_as_index():
    t1 = torch.randn(2, 3)
    i = defop(int, name="i")
    t2 = Indexable(torch.tensor([0, 1]))[i()]

    t3 = t1[t2]
    assert sizesof(t3) == sizesof(t2)
    assert (to_tensor(t3, [i]) == t1).all()
