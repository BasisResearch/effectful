import torch

from effectful.ops.core import Term
from effectful.internals.sugar import torch_getitem
from effectful.indexed.ops import name_to_sym, indices_of, IndexSet, to_tensor
from effectful.indexed.func import grad, jacfwd, jacrev, hessian, jvp, vjp


def test_grad_1():
    def sin(x):
        return torch.sin(x)

    grad_sin = grad(sin)
    i = name_to_sym("i")
    x = torch_getitem(torch.randn([10]), [i()])
    cos_x_actual = grad_sin(x)

    assert isinstance(cos_x_actual, Term)
    assert indices_of(cos_x_actual) == IndexSet(i=(set(range(10))))

    cos_x_expected = x.cos()

    assert torch.allclose(to_tensor(cos_x_actual, [i]), to_tensor(cos_x_expected, [i]))

    # Second-order gradients
    neg_sin_x_actual = grad(grad(lambda x: torch.sin(x)))(x)
    neg_sin_x_expected = -x.sin()

    assert torch.allclose(
        to_tensor(neg_sin_x_actual, [i]), to_tensor(neg_sin_x_expected, [i])
    )


def test_jacfwd_1():
    i = name_to_sym("i")
    x = torch_getitem(torch.randn(11, 5), [i()])
    jacobian = jacfwd(torch.sin)(x)
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian, [i]), to_tensor(expected, [i]))


def test_jacfwd_nested_1():
    i = name_to_sym("i")
    a = name_to_sym("a")
    y = torch_getitem(torch.randn(7, 5), [a()])
    x = torch_getitem(torch.randn(11, 5), [i()])

    def sin(x):
        return torch.sin(x) + y

    jacobian = jacfwd(sin)(x)
    expected = torch.diag(torch.cos(x) + 0 * y)

    assert torch.allclose(to_tensor(jacobian, [i, a]), to_tensor(expected, [i, a]))


def test_jacfwd_nested_2():
    i = name_to_sym("i")
    a = name_to_sym("a")
    y = torch_getitem(torch.randn(7, 5), [a()])
    x = torch_getitem(torch.randn(11, 5), [i()])

    def sin(x):
        return [torch.sin(x), y]

    jacobian = jacfwd(sin)(x)[0]
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian, [i]), to_tensor(expected, [i]))


def test_jacrev_1():
    i = name_to_sym("i")
    x = torch_getitem(torch.randn(11, 5), [i()])
    jacobian = jacrev(torch.sin)(x)
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian, [i]), to_tensor(expected, [i]))


def test_hessian_1():
    def f(x):
        return x.sin().sum()

    i = name_to_sym("i")
    x = torch_getitem(torch.randn(11, 5), [i()])
    hess = hessian(f)(x)  # equivalent to jacfwd(jacrev(f))(x)
    assert torch.allclose(to_tensor(hess, [i]), to_tensor(torch.diag(-x.sin()), [i]))


def test_jvp_1():
    i = name_to_sym("i")
    x = torch_getitem(torch.randn([10]), [i()])
    f = lambda x: x * torch.tensor([1.0, 2.0, 3])
    value, grad = jvp(f, (x,), (torch.tensor(1.0),))

    assert torch.allclose(to_tensor(value, [i]), to_tensor(f(x), [i]))
    assert torch.allclose(to_tensor(grad, [i]), torch.tensor([1.0, 2, 3]))


def test_jvp_nested():
    i = name_to_sym("i")
    j = name_to_sym("j")
    x = torch_getitem(torch.randn([10]), [i()])
    a = torch_getitem(torch.ones([7]), [j()])
    f = lambda x: a + x * torch.tensor([1.0, 2.0, 3])
    value, grad = jvp(f, (x,), (torch.tensor(1.0),))

    assert torch.allclose(to_tensor(value, [i, j]), to_tensor(f(x), [i, j]))
    assert torch.allclose(to_tensor(grad, [i, j]), torch.tensor([1.0, 2, 3]))


def test_vjp_1():
    i = name_to_sym("i")
    x = torch_getitem(torch.randn([10, 5]), [i()])
    y = torch_getitem(torch.ones([10, 5]), [i()])
    z = torch_getitem(torch.ones([10, 5]), [i()])

    f = lambda x: (x.sin(), x.cos())
    (_, vjpfunc) = vjp(f, x)
    vjps = vjpfunc((y, z))
    assert torch.allclose(to_tensor(vjps[0], [i]), to_tensor(x.cos() + -x.sin(), [i]))


def test_vjp_nested():
    i = name_to_sym("i")
    a = name_to_sym("a")
    x = torch_getitem(torch.randn([10, 5]), [i()])
    z = torch_getitem(torch.ones([7, 5]), [a()])
    y = torch_getitem(torch.ones([10, 7, 5]), [i(), a()])

    def f(x):
        return x * z

    (result, vjpfunc) = vjp(f, x)
    vjps = vjpfunc(y)
    assert torch.allclose(to_tensor(vjps[0], [i]), torch.tensor(7.0))
