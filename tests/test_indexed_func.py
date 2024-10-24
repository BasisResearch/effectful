import torch

from effectful.ops.core import Term
from effectful.internals.sugar import torch_getitem
from effectful.indexed.ops import name_to_sym, indices_of, IndexSet, to_tensor
from effectful.indexed.func import grad, jacfwd, jacrev, hessian, jvp, vjp


def test_grad_1():
    def sin(x):
        return torch.sin(x)

    grad_sin = grad(sin)
    x = torch_getitem(torch.randn([10]), [name_to_sym("i")()])
    cos_x_actual = grad_sin(x)

    assert isinstance(cos_x_actual, Term)
    assert indices_of(cos_x_actual) == IndexSet(i=(set(range(10))))

    cos_x_expected = x.cos()

    assert torch.allclose(to_tensor(cos_x_actual), to_tensor(cos_x_expected))

    # Second-order gradients
    neg_sin_x_actual = grad(grad(lambda x: torch.sin(x)))(x)
    neg_sin_x_expected = -x.sin()

    assert torch.allclose(to_tensor(neg_sin_x_actual), to_tensor(neg_sin_x_expected))


def test_jacfwd_1():
    x = torch_getitem(torch.randn(11, 5), [name_to_sym("i")()])
    jacobian = jacfwd(torch.sin)(x)
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian), to_tensor(expected))


def test_jacfwd_nested_1():
    i = name_to_sym("i")
    a = torch_getitem(torch.randn(7, 5), [name_to_sym("a")()])
    x = torch_getitem(torch.randn(11, 5), [i()])

    def sin(x):
        return torch.sin(x) + a

    jacobian = jacfwd(sin)(x)
    expected = torch.diag(torch.cos(x) + 0 * a)

    assert torch.allclose(to_tensor(jacobian), to_tensor(expected))


def test_jacfwd_nested_2():
    i = name_to_sym("i")
    a = torch_getitem(torch.randn(7, 5), [name_to_sym("a")()])
    x = torch_getitem(torch.randn(11, 5), [i()])

    def sin(x):
        return [torch.sin(x), a]

    jacobian = jacfwd(sin)(x)[0]
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian), to_tensor(expected))


def test_jacrev_1():
    x = torch_getitem(torch.randn(11, 5), [name_to_sym("i")()])
    jacobian = jacrev(torch.sin)(x)
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian), to_tensor(expected))


def test_hessian_1():
    def f(x):
        return x.sin().sum()

    x = torch_getitem(torch.randn(11, 5), [name_to_sym("i")()])
    hess = hessian(f)(x)  # equivalent to jacfwd(jacrev(f))(x)
    assert torch.allclose(to_tensor(hess), to_tensor(torch.diag(-x.sin())))


def test_jvp_1():
    x = torch_getitem(torch.randn([10]), [name_to_sym("i")()])
    f = lambda x: x * torch.tensor([1.0, 2.0, 3])
    value, grad = jvp(f, (x,), (torch.tensor(1.0),))

    assert torch.allclose(to_tensor(value), to_tensor(f(x)))
    assert torch.allclose(to_tensor(grad), torch.tensor([1.0, 2, 3]))


def test_jvp_nested():
    x = torch_getitem(torch.randn([10]), [name_to_sym("i")()])
    a = torch_getitem(torch.ones([7]), [name_to_sym("a")()])
    f = lambda x: a + x * torch.tensor([1.0, 2.0, 3])
    value, grad = jvp(f, (x,), (torch.tensor(1.0),))

    assert torch.allclose(to_tensor(value), to_tensor(f(x)))
    assert torch.allclose(to_tensor(grad), torch.tensor([1.0, 2, 3]))


def test_vjp_1():
    x = torch_getitem(torch.randn([10, 5]), [name_to_sym("i")()])
    y = torch_getitem(torch.ones([10, 5]), [name_to_sym("i")()])
    z = torch_getitem(torch.ones([10, 5]), [name_to_sym("i")()])

    f = lambda x: (x.sin(), x.cos())
    (_, vjpfunc) = vjp(f, x)
    vjps = vjpfunc((y, z))
    assert torch.allclose(to_tensor(vjps[0]), to_tensor(x.cos() + -x.sin()))


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
    assert torch.allclose(to_tensor(vjps[0]), torch.tensor(7.0))
