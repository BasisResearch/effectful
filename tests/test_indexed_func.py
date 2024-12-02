import torch

from effectful.indexed.func import grad, hessian, jacfwd, jacrev, jvp, vjp, vmap
from effectful.indexed.ops import Indexable, IndexSet, indices_of, to_tensor
from effectful.internals.sugar import gensym
from effectful.ops.core import Term


def test_grad_1():
    def sin(x):
        return torch.sin(x)

    grad_sin = grad(sin)
    i = gensym(int, name="i")
    x = Indexable(torch.randn([10]))[i()]
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
    i = gensym(int, name="i")
    x = Indexable(torch.randn(11, 5))[i()]
    jacobian = jacfwd(torch.sin)(x)
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian, [i]), to_tensor(expected, [i]))


def test_jacfwd_nested_1():
    i = gensym(int, name="i")
    a = gensym(int, name="a")
    y = Indexable(torch.randn(7, 5))[a()]
    x = Indexable(torch.randn(11, 5))[i()]

    def sin(x):
        return torch.sin(x) + y

    jacobian = jacfwd(sin)(x)
    expected = torch.diag(torch.cos(x) + 0 * y)

    assert torch.allclose(to_tensor(jacobian, [i, a]), to_tensor(expected, [i, a]))


def test_jacfwd_nested_2():
    i = gensym(int, name="i")
    a = gensym(int, name="a")
    y = Indexable(torch.randn(7, 5))[a()]
    x = Indexable(torch.randn(11, 5))[i()]

    def sin(x):
        return [torch.sin(x), y]

    jacobian = jacfwd(sin)(x)[0]
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian, [i]), to_tensor(expected, [i]))


def test_jacrev_1():
    i = gensym(int, name="i")
    x = Indexable(torch.randn(11, 5))[i()]
    jacobian = jacrev(torch.sin)(x)
    expected = torch.diag(torch.cos(x))

    assert torch.allclose(to_tensor(jacobian, [i]), to_tensor(expected, [i]))


def test_hessian_1():
    def f(x):
        return x.sin().sum()

    i = gensym(int, name="i")
    x = Indexable(torch.randn(11, 5))[i()]
    hess = hessian(f)(x)  # equivalent to jacfwd(jacrev(f))(x)
    assert torch.allclose(to_tensor(hess, [i]), to_tensor(torch.diag(-x.sin()), [i]))


def test_jvp_1():
    i = gensym(int, name="i")
    x = Indexable(torch.randn([10]))[i()]

    def f(x):
        return x * torch.tensor([1.0, 2.0, 3])

    value, grad = jvp(f, (x,), (torch.tensor(1.0),))

    assert torch.allclose(to_tensor(value, [i]), to_tensor(f(x), [i]))
    assert torch.allclose(to_tensor(grad, [i]), torch.tensor([1.0, 2, 3]))


def test_jvp_nested():
    i = gensym(int, name="i")
    j = gensym(int, name="j")
    x = Indexable(torch.randn([10]))[i()]
    a = Indexable(torch.ones([7]))[j()]

    def f(x):
        return a + x * torch.tensor([1.0, 2.0, 3])

    value, grad = jvp(f, (x,), (torch.tensor(1.0),))

    assert torch.allclose(to_tensor(value, [i, j]), to_tensor(f(x), [i, j]))
    assert torch.allclose(to_tensor(grad, [i, j]), torch.tensor([1.0, 2, 3]))


def test_vjp_1():
    i = gensym(int, name="i")
    x = Indexable(torch.randn([10, 5]))[i()]
    y = Indexable(torch.ones([10, 5]))[i()]
    z = Indexable(torch.ones([10, 5]))[i()]

    def f(x):
        return (x.sin(), x.cos())

    (_, vjpfunc) = vjp(f, x)
    vjps = vjpfunc((y, z))
    assert torch.allclose(to_tensor(vjps[0], [i]), to_tensor(x.cos() + -x.sin(), [i]))


def test_vjp_nested():
    i = gensym(int, name="i")
    a = gensym(int, name="a")
    x = Indexable(torch.randn([10, 5]))[i()]
    z = Indexable(torch.ones([7, 5]))[a()]
    y = Indexable(torch.ones([10, 7, 5]))[i(), a()]

    def f(x):
        return x * z

    (result, vjpfunc) = vjp(f, x)
    vjps = vjpfunc(y)
    assert torch.allclose(to_tensor(vjps[0], [i]), torch.tensor(7.0))


def test_vmap_1():
    i = gensym(int, name="i")
    x = torch.randn([10, 5])
    x_i = Indexable(x)[i()]

    def f(x):
        return x + 1

    actual = vmap(f)(x_i)
    expected = x + 1
    assert torch.allclose(to_tensor(actual, [i]), expected)


def test_vmap_nested():
    i = gensym(int, name="i")
    j = gensym(int, name="j")
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

    i = gensym(int, name="i")
    x = Indexable(torch.randn([10, 7]))[i()]

    # implicit vmap over i and explicit vmap over first positional dim
    actual = vmap(grad_sin)(x)
    assert actual.shape == torch.Size([7])

    actual_t = to_tensor(actual, [i])
    x_t = to_tensor(x, [i])
    for ii in range(10):
        for jj in range(7):
            assert torch.allclose(actual_t[ii, jj], x_t[ii, jj].cos())
