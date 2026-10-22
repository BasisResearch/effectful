"""Array methods shared by symbolic terms and custom pytrees in jax._terms."""

import copy
import inspect
import operator
from dataclasses import dataclass

import jax
import numpy as np
import pytest

import effectful.handlers.jax.numpy as jnp
from effectful.handlers import jax as ejax
from effectful.handlers.jax import ArrayIndex, ArrayMethods, jax_getitem
from effectful.ops.semantics import evaluate, fwd, handler
from effectful.ops.syntax import defop
from effectful.ops.types import Operation, Term


@jax.tree_util.register_dataclass
@dataclass(eq=False)
class _CustomArray(ArrayMethods):
    data: jax.Array


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
        operator.matmul,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.and_,
        operator.or_,
        operator.xor,
        operator.lshift,
        operator.rshift,
        divmod,
    ],
)
def test_custom_array_binary_methods(op, reverse):
    data = jax.numpy.array([[1, 2], [3, 4]])
    other = jax.numpy.array([[2, 3], [4, 5]])

    def run(value):
        return op(other, value) if reverse else op(value, other)

    with handler(
        {
            ejax.is_custom_array: lambda x: isinstance(x, _CustomArray),
            ejax.to_jax_array: lambda x: x.data if isinstance(x, _CustomArray) else x,
        }
    ):
        expected = run(data)
        np.testing.assert_array_equal(run(_CustomArray(data)), expected)
        np.testing.assert_array_equal(ejax.jit(run)(_CustomArray(data)), expected)


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(operator.neg, id="neg"),
        pytest.param(operator.pos, id="pos"),
        pytest.param(operator.abs, id="abs"),
        pytest.param(operator.invert, id="invert"),
        pytest.param(lambda x: x.sum(axis=0, keepdims=True), id="sum"),
        pytest.param(lambda x: x.reshape(4), id="reshape"),
        pytest.param(lambda x: x.reshape((4,)), id="reshape_tuple"),
        pytest.param(lambda x: x.transpose((1, 0)), id="transpose"),
        pytest.param(lambda x: x.T, id="T"),
        pytest.param(lambda x: x.astype(jax.numpy.float32), id="astype"),
        pytest.param(lambda x: x.dot(jax.numpy.eye(2)), id="dot"),
        pytest.param(lambda x: x.clip(min=2, max=3), id="clip"),
        pytest.param(lambda x: x[0], id="getitem"),
        pytest.param(lambda x: x.at[0, 1].set(10), id="at_set"),
        pytest.param(lambda x: x.shape, id="shape"),
        pytest.param(lambda x: x.size, id="size"),
        pytest.param(lambda x: x.ndim, id="ndim"),
        pytest.param(len, id="len"),
        pytest.param(lambda x: tuple(x), id="iter"),
        pytest.param(lambda x: x.mT, id="mT"),
        pytest.param(lambda x: x.real, id="real"),
        pytest.param(lambda x: x.imag, id="imag"),
        pytest.param(lambda x: x.itemsize, id="itemsize"),
        pytest.param(lambda x: x.nbytes, id="nbytes"),
        pytest.param(lambda x: x.flatten(order="F"), id="flatten"),
        pytest.param(lambda x: x.view(dtype=jax.numpy.uint8), id="view"),
        pytest.param(lambda x: x.to_device(jax.devices()[0]), id="to_device"),
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(round, id="round_integer"),
        pytest.param(lambda x: round(x, 1), id="round_digits"),
        pytest.param(lambda x: x.transpose(1, 0), id="transpose_variadic"),
        pytest.param(lambda x: x.all(0, None, True), id="all_options"),
        pytest.param(lambda x: x.any(0, None, True), id="any_options"),
        pytest.param(lambda x: x.argmax(0, None, True), id="argmax_options"),
        pytest.param(lambda x: x.argmin(0, None, True), id="argmin_options"),
        pytest.param(
            lambda x: x.astype(jax.numpy.float32, copy=True, device=jax.devices()[0]),
            id="astype_options",
        ),
        pytest.param(
            lambda x: x.choose(
                [jax.numpy.ones((2, 2)), jax.numpy.zeros((2, 2))], mode="clip"
            ),
            id="choose_options",
        ),
        pytest.param(
            lambda x: x.compress(
                jax.numpy.array([True, False]), axis=0, size=2, fill_value=-1
            ),
            id="compress_options",
        ),
        pytest.param(
            lambda x: x.cumsum(0, jax.numpy.float32, None), id="cumsum_options"
        ),
        pytest.param(
            lambda x: x.cumprod(0, jax.numpy.float32, None), id="cumprod_options"
        ),
        pytest.param(
            lambda x: x.dot(
                jax.numpy.eye(2),
                precision=jax.lax.Precision.HIGH,
                preferred_element_type=jax.numpy.float32,
            ),
            id="dot_options",
        ),
        pytest.param(lambda x: x.max(0, None, True), id="max_options"),
        pytest.param(lambda x: x.min(0, None, True), id="min_options"),
        pytest.param(
            lambda x: x.mean(0, jax.numpy.float32, None, True), id="mean_options"
        ),
        pytest.param(
            lambda x: x.sum(0, jax.numpy.float32, None, True), id="sum_options"
        ),
        pytest.param(
            lambda x: x.prod(0, jax.numpy.float32, None, True), id="prod_options"
        ),
        pytest.param(lambda x: x.ptp(0, None, True), id="ptp_options"),
        pytest.param(
            lambda x: x.std(axis=0, dtype=jax.numpy.float32, correction=1),
            id="std_options",
        ),
        pytest.param(
            lambda x: x.var(axis=0, dtype=jax.numpy.float32, correction=1),
            id="var_options",
        ),
        pytest.param(
            lambda x: x.take(jax.numpy.array([1, 0]), 0, None, "clip"),
            id="take_options",
        ),
        pytest.param(
            lambda x: x.trace(0, 0, 1, jax.numpy.float32, None), id="trace_options"
        ),
        pytest.param(
            lambda x: x.argsort(kind=None, order=None, descending=True),
            id="argsort_options",
        ),
        pytest.param(
            lambda x: x.sort(kind=None, order=None, descending=True), id="sort_options"
        ),
        pytest.param(
            lambda x: x.reshape((4,), out_sharding=None), id="reshape_options"
        ),
        pytest.param(lambda x: x.ravel(out_sharding=None), id="ravel_options"),
        pytest.param(
            lambda x: x.repeat(2, axis=0, total_repeat_length=4, out_sharding=None),
            id="repeat_options",
        ),
        pytest.param(lambda x: x.round(1, None), id="round_options"),
    ],
)
def test_custom_array_methods(run):
    data = jax.numpy.array([[1, 2], [3, 4]])
    with handler(
        {
            ejax.is_custom_array: lambda x: isinstance(x, _CustomArray),
            ejax.to_jax_array: lambda x: x.data if isinstance(x, _CustomArray) else x,
        }
    ):
        expected = run(data)
        np.testing.assert_array_equal(run(_CustomArray(data)), expected)
        np.testing.assert_array_equal(ejax.jit(run)(_CustomArray(data)), expected)


def test_array_methods_cover_jax_surface():
    from jax._src.numpy import array_methods

    names = {
        name
        for name in array_methods._array_methods
        if not name.startswith("_") or name == "__array_namespace__"
    }
    names.update(array_methods._array_properties)
    names.update(f"__{name}__" for name in array_methods._array_operators)
    assert names <= set(dir(ArrayMethods))

    for name in names & array_methods._array_methods.keys():
        expected = inspect.signature(array_methods._array_methods[name])
        actual = inspect.signature(getattr(ArrayMethods, name))
        for parameter in expected.parameters.values():
            if parameter.kind == parameter.VAR_POSITIONAL:
                assert any(
                    p.kind == p.VAR_POSITIONAL for p in actual.parameters.values()
                ), name
            else:
                assert parameter.name in actual.parameters, (name, parameter.name)


@pytest.mark.parametrize(
    "method",
    [
        "get",
        "set",
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "min",
        "max",
        "apply",
    ],
)
def test_custom_array_at_methods(method):
    data = jax.numpy.array([[1.0, 2.0], [3.0, 4.0]])
    indices = jax.numpy.array([-1, 0, 3])
    operation = getattr(ArrayIndex, method)
    assert isinstance(operation, Operation)
    seen = []

    def update(ref, *args, **kwargs):
        assert isinstance(ref.array, _CustomArray)
        seen.append(method)
        return fwd()

    def run(value):
        ref = value.at[indices, 1]
        kwargs = dict(
            indices_are_sorted=False, unique_indices=True, wrap_negative_indices=False
        )
        if method == "get":
            return ref.get(mode="fill", fill_value=-99, **kwargs)
        argument = (lambda x: x * 2) if method == "apply" else 2.0
        return getattr(ref, method)(argument, mode="drop", **kwargs)

    with handler(
        {
            ejax.is_custom_array: lambda x: isinstance(x, _CustomArray),
            ejax.to_jax_array: lambda x: x.data if isinstance(x, _CustomArray) else x,
        }
    ):
        expected = run(data)
        with handler({operation: update}):
            np.testing.assert_array_equal(run(_CustomArray(data)), expected)
            np.testing.assert_array_equal(ejax.jit(run)(_CustomArray(data)), expected)
    assert seen == [method, method]


def test_custom_array_at_effect_preserves_whole_operands():
    seen = []

    def update(ref, values, **kwargs):
        assert isinstance(ref.array, _CustomArray)
        seen.append("set")
        return _CustomArray(ref.array.data.at[ref.key].set(values, **kwargs))

    data = jax.numpy.ones((2, 2))

    def run(value):
        return value.at[0, 1].set(4, mode="drop")

    with handler({ArrayIndex.set: update}):
        for fn in (run, ejax.jit(run)):
            result = fn(_CustomArray(data))
            assert isinstance(result, _CustomArray)
            np.testing.assert_array_equal(result.data, [[1, 4], [1, 1]])
    assert seen == ["set", "set"]


def test_custom_array_at_apply_callback_is_effectful():
    data = jax.numpy.array([1.0, 2.0])

    def run(value):
        return value.at[0].apply(lambda x: jnp.add(x, 1))

    with handler(
        {
            ejax.is_custom_array: lambda x: isinstance(x, _CustomArray),
            ejax.to_jax_array: lambda x: x.data if isinstance(x, _CustomArray) else x,
            jnp.add: lambda *args, **kwargs: fwd() * 2,
        }
    ):
        np.testing.assert_array_equal(ejax.jit(run)(_CustomArray(data)), [4, 2])


def test_custom_array_host_methods_and_namespace():
    data = jax.numpy.array([1, 2])
    value = _CustomArray(data)
    with handler(
        {
            ejax.is_custom_array: lambda x: isinstance(x, _CustomArray),
            ejax.to_jax_array: lambda x: x.data if isinstance(x, _CustomArray) else x,
        }
    ):
        assert value.item(1) == data.item(1)
        assert value.dtype == data.dtype
        np.testing.assert_array_equal(
            value.searchsorted(jax.numpy.array([2, 3]), method="compare_all"),
            data.searchsorted(jax.numpy.array([2, 3]), method="compare_all"),
        )
    namespace = value.__array_namespace__()
    assert namespace is jnp
    assert namespace.__array_api_version__ == jax.numpy.__array_api_version__
    assert (
        value.__array_namespace__(api_version=namespace.__array_api_version__)
        is namespace
    )
    with pytest.raises(ValueError, match="not available"):
        value.__array_namespace__(api_version="invalid")
    with pytest.raises(NotImplementedError, match="flat"):
        _ = value.flat
    with pytest.raises(TypeError, match="immutable"):
        value[0] = 3
    assert not hasattr(value, "__jax_array__")


def test_custom_array_methods_preserve_whole_operands():
    seen = []

    def add(left, right):
        assert isinstance(right, _CustomArray)
        seen.append("add")
        return _CustomArray(jax.numpy.add(left, right.data))

    def sum_(value, **kwargs):
        assert isinstance(value, _CustomArray)
        seen.append("sum")
        return jax.numpy.sum(value.data, **kwargs)

    other = jax.numpy.ones((2, 2))
    data = jax.numpy.ones((2, 2))

    def run(data):
        return (other + _CustomArray(data)).sum()

    assert not hasattr(_CustomArray(data), "__jax_array__")
    with handler({jnp.add: add, jnp.sum: sum_}):
        assert run(data) == 8
        assert ejax.jit(run)(data) == 8
        np.testing.assert_array_equal(ejax.grad(run)(data), jax.numpy.ones((2, 2)))
    assert seen == ["add", "sum"] * 3


def test_symbolic_array_metadata():
    index = defop(jax.Array, name="index")
    data = jax.numpy.ones((2, 3), dtype=jax.numpy.float32)
    value = jax_getitem(data, (index(),))
    assert value.dtype == data.dtype
    assert value.itemsize == data.itemsize
    assert value.nbytes == 3 * data.itemsize

    variable = defop(jax.Array, name="variable")
    dtype = variable().dtype
    with handler({variable: lambda: data}):
        assert evaluate(dtype) == data.dtype


def test_jax_at_updates():
    """Test .at array update functionality for indexed arrays."""
    i, j, k = defop(jax.Array), defop(jax.Array), defop(jax.Array)

    # Test the exact case from the original issue
    a = jax_getitem(jnp.ones((5, 4, 3)), (i(), j()))
    a = a.at[1].set(0)
    b = jax_getitem(jnp.array([0, 1]), (k(),))
    a = a.at[b].set(0)

    # Verify the result has the expected properties
    assert isinstance(a, Term)
    assert a.shape == (3,)

    # Test with 1D remaining dimension
    arr_2d = jnp.ones((3, 5))
    indexed_2d = jax_getitem(arr_2d, (i(),))  # Shape (5,)
    updated_2d = indexed_2d.at[2].set(99.0)
    assert isinstance(updated_2d, Term)
    assert updated_2d.shape == (5,)

    # Test with 2D remaining dimensions
    arr_3d = jnp.ones((2, 3, 4))
    indexed_3d = jax_getitem(arr_3d, (i(),))  # Shape (3, 4)
    updated_3d = indexed_3d.at[1, 2].set(99.0)
    assert isinstance(updated_3d, Term)
    assert updated_3d.shape == (3, 4)

    # Test using term as index
    arr = jnp.ones((5, 3))
    a = jax_getitem(arr, (i(),))  # Shape (3,)
    k = defop(jax.Array)
    b = jax_getitem(jnp.array([0, 1, 2]), (k(),))  # Shape ()
    updated = a.at[b].set(99.0)
    assert isinstance(updated, Term)
    assert updated.shape == (3,)


def test_jax_len():
    i = defop(jax.Array, name="i")
    with pytest.raises(TypeError):
        len(i())

    t = jnp.ones((2, 3, 4))
    t_i = jax_getitem(t, (i(),))
    assert len(t_i) == 3

    for row in t_i:
        assert len(row) == 4


def test_jax_iter():
    i = defop(jax.Array, name="i")
    with pytest.raises(TypeError):
        tuple(i())


def test_arrayterm_all():
    """Test .all() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[True, False, True], [True, True, True]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.all()
    jnp_result = jnp.all(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_any():
    """Test .any() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[False, False, True], [False, False, False]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.any()
    jnp_result = jnp.any(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_argmax():
    """Test .argmax() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 5, 3], [4, 2, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.argmax()
    jnp_result = jnp.argmax(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_argmin():
    """Test .argmin() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[5, 1, 3], [4, 6, 2]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.argmin()
    jnp_result = jnp.argmin(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_argpartition():
    """Test .argpartition() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[5, 1, 3, 2], [4, 6, 2, 8]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.argpartition(2)
    jnp_result = jnp.argpartition(array_term, 2)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        # argpartition results can vary, so just check they have the same shape
        assert eval_result.shape == eval_jnp_result.shape


def test_arrayterm_argsort():
    """Test .argsort() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[5, 1, 3], [4, 6, 2]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.argsort()
    jnp_result = jnp.argsort(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_astype():
    """Test .astype() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.astype(jnp.float32)
    jnp_result = jnp.astype(array_term, jnp.float32)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert eval_result.dtype == jnp.float32
        assert eval_jnp_result.dtype == jnp.float32


def test_arrayterm_choose():
    """Test .choose() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[0, 1, 2], [1, 0, 2]])
    array_term = jax_getitem(base_array, (i(),))
    choices = [
        jnp.array([10, 20, 30]),
        jnp.array([40, 50, 60]),
        jnp.array([70, 80, 90]),
    ]

    result = array_term.choose(choices, mode="wrap")
    jnp_result = jnp.choose(array_term, choices, mode="wrap")

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_clip():
    """Test .clip() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 5, 3], [4, 2, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.clip(min=2, max=5)
    jnp_result = jnp.clip(array_term, min=2, max=5)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_compress():
    """Test .compress() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))
    condition = jnp.array([True, False, True])

    result = array_term.compress(condition)
    jnp_result = jnp.compress(condition, array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_conj():
    """Test .conj() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.conj()
    jnp_result = jnp.conj(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_conjugate():
    """Test .conjugate() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.conjugate()
    jnp_result = jnp.conjugate(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_copy():
    """Test .copy() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.copy()
    jnp_result = jnp.copy(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_cumprod():
    """Test .cumprod() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [2, 3, 4]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.cumprod()
    jnp_result = jnp.cumprod(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_cumsum():
    """Test .cumsum() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.cumsum()
    jnp_result = jnp.cumsum(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_diagonal():
    """Test .diagonal() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.diagonal()
    jnp_result = jnp.diagonal(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_dot():
    """Test .dot() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))
    other = jnp.array([1, 1, 1])

    result = array_term.dot(other)
    jnp_result = jnp.dot(array_term, other)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_max():
    """Test .max() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 5, 3], [4, 2, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.max()
    jnp_result = jnp.max(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_mean():
    """Test .mean() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.mean()
    jnp_result = jnp.mean(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_min():
    """Test .min() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 5, 3], [4, 2, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.min()
    jnp_result = jnp.min(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_nonzero():
    """Test .nonzero() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[0, 1, 0], [2, 0, 3]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.nonzero(size=4)
    jnp_result = jnp.nonzero(array_term, size=4)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        for r, j in zip(eval_result, eval_jnp_result):
            assert jnp.allclose(r, j)


def test_arrayterm_prod():
    """Test .prod() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [2, 3, 4]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.prod()
    jnp_result = jnp.prod(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_ptp():
    """Test .ptp() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 5, 3], [4, 2, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.ptp()
    jnp_result = jnp.ptp(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_ravel():
    """Test .ravel() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.ravel()
    jnp_result = jnp.ravel(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_repeat():
    """Test .repeat() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.repeat(2)
    jnp_result = jnp.repeat(array_term, 2)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_reshape():
    """Test .reshape() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.reshape((3,))
    jnp_result = jnp.reshape(array_term, (3,))

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_round():
    """Test .round() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1.234, 2.567], [3.891, 4.123]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.round(decimals=1)
    jnp_result = jnp.round(array_term, decimals=1)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_searchsorted():
    """Test .searchsorted() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 3, 5], [2, 4, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.searchsorted(4)
    jnp_result = jnp.searchsorted(array_term, 4)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_sort():
    """Test .sort() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[3, 1, 5], [6, 2, 4]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.sort()
    jnp_result = jnp.sort(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_squeeze():
    """Test .squeeze() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1], [2]], [[3], [4]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.squeeze()
    jnp_result = jnp.squeeze(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_std():
    """Test .std() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.std()
    jnp_result = jnp.std(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_sum():
    """Test .sum() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.sum()
    jnp_result = jnp.sum(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_swapaxes():
    """Test .swapaxes() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.swapaxes(0, 1)
    jnp_result = jnp.swapaxes(array_term, 0, 1)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_take():
    """Test .take() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[10, 20, 30], [40, 50, 60]])
    array_term = jax_getitem(base_array, (i(),))
    indices = jnp.array([0, 2])

    result = array_term.take(indices)
    jnp_result = jnp.take(array_term, indices)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_trace():
    """Test .trace() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.trace()
    jnp_result = jnp.trace(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)


def test_arrayterm_transpose():
    """Test .transpose() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.transpose()
    result_T = array_term.T
    jnp_result = jnp.transpose(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_result_T = evaluate(result_T)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)
        assert jnp.allclose(eval_result_T, eval_jnp_result)


def test_arrayterm_var():
    """Test .var() method on _ArrayTerm."""
    i = defop(jax.Array, name="i")
    base_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_term = jax_getitem(base_array, (i(),))

    result = array_term.var()
    jnp_result = jnp.var(array_term)

    with handler({i: lambda: jnp.array([0, 1])}):
        eval_result = evaluate(result)
        eval_jnp_result = evaluate(jnp_result)
        assert jnp.allclose(eval_result, eval_jnp_result)
