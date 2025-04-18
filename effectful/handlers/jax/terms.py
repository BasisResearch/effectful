import functools
import operator
import typing
from typing import Any, Sequence

import jax
import tree

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax.handlers import (
    IndexElement,
    _partial_eval,
    is_eager_array,
    jax_getitem,
)
from effectful.internals.tensor_utils import _desugar_tensor_index
from effectful.ops.dims import _bind_dims, _unbind_dims, bind_dims, unbind_dims
from effectful.ops.syntax import defdata
from effectful.ops.types import Expr, Operation, Term


@defdata.register(jax.Array)
def _embed_array(op, *args, **kwargs):
    if is_eager_array(op, *args, **kwargs):
        return _EagerArrayTerm(jax_getitem, args[0], args[1])
    else:
        return _ArrayTerm(op, *args, **kwargs)


class _ArrayTerm(Term[jax.Array]):
    def __init__(self, op: Operation[..., jax.Array], *args: Expr, **kwargs: Expr):
        self._op = op
        self._args = args
        self._kwargs = kwargs

    @property
    def op(self) -> Operation[..., jax.Array]:
        return self._op

    @property
    def args(self) -> tuple:
        return self._args

    @property
    def kwargs(self) -> dict:
        return self._kwargs

    def __getitem__(
        self, key: Expr[IndexElement] | tuple[Expr[IndexElement], ...]
    ) -> Expr[jax.Array]:
        return jax_getitem(self, key if isinstance(key, tuple) else (key,))

    @property
    def shape(self) -> Expr[tuple[int, ...]]:
        return jnp.shape(self)  # type: ignore [attr-defined]

    @property
    def size(self) -> Expr[int]:
        return jnp.size(self)  # type: ignore [attr-defined]

    @property
    def ndim(self) -> Expr[int]:
        return jnp.ndim(self)  # type: ignore [attr-defined]

    def __add__(self, other: jax.Array) -> jax.Array:
        return jnp.add(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __radd__(self, other: jax.Array) -> jax.Array:
        return jnp.add(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __neg__(self) -> jax.Array:
        return jnp.negative(typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __pos__(self) -> jax.Array:
        return typing.cast(jax.Array, jnp.positive(self))  # type: ignore [attr-defined]

    def __sub__(self, other: jax.Array) -> jax.Array:
        return jnp.subtract(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rsub__(self, other: jax.Array) -> jax.Array:
        return jnp.subtract(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __mul__(self, other: jax.Array) -> jax.Array:
        return jnp.multiply(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rmul__(self, other: jax.Array) -> jax.Array:
        return jnp.multiply(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __truediv__(self, other: jax.Array) -> jax.Array:
        return jnp.divide(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rtruediv__(self, other: jax.Array) -> jax.Array:
        return jnp.divide(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __pow__(self, other: jax.Array) -> jax.Array:
        return jnp.power(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rpow__(self, other: jax.Array) -> jax.Array:
        return jnp.power(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __abs__(self) -> jax.Array:
        return jnp.abs(typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __eq__(self, other: Any):
        return jnp.equal(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __ne__(self, other: Any):
        return jnp.not_equal(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __floordiv__(self, other: jax.Array) -> jax.Array:
        return jnp.floor_divide(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rfloordiv__(self, other: jax.Array) -> jax.Array:
        return jnp.floor_divide(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __mod__(self, other: jax.Array) -> jax.Array:
        return jnp.mod(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rmod__(self, other: jax.Array) -> jax.Array:
        return jnp.mod(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __lt__(self, other: jax.Array) -> jax.Array:
        return jnp.less(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __le__(self, other: jax.Array) -> jax.Array:
        return jnp.less_equal(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __gt__(self, other: jax.Array) -> jax.Array:
        return jnp.greater(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __ge__(self, other: jax.Array) -> jax.Array:
        return jnp.greater_equal(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __lshift__(self, other: jax.Array) -> jax.Array:
        return jnp.left_shift(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rlshift__(self, other: jax.Array) -> jax.Array:
        return jnp.left_shift(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __rshift__(self, other: jax.Array) -> jax.Array:
        return jnp.right_shift(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rrshift__(self, other: jax.Array) -> jax.Array:
        return jnp.right_shift(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __and__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_and(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rand__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_and(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __xor__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_xor(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rxor__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_xor(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __or__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_or(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __ror__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_or(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __invert__(self) -> jax.Array:
        return jnp.bitwise_not(typing.cast(jax.Array, self))  # type: ignore [attr-defined]

    def __matmul__(self, other: jax.Array) -> jax.Array:
        return jnp.matmul(typing.cast(jax.Array, self), other)  # type: ignore [attr-defined]

    def __rmatmul__(self, other: jax.Array) -> jax.Array:
        return jnp.matmul(other, typing.cast(jax.Array, self))  # type: ignore [attr-defined]


class _EagerArrayTerm(_ArrayTerm):
    def __init__(self, op, tensor, key):
        new_shape, new_key = _desugar_tensor_index(tensor.shape, key)
        super().__init__(op, jnp.reshape(tensor, new_shape), new_key)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            s
            for s, k in zip(self.args[0].shape, self.args[1])
            if not isinstance(k, Term)
        )

    @property
    def size(self) -> int:
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def ndim(self) -> int:
        return len(self.shape)


@_bind_dims.register(_ArrayTerm)
@_bind_dims.register(_EagerArrayTerm)
@_bind_dims.register(jax.Array)
def _bind_dims_array(t: jax.Array, *args: Operation[[], jax.Array]) -> jax.Array:
    """Convert named dimensions to positional dimensions.

    :param t: An array.
    :type t: T
    :param args: Named dimensions to convert to positional dimensions.
                  These positional dimensions will appear at the beginning of the
                  shape.
    :type args: Operation[[], jax.Array]
    :return: An array with the named dimensions in ``args`` converted to positional dimensions.

    **Example usage**:

    >>> a, b = defop(jax.Array, name='a'), defop(jax.Array, name='b')
    >>> t = jnp.ones((2, 3))
    >>> bind_dims(t[a(), b()], b, a).shape
    (3, 2)
    """

    def _evaluate(expr):
        if isinstance(expr, Term):
            (args, kwargs) = tree.map_structure(_evaluate, (expr.args, expr.kwargs))
            return _partial_eval(expr)
        if tree.is_nested(expr):
            return tree.map_structure(_evaluate, expr)
        return expr

    if not isinstance(t, Term):
        return t

    result = _evaluate(t)
    if not isinstance(result, Term) or not args:
        return result

    # ensure that the result is a jax_getitem with an array as the first argument
    if not (result.op is jax_getitem and isinstance(result.args[0], jax.Array)):
        raise NotImplementedError

    array = result.args[0]
    dims = result.args[1]
    assert isinstance(dims, Sequence)

    # ensure that the order is a subset of the named dimensions
    order_set = set(args)
    if not order_set <= set(a.op for a in dims if isinstance(a, Term)):
        raise NotImplementedError

    # permute the inner array so that the leading dimensions are in the order
    # specified and the trailing dimensions are the remaining named dimensions
    # (or slices)
    reindex_dims = [
        i
        for i, o in enumerate(dims)
        if not isinstance(o, Term) or o.op not in order_set
    ]
    dim_ops = [a.op if isinstance(a, Term) else None for a in dims]
    perm = (
        [dim_ops.index(o) for o in args]
        + reindex_dims
        + list(range(len(dims), len(array.shape)))
    )
    array = jnp.transpose(array, perm)
    reindexed = jax_getitem(
        array, (slice(None),) * len(args) + tuple(dims[i] for i in reindex_dims)
    )
    return reindexed


@_unbind_dims.register(_ArrayTerm)
@_unbind_dims.register(_EagerArrayTerm)
@_unbind_dims.register(jax.Array)
def _unbind_dims_array(t: jax.Array, *args: Operation[[], jax.Array]) -> jax.Array:
    return jax_getitem(t, tuple(n() for n in args))
