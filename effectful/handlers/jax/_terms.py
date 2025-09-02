import functools
import operator
from collections.abc import Sequence
from typing import Any, cast

import jax
import tree

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax._handlers import (
    IndexElement,
    _partial_eval,
    _register_jax_op,
    bind_dims,
    jax_getitem,
    unbind_dims,
)
from effectful.internals.tensor_utils import _desugar_tensor_index
from effectful.ops.syntax import defdata
from effectful.ops.types import Expr, NotHandled, Operation, Term


class _IndexUpdateHelper:
    """Helper class to implement array-style .at[index].set() updates for effectful arrays."""

    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        return _IndexUpdateRef(self.array, key)


class _IndexUpdateRef:
    """Reference to an array position for updates via .at[index]."""

    def __init__(self, array, key):
        self.array = array
        self.key = key

    def set(self, value):
        """Set values at the indexed positions."""

        # Create a JAX at operation that properly handles the indexing
        @_register_jax_op
        def jax_at_set(arr, index_key, val):
            # JAX's at expects the index to be unpacked correctly
            if isinstance(index_key, tuple) and len(index_key) == 1:
                # Single index case
                return arr.at[index_key[0]].set(val)
            elif isinstance(index_key, tuple):
                # Multiple indices case
                return arr.at[index_key].set(val)
            else:
                # Direct index case
                return arr.at[index_key].set(val)

        return jax_at_set(self.array, self.key, value)


@defdata.register(jax.Array)
def _embed_array(op, *args, **kwargs):
    if (
        op is jax_getitem
        and not isinstance(args[0], Term)
        and all(not k.args and not k.kwargs for k in args[1] if isinstance(k, Term))
    ):
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
        return jnp.shape(cast(jax.Array, self))

    @property
    def size(self) -> Expr[int]:
        return jnp.size(cast(jax.Array, self))

    def __len__(self):
        return self.shape[0]

    @property
    def ndim(self) -> Expr[int]:
        return jnp.ndim(cast(jax.Array, self))

    def __add__(self, other: jax.Array) -> jax.Array:
        return jnp.add(cast(jax.Array, self), other)

    def __radd__(self, other: jax.Array) -> jax.Array:
        return jnp.add(other, cast(jax.Array, self))

    def __neg__(self) -> jax.Array:
        return jnp.negative(cast(jax.Array, self))

    def __pos__(self) -> jax.Array:
        return jnp.positive(cast(jax.Array, self))

    def __sub__(self, other: jax.Array) -> jax.Array:
        return jnp.subtract(cast(jax.Array, self), other)

    def __rsub__(self, other: jax.Array) -> jax.Array:
        return jnp.subtract(other, cast(jax.Array, self))

    def __mul__(self, other: jax.Array) -> jax.Array:
        return jnp.multiply(cast(jax.Array, self), other)

    def __rmul__(self, other: jax.Array) -> jax.Array:
        return jnp.multiply(other, cast(jax.Array, self))

    def __truediv__(self, other: jax.Array) -> jax.Array:
        return jnp.divide(cast(jax.Array, self), other)

    def __rtruediv__(self, other: jax.Array) -> jax.Array:
        return jnp.divide(other, cast(jax.Array, self))

    def __pow__(self, other: jax.Array) -> jax.Array:
        return jnp.power(cast(jax.Array, self), other)

    def __rpow__(self, other: jax.Array) -> jax.Array:
        return jnp.power(other, cast(jax.Array, self))

    def __abs__(self) -> jax.Array:
        return jnp.abs(cast(jax.Array, self))

    def __eq__(self, other: Any):
        return jnp.equal(cast(jax.Array, self), other)

    def __ne__(self, other: Any):
        return jnp.not_equal(cast(jax.Array, self), other)

    def __floordiv__(self, other: jax.Array) -> jax.Array:
        return jnp.floor_divide(cast(jax.Array, self), other)

    def __rfloordiv__(self, other: jax.Array) -> jax.Array:
        return jnp.floor_divide(other, cast(jax.Array, self))

    def __mod__(self, other: jax.Array) -> jax.Array:
        return jnp.mod(cast(jax.Array, self), other)

    def __rmod__(self, other: jax.Array) -> jax.Array:
        return jnp.mod(other, cast(jax.Array, self))

    def __lt__(self, other: jax.Array) -> jax.Array:
        return jnp.less(cast(jax.Array, self), other)

    def __le__(self, other: jax.Array) -> jax.Array:
        return jnp.less_equal(cast(jax.Array, self), other)

    def __gt__(self, other: jax.Array) -> jax.Array:
        return jnp.greater(cast(jax.Array, self), other)

    def __ge__(self, other: jax.Array) -> jax.Array:
        return jnp.greater_equal(cast(jax.Array, self), other)

    def __lshift__(self, other: jax.Array) -> jax.Array:
        return jnp.left_shift(cast(jax.Array, self), other)

    def __rlshift__(self, other: jax.Array) -> jax.Array:
        return jnp.left_shift(other, cast(jax.Array, self))

    def __rshift__(self, other: jax.Array) -> jax.Array:
        return jnp.right_shift(cast(jax.Array, self), other)

    def __rrshift__(self, other: jax.Array) -> jax.Array:
        return jnp.right_shift(other, cast(jax.Array, self))

    def __and__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_and(cast(jax.Array, self), other)

    def __rand__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_and(other, cast(jax.Array, self))

    def __xor__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_xor(cast(jax.Array, self), other)

    def __rxor__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_xor(other, cast(jax.Array, self))

    def __or__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_or(cast(jax.Array, self), other)

    def __ror__(self, other: jax.Array) -> jax.Array:
        return jnp.bitwise_or(other, cast(jax.Array, self))

    def __invert__(self) -> jax.Array:
        return jnp.bitwise_not(cast(jax.Array, self))

    def __matmul__(self, other: jax.Array) -> jax.Array:
        return jnp.matmul(cast(jax.Array, self), other)

    def __rmatmul__(self, other: jax.Array) -> jax.Array:
        return jnp.matmul(other, cast(jax.Array, self))

    @property
    def at(self) -> _IndexUpdateHelper:
        """Return an IndexUpdateHelper for array updates."""
        return _IndexUpdateHelper(self)

    def __iter__(self):
        raise TypeError("A free array is not iterable.")

    def all(self, axis=None, keepdims=False, *, where=None):
        return jnp.all(cast(jax.Array, self), axis=axis, keepdims=keepdims, where=where)

    def any(self, axis=None, keepdims=False, *, where=None):
        return jnp.any(cast(jax.Array, self), axis=axis, keepdims=keepdims, where=where)

    def argmax(self, axis=None, keepdims=False):
        return jnp.argmax(cast(jax.Array, self), axis=axis, keepdims=keepdims)

    def argmin(self, axis=None, keepdims=False):
        return jnp.argmin(cast(jax.Array, self), axis=axis, keepdims=keepdims)

    def argpartition(self, kth, axis=-1):
        return jnp.argpartition(cast(jax.Array, self), kth, axis=axis)

    def argsort(self, axis=-1, descending=False, stable=True):
        return jnp.argsort(
            cast(jax.Array, self), axis=axis, descending=descending, stable=stable
        )

    def astype(self, dtype):
        return jnp.astype(cast(jax.Array, self), dtype)

    def choose(self, choices, mode="raise"):
        return jnp.choose(cast(jax.Array, self), choices, mode=mode)

    def clip(self, min=None, max=None):
        return jnp.clip(cast(jax.Array, self), min=min, max=max)

    def compress(self, condition, axis=None):
        return jnp.compress(condition, cast(jax.Array, self), axis=axis)

    def conj(self):
        return jnp.conj(cast(jax.Array, self))

    def conjugate(self):
        return jnp.conjugate(cast(jax.Array, self))

    def copy(self):
        return jnp.copy(cast(jax.Array, self))

    def cumprod(self, axis=None, dtype=None):
        return jnp.cumprod(cast(jax.Array, self), axis=axis, dtype=dtype)

    def cumsum(self, axis=None, dtype=None):
        return jnp.cumsum(cast(jax.Array, self), axis=axis, dtype=dtype)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return jnp.diagonal(
            cast(jax.Array, self), offset=offset, axis1=axis1, axis2=axis2
        )

    def dot(self, b):
        return jnp.dot(cast(jax.Array, self), b)

    def max(self, axis=None, keepdims=False, initial=None, where=None):
        return jnp.max(
            cast(jax.Array, self),
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def mean(self, axis=None, keepdims=False, *, where=None):
        return jnp.mean(
            cast(jax.Array, self), axis=axis, keepdims=keepdims, where=where
        )

    def min(self, axis=None, keepdims=False, initial=None, where=None):
        return jnp.min(
            cast(jax.Array, self),
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def nonzero(self, *, size=None, fill_value=None):
        return jnp.nonzero(cast(jax.Array, self), size=size, fill_value=fill_value)

    def prod(
        self, axis=None, keepdims=False, initial=None, where=None, promote_integers=True
    ):
        return jnp.prod(
            cast(jax.Array, self),
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            where=where,
            promote_integers=promote_integers,
        )

    def ptp(self, axis=None, keepdims=False):
        return jnp.ptp(cast(jax.Array, self), axis=axis, keepdims=keepdims)

    def ravel(self, order="C"):
        return jnp.ravel(cast(jax.Array, self), order=order)

    def repeat(self, repeats, axis=None, *, total_repeat_length=None):
        return jnp.repeat(
            cast(jax.Array, self),
            repeats,
            axis=axis,
            total_repeat_length=total_repeat_length,
        )

    def reshape(self, *shape, order="C"):
        if len(shape) == 1 and isinstance(shape[0], tuple | list):
            shape = shape[0]
        return jnp.reshape(cast(jax.Array, self), shape, order=order)

    def round(self, decimals=0):
        return jnp.round(cast(jax.Array, self), decimals=decimals)

    def searchsorted(self, v, side="left", sorter=None):
        return jnp.searchsorted(cast(jax.Array, self), v, side=side, sorter=sorter)

    def sort(self, axis=-1, descending=False, stable=True):
        return jnp.sort(
            cast(jax.Array, self), axis=axis, descending=descending, stable=stable
        )

    def squeeze(self, axis=None):
        return jnp.squeeze(cast(jax.Array, self), axis=axis)

    def std(self, axis=None, keepdims=False, ddof=0, *, where=None):
        return jnp.std(
            cast(jax.Array, self), axis=axis, keepdims=keepdims, ddof=ddof, where=where
        )

    def sum(
        self, axis=None, keepdims=False, initial=None, where=None, promote_integers=True
    ):
        return jnp.sum(
            cast(jax.Array, self),
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            where=where,
            promote_integers=promote_integers,
        )

    def swapaxes(self, axis1, axis2):
        return jnp.swapaxes(cast(jax.Array, self), axis1, axis2)

    def take(
        self,
        indices,
        axis=None,
        mode=None,
        unique_indices=False,
        indices_are_sorted=False,
        fill_value=None,
    ):
        return jnp.take(
            cast(jax.Array, self),
            indices,
            axis=axis,
            mode=mode,
            unique_indices=unique_indices,
            indices_are_sorted=indices_are_sorted,
            fill_value=fill_value,
        )

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        return jnp.trace(
            cast(jax.Array, self), offset=offset, axis1=axis1, axis2=axis2, dtype=dtype
        )

    def transpose(self, axes=None):
        return jnp.transpose(cast(jax.Array, self), axes=axes)

    def var(self, axis=None, keepdims=False, ddof=0, *, where=None):
        return jnp.var(
            cast(jax.Array, self), axis=axis, keepdims=keepdims, ddof=ddof, where=where
        )


class _EagerArrayTerm(_ArrayTerm):
    def __init__(self, op, tensor, key):
        new_shape, new_key = _desugar_tensor_index(tensor.shape, key)
        super().__init__(op, jax.numpy.reshape(tensor, new_shape), new_key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

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


@bind_dims.register  # type: ignore
def _bind_dims_array(t: jax.Array, *args: Operation[[], jax.Array]) -> jax.Array:
    """Convert named dimensions to positional dimensions.

    :param t: An array.
    :param args: Named dimensions to convert to positional dimensions.
                  These positional dimensions will appear at the beginning of the
                  shape.
    :return: An array with the named dimensions in ``args`` converted to positional dimensions.

    **Example usage**:

    >>> from effectful.ops.syntax import defop
    >>> from effectful.handlers.jax import bind_dims, unbind_dims
    >>> a, b = defop(jax.Array, name='a'), defop(jax.Array, name='b')
    >>> t = unbind_dims(jnp.ones((2, 3)), a, b)
    >>> bind_dims(t, b, a).shape
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
        raise NotHandled

    array = result.args[0]
    dims = result.args[1]
    assert isinstance(dims, Sequence)

    # ensure that the order is a subset of the named dimensions
    order_set = set(args)
    if not order_set <= set(a.op for a in dims if isinstance(a, Term)):
        raise NotHandled

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


@unbind_dims.register  # type: ignore
def _unbind_dims_array(t: jax.Array, *args: Operation[[], jax.Array]) -> jax.Array:
    return jax_getitem(t, tuple(n() for n in args))
