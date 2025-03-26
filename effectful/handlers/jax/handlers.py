import functools
import typing
from collections.abc import Callable, Mapping, Sequence
from types import EllipsisType
from typing import Annotated, TypeVar

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError("JAX is required to use effectful.handlers.jax")

import tree
from typing_extensions import ParamSpec

import effectful.handlers.numbers  # noqa: F401
from effectful.internals.runtime import interpreter
from effectful.internals.tensor_utils import _desugar_tensor_index
from effectful.ops.semantics import apply, evaluate, fvsof
from effectful.ops.syntax import Scoped, defdata, deffn, defop, defterm
from effectful.ops.types import Expr, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
A = TypeVar("A")
B = TypeVar("B")


# + An element of an array index expression.
IndexElement = None | int | slice | Sequence[int] | EllipsisType | jax.Array


def sizesof(value) -> Mapping[Operation[[], jax.Array], int]:
    """Return the sizes of named dimensions in an array expression.

    Sizes are inferred from the array shape.

    :param value: An array expression.
    :return: A mapping from named dimensions to their sizes.

    **Example usage**:

    >>> a, b = defop(jax.Array, name='a'), defop(jax.Array, name='b')
    >>> sizes = sizesof(jnp.ones((2, 3))[a(), b()])
    >>> assert sizes[a] == 2 and sizes[b] == 3
    """
    sizes: dict[Operation[[], jax.Array], int] = {}

    def update_sizes(sizes, op, size):
        old_size = sizes.get(op)
        if old_size is not None and size != old_size:
            raise ValueError(
                f"Named index {op} used in incompatible dimensions of size {old_size} and {size}"
            )
        sizes[op] = size

    def _getitem_sizeof(
        x: Expr[jax.Array], key: tuple[Expr[IndexElement], ...]
    ) -> Expr[jax.Array]:
        if isinstance(x, jax.Array):
            for i, k in enumerate(key):
                if isinstance(k, Term) and len(k.args) == 0 and len(k.kwargs) == 0:
                    update_sizes(sizes, k.op, x.shape[i])

        return defdata(jax_getitem, x, key)

    def _apply(_, op, *args, **kwargs):
        args, kwargs = tree.map_structure(defterm, (args, kwargs))
        return defdata(op, *args, **kwargs)

    with interpreter({jax_getitem: _getitem_sizeof, apply: _apply}):
        evaluate(defterm(value))

    return sizes


def _partial_eval(t: Expr[jax.Array]) -> Expr[jax.Array]:
    """Partially evaluate a term with respect to its sized free variables."""

    sized_fvs = sizesof(t)
    if not sized_fvs:
        return t

    def _is_eager(t):
        return not isinstance(t, Term) or (
            t.op is jax_getitem
            and all(
                isinstance(a, Term) and len(a.args) == 0 and len(a.kwargs) == 0
                for a in t.args[1]
            )
        )

    if not (
        isinstance(t, Term)
        and all(_is_eager(a) for a in tree.flatten((t.args, t.kwargs)))
    ):
        return t

    tpe_jax_fn = jax.vmap(deffn(t, *sized_fvs.keys()))

    # Create indices for each dimension
    indices = jnp.meshgrid(
        *[jnp.arange(size) for size in sized_fvs.values()], indexing="ij"
    )

    # Flatten indices for vmap
    flat_indices = [idx.reshape(-1) for idx in indices]

    # Apply vmap
    flat_result = tpe_jax_fn(*flat_indices)

    def reindex_flat_array(t):
        if not isinstance(t, jax.Array):
            return t

        result_shape = indices[0].shape + t.shape[1:]
        result = jnp.reshape(t, result_shape)
        return jax_getitem(result, tuple(k() for k in sized_fvs.keys()))

    result = tree.map_structure(reindex_flat_array, flat_result)
    return result


@defop
def to_array(
    t: Annotated[jax.Array, Scoped[A | B]],
    *args: Annotated[Operation[[], jax.Array], Scoped[A]],
) -> Annotated[jax.Array, Scoped[B]]:
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
    >>> to_array(t[a(), b()], b, a).shape
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
    perm = [dim_ops.index(o) for o in args] + reindex_dims
    array = jnp.transpose(array, perm)
    return array[(slice(None),) * len(args) + tuple(dims[i] for i in reindex_dims)]


@functools.cache
def _register_jax_op(jax_fn: Callable[P, T]):
    if getattr(jax_fn, "__name__", None) == "__getitem__":
        return jax_getitem

    @defop
    def _jax_op(*args, **kwargs) -> jax.Array:
        tm = defdata(_jax_op, *args, **kwargs)
        sized_fvs = sizesof(tm)

        if (
            _jax_op is jax_getitem
            and not isinstance(args[0], Term)
            and sized_fvs
            and args[1]
            and all(isinstance(k, Term) and k.op in sized_fvs for k in args[1])
        ):
            raise NotImplementedError
        elif sized_fvs and set(sized_fvs.keys()) == fvsof(tm) - {
            jax_getitem,
            _jax_op,
        }:
            # note: this cast is a lie. partial_eval can return non-arrays, as
            # can jax_fn. for example, some jax functions return tuples,
            # which partial_eval handles.
            return typing.cast(jax.Array, _partial_eval(tm))
        elif not any(
            tree.flatten(
                tree.map_structure(lambda x: isinstance(x, Term), (args, kwargs))
            )
        ):
            return typing.cast(jax.Array, jax_fn(*args, **kwargs))
        else:
            raise NotImplementedError

    functools.update_wrapper(_jax_op, jax_fn)
    return _jax_op


@_register_jax_op
def jax_getitem(x: jax.Array, key: tuple[IndexElement, ...]) -> jax.Array:
    """Operation for indexing an array.

    .. note::

      This operation is not intended to be called directly. Instead, it is
      exposed so that it can be handled.

    """
    if not isinstance(x, jax.Array):
        raise TypeError(f"expected an array but got {type(x)}")

    for k in key:
        if isinstance(k, Operation):
            raise TypeError(
                f"Got operation symbol {str(k)}. You probably meant {str(k)}()."
            )

    # fast path for simple cases
    if len(key) == 0:
        return x
    elif not any(isinstance(k, jax.Array) for k in key):
        return x[tuple(key)]

    # handle None, Ellipsis, and missing dimensions
    new_shape, key = _desugar_tensor_index(x.shape, key)
    x = jnp.reshape(x, new_shape)

    # Convert non-array args to arrays and handle advanced indexing
    # JAX's advanced indexing works differently than PyTorch's, so we need to adapt
    indices: list[IndexElement] = []
    for i, k in enumerate(key):
        if isinstance(k, slice):
            if k == slice(None):
                indices.append(k)
            else:
                start = 0 if k.start is None else k.start
                stop = x.shape[i] if k.stop is None else k.stop
                step = 1 if k.step is None else k.step
                indices.append(jnp.arange(start, stop, step))
        elif isinstance(k, int):
            indices.append(k)
        elif isinstance(k, list | tuple):
            indices.append(jnp.array(k))
        elif isinstance(k, jax.Array):
            indices.append(k)
        else:
            indices.append(k)

    return x[tuple(indices)]


# time for crime
old_getitem = jax._src.array.ArrayImpl.__getitem__


def _jax_getitem_override(self, key):
    key_ = key if isinstance(key, tuple) else (key,)
    if any(isinstance(k, Term) for k in key_):
        return jax_getitem(self, key_)
    return old_getitem(self, key)


jax._src.array.ArrayImpl.__getitem__ = _jax_getitem_override  # type: ignore
