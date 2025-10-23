import functools
import typing
from collections.abc import Callable, Mapping, Sequence
from types import EllipsisType
from typing import Annotated

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError("JAX is required to use effectful.handlers.jax")

import tree

from effectful.ops.semantics import fvsof, typeof
from effectful.ops.syntax import (
    Scoped,
    _CustomSingleDispatchCallable,
    defdata,
    deffn,
    defop,
    defterm,
    syntactic_eq,
)
from effectful.ops.types import Expr, NotHandled, Operation, Term

# + An element of an array index expression.
IndexElement = None | int | slice | Sequence[int] | EllipsisType | jax.Array


def is_eager_array(x):
    return isinstance(x, jax.Array) or (
        isinstance(x, Term)
        and x.op is jax_getitem
        and isinstance(x.args[0], jax.Array)
        and all(
            (not isinstance(k, Term)) or (not k.args and not k.kwargs)
            for k in x.args[1]
        )
        and not x.kwargs
    )


def sizesof(value) -> Mapping[Operation[[], jax.Array], int]:
    """Return the sizes of named dimensions in an array expression.

    Sizes are inferred from the array shape.

    :param value: An array expression.
    :return: A mapping from named dimensions to their sizes.

    **Example usage**:

    >>> a, b = defop(jax.Array, name='a'), defop(jax.Array, name='b')
    >>> sizes = sizesof(jax_getitem(jnp.ones((2, 3)), [a(), b()]))
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

    def _getitem_sizeof(x: jax.Array, key: tuple[Expr[IndexElement], ...]):
        if is_eager_array(x):
            for i, k in enumerate(key):
                if isinstance(k, Term) and len(k.args) == 0 and len(k.kwargs) == 0:
                    update_sizes(sizes, k.op, x.shape[i])

    def _sizesof(expr):
        expr = defterm(expr)
        if isinstance(expr, Term):
            for x in tree.flatten((expr.args, expr.kwargs)):
                _sizesof(x)
            if expr.op is jax_getitem:
                _getitem_sizeof(*expr.args)
        elif tree.is_nested(expr):
            for x in tree.flatten(expr):
                _sizesof(x)

    _sizesof(value)
    return sizes


def _partial_eval(t: Expr[jax.Array]) -> Expr[jax.Array]:
    """Partially evaluate a term with respect to its sized free variables."""

    sized_fvs = sizesof(t)
    if not sized_fvs:
        return t

    def _is_eager(t):
        return not isinstance(t, Term) or t.op in sized_fvs or is_eager_array(t)

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


@functools.cache
def _register_jax_op[**P, T](jax_fn: Callable[P, T]):
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
            raise NotHandled
        elif sized_fvs and set(sized_fvs.keys()) == fvsof(tm) - {jax_getitem, _jax_op}:
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
            raise NotHandled

    functools.update_wrapper(_jax_op, jax_fn)
    return _jax_op


@functools.cache
def _register_jax_op_no_partial_eval[**P, T](jax_fn: Callable[P, T]):
    # FIXME: Presumably not all jax ops return arrays. In other cases, we won't
    # get the right kind of term.
    @defop
    def _jax_op(*args, **kwargs) -> jax.Array:
        if not any(
            tree.flatten(
                tree.map_structure(lambda x: isinstance(x, Term), (args, kwargs))
            )
        ):
            return typing.cast(jax.Array, jax_fn(*args, **kwargs))
        else:
            raise NotHandled

    functools.update_wrapper(_jax_op, jax_fn)
    return _jax_op


@_register_jax_op
def jax_getitem(x: jax.Array, key: tuple[IndexElement, ...]) -> jax.Array:
    """Operation for indexing an array. Unlike the standard __getitem__ method,
    this operation correctly handles indexing with terms.

    """
    return x[tuple(key)]


@defop
@_CustomSingleDispatchCallable
def bind_dims[T, A, B](
    __dispatch: Callable[[type], Callable[..., T]],
    value: Annotated[T, Scoped[A | B]],
    *names: Annotated[Operation[[], jax.Array], Scoped[B]],
) -> Annotated[T, Scoped[A]]:
    """Convert named dimensions to positional dimensions.

    :param t: An array.
    :param args: Named dimensions to convert to positional dimensions.
                  These positional dimensions will appear at the beginning of the
                  shape.
    :return: An array with the named dimensions in ``args`` converted to positional dimensions.

    **Example usage**:

    >>> import jax.numpy as jnp
    >>> from effectful.ops.syntax import defop
    >>> a, b = defop(jax.Array, name='a'), defop(jax.Array, name='b')
    >>> t = jax_getitem(jnp.ones((2, 3)), [a(), b()])
    >>> bind_dims(t, b, a).shape
    (3, 2)
    """
    if tree.is_nested(value):
        return tree.map_structure(lambda v: bind_dims(v, *names), value)

    semantic_type = typeof(value)
    return __dispatch(semantic_type)(value, *names)


@defop
@_CustomSingleDispatchCallable
def unbind_dims[T, A, B](
    __dispatch: Callable[[type], Callable[..., T]],
    value: Annotated[T, Scoped[A | B]],
    *names: Annotated[Operation[[], jax.Array], Scoped[B]],
) -> Annotated[T, Scoped[A | B]]:
    """Convert positional dimensions to named dimensions."""
    if tree.is_nested(value):
        return tree.map_structure(lambda v: unbind_dims(v, *names), value)

    semantic_type = typeof(value)
    return __dispatch(semantic_type)(value, *names)


def jit(f, *args, **kwargs):
    f_noindex, f_reindex = _indexed_func_wrapper(f, jax_getitem, sizesof)
    f_noindex_jitted = jax.jit(f_noindex, *args, **kwargs)
    return lambda *args, **kwargs: f_reindex(f_noindex_jitted(*args, **kwargs))


def _indexed_func_wrapper[**P, S, T](
    func: Callable[P, T], getitem, sizesof
) -> tuple[Callable[P, S], Callable[[S], T]]:
    # index expressions for the result of the function
    indexes = None

    # hide index lists from tree.map_structure
    class Indexes:
        def __init__(self, sizes):
            self.sizes = sizes
            self.indexes = list(sizes.keys())

    # strip named indexes from the result of the function and store them
    def deindexed(*args, **kwargs):
        nonlocal indexes

        def deindex_tensor(t, i):
            t_ = bind_dims(t, *i.sizes.keys())
            assert all(t_.shape[j] == i.sizes[v] for j, v in enumerate(i.sizes))
            return t_

        ret = func(*args, **kwargs)
        indexes = tree.map_structure(lambda t: Indexes(sizesof(t)), ret)
        tensors = tree.map_structure(lambda t, i: deindex_tensor(t, i), ret, indexes)
        return tensors

    # reapply the stored indexes to a result
    def reindex(ret, starting_dim=0):
        def index_expr(i):
            return (slice(None),) * (starting_dim) + tuple(x() for x in i.indexes)

        if tree.is_nested(ret):
            indexed_ret = tree.map_structure(
                lambda t, i: getitem(t, index_expr(i)), ret, indexes
            )
        else:
            indexed_ret = getitem(ret, index_expr(indexes))

        return indexed_ret

    return deindexed, reindex


@syntactic_eq.register
def _(x: jax.typing.ArrayLike, other) -> bool:
    return isinstance(other, jax.typing.ArrayLike) and bool(  # type: ignore[arg-type]
        (jnp.asarray(x) == jnp.asarray(other)).all()
    )
