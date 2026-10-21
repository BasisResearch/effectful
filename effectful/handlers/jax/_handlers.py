import functools
import typing
from collections.abc import Callable, Mapping
from typing import Annotated

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError("JAX is required to use effectful.handlers.jax")

from effectful.internals.tensor_utils import IndexElement, _BaseSizesofIntp, _sizesof
from effectful.ops.semantics import fvsof, typeof
from effectful.ops.syntax import (
    Scoped,
    _CustomSingleDispatchCallable,
    defdata,
    deffn,
    defop,
    implements,
    syntactic_eq,
)
from effectful.ops.types import Expr, NotHandled, Operation, Term


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


def _partial_eval(t: Expr[jax.Array]) -> Expr[jax.Array]:
    """Partially evaluate a term with respect to its sized free variables."""

    sized_fvs = sizesof(t)
    if not sized_fvs:
        return t

    def _is_eager(t):
        return not isinstance(t, Term) or t.op in sized_fvs or is_eager_array(t)

    if not (
        isinstance(t, Term)
        and all(_is_eager(a) for a in jax.tree.flatten((t.args, t.kwargs))[0])
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

    result = jax.tree.map(reindex_flat_array, flat_result)
    return result


@functools.cache
def _register_jax_op[**P, T](jax_fn: Callable[P, T]):
    if getattr(jax_fn, "__name__", None) == "__getitem__":
        return jax_getitem

    @defop
    def _jax_op(*args, **kwargs) -> jax.Array:
        from ._custom_array import dense_arguments, is_custom_array

        if any(
            is_custom_array(x)
            for x in jax.tree.leaves((args, kwargs), is_leaf=is_custom_array)
        ):
            dense_args, dense_kwargs = dense_arguments(args, kwargs)
            return typing.cast(jax.Array, jax_fn(*dense_args, **dense_kwargs))
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
            jax.tree.flatten(
                jax.tree.map(lambda x: isinstance(x, Term), (args, kwargs))
            )[0]
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
            jax.tree.flatten(
                jax.tree.map(lambda x: isinstance(x, Term), (args, kwargs))
            )[0]
        ):
            from ._custom_array import dense_arguments

            dense_args, dense_kwargs = dense_arguments(args, kwargs)
            return typing.cast(jax.Array, jax_fn(*dense_args, **dense_kwargs))
        else:
            raise NotHandled

    functools.update_wrapper(_jax_op, jax_fn)
    return _jax_op


@_register_jax_op
def jax_getitem(x: jax.Array, key: tuple[IndexElement[jax.Array], ...]) -> jax.Array:
    """Operation for indexing an array. Unlike the standard __getitem__ method,
    this operation correctly handles indexing with terms.

    """
    return x[tuple(key)]


class _SizesofIntp(_BaseSizesofIntp[jax.Array]):
    arr_type: typing.ClassVar[type] = jax.Array

    @implements(jax_getitem)
    def _jax_getitem(self, arr, key):
        return self._getitem(arr, key)


_SIZESOF_INTP = _SizesofIntp()


def sizesof(term: Expr) -> Mapping[Operation[[], jax.Array], int]:
    """Return the sizes of named dimensions in an array expression.

    Sizes are inferred from the array shape.

    :param value: An array expression.
    :return: A mapping from named dimensions to their sizes.

    **Example usage**:

    >>> a, b = defop(jax.Array, name='a'), defop(jax.Array, name='b')
    >>> sizes = sizesof(jax_getitem(jnp.ones((2, 3)), [a(), b()]))
    >>> assert sizes[a] == 2 and sizes[b] == 3
    """
    return _sizesof(term, analysis=_SIZESOF_INTP)


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
    if jax.tree_util.treedef_is_leaf(jax.tree.structure(value)):
        return __dispatch(typeof(value))(value, *names)
    return jax.tree.map(lambda v: bind_dims(v, *names), value)


@defop
@_CustomSingleDispatchCallable
def unbind_dims[T, A, B](
    __dispatch: Callable[[type], Callable[..., T]],
    value: Annotated[T, Scoped[A | B]],
    *names: Annotated[Operation[[], jax.Array], Scoped[B]],
) -> Annotated[T, Scoped[A | B]]:
    """Convert positional dimensions to named dimensions."""
    if jax.tree_util.treedef_is_leaf(jax.tree.structure(value)):
        return __dispatch(typeof(value))(value, *names)
    return jax.tree.map(lambda v: unbind_dims(v, *names), value)


@syntactic_eq.register
def _(x: jax.Array, other) -> bool:
    return (
        isinstance(other, jax.Array)
        and x.shape == other.shape
        and bool((jnp.asarray(x) == jnp.asarray(other)).all())
    )
