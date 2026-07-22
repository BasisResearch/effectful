import functools
import itertools
import typing
from collections.abc import Callable, Mapping, Sequence
from types import EllipsisType
from typing import Annotated

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError("JAX is required to use effectful.handlers.jax")

from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, evaluate, fvsof, typeof
from effectful.ops.syntax import (
    CollectionConstrOperation,
    Scoped,
    _BaseTerm,
    _CustomSingleDispatchCallable,
    defdata,
    deffn,
    defop,
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
    from effectful.internals.product_n import _unpack, argsof, productN

    # Analysis for type computation and term reconstruction
    _sizes = defop(object, name="sizes")
    _getitem_term = defop(object, name="getitem_args")

    def _retain(op, *args, **kwargs):
        # Non-getitem subterms are opaque to this analysis. Keeping their
        # arguments would retain the entire input term unnecessarily.
        return _BaseTerm(op)

    def _retain_getitem(*args, **kwargs):
        return defdata(jax_getitem, *args, **kwargs)

    def _merge(s1, s2):
        s3 = s1.copy()
        for k, v in s2.items():
            if k in s3 and s3[k] != v:
                raise ValueError(
                    f"Named index {k} used in incompatible dimensions of size {s3[k]} and {v}"
                )
            s3[k] = v
        return s3

    def _apply_sizes(op, *args, **kwargs):
        analyses = (x for x in (*args, *kwargs.values()) if isinstance(x, dict))
        return functools.reduce(_merge, analyses, {})

    def _getitem(arr, index):
        # Inspect this getitem's arguments in the term projection without
        # forcing that projection to retain the getitem result.
        term_args, _ = argsof(_getitem_term)
        term_arr, term_index = term_args

        arg_sizes = (x for x in (arr, index) if isinstance(x, dict))
        if not is_eager_array(term_arr):
            return functools.reduce(_merge, arg_sizes, {})

        sizes = (
            {k.op: term_arr.shape[i]}
            for i, k in enumerate(term_index)
            if isinstance(k, Term) and len(k.args) == 0 and len(k.kwargs) == 0
        )
        return functools.reduce(_merge, itertools.chain(arg_sizes, sizes), {})

    _intp = productN(
        {
            _sizes: {apply: _apply_sizes, jax_getitem: _getitem},
            _getitem_term: {
                apply: _retain,
                jax_getitem: _retain_getitem,
                CollectionConstrOperation.__apply__: apply.__default_rule__,
            },
        }
    )

    with interpreter(_intp):
        result = evaluate(term)

    fvs = _unpack(result, _sizes)
    if not isinstance(fvs, dict):
        return {}
    return fvs


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


def jit(f, *args, **kwargs):
    f_noindex, f_reindex = _indexed_func_wrapper(f, jax_getitem, sizesof)
    f_noindex_jitted = jax.jit(f_noindex, *args, **kwargs)
    return lambda *args, **kwargs: f_reindex(f_noindex_jitted(*args, **kwargs))


def _indexed_func_wrapper[**P, S, T](
    func: Callable[P, T], getitem, sizesof
) -> tuple[Callable[P, S], Callable[[S], T]]:
    # index expressions for the result of the function
    indexes = None

    # hide index lists from jax.tree.mapping
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
        indexes = jax.tree.map(lambda t: Indexes(sizesof(t)), ret)
        tensors = jax.tree.map(lambda t, i: deindex_tensor(t, i), ret, indexes)
        return tensors

    # reapply the stored indexes to a result
    def reindex(ret, starting_dim=0):
        def index_expr(i):
            return (slice(None),) * (starting_dim) + tuple(x() for x in i.indexes)

        indexed_ret = jax.tree.map(lambda t, i: getitem(t, index_expr(i)), ret, indexes)

        return indexed_ret

    return deindexed, reindex


@syntactic_eq.register
def _(x: jax.Array, other) -> bool:
    return (
        isinstance(other, jax.Array)
        and x.shape == other.shape
        and bool((jnp.asarray(x) == jnp.asarray(other)).all())
    )
