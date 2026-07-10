import functools
import itertools
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import EllipsisType
from typing import Annotated, Any

from opt_einsum import get_symbol
from opt_einsum.parser import parse_einsum_input

try:
    import jax
    import jax.core
    import jax.numpy as jnp
except ImportError:
    raise ImportError("JAX is required to use effectful.handlers.jax")

from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, evaluate, fvsof, fwd, typeof
from effectful.ops.syntax import (
    Scoped,
    _CustomSingleDispatchCallable,
    defdata,
    deffn,
    defop,
    syntactic_eq,
    syntactic_hash,
)
from effectful.ops.types import Expr, NotHandled, Operation, Term

# + An element of an array index expression.
IndexElement = None | int | slice | Sequence[int] | EllipsisType | jax.Array


class JaxOperation[**P, T](Operation[P, T]): ...


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
            i = 0
            for k in key:
                if isinstance(k, Term) and len(k.args) == 0 and len(k.kwargs) == 0:
                    update_sizes(sizes, k.op, x.shape[i])
                if k is not None:
                    i += 1

        return defdata(jax_getitem, x, key)

    def _apply(op, *args, **kwargs):
        return defdata(op, *args, **kwargs)

    with interpreter({jax_getitem: _getitem_sizeof, apply: _apply}):
        evaluate(value)

    return sizes


def _partial_eval(t: Expr[jax.Array]) -> Expr[jax.Array]:
    """Partially evaluate a term with respect to its sized free variables."""

    sized_fvs = sizesof(t)
    if not sized_fvs:
        return t

    # if any dimension is zero sized, the result is empty
    if any(size == 0 for size in sized_fvs.values()):
        ops = tuple(sized_fvs.keys())
        key = tuple(k() for k in ops)
        shape = tuple(sized_fvs[k] for k in ops)
        return jax_getitem(jnp.empty(shape), key)

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

    @JaxOperation.define
    def _jax_op(*args, **kwargs) -> jax.Array:
        tm = defdata(_jax_op, *args, **kwargs)
        sized_fvs = sizesof(tm)

        if (
            _jax_op is jax_getitem
            and not isinstance(args[0], Term)
            and sized_fvs
            and args[1]
            and all(
                (isinstance(k, Term) and k.op in sized_fvs)
                or (isinstance(k, slice) and k == slice(None))
                for k in args[1]
            )
        ):
            raise NotHandled
        elif _jax_op is jax_getitem and not args[1]:
            return args[0]
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


def _named_dims(term: Expr[jax.Array]) -> tuple[Operation, ...]:
    if not (isinstance(term, Term) and term.op == jax_getitem):
        return ()
    index = term.args[1]
    assert isinstance(index, Iterable)
    return tuple(i.op for i in index if isinstance(i, Term) and not i.args)


def _reduce_named(array, axis=None, **kwargs) -> jax.Array:
    if axis is None:
        return fwd()

    named_dims = _named_dims(array)
    if not named_dims:
        return fwd()

    bound_arr = bind_dims(array, *named_dims)

    if isinstance(axis, int):
        axis = (axis,)
    shifted_axis = tuple(a + len(named_dims) if a >= 0 else a for a in axis)

    reduced = fwd(bound_arr, axis=shifted_axis, **kwargs)
    return unbind_dims(reduced, *named_dims)


def _einsum_named(subscripts, *operands, **kwargs) -> jax.Array:
    # only the string-subscripts form is handled; forward the interleaved form
    if not isinstance(subscripts, str):
        if any(isinstance(x, Term) for x in (subscripts, *operands)):
            raise ValueError("Interleaved einsum is not implemented with named tensors")
        return jax.numpy.einsum(subscripts, *operands, **kwargs)

    # forward if any operand has a symbolic (Term) shape
    if any(isinstance(arr, Term) and not is_eager_array(arr) for arr in operands):
        raise NotHandled

    named = [_named_dims(op) for op in operands]

    # normalize: expand ellipses and make the output explicit, using the
    # positional shapes (shapes=True avoids materializing the operands)
    shapes = [op.shape for op in operands]
    in_part, out_part, _ = parse_einsum_input([subscripts, *shapes], shapes=True)
    in_specs = in_part.split(",")
    assert len(in_specs) == len(operands)

    # fresh symbols for named dims, avoiding every symbol already in use;
    # get_symbol gives an effectively unlimited supply (spills into unicode)
    used = {c for c in (in_part + out_part) if c not in ",->"}
    counter = itertools.count()

    def next_symbol():
        while True:
            s = get_symbol(next(counter))
            if s not in used:
                used.add(s)
                return s

    # assign a letter per unique named dim; shared names reuse the same letter
    # so einsum aligns them as batch dims rather than contracting
    letter_of, order = {}, []
    for dims in named:
        for d in dims:
            if d not in letter_of:
                letter_of[d] = next_symbol()
                order.append(d)

    # bind named dims to leading positional axes and prepend their letters
    bound, new_in_specs = [], []
    for op, dims, spec in zip(operands, named, in_specs):
        bound.append(bind_dims(op, *dims) if dims else op)
        new_in_specs.append("".join(letter_of[d] for d in dims) + spec)

    # add every named dim to the front of the output as passthrough
    out_prefix = "".join(letter_of[d] for d in order)
    new_subscripts = ",".join(new_in_specs) + "->" + out_prefix + out_part

    result = jax.numpy.einsum(new_subscripts, *bound, **kwargs)

    # unbind: leading axes correspond to `order`, reindex them back to named
    reindexed = jax_getitem(
        result,
        tuple(d() for d in order) + tuple(slice(None) for _ in range(len(out_part))),
    )
    return reindexed


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
    *names: Annotated[Operation[[], Any], Scoped[B]],
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
    if not names:
        return value
    if isinstance(value, Term) and value.op == bind_dims:
        return bind_dims(value.args[0], *(names + tuple(value.args[1:])))
    if jax.tree_util.treedef_is_leaf(jax.tree.structure(value)):
        return __dispatch(typeof(value))(value, *names)
    return jax.tree.map(lambda v: bind_dims(v, *names), value)


@bind_dims.register(object)
def _(*args, **kwargs):
    raise NotHandled


@defop
@_CustomSingleDispatchCallable
def unbind_dims[T, A, B](
    __dispatch: Callable[[type], Callable[..., T]],
    value: Annotated[T, Scoped[A | B]],
    *names: Annotated[Operation[[], Any], Scoped[B]],
) -> Annotated[T, Scoped[A | B]]:
    """Convert positional dimensions to named dimensions."""
    if not names:
        return value
    if jax.tree_util.treedef_is_leaf(jax.tree.structure(value)):
        return __dispatch(typeof(value))(value, *names)
    return jax.tree.map(lambda v: unbind_dims(v, *names), value)


@unbind_dims.register(object)
def _(*args, **kwargs):
    raise NotHandled


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


@syntactic_eq.register
def _(x: jax.core.Tracer, other) -> bool:
    return isinstance(other, jax.core.Tracer) and id(x) == id(other)


@syntactic_hash.register
def _(x: jax.Array) -> int:
    # Concrete arrays aren't hashable; hash by shape, dtype, and bytes.
    return hash(("jax.Array", x.shape, str(x.dtype), x.tobytes()))


@syntactic_hash.register
def _(x: jax.core.Tracer) -> int:
    # Concrete arrays aren't hashable; hash by shape, dtype, and bytes.
    return hash(("jax.core.Tracer", id(x)))
