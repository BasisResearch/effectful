import functools
import typing
from collections.abc import Callable, Mapping, Sequence
from types import EllipsisType
from typing import Annotated, Any

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required to use effectful.handlers.torch")

import tree

from effectful.internals.runtime import interpreter
from effectful.internals.tensor_utils import _desugar_tensor_index
from effectful.ops.semantics import apply, evaluate, fvsof, handler, typeof
from effectful.ops.syntax import Scoped, defdata, defop, defterm, syntactic_eq
from effectful.ops.types import Expr, NotHandled, Operation, Term

# + An element of a tensor index expression.
IndexElement = None | int | slice | Sequence[int] | EllipsisType | torch.Tensor


def _getitem_ellipsis_and_none(
    x: torch.Tensor, key: tuple[IndexElement, ...]
) -> tuple[torch.Tensor, tuple[IndexElement, ...]]:
    """Eliminate ellipses and None in an index expression x[key].

    Returns x1, key1 such that x1[key1] == x[key] nand key1 does not contain None or Ellipsis.

    """

    new_shape, new_key = _desugar_tensor_index(x.shape, key)
    return torch.reshape(x, new_shape), new_key


def sizesof(value) -> Mapping[Operation[[], torch.Tensor], int]:
    """Return the sizes of named dimensions in a tensor expression.

    Sizes are inferred from the tensor shape.

    :param value: A tensor expression.
    :return: A mapping from named dimensions to their sizes.

    **Example usage**:

    >>> a, b = defop(torch.Tensor, name='a'), defop(torch.Tensor, name='b')
    >>> sizes = sizesof(torch.ones(2, 3)[a(), b()])
    >>> assert sizes[a] == 2 and sizes[b] == 3
    """
    sizes: dict[Operation[[], torch.Tensor], int] = {}

    def _torch_getitem_sizeof(
        x: Expr[torch.Tensor], key: tuple[Expr[IndexElement], ...]
    ) -> Expr[torch.Tensor]:
        if isinstance(x, torch.Tensor):
            shape, key_ = _desugar_tensor_index(x.shape, key)

            for i, k in enumerate(key_):
                if (
                    isinstance(k, Term)
                    and len(k.args) == 0
                    and len(k.kwargs) == 0
                    and issubclass(typeof(k), torch.Tensor)
                ):
                    if k.op in sizes and sizes[k.op] != shape[i]:
                        raise ValueError(
                            f"Named index {k.op} used in incompatible dimensions of size {sizes[k.op]} and {shape[i]}"
                        )
                    sizes[k.op] = shape[i]

        return defdata(torch_getitem, x, key)

    def _apply(op, *args, **kwargs):
        args, kwargs = tree.map_structure(defterm, (args, kwargs))
        return defdata(op, *args, **kwargs)

    with interpreter({torch_getitem: _torch_getitem_sizeof, apply: _apply}):
        evaluate(defterm(value))

    return sizes


def _partial_eval(t: Expr[torch.Tensor]) -> Expr[torch.Tensor]:
    """Partially evaluate a term with respect to its sized free variables."""

    sized_fvs = sizesof(t)
    if not sized_fvs:
        return t

    if not (
        isinstance(t, Term)
        and all(
            isinstance(a, torch.Tensor) or not isinstance(a, Term) or a.op in sized_fvs
            for a in tree.flatten((t.args, t.kwargs))
        )
    ):
        return t

    # note: torch.func.vmap will call repr on the callable, so it's important
    # that we don't pass something with a slow repr (like a large tensor wrapped
    # in a deffn)
    def wrapper(*sized_values):
        with handler(
            {
                k: functools.partial(lambda x: x, v)
                for (k, v) in zip(sized_fvs.keys(), sized_values)
            }
        ):
            return evaluate(t)

    tpe_torch_fn = torch.func.vmap(wrapper, randomness="different")

    inds = torch.broadcast_tensors(
        *(
            torch.arange(size)[(...,) + (None,) * (len(sized_fvs) - i - 1)]
            for i, size in enumerate(sized_fvs.values())
        )
    )

    flat_result = tpe_torch_fn(*[i.reshape(-1) for i in inds])

    def reindex_flat_tensor(t):
        if not isinstance(t, torch.Tensor):
            return t

        result = t.reshape(inds[0].shape + t.shape[1:])
        return torch_getitem(result, tuple(k() for k in sized_fvs.keys()))

    result = tree.map_structure(reindex_flat_tensor, flat_result)
    return result


@defop
@functools.singledispatch
def bind_dims[
    A,
    B,
    HasDims: torch.Tensor
    | torch.distributions.Distribution
    | tree.Structure[torch.Tensor | torch.distributions.Distribution],
](
    value: Annotated[HasDims, Scoped[A | B]],
    *names: Annotated[Operation[[], torch.Tensor], Scoped[B]],
) -> Annotated[HasDims, Scoped[A]]:
    """Convert named dimensions to positional dimensions.

    :param t: A tensor.
    :param args: Named dimensions to convert to positional dimensions.
                  These positional dimensions will appear at the beginning of the
                  shape.
    :return: A tensor with the named dimensions in ``args`` converted to positional dimensions.

    **Example usage**:

    >>> a, b = defop(torch.Tensor, name='a'), defop(torch.Tensor, name='b')
    >>> t = torch.ones(2, 3)
    >>> bind_dims(t[a(), b()], b, a).shape
    torch.Size([3, 2])
    """
    if tree.is_nested(value):
        return tree.map_structure(lambda v: bind_dims(v, *names), value)
    raise NotHandled


@bind_dims.register  # type: ignore
def _bind_dims_tensor(
    value: torch.Tensor, *names: Operation[[], torch.Tensor]
) -> torch.Tensor:
    def _evaluate(expr):
        if isinstance(expr, Term):
            (args, kwargs) = tree.map_structure(_evaluate, (expr.args, expr.kwargs))
            return _partial_eval(expr)
        if tree.is_nested(expr):
            return tree.map_structure(_evaluate, expr)
        return expr

    t = value
    args = names

    if not isinstance(t, Term):
        return t

    result = _evaluate(t)
    if not isinstance(result, Term) or not args:
        return result

    # ensure that the result is a torch_getitem with a tensor as the first argument
    if not (result.op is torch_getitem and isinstance(result.args[0], torch.Tensor)):
        raise NotHandled

    tensor = result.args[0]
    dims = result.args[1]
    assert isinstance(dims, Sequence)

    # ensure that the order is a subset of the named dimensions
    order_set = set(args)
    if not order_set <= set(a.op for a in dims if isinstance(a, Term)):
        raise NotHandled

    # permute the inner tensor so that the leading dimensions are in the order
    # specified and the trailing dimensions are the remaining named dimensions
    # (or slices)
    reindex_dims = [
        i
        for i, o in enumerate(dims)
        if not isinstance(o, Term) or o.op not in order_set
    ]
    dim_ops = [a.op if isinstance(a, Term) else None for a in dims]
    perm = [dim_ops.index(o) for o in args] + reindex_dims
    tensor = tensor.permute(perm)
    return tensor[(slice(None),) * len(args) + tuple(dims[i] for i in reindex_dims)]


@defop
@functools.singledispatch
def unbind_dims[
    A,
    B,
    HasDims: torch.Tensor
    | torch.distributions.Distribution
    | tree.Structure[torch.Tensor | torch.distributions.Distribution],
](
    value: Annotated[HasDims, Scoped[A | B]],
    *names: Annotated[Operation[[], torch.Tensor], Scoped[B]],
) -> Annotated[HasDims, Scoped[A | B]]:
    if tree.is_nested(value):
        return tree.map_structure(lambda v: unbind_dims(v, *names), value)
    raise NotHandled


@unbind_dims.register  # type: ignore
def _unbind_dims_tensor[A, B](
    value: torch.Tensor,
    *names: Annotated[Operation[[], torch.Tensor], Scoped[B]],
) -> Annotated[torch.Tensor, Scoped[A | B]]:
    return value[tuple(n() for n in names)]


@functools.cache
def _register_torch_op[**P, T](torch_fn: Callable[P, T]):
    if torch_fn is torch._C.TensorBase.__getitem__:
        return torch_getitem

    @defop
    def _torch_op(*args, **kwargs) -> torch.Tensor:
        tm = defdata(_torch_op, *args, **kwargs)
        sized_fvs = sizesof(tm)

        if (
            _torch_op is torch_getitem
            and not isinstance(args[0], Term)
            and sized_fvs
            and args[1]
            and all(isinstance(k, Term) and k.op in sized_fvs for k in args[1])
        ):
            raise NotHandled
        elif sized_fvs and set(sized_fvs.keys()) == fvsof(tm) - {
            torch_getitem,
            _torch_op,
        }:
            # note: this cast is a lie. partial_eval can return non-tensors, as
            # can torch_fn. for example, some torch functions return tuples,
            # which partial_eval handles.
            return typing.cast(torch.Tensor, _partial_eval(tm))
        elif not any(
            tree.flatten(
                tree.map_structure(lambda x: isinstance(x, Term), (args, kwargs))
            )
        ):
            return typing.cast(torch.Tensor, torch_fn(*args, **kwargs))
        else:
            raise NotHandled

    functools.update_wrapper(_torch_op, torch_fn)
    return _torch_op


@_register_torch_op
def torch_getitem(x: torch.Tensor, key: tuple[IndexElement, ...]) -> torch.Tensor:
    """Operation for indexing a tensor.

    .. note::

      This operation is not intended to be called directly. Instead, it is
      exposed so that it can be handled.

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected a tensor but got {type(x)}")

    for k in key:
        if isinstance(k, Operation):
            raise TypeError(
                f"Got operation symbol {str(k)}. You probably meant {str(k)}()."
            )

    # fast path for simple cases
    if len(key) == 0:
        return x
    elif not any(isinstance(k, torch.Tensor) for k in key):
        return x[tuple(key)]
    elif all(isinstance(k, torch.Tensor) for k in key):
        return torch.ops.aten.index(x, key)

    # handle None, Ellipsis, and missing dimensions
    x, key = _getitem_ellipsis_and_none(x, key)

    # Convert non-tensor args to tensors
    key_l = list(key)
    for i, arg in list(enumerate(key)):
        if isinstance(arg, slice):
            if arg == slice(None):
                key_l[i] = None
            else:
                # Convert slices to torch.arange()s.
                start = arg.start if arg.start is not None else 0
                stop = arg.stop if arg.stop is not None else x.shape[i]
                step = arg.step if arg.step is not None else 1
                flat_arg = torch.arange(
                    start, stop, step, dtype=torch.long, device=x.device
                )
                key_l[i] = flat_arg.reshape((-1,) + (1,) * i)
        elif isinstance(arg, int):
            key_l[i] = torch.tensor(arg, dtype=torch.long, device=x.device)
        elif isinstance(arg, list | tuple):
            flat_arg = torch.tensor(arg, dtype=torch.long, device=x.device)
            key_l[i] = flat_arg.reshape(flat_arg.shape + (1,) * i)

    return torch.ops.aten.index(x, tuple(key_l))


@defdata.register(torch.Tensor)
def _embed_tensor(op, *args, **kwargs):
    if (
        op is torch_getitem
        and not isinstance(args[0], Term)
        and len(args[1]) > 0
        and all(
            typeof(k) is torch.Tensor and not k.args and not k.kwargs
            for k in args[1]
            if isinstance(k, Term)
        )
    ):
        return _EagerTensorTerm(args[0], args[1])
    else:
        return _TensorTerm(op, *args, **kwargs)


class _TensorTerm(Term[torch.Tensor]):
    def __init__(
        self, op: Operation[..., torch.Tensor], *args: Expr, **kwargs: Expr
    ) -> None:
        self._op = op
        self._args = args
        self._kwargs = kwargs

    @property
    def op(self) -> Operation[..., torch.Tensor]:
        return self._op

    @property
    def args(self) -> tuple:
        return self._args

    @property
    def kwargs(self) -> dict:
        return self._kwargs

    def __getitem__(
        self, key: Expr[IndexElement] | tuple[Expr[IndexElement], ...]
    ) -> Expr[torch.Tensor]:
        return torch_getitem(self, key if isinstance(key, tuple) else (key,))

    @classmethod
    def __torch_function__[T](
        cls, func: Callable[..., T], types, args=(), kwargs=None
    ) -> Expr[T]:
        return _register_torch_op(func)(*args, **({} if kwargs is None else kwargs))

    def __add__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.add(typing.cast(torch.Tensor, self), other)

    def __radd__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.add(other, typing.cast(torch.Tensor, self))

    def __neg__(self) -> torch.Tensor:
        return torch.neg(typing.cast(torch.Tensor, self))

    def __pos__(self) -> torch.Tensor:
        return torch.positive(typing.cast(torch.Tensor, self))

    def __sub__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.sub(typing.cast(torch.Tensor, self), other)

    def __rsub__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.sub(other, typing.cast(torch.Tensor, self))

    def __mul__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.mul(typing.cast(torch.Tensor, self), other)

    def __rmul__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.mul(other, typing.cast(torch.Tensor, self))

    def __truediv__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.div(typing.cast(torch.Tensor, self), other)

    def __rtruediv__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.div(other, typing.cast(torch.Tensor, self))

    def __pow__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.pow(typing.cast(torch.Tensor, self), other)

    def __rpow__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.pow(other, typing.cast(torch.Tensor, self))

    def __abs__(self) -> torch.Tensor:
        return torch.abs(typing.cast(torch.Tensor, self))

    def __eq__(self, other: Any):
        return torch.eq(typing.cast(torch.Tensor, self), other)

    def __ne__(self, other: Any):
        return torch.ne(typing.cast(torch.Tensor, self), other)

    def __floordiv__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.floor_divide(typing.cast(torch.Tensor, self), other)

    def __rfloordiv__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.floor_divide(other, typing.cast(torch.Tensor, self))

    def __mod__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.fmod(typing.cast(torch.Tensor, self), other)

    def __rmod__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.fmod(other, typing.cast(torch.Tensor, self))

    def __lt__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.lt(typing.cast(torch.Tensor, self), other)

    def __le__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.le(typing.cast(torch.Tensor, self), other)

    def __gt__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.gt(typing.cast(torch.Tensor, self), other)

    def __ge__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.ge(typing.cast(torch.Tensor, self), other)

    def __lshift__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_left_shift(typing.cast(torch.Tensor, self), other)

    def __rlshift__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_left_shift(other, typing.cast(torch.Tensor, self))

    def __rshift__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_right_shift(typing.cast(torch.Tensor, self), other)

    def __rrshift__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_right_shift(other, typing.cast(torch.Tensor, self))

    def __and__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_and(typing.cast(torch.Tensor, self), other)

    def __rand__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_and(other, typing.cast(torch.Tensor, self))

    def __xor__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_xor(typing.cast(torch.Tensor, self), other)

    def __rxor__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_xor(other, typing.cast(torch.Tensor, self))

    def __or__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_or(typing.cast(torch.Tensor, self), other)

    def __ror__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_or(other, typing.cast(torch.Tensor, self))

    def __invert__(self) -> torch.Tensor:
        return torch.bitwise_not(typing.cast(torch.Tensor, self))

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.matmul(typing.cast(torch.Tensor, self), other)

    def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
        return torch.matmul(other, typing.cast(torch.Tensor, self))

    def __iter__(self):
        raise TypeError("A free tensor is not iterable.")


@Term.register
class _EagerTensorTerm(torch.Tensor):
    args: tuple[torch.Tensor, tuple[IndexElement, ...]]
    kwargs: Mapping[str, object] = {}

    __match_args__ = ("op", "args", "kwargs")

    def __new__(cls, x: torch.Tensor, key: tuple[IndexElement, ...]):
        assert not isinstance(x, Term)

        for k in key:
            if isinstance(k, Term):
                assert typeof(k) is torch.Tensor and not k.args and not k.kwargs

        x, key = _getitem_ellipsis_and_none(x, key)
        ret = x.as_subclass(cls)
        ret.args = (x, key)
        return ret

    @property
    def op(self) -> Operation[..., torch.Tensor]:
        return torch_getitem

    def __str__(self):
        tensor_str = str(self.args[0])
        key_str = ", ".join(str(k) for k in self.args[1])
        return f"{tensor_str}[{key_str}]"

    def __repr__(self):
        return str(self)

    @classmethod
    def __torch_function__[T](
        cls, func: Callable[..., T], types, args=(), kwargs=None
    ) -> Expr[T]:
        return _register_torch_op(func)(*args, **({} if kwargs is None else kwargs))

    def __getitem__(self, key) -> torch.Tensor:
        return torch_getitem(self, key if isinstance(key, tuple) else (key,))

    def __format__(self, format_spec: str) -> str:
        return (
            format(torch.Tensor(self), format_spec)
            + "["
            + ", ".join(str(a) for a in self.args[1])
            + "]"
        )

    @property
    def shape(self) -> torch.Size:  # type: ignore
        x, key = self.args
        return torch.Size([s for s, k in zip(x.shape, key) if not isinstance(k, Term)])

    def size(self, dim: int | None = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def numel(self) -> int:
        return self.shape.numel()

    def dim(self) -> int:
        return len(self.shape)

    @property
    def ndim(self) -> int:  # type: ignore
        return self.dim()

    def ndimension(self):
        return self.dim()

    def item(self):
        raise ValueError(f"cannot convert {self} to a Python scalar")

    @property
    def dtype(self):
        return self.args[0].dtype

    @property
    def device(self):
        return self.args[0].device

    def new(self, *args, **kwargs):
        return self.args[0].new(*args, **kwargs)

    @property
    def requires_grad(self):
        return self.args[0].requires_grad

    def requires_grad_(self, requires_grad=True):
        return self.args[0].requires_grad_(requires_grad=requires_grad)

    @property
    def grad_fn(self):
        return self.args[0].grad_fn


def _indexed_func_wrapper[**P, S, T](
    func: Callable[P, T],
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
                lambda t, i: torch_getitem(t, index_expr(i)), ret, indexes
            )
        else:
            indexed_ret = torch_getitem(ret, index_expr(indexes))

        return indexed_ret

    return deindexed, reindex


@functools.wraps(torch.func.grad)
def grad(func, *args, **kwargs):
    """Compute the gradient of a function with respect to its arguments. This is
    a wrapper around `torch.func.grad` that allows the function to be called
    with indexed arguments.

    """
    (deindexed_func, reindex) = _indexed_func_wrapper(func)
    f = _register_torch_op(torch.func.grad(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(f(*a, *k))


@functools.wraps(torch.func.jacfwd)
def jacfwd(func, *args, **kwargs):
    (deindexed_func, reindex) = _indexed_func_wrapper(func)
    jacobian = _register_torch_op(torch.func.jacfwd(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(jacobian(*a, *k))


@functools.wraps(torch.func.jacrev)
def jacrev(func, *args, **kwargs):
    (deindexed_func, reindex) = _indexed_func_wrapper(func)
    jacobian = _register_torch_op(torch.func.jacrev(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(jacobian(*a, *k))


@functools.wraps(torch.func.hessian)
def hessian(func, *args, **kwargs):
    (deindexed_func, reindex) = _indexed_func_wrapper(func)
    h = _register_torch_op(torch.func.hessian(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(h(*a, *k))


@functools.wraps(torch.func.jvp)
def jvp(func, *args, **kwargs):
    (deindexed_func, reindex) = _indexed_func_wrapper(func)

    # hide deindexed_func from _register_torch_op
    jvp_func = functools.partial(torch.func.jvp, deindexed_func)
    ret = _register_torch_op(jvp_func)(*args, **kwargs)
    return tree.map_structure(reindex, ret)


@functools.wraps(torch.func.vjp)
def vjp(func, *indexed_primals, **kwargs):
    unpacked_primals = []
    for t in indexed_primals:
        indices = list(sizesof(t).keys())
        unpacked = bind_dims(t, *indices)
        unpacked_primals.append((unpacked, indices))

    indexed_result = None

    def repack_primals(primals):
        return [
            torch_getitem(p, tuple(x() for x in unpacked_primals[i][1]))
            for i, p in enumerate(primals)
        ]

    def wrapper(*primals):
        nonlocal indexed_result
        indexed_result = func(*repack_primals(primals))
        return tree.map_structure(
            lambda t: bind_dims(t, *list(sizesof(t).keys())), indexed_result
        )

    unindexed_primals = [t[0] for t in unpacked_primals]
    _, vjpfunc = torch.func.vjp(wrapper, *unindexed_primals, **kwargs)

    def vjpfunc_wrapper(*tangents):
        unindexed_tangents = tree.map_structure(
            lambda t: bind_dims(t, *list(sizesof(t).keys())), tangents
        )
        grads = vjpfunc(*unindexed_tangents)
        return repack_primals(grads)

    return indexed_result, vjpfunc_wrapper


@functools.wraps(torch.func.vmap)
def vmap(func, *args, **kwargs):
    (deindexed_func, reindex) = _indexed_func_wrapper(func)
    vmap_func = _register_torch_op(torch.func.vmap(deindexed_func, *args, **kwargs))
    # vmap_func returns tensors of shape [vmap_dim, indexed_dim_1, ...,
    # indexed_dim_n, pos_dim_1, ..., pos_dim_m], so we reapply indexes starting
    # at dim 1
    return lambda *a, **k: reindex(vmap_func(*a, *k), starting_dim=1)


@syntactic_eq.register
def _(x: torch.Tensor, other) -> bool:
    return isinstance(other, torch.Tensor) and bool((x == other).all())
