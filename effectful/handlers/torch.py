import functools
import typing
from types import EllipsisType
from typing import Callable, Mapping, Optional, Sequence, Tuple, TypeVar, Union

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required to use effectful.handlers.torch")

import tree
from typing_extensions import ParamSpec

import effectful.handlers.numbers  # noqa: F401
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, evaluate, fvsof, typeof
from effectful.ops.syntax import defdata, defop
from effectful.ops.types import Expr, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


# + An element of a tensor index expression.
IndexElement = Union[None, int, slice, Sequence[int], EllipsisType, torch.Tensor]


def _desugar_tensor_index(shape, key):
    new_shape = []
    new_key = []

    def extra_dims(key):
        return sum(1 for k in key if k is None)

    # handle any missing dimensions by adding a trailing Ellipsis
    if not any(k is Ellipsis for k in key):
        key = tuple(key) + (...,)

    for i, k in enumerate(key):
        if k is None:  # add a new singleton dimension
            new_shape.append(1)
            new_key.append(slice(None))
        elif k is Ellipsis:
            assert not any(
                k is Ellipsis for k in key[i + 1 :]
            ), "only one Ellipsis allowed"

            # determine which of the original dimensions this ellipsis refers to
            pre_dims = i - extra_dims(key[:i])  # dimensions that precede the ellipsis
            elided_dims = (
                len(shape) - pre_dims - (len(key) - i - 1 - extra_dims(key[i + 1 :]))
            )  #
            new_shape += shape[pre_dims : pre_dims + elided_dims]
            new_key += [slice(None)] * elided_dims
        else:
            new_shape.append(shape[len(new_shape) - extra_dims(key[:i])])
            new_key.append(k)

    return new_shape, new_key


def _getitem_ellipsis_and_none(
    x: torch.Tensor, key: Tuple[IndexElement, ...]
) -> Tuple[torch.Tensor, Tuple[IndexElement, ...]]:
    """Eliminate ellipses and None in an index expression x[key].

    Returns x1, key1 such that x1[key1] == x[key] nand key1 does not contain None or Ellipsis.

    """

    new_shape, new_key = _desugar_tensor_index(x.shape, key)
    return torch.reshape(x, new_shape), new_key


def sizesof(value: Expr) -> Mapping[Operation[[], int], int]:
    """Return the sizes of named dimensions in a tensor expression.

    Sizes are inferred from the tensor shape.

    :param value: A tensor expression.
    :return: A mapping from named dimensions to their sizes.

    **Example usage**:

    >>> a, b = defop(int, name='a'), defop(int, name='b')
    >>> sizesof(Indexable(torch.ones(2, 3))[a(), b()])
    {a: 2, b: 3}
    """
    if isinstance(value, torch.distributions.Distribution) and not isinstance(
        value, Term
    ):
        return {v: s for a in value.__dict__.values() for v, s in sizesof(a).items()}

    sizes: dict[Operation[[], int], int] = {}

    def _torch_getitem_sizeof(
        x: Expr[torch.Tensor], key: Tuple[Expr[IndexElement], ...]
    ) -> Expr[torch.Tensor]:
        if isinstance(x, torch.Tensor):
            shape, key_ = _desugar_tensor_index(x.shape, key)

            for i, k in enumerate(key_):
                if (
                    isinstance(k, Term)
                    and len(k.args) == 0
                    and len(k.kwargs) == 0
                    and issubclass(typeof(k), int)
                ):
                    if k.op in sizes and sizes[k.op] != shape[i]:
                        raise ValueError(
                            f"Named index {k.op} used in incompatible dimensions of size {sizes[k.op]} and {shape[i]}"
                        )
                    sizes[k.op] = shape[i]

        return defdata(torch_getitem, x, key)

    with interpreter(
        {
            torch_getitem: _torch_getitem_sizeof,
            apply: lambda _, op, *a, **k: defdata(op, *a, **k),
        }
    ):
        evaluate(value)

    return sizes


def _partial_eval(t: T, order: Optional[Sequence[Operation[[], int]]] = None) -> T:
    """Partially evaluate a term with respect to its sized free variables.

    Variables in `order` are converted to positional dimensions in the result
    tensor, in the order they appear. All other variables remain free.

    """
    from effectful.ops.syntax import deffn

    if order is None:
        order = []

    sized_fvs = sizesof(t)

    for x in order:
        if x not in sized_fvs:
            raise ValueError(
                f"Tried to partially evaluate nonexistent free variable {x} (free={sized_fvs})"
            )

    # if there are no sized free variables, then nothing to do
    if len(sized_fvs) == 0:
        return t

    order_set = set(order)
    reindex_fvs = [
        (var, size) for var, size in sized_fvs.items() if var not in order_set
    ]
    ordered_sized_fvs = reindex_fvs + [(var, sized_fvs[var]) for var in order]

    tpe_torch_fn = torch.func.vmap(
        deffn(t, *[var for (var, _) in ordered_sized_fvs]), randomness="different"
    )

    inds = torch.broadcast_tensors(
        *(
            torch.arange(size)[(...,) + (None,) * (len(ordered_sized_fvs) - i - 1)]
            for i, (_, size) in enumerate(ordered_sized_fvs)
        )
    )

    flat_result = tpe_torch_fn(*[i.reshape(-1) for i in inds])

    def reindex_flat_tensor(t):
        if not isinstance(t, torch.Tensor):
            return t

        result = t.reshape(inds[0].shape + t.shape[1:])
        return torch_getitem(result, tuple(var() for (var, _) in reindex_fvs))

    return tree.map_structure(reindex_flat_tensor, flat_result)


def to_tensor(*args, **kwargs) -> torch.Tensor:
    """Convert named dimensions to positional dimensions.

    :param t: A tensor.
    :type t: T
    :param order: A list of named dimensions to convert to positional dimensions.
                  These positional dimensions will appear at the beginning of the
                  shape.
    :type order: Optional[Sequence[Operation[[], int]]]
    :return: A tensor with the named dimensions in ``order`` converted to positional dimensions.

    **Example usage**:

    >>> a, b = defop(int, name='a'), defop(int, name='b')
    >>> t = torch.ones(2, 3)
    >>> to_tensor(Indexable(t)[a(), b()], [b, a]).shape
    torch.Size([3, 2])
    """
    return _partial_eval(*args, **kwargs)


@functools.cache
def _register_torch_op(torch_fn: Callable[P, T]):

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
            raise NotImplementedError
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
            raise NotImplementedError

    functools.update_wrapper(_torch_op, torch_fn)
    return _torch_op


@_register_torch_op
def torch_getitem(x: torch.Tensor, key: Tuple[IndexElement, ...]) -> torch.Tensor:
    """Operation for indexing a tensor.

    .. note::

      This operation is not intended to be called directly. Instead, use
      :class:`Indexable` to create indexed tensors. :func:`torch_getitem` is
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
        elif isinstance(arg, (list, tuple)):
            flat_arg = torch.tensor(arg, dtype=torch.long, device=x.device)
            key_l[i] = flat_arg.reshape(flat_arg.shape + (1,) * i)

    return torch.ops.aten.index(x, tuple(key_l))


class Indexable:
    """Helper class for constructing indexed tensors.

    **Example usage**:

    >>> width, height = defop(int, name='width'), defop(int, name='height')
    >>> t = Indexable(torch.ones(2, 3))[width(), height()]
    >>> t
    Indexable(tensor([[1., 1., 1.],
                      [1., 1., 1.]]))[width(), height()]
    """

    def __init__(self, t: torch.Tensor):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"Expected a torch.Tensor, got {type(t)}")
        self.t = t

    def __getitem__(self, key) -> torch.Tensor:
        if not isinstance(key, tuple):
            key = (key,)
        return torch_getitem(self.t, key)


@defdata.register(torch.Tensor)
def _embed_tensor(op, *args, **kwargs):
    if (
        op is torch_getitem
        and not isinstance(args[0], Term)
        and len(args[1]) > 0
        and all(
            typeof(k) is int and not k.args and not k.kwargs
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
        self, key: Union[Expr[IndexElement], Tuple[Expr[IndexElement], ...]]
    ) -> Expr[torch.Tensor]:
        return torch_getitem(self, key if isinstance(key, tuple) else (key,))

    @classmethod
    def __torch_function__(
        cls, func: Callable[..., T], types, args=(), kwargs=None
    ) -> Expr[T]:
        return _register_torch_op(func)(*args, **({} if kwargs is None else kwargs))


@Term.register
class _EagerTensorTerm(torch.Tensor):

    op: Operation[..., torch.Tensor] = torch_getitem
    args: Tuple[torch.Tensor, Tuple[IndexElement, ...]]
    kwargs: Mapping[str, object] = {}

    __match_args__ = ("op", "args", "kwargs")

    def __new__(cls, x: torch.Tensor, key: Tuple[IndexElement, ...]):
        assert not isinstance(x, Term)

        for k in key:
            if isinstance(k, Term):
                assert typeof(k) is int and not k.args and not k.kwargs

        x, key = _getitem_ellipsis_and_none(x, key)
        ret = x.as_subclass(cls)
        ret.args = (x, key)
        return ret

    def __repr__(self):
        indexed_constr = "Indexable"

        # correct indentation
        parts = str(self.args[0]).split("\n")
        tensor_str = "\n".join(
            [parts[0]] + [(len(indexed_constr) + 1) * " " + p for p in parts[1:]]
        )

        key_str = ", ".join(str(k) for k in self.args[1])
        return f"{indexed_constr}({tensor_str})[{key_str}]"

    @classmethod
    def __torch_function__(
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

    def size(self, dim: Optional[int] = None):
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

    @property
    def grad_fn(self):
        return self.args[0].grad_fn


def _indexed_func_wrapper(
    func: Callable[P, T]
) -> Tuple[Callable[P, S], Callable[[S], T]]:
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
            t_ = to_tensor(t, i.sizes.keys())
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
        unpacked = to_tensor(t, indices)
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
            lambda t: to_tensor(t, list(sizesof(t).keys())), indexed_result
        )

    unindexed_primals = [t[0] for t in unpacked_primals]
    _, vjpfunc = torch.func.vjp(wrapper, *unindexed_primals, **kwargs)

    def vjpfunc_wrapper(*tangents):
        unindexed_tangents = tree.map_structure(
            lambda t: to_tensor(t, list(sizesof(t).keys())), tangents
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
