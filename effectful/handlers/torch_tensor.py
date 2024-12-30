import functools
import typing
from types import EllipsisType
from typing import Callable, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import torch
import tree
from typing_extensions import ParamSpec

import effectful.handlers.operator  # noqa: F401
from effectful.internals.base_impl import BaseTerm, as_data_register
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, evaluate, fvsof, typeof
from effectful.ops.syntax import NoDefaultRule, defop
from effectful.ops.types import Expr, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


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

        return torch_getitem.__free_rule__(x, key)

    with interpreter(
        {
            torch_getitem: _torch_getitem_sizeof,
            apply: lambda _, op, *a, **k: op.__free_rule__(*a, **k),
        }
    ):
        evaluate(value)

    return sizes


def partial_eval(t: T, order=None) -> T:
    """Partially evaluate a term with respect to its sized free variables.

    Variables in `order` are converted to positional dimensions in the result
    tensor, in the order they appear. All other variables remain free.

    """
    from effectful.ops.syntax import defun

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
        defun(t, *[var for (var, _) in ordered_sized_fvs]), randomness="different"
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


@functools.cache
def _register_torch_op(torch_fn: Callable[P, T]):

    @defop
    def _torch_op(*args, **kwargs) -> torch.Tensor:

        tm = _torch_op.__free_rule__(*args, **kwargs)
        sized_fvs = sizesof(tm)

        if (
            _torch_op is torch_getitem
            and not isinstance(args[0], Term)
            and sized_fvs
            and args[1]
            and all(isinstance(k, Term) and k.op in sized_fvs for k in args[1])
        ):
            raise NoDefaultRule
        elif sized_fvs and set(sized_fvs.keys()) == set(fvsof(tm).keys()) - {
            torch_getitem,
            _torch_op,
        }:
            # note: this cast is a lie. partial_eval can return non-tensors, as
            # can torch_fn. for example, some torch functions return tuples,
            # which partial_eval handles.
            return typing.cast(torch.Tensor, partial_eval(tm))
        elif not any(
            tree.flatten(
                tree.map_structure(lambda x: isinstance(x, Term), (args, kwargs))
            )
        ):
            return typing.cast(torch.Tensor, torch_fn(*args, **kwargs))
        else:
            raise NoDefaultRule

    return _torch_op


@_register_torch_op
def torch_getitem(x: torch.Tensor, key: Tuple[IndexElement, ...]) -> torch.Tensor:
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

    Example:
    >>> width, height = gensym(int, name='width'), gensym(int, name='height')
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


@as_data_register(torch.Tensor)
def _embed_tensor(op, args, kwargs):
    match op, args, kwargs:
        case torch_getitem_, (torch.Tensor() as x, key), () if (
            torch_getitem_ is torch_getitem
            and len(key) >= 1
            and not isinstance(x, Term)
            and all(
                typeof(k) is int and not k.args and not k.kwargs
                for k in key
                if isinstance(k, Term)
            )
        ):
            return EagerTensorTerm(x, key)
        case _:
            return TensorTerm(op, args, kwargs)


class TensorTerm(BaseTerm[torch.Tensor]):
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
class EagerTensorTerm(torch.Tensor):

    op: Operation[..., torch.Tensor] = torch_getitem
    args: Tuple[torch.Tensor, Tuple[IndexElement, ...]]
    kwargs: Tuple = ()

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
