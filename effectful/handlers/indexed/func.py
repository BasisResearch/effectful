import functools
from typing import Callable, ParamSpec, Tuple, TypeVar

import torch
import tree

from ..handlers.torch_tensor import _register_torch_op, sizesof, torch_getitem
from .ops import to_tensor

P = ParamSpec("P")
T = TypeVar("T")
S = TypeVar("S")


def indexed_func_wrapper(
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
    (deindexed_func, reindex) = indexed_func_wrapper(func)
    f = _register_torch_op(torch.func.grad(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(f(*a, *k))


@functools.wraps(torch.func.jacfwd)
def jacfwd(func, *args, **kwargs):
    (deindexed_func, reindex) = indexed_func_wrapper(func)
    jacobian = _register_torch_op(torch.func.jacfwd(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(jacobian(*a, *k))


@functools.wraps(torch.func.jacrev)
def jacrev(func, *args, **kwargs):
    (deindexed_func, reindex) = indexed_func_wrapper(func)
    jacobian = _register_torch_op(torch.func.jacrev(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(jacobian(*a, *k))


@functools.wraps(torch.func.hessian)
def hessian(func, *args, **kwargs):
    (deindexed_func, reindex) = indexed_func_wrapper(func)
    h = _register_torch_op(torch.func.hessian(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(h(*a, *k))


@functools.wraps(torch.func.jvp)
def jvp(func, *args, **kwargs):
    (deindexed_func, reindex) = indexed_func_wrapper(func)

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
    (deindexed_func, reindex) = indexed_func_wrapper(func)
    vmap_func = _register_torch_op(torch.func.vmap(deindexed_func, *args, **kwargs))
    # vmap_func returns tensors of shape [vmap_dim, indexed_dim_1, ...,
    # indexed_dim_n, pos_dim_1, ..., pos_dim_m], so we reapply indexes starting
    # at dim 1
    return lambda *a, **k: reindex(vmap_func(*a, *k), starting_dim=1)
