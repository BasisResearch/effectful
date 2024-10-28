import functools

import torch
import tree

from ..internals.sugar import (
    _register_torch_op,
    torch_getitem,
    sizesof,
)
from .ops import to_tensor


def indexed_func_wrapper(func):
    # index expressions for the result of the function
    indexes = None

    # hide index lists from tree.map_structure
    class Indexes:
        def __init__(self, indexes):
            self.indexes = indexes

    # strip named indexes from the result of the function and store them
    def deindexed(*args, **kwargs):
        nonlocal indexes
        ret = func(*args, **kwargs)
        indexes = tree.map_structure(lambda t: Indexes(list(sizesof(t).keys())), ret)
        return tree.map_structure(lambda t, i: to_tensor(t, i.indexes), ret, indexes)

    # reapply the stored indexes to a result
    def reindex(ret):
        if tree.is_nested(ret):
            return tree.map_structure(
                lambda t, i: torch_getitem(t, tuple(x() for x in i.indexes)),
                ret,
                indexes,
            )
        if indexes is not None:
            return torch_getitem(ret, tuple(i() for i in indexes.indexes))
        return ret

    return deindexed, reindex


def grad(func, *args, **kwargs):
    """Compute the gradient of a function with respect to its arguments. This is
    a wrapper around `torch.func.grad` that allows the function to be called
    with indexed arguments.

    """
    (deindexed_func, reindex) = indexed_func_wrapper(func)
    f = _register_torch_op(torch.func.grad(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(f(*a, *k))


def jacfwd(func, *args, **kwargs):
    (deindexed_func, reindex) = indexed_func_wrapper(func)
    jacobian = _register_torch_op(torch.func.jacfwd(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(jacobian(*a, *k))


def jacrev(func, *args, **kwargs):
    (deindexed_func, reindex) = indexed_func_wrapper(func)
    jacobian = _register_torch_op(torch.func.jacrev(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(jacobian(*a, *k))


def hessian(func, *args, **kwargs):
    (deindexed_func, reindex) = indexed_func_wrapper(func)
    h = _register_torch_op(torch.func.hessian(deindexed_func, *args, **kwargs))
    return lambda *a, **k: reindex(h(*a, *k))


def jvp(func, *args, **kwargs):
    (deindexed_func, reindex) = indexed_func_wrapper(func)

    # hide deindexed_func from _register_torch_op
    jvp_func = functools.partial(torch.func.jvp, deindexed_func)
    ret = _register_torch_op(jvp_func)(*args, **kwargs)
    return tree.map_structure(reindex, ret)


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
