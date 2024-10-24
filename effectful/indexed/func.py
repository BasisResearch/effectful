import functools

import torch
import tree

from ..internals.sugar import EagerTensorTerm, _register_torch_op, torch_getitem
from .ops import to_tensor


def add_indexes(func, unindexed, indexed):
    def add_single_index(u, i):
        """Given an unindexed tensor and an index expression, expand the tensor
        to have leading dimensions corresponding to the named indexes and create
        an index expression.

        """
        if isinstance(i, EagerTensorTerm):
            indexes = i.indices()
            return torch_getitem(
                u[(None,) * len(indexes) + (slice(None),) * len(i.shape)],
                tuple(indexes),
            )
        return u

    reindexed = tree.map_structure(add_single_index, unindexed, indexed)
    reindexed_args, reindexed_kwargs = reindexed
    ret = func(*reindexed_args, **reindexed_kwargs)
    return tree.map_structure(lambda t: t.to_tensor().reshape(t.shape), ret)


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
        indexes = tree.map_structure(
            lambda t: Indexes(t.indices()) if isinstance(t, EagerTensorTerm) else None,
            ret,
        )
        return tree.map_structure(to_tensor, ret)

    # reapply the stored indexes to a result
    def reindex(ret):
        if tree.is_nested(ret):
            return tree.map_structure(
                lambda t, i: torch_getitem(t, i.indexes) if i is not None else t,
                ret,
                indexes,
            )
        if indexes is not None:
            return torch_getitem(ret, indexes.indexes)
        return ret

    return deindexed, reindex


def torch_func_wrapper(torch_func, func, *args, **kwargs):
    return lambda *indexed_args, **indexed_kwargs: _register_torch_op(
        torch_func(
            lambda *unindexed_args, **unindexed_kwargs: add_indexes(
                func, (unindexed_args, unindexed_kwargs), (indexed_args, indexed_kwargs)
            ),
            *args,
            **kwargs,
        )
    )(*indexed_args, **indexed_kwargs)


def grad(func, *args, **kwargs):
    """Compute the gradient of a function with respect to its arguments. This is
    a wrapper around `torch.func.grad` that allows the function to be called
    with indexed arguments.

    """
    return torch_func_wrapper(torch.func.grad, func, *args, **kwargs)


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
    unpacked_primals = [
        (t.args[0], t.args[1]) if isinstance(t, EagerTensorTerm) else None
        for t in indexed_primals
    ]

    indexed_result = None

    def repack_primals(primals):
        return [
            (
                torch_getitem(p, unpacked_primals[i][1])
                if unpacked_primals[i] is not None
                else p
            )
            for i, p in enumerate(primals)
        ]

    def wrapper(*primals):
        nonlocal indexed_result
        indexed_result = func(*repack_primals(primals))
        return tree.map_structure(to_tensor, indexed_result)

    unindexed_primals = [to_tensor(t) for t in indexed_primals]
    _, vjpfunc = torch.func.vjp(wrapper, *unindexed_primals, **kwargs)

    def vjpfunc_wrapper(*tangents):
        unindexed_tangents = tree.map_structure(to_tensor, tangents)
        grads = vjpfunc(*unindexed_tangents)
        return repack_primals(grads)

    return indexed_result, vjpfunc_wrapper
