import functools

import torch
import tree

from ..internals.sugar import (
    EagerTensorTerm,
    _register_torch_op,
    torch_getitem,
)
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
    return torch_func_wrapper(torch.func.jacfwd, func, *args, **kwargs)


def jacrev(func, *args, **kwargs):
    return torch_func_wrapper(torch.func.jacrev, func, *args, **kwargs)


def hessian(func, *args, **kwargs):
    return torch_func_wrapper(torch.func.hessian, func, *args, **kwargs)


def jvp(func, indexed_primals, indexed_tangents, **kwargs):
    # unembed doesn't like this lambda, so hide it by partially applying
    jvp_func = functools.partial(
        torch.func.jvp,
        lambda *unindexed_primals: add_indexes(
            func,
            (unindexed_primals, {}),
            (indexed_primals, {}),
        ),
    )

    return _register_torch_op(jvp_func)(
        indexed_primals,
        indexed_tangents,
        **kwargs,
    )


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
