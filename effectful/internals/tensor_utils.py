from typing import Any, Callable, Collection, Generic, Mapping, Protocol, TypeVar

import tree
from typing_extensions import ParamSpec

from effectful.ops.types import Operation

P = ParamSpec("P")
T = TypeVar("T")
S = TypeVar("S")


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
            assert not any(k is Ellipsis for k in key[i + 1 :]), (
                "only one Ellipsis allowed"
            )

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


def _indexed_func_wrapper(
    func: Callable[P, T], getitem, to_tensor, sizesof
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
            t_ = to_tensor(t, *i.sizes.keys())
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
                lambda t, i: getitem(t, index_expr(i)), ret, indexes
            )
        else:
            indexed_ret = getitem(ret, index_expr(indexes))

        return indexed_ret

    return deindexed, reindex
