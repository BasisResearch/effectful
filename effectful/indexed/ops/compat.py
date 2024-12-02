import functools
from typing import Any, Dict, Iterable, Sequence, Set, TypeVar, Union

import torch

from ..internals.utils import name_to_sym

import effectful.indexed.ops.impl as impl

K = TypeVar("K")
T = TypeVar("T")


class IndexSet(Dict[str, Set[int]]):
    def __init__(self, **mapping: Union[int, Iterable[int]]):
        index_set = {}
        for k, vs in mapping.items():
            indexes = {vs} if isinstance(vs, int) else set(vs)
            if len(indexes) > 0:
                index_set[k] = indexes
        super().__init__(**index_set)

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    def __hash__(self):
        return hash(frozenset((k, frozenset(vs)) for k, vs in self.items()))

    def _to_handler(self):
        """Return an effectful handler that binds each index variable to a
        tensor of its possible index values.

        """
        return {
            k: functools.partial(lambda v: v, torch.tensor(list(v)))
            for k, v in self.items()
        }


def union(*indexsets: IndexSet) -> IndexSet:
    return IndexSet(
        **{
            k: set.union(*[vs[k] for vs in indexsets if k in vs])
            for k in set.union(*(set(vs) for vs in indexsets))
        }
    )


def indices_of(value: Any) -> IndexSet:
    return {k.__name__: v for (k, v) in impl.indices_of(value).items()}


def gather(value: torch.Tensor, indexset: IndexSet, **kwargs) -> torch.Tensor:
    indexset_vars = {name_to_sym(name): inds for name, inds in indexset.items()}
    return impl.gather(value, indexset_vars)


def stack(values: Sequence[torch.Tensor], name: str, **kwargs) -> torch.Tensor:
    return impl.stack(values, name_to_sym(name))


cond = impl.cond
cond_n = impl.cond_n
to_tensor = impl.to_tensor
Indexable = impl.Indexable
