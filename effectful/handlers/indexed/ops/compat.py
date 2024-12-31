import functools
from typing import Any, Dict, Iterable, Set, TypeVar, Union

import torch

import effectful.handlers.indexed.ops.impl as impl
from effectful.ops.syntax import defop
from effectful.ops.types import Operation

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
    return IndexSet(**{k.__name__: v for (k, v) in impl.indices_of(value).items()})  # type: ignore


@functools.lru_cache(maxsize=None)
def name_to_sym(name: str) -> Operation[[], int]:
    return defop(int, name=name)


def gather(value: torch.Tensor, indexset: IndexSet, **kwargs) -> torch.Tensor:
    indexset_vars = {name_to_sym(name): inds for name, inds in indexset.items()}
    return impl.gather(value, impl.IndexSet(indexset_vars))


def stack(
    values: Union[tuple[torch.Tensor, ...], list[torch.Tensor]], name: str, **kwargs
) -> torch.Tensor:
    return impl.stack(values, name_to_sym(name))


def to_tensor(value: torch.Tensor, indices: list[str]) -> torch.Tensor:
    from effectful.handlers.torch import to_tensor

    return to_tensor(value, [name_to_sym(name) for name in indices])


cond = impl.cond
cond_n = impl.cond_n
