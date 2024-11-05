import numbers
from typing import Optional, TypeVar, Union, Dict

import pyro
import pyro.infer.reparam
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame, IndepMessenger

from ..ops.core import Term
from ..internals.sugar import sizesof
from .ops import IndexSet, indices_of, union, get_index_plates

K = TypeVar("K")
T = TypeVar("T")


@indices_of.register
def _indices_of_number(value: numbers.Number, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_bool(value: bool, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_none(value: None, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_tuple(value: tuple, **kwargs) -> IndexSet:
    if all(isinstance(v, int) for v in value):
        return indices_of(torch.Size(value), **kwargs)
    return union(*(indices_of(v, **kwargs) for v in value))


@indices_of.register
def _indices_of_shape(value: torch.Size, **kwargs) -> IndexSet:
    name_to_dim = (
        kwargs["name_to_dim"]
        if "name_to_dim" in kwargs
        else {name: f.dim for name, f in get_index_plates().items()}
    )
    value = value[: len(value) - kwargs.get("event_dim", 0)]
    return IndexSet(
        **{
            name: set(range(value[dim]))
            for name, dim in name_to_dim.items()
            if -dim <= len(value) and value[dim] > 1
        }
    )


@indices_of.register
def _indices_of_term(value: Term, **kwargs) -> IndexSet:
    return IndexSet(**{k._name: set(range(v)) for (k, v) in sizesof(value).items()})


@indices_of.register
def _indices_of_tensor(value: torch.Tensor, **kwargs) -> IndexSet:
    if isinstance(value, Term):
        return _indices_of_term(value, **kwargs)
    return indices_of(value.shape, **kwargs)


@indices_of.register
def _indices_of_distribution(
    value: pyro.distributions.Distribution, **kwargs
) -> IndexSet:
    kwargs.pop("event_dim", None)
    return indices_of(value.batch_shape, event_dim=0, **kwargs)


@indices_of.register(dict)
def _indices_of_state(value: Dict[K, T], *, event_dim: int = 0, **kwargs) -> IndexSet:
    return union(
        *(indices_of(value[k], event_dim=event_dim, **kwargs) for k in value.keys())
    )


class _LazyPlateMessenger(IndepMessenger):
    prefix: str = "__index_plate__"

    def __init__(self, name: str, *args, **kwargs):
        self._orig_name: str = name
        super().__init__(f"{self.prefix}_{name}", *args, **kwargs)

    @property
    def frame(self) -> CondIndepStackFrame:
        return CondIndepStackFrame(
            name=self.name, dim=self.dim, size=self.size, counter=0
        )

    def _process_message(self, msg):
        if msg["type"] not in ("sample",) or pyro.poutine.util.site_is_subsample(msg):
            return
        if self._orig_name in union(
            indices_of(msg["value"], event_dim=msg["fn"].event_dim),
            indices_of(msg["fn"]),
        ):
            super()._process_message(msg)


def get_sample_msg_device(
    dist: pyro.distributions.Distribution,
    value: Optional[Union[torch.Tensor, float, int, bool]],
) -> torch.device:
    # some gross code to infer the device of the obs_mask tensor
    #   because distributions are hard to introspect
    if isinstance(value, torch.Tensor):
        return value.device
    else:
        dist_ = dist
        while hasattr(dist_, "base_dist"):
            dist_ = dist_.base_dist
        for param_name in dist_.arg_constraints.keys():
            p = getattr(dist_, param_name)
            if isinstance(p, torch.Tensor):
                return p.device
    raise ValueError(f"could not infer device for {dist} and {value}")


@pyro.poutine.runtime.effectful(type="add_indices")
def add_indices(indexset: IndexSet) -> IndexSet:
    return indexset
