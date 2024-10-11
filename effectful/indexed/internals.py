import numbers
from typing import Dict, Hashable, Optional, TypeVar, Union

import pyro
import pyro.infer.reparam
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame, IndepMessenger

from .ops import IndexSet, cond, gather, get_index_plates, indices_of, scatter, union

K = TypeVar("K")
T = TypeVar("T")


@scatter.register
def _scatter_number(
    value: numbers.Number,
    indexset: IndexSet,
    *,
    result: Optional[torch.Tensor] = None,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> Union[numbers.Number, torch.Tensor]:
    assert event_dim is None or event_dim == 0
    return scatter(
        torch.as_tensor(value),
        indexset,
        result=result,
        event_dim=event_dim,
        name_to_dim=name_to_dim,
    )


@scatter.register
def _scatter_tensor(
    value: torch.Tensor,
    indexset: IndexSet,
    *,
    result: Optional[torch.Tensor] = None,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> torch.Tensor:
    if event_dim is None:
        event_dim = 0

    if name_to_dim is None:
        name_to_dim = {name: f.dim for name, f in get_index_plates().items()}

    value = gather(value, indexset, event_dim=event_dim, name_to_dim=name_to_dim)
    indexset = union(
        indexset, indices_of(value, event_dim=event_dim, name_to_dim=name_to_dim)
    )

    if result is None:
        index_plates = get_index_plates()
        result_shape = list(
            torch.broadcast_shapes(
                value.shape,
                (1,) * max([event_dim - f.dim for f in index_plates.values()] + [0]),
            )
        )
        for name, indices in indexset.items():
            result_shape[name_to_dim[name] - event_dim] = index_plates[name].size
        result = value.new_zeros(result_shape)

    index = [
        torch.arange(0, result.shape[i], dtype=torch.long).reshape(
            (-1,) + (1,) * (len(result.shape) - 1 - i)
        )
        for i in range(len(result.shape))
    ]
    for name, indices in indexset.items():
        if result.shape[name_to_dim[name] - event_dim] > 1:
            index[name_to_dim[name] - event_dim] = torch.tensor(
                list(sorted(indices)), device=value.device, dtype=torch.long
            ).reshape((-1,) + (1,) * (event_dim - name_to_dim[name] - 1))

    result[tuple(index)] = value
    return result


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
