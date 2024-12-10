from typing import Optional, Union

import pyro
import pyro.infer.reparam
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame, IndepMessenger

from ..ops.compat import indices_of, union


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

        if self._orig_name in union(indices_of(msg["value"]), indices_of(msg["fn"])):
            super()._process_message(msg)


def get_sample_msg_device(
    dist: pyro.distributions.torch_distribution.TorchDistribution,
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
