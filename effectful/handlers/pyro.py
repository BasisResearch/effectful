from typing import Dict, Optional, Any, Mapping

import torch
import pyro

from effectful.ops.core import Operation


@Operation
def pyro_sample(
    name: str,
    fn: pyro.distributions.torch_distribution.TorchDistributionMixin,
    *args,
    obs: Optional[torch.Tensor] = None,
    obs_mask: Optional[torch.BoolTensor] = None,
    infer: Optional[pyro.poutine.runtime.InferDict] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Operation to sample from a Pyro distribution.
    """
    return pyro.sample(name, fn, *args, obs=obs, obs_mask=obs_mask, infer=infer, **kwargs)


class PyroShim(pyro.poutine.messenger.Messenger):
    """
    Handler for Pyro that wraps all sample sites in a custom effectful type.
    """
    pyro_type_prefix: str = "__effectful"

    _current_site: Optional[str]

    def _pyro_sample(self, msg: pyro.poutine.runtime.Message) -> None:
        if self._current_site == msg["name"] or \
                pyro.poutine.util.site_is_subsample(msg) or \
                pyro.poutine.util.site_is_factor(msg):
            return

        msg["type"] = f"{self.pyro_type_prefix}_{msg['type']}"
        msg["stop"] = True
        msg["done"] = True
        try:
            self._current_site = msg["name"]
            msg["value"] = pyro_sample(
                msg["name"],
                msg["fn"],
                obs=msg["value"] if msg["is_observed"] else None,
                obs_mask=msg["obs_mask"],
                infer=msg["infer"],
            )
        finally:
            self._current_site = None
