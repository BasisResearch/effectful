import typing
from typing import Optional

import pyro
import torch

from effectful.ops.core import defop


@defop
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
    return pyro.sample(
        name, fn, *args, obs=obs, obs_mask=obs_mask, infer=infer, **kwargs
    )


class PyroShim(pyro.poutine.messenger.Messenger):
    """
    Handler for Pyro that wraps all sample sites in a custom effectful type.
    """

    _current_site: Optional[str]
    _current_site_is_observed: Optional[bool]

    def _pyro_sample(self, msg: pyro.poutine.runtime.Message) -> None:
        if typing.TYPE_CHECKING:
            assert msg["type"] == "sample"
            assert msg["name"] is not None
            assert msg["infer"] is not None
            assert isinstance(
                msg["fn"], pyro.distributions.torch_distribution.TorchDistributionMixin
            )

        if (
            (
                getattr(self, "_current_site", None) == msg["name"]
                and getattr(self, "_current_site_is_observed", None)
                == msg["is_observed"]
            )
            or pyro.poutine.util.site_is_subsample(msg)
            or pyro.poutine.util.site_is_factor(msg)
        ):
            if "_markov_scope" in msg["infer"] and self._current_site:
                msg["infer"]["_markov_scope"].pop(self._current_site, None)
            return

        try:
            self._current_site = msg["name"]
            self._current_site_is_observed = msg["is_observed"]
            msg["value"] = pyro_sample(
                msg["name"],
                msg["fn"],
                obs=msg["value"] if msg["is_observed"] else None,
                infer=msg["infer"].copy(),
            )
        finally:
            self._current_site = None
            self._current_site_is_observed = None

        # flags to guarantee commutativity of condition, intervene, trace
        msg["stop"] = True
        msg["done"] = True
        msg["mask"] = False
        msg["is_observed"] = True
        msg["infer"]["is_auxiliary"] = True
        msg["infer"]["_do_not_trace"] = True
