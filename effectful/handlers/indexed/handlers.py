from typing import Optional, Protocol

import pyro
import torch
from typing_extensions import ParamSpec

from effectful.handlers.pyro import get_sample_msg_device, pyro_sample
from effectful.ops.semantics import fwd
from effectful.ops.types import Interpretation

P = ParamSpec("P")


class GetMask(Protocol):
    def __call__(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor],
        device: torch.device = torch.device("cpu"),
        name: Optional[str] = None,
    ): ...


def dependent_mask(get_mask: GetMask) -> Interpretation[torch.Tensor, torch.Tensor]:
    """
    Helper function for effect handlers that select a subset of worlds.
    """

    def pyro_sample_handler(
        name: str,
        dist: pyro.distributions.torch_distribution.TorchDistributionMixin,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        obs = kwargs.get("obs")
        device = get_sample_msg_device(dist, obs)
        mask = get_mask(dist, obs, device=device, name=name)
        dist = dist.expand(torch.broadcast_shapes(dist.batch_shape, mask.shape))
        return fwd(None, name, dist, *args, **kwargs)

    return {pyro_sample: pyro_sample_handler}


class DependentMaskMessenger(pyro.poutine.messenger.Messenger):
    """
    Abstract base class for effect handlers that select a subset of worlds.
    """

    def get_mask(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor],
        device: torch.device = torch.device("cpu"),
        name: Optional[str] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _pyro_sample(self, msg: pyro.poutine.runtime.Message) -> None:
        if pyro.poutine.util.site_is_subsample(msg):
            return

        assert isinstance(
            msg["fn"], pyro.distributions.torch_distribution.TorchDistributionMixin
        )

        device = get_sample_msg_device(msg["fn"], msg["value"])
        name = msg["name"] if "name" in msg else None
        mask = self.get_mask(msg["fn"], msg["value"], device=device, name=name)
        msg["mask"] = mask if msg["mask"] is None else msg["mask"] & mask

        # expand distribution to make sure two copies of a variable are sampled
        msg["fn"] = msg["fn"].expand(
            torch.broadcast_shapes(msg["fn"].batch_shape, mask.shape)
        )
