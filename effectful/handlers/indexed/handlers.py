from typing import Optional, Protocol

import pyro
import torch
from typing_extensions import ParamSpec

from effectful.handlers.pyro import (
    NamedDistribution,
    get_sample_msg_device,
    pyro_sample,
)
from effectful.handlers.torch import sizesof
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

    def _pyro_sample(
        name: str,
        dist: pyro.distributions.torch_distribution.TorchDistributionMixin,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        obs = kwargs.get("obs")
        mask = get_mask(dist, obs, device=get_sample_msg_device(dist, obs), name=name)
        assert mask.shape == torch.Size([])

        # expand distribution with any named dimensions not already present
        dist_indices = sizesof(dist.sample())
        mask_extra_indices = {
            k: v for (k, v) in sizesof(mask).items() if k not in dist_indices
        }

        if len(mask_extra_indices) > 0:
            mask_expanded_shape = torch.Size([v for v in mask_extra_indices.values()])
            expanded_dist = dist.expand(mask_expanded_shape + dist.batch_shape)
            dist = NamedDistribution(expanded_dist, mask_extra_indices.keys())

        return fwd(None, name, dist, *args, mask=mask, **kwargs)

    return {pyro_sample: _pyro_sample}
