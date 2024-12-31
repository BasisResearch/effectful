from typing import Any, Callable, Optional, Protocol

import pyro
import torch
from typing_extensions import ParamSpec

from ..handlers.pyro import pyro_sample
from ..ops.core import Interpretation
from ..ops.handler import fwd
from .distributions import NamedDistribution
from .internals.handlers import get_sample_msg_device
from .ops import indices_of

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
        dist_indices = indices_of(dist)
        mask_extra_indices = {
            k: v for (k, v) in indices_of(mask).items() if k not in dist_indices
        }

        if len(mask_extra_indices) > 0:
            mask_expanded_shape = torch.Size(
                [len(v) for v in mask_extra_indices.values()]
            )
            expanded_dist = dist.expand(mask_expanded_shape + dist.batch_shape)
            dist = NamedDistribution(expanded_dist, mask_extra_indices.keys())

        return fwd(None, name, dist, *args, mask=mask, **kwargs)

    return {pyro_sample: _pyro_sample}


@pyro.poutine.block()
@pyro.validation_enabled(False)
@torch.no_grad()
def guess_max_plate_nesting(
    model: Callable[P, Any], guide: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
) -> int:
    """
    Guesses the maximum plate nesting level by running `pyro.infer.Trace_ELBO`

    :param model: Python callable containing Pyro primitives.
    :type model: Callable[P, Any]
    :param guide: Python callable containing Pyro primitives.
    :type guide: Callable[P, Any]
    :return: maximum plate nesting level
    :rtype: int
    """
    elbo = pyro.infer.Trace_ELBO()
    elbo._guess_max_plate_nesting(model, guide, args, kwargs)
    return elbo.max_plate_nesting
