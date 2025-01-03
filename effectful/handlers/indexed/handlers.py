from typing import Any, Callable, Optional, Protocol, Union

import pyro
import torch
from typing_extensions import ParamSpec

from effectful.handlers.pyro import NamedDistribution, pyro_sample
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
