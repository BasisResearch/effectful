from typing import Any, Callable, Optional, Protocol

import pyro
import torch
from typing_extensions import ParamSpec

from ..handlers.pyro import pyro_sample
from ..ops.core import Interpretation
from ..ops.handler import fwd
from .internals.handlers import _LazyPlateMessenger, get_sample_msg_device
from .ops import Indexable, indices_of, to_tensor

P = ParamSpec("P")


class PositionalDistribution(pyro.distributions.torch_distribution.TorchDistribution):
    """A distribution wrapper that lazily converts indexed dimensions to
    positional.

    """

    def __init__(self, base_dist):
        self.base_dist = base_dist
        self._names = None
        self._vars = None
        self.enumerate_support = base_dist.enumerate_support

    def _get_vars_sizes(self, value=None):
        if self._vars is None:
            if value is None:
                value = self.base_dist.sample()

            free = indices_of(value)
            self._vars = list(free.keys())
            self._sizes = [free[v] for v in self._vars]

        return (self._vars, self._sizes)

    def _to_positional(self, value):
        (vars_, _) = self._get_vars_sizes(value)
        return to_tensor(value, vars_)

    def _from_positional(self, value):
        (vars_, _) = self._get_vars_sizes()
        return Indexable(value)[tuple(v() for v in vars_)]

    @property
    def batch_shape(self):
        return (
            torch.Size([len(s) for s in self._get_vars_sizes()[1]])
            + self.base_dist.batch_shape
        )

    @property
    def event_shape(self):
        return self.base_dist.event_shape

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @property
    def arg_constraints(self):
        return self.base_dist.arg_constraints

    def __repr__(self):
        return f"PositionalDistribution({self.base_dist})"

    def sample(self, sample_shape=torch.Size()):
        return self._to_positional(self.base_dist.sample(sample_shape))

    def rsample(self, sample_shape=torch.Size()):
        return self._to_positional(self.base_dist.rsample(sample_shape))

    def log_prob(self, value):
        return self.base_dist.log_prob(self._from_positional(value))

    def enumerate_support(self, expand=True):
        return self._to_positional(self.base_dist.enumerate_support(expand))


def _indexed_pyro_sample_handler(
    name: str,
    dist: pyro.distributions.torch_distribution.TorchDistributionMixin,
    infer: Optional[pyro.poutine.runtime.InferDict] = None,
    **kwargs,
) -> torch.Tensor:
    infer = infer or {}

    if "_index_expanded" in infer:
        return fwd(None)

    pdist = PositionalDistribution(dist)
    (vars_, sizes) = pdist._get_vars_sizes()

    shape_len = len(pdist.shape())
    plates = []
    for dim_offset, (var, size) in enumerate(zip(vars_, sizes)):
        plate = _LazyPlateMessenger(
            str(var),
            dim=-shape_len + dim_offset,
            size=len(size),
        )
        plate.__enter__()
        plates.append(plate)

    infer["_index_expanded"] = True  # type: ignore

    try:
        return pdist._from_positional(pyro.sample(name, pdist, infer=infer, **kwargs))
    finally:
        for plate in reversed(plates):
            plate.__exit__(None, None, None)


#: Allow distributions with indexed batch dimensions to be used with
#: `pyro.sample` by lazily converting the indexed dimensions to positional
#: dimensions.
indexed = {pyro_sample: _indexed_pyro_sample_handler}


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
