from typing import Any, Callable, List, Optional, Protocol, Sequence

import pyro
import torch
from typing_extensions import ParamSpec

from ..handlers.pyro import pyro_sample
from ..ops.semantics import fwd
from ..ops.types import Interpretation, Operation
from .internals.handlers import _LazyPlateMessenger, get_sample_msg_device
from .ops import Indexable, indices_of, to_tensor

P = ParamSpec("P")


class PositionalDistribution(pyro.distributions.torch_distribution.TorchDistribution):
    """A distribution wrapper that lazily converts indexed dimensions to
    positional.

    """

    def __init__(
        self, base_dist: pyro.distributions.torch_distribution.TorchDistribution
    ):
        self.base_dist = base_dist
        self.indices = indices_of(base_dist)
        super().__init__()

    def _to_positional(self, value: torch.Tensor) -> torch.Tensor:
        # self.base_dist has shape: | batch_shape | event_shape | & named
        # assume value comes from base_dist with shape:
        # | sample_shape | batch_shape | event_shape | & named
        # return a tensor of shape | sample_shape | named | batch_shape | event_shape |

        n_named = len(self.indices)
        dims = list(range(n_named + len(value.shape)))

        n_event = len(self.event_shape)
        n_batch = len(self.base_dist.batch_shape)
        n_sample = len(value.shape) - n_batch - n_event

        event_dims = dims[len(dims) - n_event :]
        batch_dims = dims[len(dims) - n_event - n_batch : len(dims) - n_event]
        named_dims = dims[:n_named]
        sample_dims = dims[n_named : n_named + n_sample]

        # shape: | named | sample_shape | batch_shape | event_shape |
        pos_tensor = to_tensor(value, self.indices.keys())

        # shape: | sample_shape | named | batch_shape | event_shape |
        pos_tensor_r = torch.permute(
            pos_tensor, sample_dims + named_dims + batch_dims + event_dims
        )

        return pos_tensor_r

    def _from_positional(self, value: torch.Tensor) -> torch.Tensor:
        # maximal value shape: | sample_shape | named | batch_shape | event_shape |
        shape = self.shape()
        if len(value.shape) < len(shape):
            value = value.expand(shape)

        # check that the rightmost dimensions match
        assert value.shape[len(value.shape) - len(shape) :] == shape

        indexes: List[Any] = [slice(None)] * (len(value.shape))
        for i, n in enumerate(self.indices.keys()):
            indexes[
                len(value.shape)
                - len(self.indices)
                - len(self.event_shape)
                - len(self.base_dist.batch_shape)
                + i
            ] = n()

        return Indexable(value)[tuple(indexes)]

    @property
    def batch_shape(self):
        return (
            torch.Size([len(s) for s in self.indices.values()])
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


class NamedDistribution(pyro.distributions.torch_distribution.TorchDistribution):
    """A distribution wrapper that lazily names leftmost dimensions."""

    def __init__(
        self,
        base_dist: pyro.distributions.torch_distribution.TorchDistribution,
        names: Sequence[Operation[[], int]],
    ):
        """
        :param base_dist: A distribution with batch dimensions.

        :param names: A list of names.

        """
        assert len(names) <= len(base_dist.batch_shape)

        self.base_dist = base_dist
        self.names = names
        super().__init__()

    def _to_named(self, value: torch.Tensor, offset=0) -> torch.Tensor:
        return Indexable(value)[
            tuple([slice(None)] * offset + [n() for n in self.names])
        ]

    def _from_named(self, value: torch.Tensor) -> torch.Tensor:
        return to_tensor(value, self.names)

    @property
    def batch_shape(self):
        return self.base_dist.batch_shape[len(self.names) :]

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
        return f"NamedDistribution({self.base_dist}, {self.names})"

    def sample(self, sample_shape=torch.Size()):
        return self._to_named(
            self.base_dist.sample(sample_shape), offset=len(sample_shape)
        )

    def rsample(self, sample_shape=torch.Size()):
        return self._to_named(
            self.base_dist.rsample(sample_shape), offset=len(sample_shape)
        )

    def log_prob(self, value):
        return self._to_named(self.base_dist.log_prob(self._from_named(value)))

    def enumerate_support(self, expand=True):
        return self._to_named(self.base_dist.enumerate_support(expand))


def _indexed_pyro_sample_handler(
    name: str,
    dist: pyro.distributions.torch_distribution.TorchDistributionMixin,
    infer: Optional[pyro.poutine.runtime.InferDict] = None,
    obs: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    infer = infer or {}

    if "_index_expanded" in infer:
        return fwd(None)

    pdist = PositionalDistribution(dist)
    obs = None if obs is None else pdist._to_positional(obs)

    shape_len = len(pdist.shape())
    plates = []
    for dim_offset, (var, size) in enumerate(pdist.indices.items()):
        plate = _LazyPlateMessenger(
            str(var),
            dim=-shape_len + dim_offset,
            size=len(size),
        )
        plate.__enter__()
        plates.append(plate)

    infer["_index_expanded"] = True  # type: ignore

    try:
        t = pyro.sample(name, pdist, infer=infer, obs=obs, **kwargs)
        return pdist._from_positional(t)
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
