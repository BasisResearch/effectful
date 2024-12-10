from typing import Any, Callable, List, Mapping, Optional, Protocol, Sequence

import pyro
import torch
from typing_extensions import ParamSpec

from ..handlers.pyro import pyro_sample
from ..ops.core import Interpretation, Operation
from ..ops.handler import fwd
from .internals.handlers import _LazyPlateMessenger, get_sample_msg_device
from .ops import Indexable, IndexSet, indices_of, to_tensor, union

P = ParamSpec("P")


class Naming:
    """
    A mapping from dimensions (indexed from the right) to names.
    """

    def __init__(self, name_to_dim: Mapping[Operation[[], int], int]):
        assert all(v < 0 for v in name_to_dim.values())
        self.name_to_dim = name_to_dim

    @staticmethod
    def from_shape(names: Sequence[Operation[[], int]], event_dims: int) -> "Naming":
        """Create a naming from a set of indices and the number of event dimensions.

        The resulting naming converts tensors of shape
        | batch_shape | named | event_shape |
        to tensors of shape | batch_shape | event_shape |, | named |.

        """
        assert event_dims >= 0
        return Naming({n: -event_dims - len(names) + i for i, n in enumerate(names)})

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        indexes: List[Any] = [slice(None)] * (len(value.shape))
        for n, d in self.name_to_dim.items():
            indexes[len(value.shape) + d] = n()
        return Indexable(value)[tuple(indexes)]

    def __repr__(self):
        return f"Naming({self.name_to_dim})"


class PositionalDistribution(pyro.distributions.torch_distribution.TorchDistribution):
    """A distribution wrapper that lazily converts indexed dimensions to
    positional.

    """

    def __init__(
        self, base_dist: pyro.distributions.torch_distribution.TorchDistribution
    ):
        self.base_dist = base_dist
        self.indices = indices_of(base_dist)

        n_base = len(base_dist.batch_shape) + len(base_dist.event_shape)
        self.naming = Naming.from_shape(self.indices.keys(), n_base)

        super().__init__()

    def _to_positional(self, value: torch.Tensor) -> torch.Tensor:
        # self.base_dist has shape: | batch_shape | event_shape | & named
        # assume value comes from base_dist with shape:
        # | sample_shape | batch_shape | event_shape | & named
        # return a tensor of shape | sample_shape | named | batch_shape | event_shape |
        n_named = len(self.indices)
        dims = list(range(n_named + len(value.shape)))

        n_base = len(self.event_shape) + len(self.base_dist.batch_shape)
        n_sample = len(value.shape) - n_base

        base_dims = dims[len(dims) - n_base :]
        named_dims = dims[:n_named]
        sample_dims = dims[n_named : n_named + n_sample]

        # shape: | named | sample_shape | batch_shape | event_shape |
        # TODO: replace with something more efficient
        pos_tensor = to_tensor(value, self.indices.keys())

        # shape: | sample_shape | named | batch_shape | event_shape |
        pos_tensor_r = torch.permute(pos_tensor, sample_dims + named_dims + base_dims)

        return pos_tensor_r

    def _from_positional(self, value: torch.Tensor) -> torch.Tensor:
        # maximal value shape: | sample_shape | named | batch_shape | event_shape |
        return self.naming.apply(value)

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

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
        return self._to_positional(
            self.base_dist.log_prob(self._from_positional(value))
        )

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
        self.base_dist = base_dist
        self.names = names

        assert len(names) <= len(base_dist.batch_shape)
        base_indices = indices_of(base_dist)
        assert not any(n in base_indices for n in names)

        n_base = len(base_dist.batch_shape) + len(base_dist.event_shape)
        self.naming = Naming.from_shape(names, n_base - len(names))
        super().__init__()

    def _to_named(self, value: torch.Tensor, offset=0) -> torch.Tensor:
        return self.naming.apply(value)

    def _from_named(self, value: torch.Tensor) -> torch.Tensor:
        pos_value = to_tensor(value, self.names)

        dims = list(range(len(pos_value.shape)))

        n_base = len(self.event_shape) + len(self.batch_shape)
        n_named = len(self.names)
        n_sample = len(pos_value.shape) - n_base - n_named

        base_dims = dims[len(dims) - n_base :]
        named_dims = dims[:n_named]
        sample_dims = dims[n_named : n_named + n_sample]

        pos_tensor_r = torch.permute(pos_value, sample_dims + named_dims + base_dims)

        return pos_tensor_r

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

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
        t = self._to_named(
            self.base_dist.sample(sample_shape), offset=len(sample_shape)
        )
        assert set(indices_of(t).keys()) == set(self.names) and t.shape == self.shape()
        return t

    def rsample(self, sample_shape=torch.Size()):
        return self._to_named(
            self.base_dist.rsample(sample_shape), offset=len(sample_shape)
        )

    def log_prob(self, value):
        v1 = self._from_named(value)
        v2 = self.base_dist.log_prob(v1)
        v3 = self._to_named(v2)
        return v3

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

    print("indexed_sample", name, dist, infer, obs)

    if "_index_dist" in infer:
        return fwd(None)

    # pdist shape: | named1 | batch_shape | event_shape |
    # obs shape: | batch_shape | event_shape |, | named2 | where named2 may overlap named1
    pdist = PositionalDistribution(dist)
    assert indices_of(pdist) == IndexSet({})
    naming = pdist.naming

    # convert remaining named dimensions to positional
    obs_indices = indices_of(obs)
    pos_obs = obs
    if obs is not None:
        # ensure obs has the same | batch_shape | event_shape | as dist
        # it should now differ only in named dimensions
        batch_dims = dist.shape()
        if len(pos_obs.shape) < len(batch_dims):
            pos_obs = pos_obs.expand(batch_dims)

        name_to_dim = {}
        for i, (k, v) in enumerate(reversed(pdist.indices.items())):
            if k in obs_indices:
                pos_obs = to_tensor(pos_obs, [k])
            else:
                pos_obs = pos_obs.expand((len(v),) + pos_obs.shape)
            name_to_dim[k] = -len(batch_dims) - i - 1

        n_batch_and_dist_named = len(pos_obs.shape)
        for i, k in enumerate(reversed(indices_of(pos_obs).keys())):
            pos_obs = to_tensor(pos_obs, [k])
            name_to_dim[k] = -n_batch_and_dist_named - i - 1

        naming = Naming(name_to_dim)

    # obs shape: | named2 | named1 | batch_shape | event_shape |
    indices = union(IndexSet(obs_indices), pdist.indices)
    plates = []
    for var, dim in naming.name_to_dim.items():
        plate = _LazyPlateMessenger(str(var), dim=dim, size=len(indices[var]))
        plate.__enter__()
        plates.append(plate)

    infer["_index_naming"] = naming  # type: ignore

    try:
        assert indices_of(pos_obs) == IndexSet({})
        t = pyro.sample(name, pdist, infer=infer, obs=pos_obs, **kwargs)
        return naming.apply(t)
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
