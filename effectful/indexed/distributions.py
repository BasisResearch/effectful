from typing import Any, List, Mapping, Sequence

import pyro
import torch

from ..ops.core import Operation
from .ops import Indexable, indices_of, to_tensor


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

    @property
    def support(self):
        return self.base_dist.support

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

    @property
    def support(self):
        return self.base_dist.support

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
