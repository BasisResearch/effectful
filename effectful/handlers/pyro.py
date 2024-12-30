import typing
from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

import pyro
import torch
from typing_extensions import ParamSpec

from effectful.handlers.torch import Indexable, sizesof, to_tensor
from effectful.ops.semantics import fwd
from effectful.ops.syntax import defop
from effectful.ops.types import Operation

P = ParamSpec("P")


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

    def _pyro_sample(self, msg: pyro.poutine.runtime.Message) -> None:
        if typing.TYPE_CHECKING:
            assert msg["type"] == "sample"
            assert msg["name"] is not None
            assert msg["infer"] is not None
            assert isinstance(
                msg["fn"], pyro.distributions.torch_distribution.TorchDistributionMixin
            )

        if (
            getattr(self, "_current_site", None) == msg["name"]
            or pyro.poutine.util.site_is_subsample(msg)
            or pyro.poutine.util.site_is_factor(msg)
        ):
            if "_markov_scope" in msg["infer"] and self._current_site:
                msg["infer"]["_markov_scope"].pop(self._current_site, None)
            return

        try:
            self._current_site = msg["name"]
            msg["value"] = pyro_sample(
                msg["name"],
                msg["fn"],
                obs=msg["value"] if msg["is_observed"] else None,
                infer=msg["infer"].copy(),
            )
        finally:
            self._current_site = None

        # flags to guarantee commutativity of condition, intervene, trace
        msg["stop"] = True
        msg["done"] = True
        msg["mask"] = False
        msg["is_observed"] = True
        msg["infer"]["is_auxiliary"] = True
        msg["infer"]["_do_not_trace"] = True


class PositionalDistribution(pyro.distributions.torch_distribution.TorchDistribution):
    """A distribution wrapper that lazily converts indexed dimensions to
    positional.

    """

    indices: Mapping[Operation[[], int], int]

    def __init__(
        self, base_dist: pyro.distributions.torch_distribution.TorchDistribution
    ):
        self.base_dist = base_dist
        self.indices = sizesof(base_dist.sample())
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
        return tuple(self.indices.values()) + self.base_dist.batch_shape

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


class _LazyPlateMessenger(pyro.poutine.indep_messenger.IndepMessenger):
    prefix: str = "__index_plate__"

    def __init__(self, name: str, *args, **kwargs):
        self._orig_name: str = name
        super().__init__(f"{self.prefix}_{name}", *args, **kwargs)

    @property
    def frame(self) -> pyro.poutine.indep_messenger.CondIndepStackFrame:
        return pyro.poutine.indep_messenger.CondIndepStackFrame(
            name=self.name, dim=self.dim, size=self.size, counter=0
        )

    def _process_message(self, msg):
        if msg["type"] not in ("sample",) or pyro.poutine.util.site_is_subsample(msg):
            return

        super()._process_message(msg)


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
            size=size,
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
