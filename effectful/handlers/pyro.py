import typing
import warnings
from typing import Any, Collection, List, Mapping, Optional, Tuple

try:
    import pyro
except ImportError:
    raise ImportError("Pyro is required to use effectful.handlers.pyro.")

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required to use effectful.handlers.pyro.")

from typing_extensions import ParamSpec

from effectful.handlers.torch import Indexable, sizesof, to_tensor
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
    mask: Optional[torch.BoolTensor] = None,
    infer: Optional[pyro.poutine.runtime.InferDict] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Operation to sample from a Pyro distribution. See :func:`pyro.sample`.
    """
    with pyro.poutine.mask(mask=mask if mask is not None else True):
        return pyro.sample(
            name, fn, *args, obs=obs, obs_mask=obs_mask, infer=infer, **kwargs
        )


class PyroShim(pyro.poutine.messenger.Messenger):
    """Pyro handler that wraps all sample sites in a custom effectful type.

    .. note::

      This handler should be installed around any Pyro model that you want to
      use effectful handlers with.

    **Example usage**:

    >>> import pyro.distributions as dist
    >>> from effectful.ops.semantics import fwd, handler
    >>> torch.distributions.Distribution.set_default_validate_args(False)

    It can be used as a decorator:

    >>> @PyroShim()
    ... def model():
    ...     return pyro.sample("x", dist.Normal(0, 1))

    It can also be used as a context manager:

    >>> with PyroShim():
    ...     x = pyro.sample("x", dist.Normal(0, 1))

    When :class:`PyroShim` is installed, all sample sites perform the
    :func:`pyro_sample` effect, which can be handled by an effectful
    interpretation.

    >>> def log_sample(name, *args, **kwargs):
    ...     print(f"Sampled {name}")
    ...     return fwd()

    >>> with PyroShim(), handler({pyro_sample: log_sample}):
    ...     x = pyro.sample("x", dist.Normal(0, 1))
    ...     y = pyro.sample("y", dist.Normal(0, 1))
    Sampled x
    Sampled y
    """

    _current_site: Optional[str]

    def __enter__(self):
        if any(isinstance(m, PyroShim) for m in pyro.poutine.runtime._PYRO_STACK):
            warnings.warn("PyroShim should be installed at most once.")
        return super().__enter__()

    @staticmethod
    def _broadcast_to_named(
        t: torch.Tensor, shape: torch.Size, indices: Mapping[Operation[[], int], int]
    ) -> Tuple[torch.Tensor, "Naming"]:
        """Convert a tensor `t` to a fully positional tensor that is
        broadcastable with the positional representation of tensors of shape
        |shape|, |indices|.

        """
        t_indices = sizesof(t)

        if len(t.shape) < len(shape):
            t = t.expand(shape)

        # create a positional dimension for every named index in the target shape
        name_to_dim = {}
        for i, (k, v) in enumerate(reversed(list(indices.items()))):
            if k in t_indices:
                t = to_tensor(t, [k])
            else:
                t = t.expand((v,) + t.shape)
            name_to_dim[k] = -len(shape) - i - 1

        # create a positional dimension for every remaining named index in `t`
        n_batch_and_dist_named = len(t.shape)
        for i, k in enumerate(reversed(list(sizesof(t).keys()))):
            t = to_tensor(t, [k])
            name_to_dim[k] = -n_batch_and_dist_named - i - 1

        return t, Naming(name_to_dim)

    def _pyro_sample(self, msg: pyro.poutine.runtime.Message) -> None:
        if typing.TYPE_CHECKING:
            assert msg["type"] == "sample"
            assert msg["name"] is not None
            assert msg["infer"] is not None
            assert isinstance(
                msg["fn"], pyro.distributions.torch_distribution.TorchDistributionMixin
            )

        if pyro.poutine.util.site_is_subsample(msg) or pyro.poutine.util.site_is_factor(
            msg
        ):
            return

        if getattr(self, "_current_site", None) == msg["name"]:
            if "_markov_scope" in msg["infer"] and self._current_site:
                msg["infer"]["_markov_scope"].pop(self._current_site, None)

            dist = msg["fn"]
            obs = msg["value"] if msg["is_observed"] else None

            # pdist shape: | named1 | batch_shape | event_shape |
            # obs shape: | batch_shape | event_shape |, | named2 | where named2 may overlap named1
            pdist = PositionalDistribution(dist)
            naming = pdist.naming

            if msg["mask"] is None:
                mask = torch.tensor(True)
            elif isinstance(msg["mask"], bool):
                mask = torch.tensor(msg["mask"])
            else:
                mask = msg["mask"]

            pos_mask, _ = PyroShim._broadcast_to_named(
                mask, dist.batch_shape, pdist.indices
            )

            pos_obs: Optional[torch.Tensor] = None
            if obs is not None:
                pos_obs, naming = PyroShim._broadcast_to_named(
                    obs, dist.shape(), pdist.indices
                )

            for var, dim in naming.name_to_dim.items():
                frame = pyro.poutine.indep_messenger.CondIndepStackFrame(
                    name=str(var), dim=dim, size=-1, counter=0
                )
                msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

            msg["fn"] = pdist
            msg["value"] = pos_obs
            msg["mask"] = pos_mask
            msg["infer"]["_index_naming"] = naming  # type: ignore

            assert sizesof(msg["value"]) == {}
            assert sizesof(msg["mask"]) == {}

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

    def _pyro_post_sample(self, msg: pyro.poutine.runtime.Message) -> None:
        infer = msg.get("infer")
        if infer is None or "_index_naming" not in infer:
            return

        # note: Pyro uses a TypedDict for infer, so it doesn't know we've stored this key
        naming = infer["_index_naming"]  # type: ignore

        value = msg["value"]

        if value is not None:
            # note: is it safe to assume that msg['fn'] is a distribution?
            assert isinstance(
                msg["fn"], pyro.distributions.torch_distribution.TorchDistribution
            )
            dist_shape: tuple[int, ...] = msg["fn"].batch_shape + msg["fn"].event_shape
            if len(value.shape) < len(dist_shape):
                value = value.broadcast_to(
                    torch.broadcast_shapes(value.shape, dist_shape)
                )
            value = naming.apply(value)
            msg["value"] = value


class Naming:
    """
    A mapping from dimensions (indexed from the right) to names.
    """

    def __init__(self, name_to_dim: Mapping[Operation[[], int], int]):
        assert all(v < 0 for v in name_to_dim.values())
        self.name_to_dim = name_to_dim

    @staticmethod
    def from_shape(names: Collection[Operation[[], int]], event_dims: int) -> "Naming":
        """Create a naming from a set of indices and the number of event dimensions.

        The resulting naming converts tensors of shape
        ``| batch_shape | named | event_shape |``
        to tensors of shape ``| batch_shape | event_shape |, | named |``.

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

    indices: Mapping[Operation[[], int], int]

    def __init__(
        self, base_dist: pyro.distributions.torch_distribution.TorchDistribution
    ):
        self.base_dist = base_dist
        self.indices = sizesof(base_dist)

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
            torch.Size([s for s in self.indices.values()]) + self.base_dist.batch_shape
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
        names: Collection[Operation[[], int]],
    ):
        """
        :param base_dist: A distribution with batch dimensions.

        :param names: A list of names.

        """
        self.base_dist = base_dist
        self.names = names

        assert 1 <= len(names) <= len(base_dist.batch_shape)
        base_indices = sizesof(base_dist)
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
        assert set(sizesof(t).keys()) == set(self.names)
        assert t.shape == self.shape() + sample_shape
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


def pyro_module_shim(
    module: type[pyro.nn.module.PyroModule],
) -> type[pyro.nn.module.PyroModule]:
    """Wrap a :class:`PyroModule` in a :class:`PyroShim`.

    Returns a new subclass of :class:`PyroModule` that wraps calls to
    :func:`forward` in a :class:`PyroShim`.

    **Example usage**:

    .. code-block:: python

        class SimpleModel(PyroModule):
            def forward(self):
                return pyro.sample("y", dist.Normal(0, 1))

        SimpleModelShim = pyro_module_shim(SimpleModel)

    """

    class PyroModuleShim(module):  # type: ignore
        def forward(self, *args, **kwargs):
            with PyroShim():
                return super().forward(*args, **kwargs)

    return PyroModuleShim
