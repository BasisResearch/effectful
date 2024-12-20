import typing
import warnings
from typing import Optional, Tuple

import pyro
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.poutine.mask_messenger import MaskMessenger

from effectful.indexed.distributions import Naming, PositionalDistribution
from effectful.indexed.ops import IndexSet, indices_of, to_tensor
from effectful.ops.core import defop


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
    Operation to sample from a Pyro distribution.
    """
    with MaskMessenger(mask if mask is not None else True):
        return pyro.sample(
            name, fn, *args, obs=obs, obs_mask=obs_mask, infer=infer, **kwargs
        )


class PyroShim(pyro.poutine.messenger.Messenger):
    """
    Handler for Pyro that wraps all sample sites in a custom effectful type.
    """

    _current_site: Optional[str]

    def __enter__(self):
        if any(isinstance(m, PyroShim) for m in pyro.poutine.runtime._PYRO_STACK):
            warnings.warn("PyroShim should be installed at most once.")
        return super().__enter__()

    @staticmethod
    def _broadcast_to_named(
        t: torch.Tensor, shape: torch.Size, indices: IndexSet
    ) -> Tuple[torch.Tensor, Naming]:
        """Convert a tensor `t` to a fully positional tensor that is
        broadcastable with the positional representation of tensors of shape
        |shape|, |indices|.

        """
        t_indices = indices_of(t)

        if len(t.shape) < len(shape):
            t = t.expand(shape)

        # create a positional dimension for every named index in the target shape
        name_to_dim = {}
        for i, (k, v) in enumerate(reversed(indices.items())):
            if k in t_indices:
                t = to_tensor(t, [k])
            else:
                t = t.expand((len(v),) + t.shape)
            name_to_dim[k] = -len(shape) - i - 1

        # create a positional dimension for every remaining named index in `t`
        n_batch_and_dist_named = len(t.shape)
        for i, k in enumerate(reversed(indices_of(t).keys())):
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
                frame = CondIndepStackFrame(name=str(var), dim=dim, size=-1, counter=0)
                msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

            msg["fn"] = pdist
            msg["value"] = pos_obs
            msg["mask"] = pos_mask
            msg["infer"]["_index_naming"] = naming  # type: ignore

            assert indices_of(msg["fn"]) == IndexSet({})
            assert indices_of(msg["value"]) == IndexSet({})
            assert indices_of(msg["mask"]) == IndexSet({})

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
            dist_shape = msg["fn"].shape()  # type: ignore
            if len(value.shape) < len(dist_shape):
                value = value.broadcast_to(dist_shape)
            value = naming.apply(value)
            msg["value"] = value


def pyro_module_shim(
    module: type[pyro.nn.module.PyroModule],
) -> type[pyro.nn.module.PyroModule]:
    """Wrap a PyroModule in a PyroShim.

    Returns a new subclass of PyroModule that wraps calls to `forward` in a PyroShim.

    """

    class PyroModuleShim(module):  # type: ignore
        def forward(self, *args, **kwargs):
            with PyroShim():
                return super().forward(*args, **kwargs)

    return PyroModuleShim
