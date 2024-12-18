import typing
import warnings
from typing import List, Optional

import pyro
from pyro.poutine.indep_messenger import CondIndepStackFrame
import torch

from effectful.ops.core import defop

from effectful.indexed.ops import IndexSet, indices_of, to_tensor
from effectful.indexed.handlers import PositionalDistribution, Naming


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

    def __enter__(self):
        if any(isinstance(m, PyroShim) for m in pyro.poutine.runtime._PYRO_STACK):
            warnings.warn("PyroShim should be installed at most once.")
        return super().__enter__()

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
            assert indices_of(pos_obs) == IndexSet({})

            for var, dim in naming.name_to_dim.items():
                frame = CondIndepStackFrame(name=var, dim=dim, size=None, counter=0)
                msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

            msg["fn"] = pdist
            msg["value"] = pos_obs
            msg["infer"]["_index_naming"] = naming  # type: ignore

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
        if "_index_naming" not in msg["infer"]:
            return

        msg["value"] = (
            msg["infer"]["_index_naming"].apply(msg["value"])
            if msg["value"] is not None
            else None
        )


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
