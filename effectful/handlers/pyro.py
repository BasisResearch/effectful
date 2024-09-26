from typing import Dict, Optional, Any, Mapping

from torch import Tensor, tensor
from torch.distributions import Distribution

import pyro.poutine.util
from pyro.poutine.messenger import Messenger

from ..ops.core import Operation


@Operation
def pyro_sample(
    name: str,
    dist: Distribution,
    obs: Optional[Tensor] = None,
    mask: Tensor = tensor(True),
) -> Tensor:
    raise NotImplementedError()


# class EffectfulSampleMessenger(Messenger):
#     data = {}

#     def _pyro_sample(self, msg: Dict[str, Any]) -> None:
#         if "name" not in msg:
#             raise RuntimeError("Sample missing required field 'name'")
#         if "fn" not in msg:
#             raise RuntimeError("Sample missing required field 'fn'")

#         if "in_pyro_sample_handler" not in msg["infer"]:
#             msg["value"] = pyro_sample(
#                 msg["name"], msg["fn"], obs=msg.get("obs"), mask=msg.get("mask")
#             )


class EffectfulSampleMessenger(Messenger):
    _current_site = None

    def _pyro_sample(self, msg):
        if (
            self._current_site == msg["name"]
            or "in_pyro_sample_handler" in msg["infer"]
        ):
            return

        if pyro.poutine.util.site_is_subsample(msg) or pyro.poutine.util.site_is_factor(
            msg
        ):
            return

        if msg["infer"].get("_do_not_observe", None):
            if (
                "_markov_scope" in msg["infer"]
                and getattr(self, "_current_site", None) is not None
            ):
                msg["infer"]["_markov_scope"].pop(self._current_site, None)
            return

        msg["stop"] = True
        msg["done"] = True

        # flags to guarantee commutativity of condition, intervene, trace
        msg["mask"] = False
        msg["is_observed"] = False
        msg["infer"]["is_auxiliary"] = True
        msg["infer"]["_do_not_trace"] = True
        msg["infer"]["_do_not_intervene"] = True
        msg["infer"]["_do_not_observe"] = True

        with pyro.poutine.infer_config(
            config_fn=lambda msg_: {
                "_do_not_observe": msg["name"] == msg_["name"]
                or msg_["infer"].get("_do_not_observe", False)
            }
        ):
            try:
                self._current_site = msg["name"]
                msg["value"] = pyro_sample(
                    msg["name"], msg["fn"], obs=msg.get("obs"), mask=msg.get("mask")
                )
            finally:
                self._current_site = None
