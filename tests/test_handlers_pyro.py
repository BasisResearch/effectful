import contextlib
import logging
from typing import Mapping, Optional

import pyro
import pyro.distributions as dist
import pytest
import torch

from effectful.handlers.pyro import PyroShim, pyro_sample
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import defop

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)


@defop
def chirho_observe_dist(
    name: str,
    rv: pyro.distributions.torch_distribution.TorchDistributionMixin,
    obs: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    return pyro.sample(name, rv, obs=obs, **kwargs)


@contextlib.contextmanager
def chirho_condition(data: Mapping[str, torch.Tensor]):

    def _handle_pyro_sample(
        name: str,
        fn: pyro.distributions.torch_distribution.TorchDistributionMixin,
        obs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if name in data:
            assert obs is None
            return chirho_observe_dist(
                name,
                fn,
                obs=data[name],
                **kwargs,
            )
        else:
            return fwd(None)

    with handler({pyro_sample: pyro_sample.__default_rule__}):
        with handler({pyro_sample: _handle_pyro_sample}):
            yield data


class HMM(pyro.nn.PyroModule):
    @pyro.nn.PyroParam(constraint=dist.constraints.simplex)  # type: ignore
    def trans_probs(self):
        return torch.tensor([[0.75, 0.25], [0.25, 0.75]])

    def forward(self, data):
        emission_probs = pyro.sample(
            "emission_probs",
            dist.Dirichlet(torch.tensor([0.5, 0.5])).expand([2]).to_event(1),
        )
        x = pyro.sample("x", dist.Categorical(torch.tensor([0.5, 0.5])))
        logger.debug(f"-1\t{tuple(x.shape)}")
        for t, y in pyro.markov(enumerate(data)):
            x = pyro.sample(
                f"x_{t}",
                dist.Categorical(pyro.ops.indexing.Vindex(self.trans_probs)[..., x, :]),
            )

            pyro.sample(
                f"y_{t}",
                dist.Categorical(pyro.ops.indexing.Vindex(emission_probs)[..., x, :]),
            )
            logger.debug(f"{t}\t{tuple(x.shape)}")


@pytest.mark.parametrize("num_particles", [1, 10])
@pytest.mark.parametrize("max_plate_nesting", [3, float("inf")])
@pytest.mark.parametrize("use_guide", [False, True])
@pytest.mark.parametrize("num_steps", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("Elbo", [pyro.infer.TraceEnum_ELBO, pyro.infer.TraceTMC_ELBO])
def test_smoke_condition_enumerate_hmm_elbo(
    num_steps, Elbo, use_guide, max_plate_nesting, num_particles
):
    data = dist.Categorical(torch.tensor([0.5, 0.5])).sample((num_steps,))
    hmm_model = HMM()

    assert issubclass(Elbo, pyro.infer.elbo.ELBO)
    elbo = Elbo(
        max_plate_nesting=max_plate_nesting,
        num_particles=num_particles,
        vectorize_particles=(num_particles > 1),
    )

    model = PyroShim()(hmm_model)
    model = chirho_condition(data={f"y_{t}": y for t, y in enumerate(data)})(model)

    tr = pyro.poutine.trace(pyro.plate("plate1", 7, dim=-1)(model)).get_trace(data)
    tr.compute_log_prob()
    for t in range(num_steps):
        assert f"x_{t}" in tr.nodes
        assert tr.nodes[f"x_{t}"]["type"] == "sample"
        assert not tr.nodes[f"x_{t}"]["is_observed"]
        assert any(f.name == "plate1" for f in tr.nodes[f"x_{t}"]["cond_indep_stack"])

        assert f"y_{t}" in tr.nodes
        assert tr.nodes[f"y_{t}"]["type"] == "sample"
        assert tr.nodes[f"y_{t}"]["is_observed"]
        assert (tr.nodes[f"y_{t}"]["value"] == data[t]).all()
        assert any(f.name == "plate1" for f in tr.nodes[f"x_{t}"]["cond_indep_stack"])

    if use_guide:
        guide = pyro.infer.config_enumerate(default="parallel")(
            pyro.infer.autoguide.AutoDiscreteParallel(
                pyro.poutine.block(expose=["x"])(chirho_condition(data={})(model))
            )
        )
        model = pyro.infer.config_enumerate(default="parallel")(model)
    else:
        model = pyro.infer.config_enumerate(default="parallel")(model)
        model = chirho_condition(data={"x": torch.as_tensor(0)})(model)

        def guide(data):
            pass

    # smoke test
    elbo.differentiable_loss(model, guide, data)
