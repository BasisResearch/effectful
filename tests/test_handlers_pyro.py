import contextlib
import logging
from collections.abc import Mapping

import pyro
import pyro.distributions as dist
import pytest
import torch

from effectful.handlers.pyro import Naming, PyroShim, pyro_sample
from effectful.handlers.torch import bind_dims, sizesof, torch_getitem, unbind_dims
from effectful.ops.semantics import fvsof, fwd, handler
from effectful.ops.syntax import defop

torch.distributions.Distribution.set_default_validate_args(False)
pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)


def setup_module():
    pyro.settings.set(module_local_params=True)
    pyro.enable_validation(False)
    torch.distributions.Distribution.set_default_validate_args(False)


@defop
def chirho_observe_dist(
    name: str,
    rv: pyro.distributions.torch_distribution.TorchDistributionMixin,
    obs: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    return pyro.sample(name, rv, obs=obs, **kwargs)


@contextlib.contextmanager
def chirho_condition(data: Mapping[str, torch.Tensor]):
    def _handle_pyro_sample(
        name: str,
        fn: pyro.distributions.torch_distribution.TorchDistributionMixin,
        obs: torch.Tensor | None = None,
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
            return fwd()

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


def test_indexed_sample():
    b = defop(torch.Tensor, name="b")

    def model():
        loc, scale = (
            (torch.tensor(0.0).expand((3, 2)))[b()],
            (torch.tensor(1.0).expand((3, 2)))[b()],
        )
        return pyro.sample("x", dist.Normal(loc, scale))

    class CheckSampleMessenger(pyro.poutine.messenger.Messenger):
        def _pyro_sample(self, msg):
            # named dimensions should not be visible to Pyro
            assert sizesof(msg["fn"].sample()) == {}
            assert any(
                f.name == "__index_plate_b" and f.dim == -2
                for f in msg["cond_indep_stack"]
            )

    with CheckSampleMessenger(), PyroShim():
        t = model()

        # samples from indexed distributions should also be indexed
        assert t.shape == torch.Size([2])
        assert b in fvsof(t)


def test_named_dist():
    x, y = defop(torch.Tensor, name="x"), defop(torch.Tensor, name="y")
    d = unbind_dims(dist.Normal(0.0, 1.0).expand((2, 3)), x, y)

    expected_indices = {x: 2, y: 3}

    s1 = d.sample()
    assert sizesof(d.sample()) == expected_indices
    assert s1.shape == torch.Size([])

    s2 = d.sample((4, 5))
    assert sizesof(s2) == expected_indices
    assert s2.shape == torch.Size([4, 5])

    s3 = d.rsample((4, 5))
    assert sizesof(s3) == expected_indices
    assert s3.shape == torch.Size([4, 5])


def test_positional_dist():
    x, y = defop(torch.Tensor, name="x"), defop(torch.Tensor, name="y")
    loc = (torch.tensor(0.0).expand((2, 3)))[x(), y()]
    scale = (torch.tensor(1.0).expand((2, 3)))[x(), y()]

    expected_indices = {x: 2, y: 3}

    i_dist = dist.Normal(loc, scale)
    sizes = sizesof(i_dist)
    naming = Naming.from_shape(sizes.keys(), len(i_dist.shape()))
    p_dist = bind_dims(i_dist, *sizes.keys())

    assert p_dist.shape() == torch.Size([2, 3])

    s1 = p_dist.sample()
    assert sizesof(s1) == {}
    assert s1.shape == torch.Size([2, 3])
    assert all(n in sizesof(naming.apply(s1)) for n in [x, y])

    d_exp = p_dist.expand((4, 5) + p_dist.batch_shape)
    s2 = d_exp.sample()
    assert sizesof(s2) == {}
    assert s2.shape == torch.Size([4, 5, 2, 3])

    s3 = p_dist.sample((4, 5))
    assert sizesof(s3) == {}
    assert s3.shape == torch.Size([4, 5, 2, 3])
    assert all(n in sizesof(naming.apply(s3)) for n in [x, y])

    loc = (torch.tensor(0.0).expand((2, 3, 4, 5)))[x(), y()]
    scale = (torch.tensor(1.0).expand((2, 3, 4, 5)))[x(), y()]
    i_dist = dist.Normal(loc, scale)
    sizes = sizesof(i_dist)
    naming = Naming.from_shape(sizes.keys(), len(i_dist.shape()))
    p_dist = bind_dims(i_dist, *sizes.keys())

    assert sizesof(naming.apply(p_dist.sample((6, 7)))) == expected_indices
    assert p_dist.sample().shape == torch.Size([2, 3, 4, 5])
    assert p_dist.sample((6, 7)).shape == torch.Size([6, 7, 2, 3, 4, 5])


def test_simple_distribution():
    i = defop(torch.Tensor)
    t = torch_getitem(torch.tensor([0.5, 0.2, 0.9]), (i(),))

    dist.Beta(t, t, validate_args=False)

    dist.Bernoulli(t, validate_args=False)


def test_sizesof_unbind_dims():
    # Create base distribution with known batch shape
    base_dist = dist.Normal(torch.zeros(3, 4, 5), torch.ones(3, 4, 5))

    # Create names for the first two dimensions
    dim0 = defop(torch.Tensor, name="dim0")
    dim1 = defop(torch.Tensor, name="dim1")

    # Create named distribution
    named_dist = unbind_dims(base_dist, dim0, dim1)

    # Get sizes
    sizes = sizesof(named_dist)

    # Check that the sizes match expected values
    assert sizes[dim0] == 3
    assert sizes[dim1] == 4
    assert len(sizes) == 2  # Only named dimensions should be included


def test_sizesof_bind_dims():
    dim0 = defop(torch.Tensor, name="dim0")
    dim1 = defop(torch.Tensor, name="dim1")

    mean = (torch.zeros(3, 4, 5))[dim0(), dim1()]
    var = (torch.ones(3, 4, 5))[dim0(), dim1()]
    base_dist = dist.Normal(mean, var)

    pos_dist = bind_dims(base_dist, dim0, dim1)

    assert sizesof(pos_dist) == {}
