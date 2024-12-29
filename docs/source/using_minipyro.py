from torch import distributions as dist
from torch import tensor

from effectful.handlers.minipyro import (
    SVI,
    Adam,
    Trace_ELBO,
    default_runner,
    get_param_store,
    param,
    sample,
)
from effectful.ops.semantics import handler


def test_optimizer():
    def model(data):
        p = param("p", tensor(0.5))
        sample("x", dist.Bernoulli(p), obs=data)

    def guide(data):
        pass

    with handler(default_runner):
        data = tensor(0.0)
        get_param_store().clear()
        elbo = Trace_ELBO(ignore_jit_warnings=True)

        optimizer = Adam({"lr": 1e-6})
        inference = SVI(model, guide, optimizer, elbo)
        for i in range(2):
            inference.step(data)
