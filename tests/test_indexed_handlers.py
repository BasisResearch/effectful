import pyro
import pyro.distributions as dist
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame

from effectful.handlers.pyro import PyroShim
from effectful.indexed.handlers import indexed
from effectful.indexed.ops import Indexable, IndexSet, indices_of
from effectful.ops.core import ctxof, gensym
from effectful.ops.handler import handler

torch.distributions.Distribution.set_default_validate_args(False)


def test_indexed_sample():
    b = gensym(int, name="b")

    def model():
        loc, scale = (
            Indexable(torch.tensor(0.0).expand((3, 2)))[b()],
            Indexable(torch.tensor(1.0).expand((3, 2)))[b()],
        )
        return pyro.sample("x", dist.Normal(loc, scale))

    class CheckSampleMessenger(pyro.poutine.messenger.Messenger):
        def _pyro_sample(self, msg):
            # named dimensions should not be visible to Pyro
            assert indices_of(msg["fn"]) == IndexSet({})
            assert (
                CondIndepStackFrame(name="__index_plate___b", dim=-2, size=3, counter=0)
                in msg["cond_indep_stack"]
            )

    with CheckSampleMessenger(), PyroShim():
        with handler(indexed):
            t = model()

            # samples from indexed distributions should also be indexed
            assert t.shape == torch.Size([2])
            assert b in ctxof(t)
