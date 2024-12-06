import pyro
import pyro.distributions as dist
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame

from effectful.handlers.pyro import PyroShim
from effectful.indexed.handlers import (
    indexed,
    NamedDistribution,
    PositionalDistribution,
)
from effectful.indexed.ops import Indexable, IndexSet, indices_of
from effectful.internals.sugar import gensym
from effectful.ops.core import ctxof
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


def test_named_dist():
    x, y = gensym(int, name="x"), gensym(int, name="y")
    d = NamedDistribution(dist.Normal(0.0, 1.0).expand((2, 3)), [x, y])

    expected_indices = IndexSet({x: {0, 1}, y: {0, 1, 2}})
    assert indices_of(d) == expected_indices

    s1 = d.sample()
    assert indices_of(d.sample()) == expected_indices
    assert s1.shape == torch.Size([])

    s2 = d.sample((4, 5))
    assert indices_of(s2) == expected_indices
    assert s2.shape == torch.Size([4, 5])

    s3 = d.rsample((4, 5))
    assert indices_of(s3) == expected_indices
    assert s3.shape == torch.Size([4, 5])


def test_positional_dist():
    x, y = gensym(int, name="x"), gensym(int, name="y")
    loc = Indexable(torch.tensor(0.0).expand((2, 3)))[x(), y()]
    scale = Indexable(torch.tensor(1.0).expand((2, 3)))[x(), y()]

    expected_indices = IndexSet({x: {0, 1}, y: {0, 1, 2}})

    d = PositionalDistribution(dist.Normal(loc, scale))

    assert d.shape() == torch.Size([2, 3])

    s1 = d.sample()
    assert indices_of(s1) == IndexSet({})
    assert s1.shape == torch.Size([2, 3])
    assert all(n in indices_of(d._from_positional(s1)) for n in [x, y])

    d_exp = d.expand((4, 5) + d.batch_shape)
    s2 = d_exp.sample()
    assert indices_of(s2) == IndexSet({})
    assert s2.shape == torch.Size([4, 5, 2, 3])

    s2_named = d_exp._from_positional(s2)
    assert all(n in indices_of(s2_named) for n in [x, y])
    assert s2_named.shape == torch.Size([4, 5])

    s3 = d.sample((4, 5))
    assert indices_of(s3) == IndexSet({})
    assert s3.shape == torch.Size([4, 5, 2, 3])
    assert all(n in indices_of(d._from_positional(s3)) for n in [x, y])

    loc = Indexable(torch.tensor(0.0).expand((2, 3, 4, 5)))[x(), y()]
    scale = Indexable(torch.tensor(1.0).expand((2, 3, 4, 5)))[x(), y()]
    d = PositionalDistribution(dist.Normal(loc, scale))

    assert indices_of(d._from_positional(d.sample((6, 7)))) == expected_indices
