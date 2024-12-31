import pyro.distributions as dist
import torch

from effectful.indexed.distributions import NamedDistribution, PositionalDistribution
from effectful.indexed.ops import Indexable, IndexSet, indices_of
from effectful.internals.sugar import gensym

torch.distributions.Distribution.set_default_validate_args(False)


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

    s3 = d.sample((4, 5))
    assert indices_of(s3) == IndexSet({})
    assert s3.shape == torch.Size([4, 5, 2, 3])
    assert all(n in indices_of(d._from_positional(s3)) for n in [x, y])

    loc = Indexable(torch.tensor(0.0).expand((2, 3, 4, 5)))[x(), y()]
    scale = Indexable(torch.tensor(1.0).expand((2, 3, 4, 5)))[x(), y()]
    d = PositionalDistribution(dist.Normal(loc, scale))

    assert indices_of(d._from_positional(d.sample((6, 7)))) == expected_indices
    assert d.sample().shape == torch.Size([2, 3, 4, 5])
    assert d.sample((6, 7)).shape == torch.Size([6, 7, 2, 3, 4, 5])
