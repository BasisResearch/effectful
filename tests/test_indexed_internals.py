import itertools
import logging

import pyro.distributions as dist
import pytest
import torch

from effectful.indexed.ops import (
    Indexable,
    IndexSet,
    cond,
    cond_n,
    gather,
    indices_of,
    stack,
    to_tensor,
)
from effectful.internals.torch import sizesof, torch_getitem
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import defop, defun

torch.distributions.Distribution.set_default_validate_args(False)

logger = logging.getLogger(__name__)

ENUM_SHAPES = [
    (),
    (2,),
    (2, 1),
    (2, 3),
]

PLATE_SHAPES = [
    (),
    (2,),
    (2, 1),
    (2, 3),
    (1, 3),
]

BATCH_SHAPES = [
    (2,),
    (2, 1),
    (2, 3),
    (1, 2, 3),
    (2, 1, 3),
    (2, 3, 1),
    (2, 2),
    (2, 2, 2),
    (2, 2, 3),
]

EVENT_SHAPES = [
    (),
    (1,),
    (2,),
    (2, 1),
    (1, 2),
    (2, 2),
    (3, 1),
    (1, 1),
    (2, 2, 1),
    (2, 1, 2),
    (2, 3, 2),
]

SHAPE_CASES = list(
    itertools.product(ENUM_SHAPES, PLATE_SHAPES, BATCH_SHAPES, EVENT_SHAPES)
)


def indexed_batch(t, batch_len, name_to_dim):
    i = [slice(None)] * batch_len
    for n, d in name_to_dim.items():
        i[d] = n()
    return Indexable(t)[tuple(i)]


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_indices_of_tensor(enum_shape, plate_shape, batch_shape, event_shape):
    batch_dim_names = {
        defop(int, name=f"b{i}"): -1 - i
        for i in range(len(plate_shape), len(plate_shape) + len(batch_shape))
    }
    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = indexed_batch(
        torch.randn(full_batch_shape + event_shape),
        len(full_batch_shape),
        batch_dim_names,
    )

    actual = indices_of(value)
    expected = IndexSet(
        {
            name: set(range(full_batch_shape[dim]))
            for name, dim in batch_dim_names.items()
        }
    )

    assert actual == expected


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_indices_of_distribution(enum_shape, plate_shape, batch_shape, event_shape):
    batch_dim_names = {
        defop(int, name=f"b{i}"): -1 - i
        for i in range(len(plate_shape), len(plate_shape) + len(batch_shape))
    }

    full_batch_shape = enum_shape + batch_shape + plate_shape
    full_shape = full_batch_shape + event_shape

    loc = indexed_batch(
        torch.tensor(0.0).expand(full_shape), len(full_batch_shape), batch_dim_names
    )
    scale = indexed_batch(
        torch.tensor(1.0).expand(full_shape), len(full_batch_shape), batch_dim_names
    )
    value = dist.Normal(loc, scale).to_event(len(event_shape))

    actual = indices_of(value)

    expected = IndexSet(
        {
            name: set(range(full_batch_shape[dim]))
            for name, dim in batch_dim_names.items()
        }
    )

    assert actual == expected


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_gather_tensor(enum_shape, plate_shape, batch_shape, event_shape):
    cf_dim = -1 - len(plate_shape)
    name_to_dim = {
        defop(int, name=f"dim_{i}"): cf_dim - i for i in range(len(batch_shape))
    }

    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = torch.randn(full_batch_shape + event_shape)

    world = IndexSet(
        {name: {max(full_batch_shape[dim] - 2, 0)} for name, dim in name_to_dim.items()}
    )

    ivalue = indexed_batch(value, len(full_batch_shape), name_to_dim)

    actual = gather(ivalue, world)

    # for each gathered index, check that the gathered value is equal to the
    # value at that index
    world_vars = []
    for sym, inds in world.items():
        world_vars.append([(sym, i) for i in range(len(inds))])

    for binding in itertools.product(*world_vars):
        with handler({sym: lambda: post_gather for (sym, post_gather) in binding}):
            actual_v = evaluate(actual)

        assert actual_v.shape == enum_shape + plate_shape + event_shape

        expected_idx = [slice(None)] * len(full_batch_shape)
        for name, dim in name_to_dim.items():
            expected_idx[dim] = list(world[name])[0]
        expected_v = value[tuple(expected_idx)]

        assert (actual_v == expected_v).all()


def indexed_to_defun(value, names):
    vars_ = sizesof(value)
    ordered_vars = [[v for v in vars_ if v is n][0] for n in names]
    return defun(value, *ordered_vars)


def test_stack():
    t1 = torch.randn(5, 3)
    t2 = torch.randn(5, 3)

    a, b, x = defop(int, name="a"), defop(int, name="b"), defop(int, name="x")
    l1 = Indexable(t1)[a(), b()]
    l2 = Indexable(t2)[a(), b()]
    l3 = stack([l1, l2], x)

    f = indexed_to_defun(l3, [x, a, b])

    for i in range(5):
        for j in range(3):
            assert f(0, i, j) == t1[i, j]
            assert f(1, i, j) == t2[i, j]


def test_index_incompatible():
    """Check that using the same index in two incompatible dimensions raises an error."""
    i = defop(int)
    with pytest.raises(ValueError):
        torch_getitem(torch.randn(2, 3), (i(), i()))

    torch_getitem(torch.randn(2, 2), (i(), i()))


def test_simple_distribution():
    i = defop(int)
    t = torch_getitem(torch.tensor([0.5, 0.2, 0.9]), (i(),))

    dist.Beta(t, t, validate_args=False)

    dist.Bernoulli(t, validate_args=False)


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_cond_tensor_associate(enum_shape, batch_shape, plate_shape, event_shape):
    cf_dim = -1 - len(plate_shape)
    new_dim = defop(int, name="new_dim")
    ind1, ind2, ind3 = (
        IndexSet({new_dim: {0}}),
        IndexSet({new_dim: {1}}),
        IndexSet({new_dim: {2}}),
    )
    name_to_dim = {
        defop(int, name=f"dim_{i}"): cf_dim - i for i in range(len(batch_shape))
    }

    full_batch_shape = enum_shape + batch_shape + plate_shape
    batch_len = len(full_batch_shape)

    case = indexed_batch(torch.randint(0, 3, full_batch_shape), batch_len, name_to_dim)
    value1 = indexed_batch(
        torch.randn(full_batch_shape + event_shape), batch_len, name_to_dim
    )
    value2 = indexed_batch(
        torch.randn(enum_shape + batch_shape + (1,) * len(plate_shape) + event_shape),
        batch_len,
        name_to_dim,
    )
    value3 = indexed_batch(
        torch.randn(full_batch_shape + event_shape), batch_len, name_to_dim
    )

    actual_full = cond_n({ind1: value1, ind2: value2, ind3: value3}, case)

    actual_left = cond(cond(value1, value2, case == 1), value3, case >= 2)

    actual_right = cond(value1, cond(value2, value3, case == 2), case >= 1)

    assert (
        indices_of(actual_full) == indices_of(actual_left) == indices_of(actual_right)
    )

    vars = list(name_to_dim.keys())
    assert (to_tensor(actual_full, vars) == to_tensor(actual_left, vars)).all()
    assert (to_tensor(actual_left, vars) == to_tensor(actual_right, vars)).all()


def test_to_tensor():
    i, j, k = defop(int, name="i"), defop(int, name="j"), defop(int, name="k")

    # test that named dimensions can be removed and reordered
    t = torch.randn([2, 3, 4])
    t1 = to_tensor(Indexable(t)[i(), j(), k()], [i, j, k])
    t2 = to_tensor(Indexable(t.permute((2, 0, 1)))[k(), i(), j()], [i, j, k])
    t3 = to_tensor(Indexable(t.permute((1, 0, 2)))[j(), i(), k()], [i, j, k])

    assert torch.allclose(t1, t2)
    assert torch.allclose(t1, t3)

    # test that to_tensor can remove some but not all named dimensions
    t_ijk = Indexable(t)[i(), j(), k()]
    t_ij = to_tensor(t_ijk, [k])
    assert set(sizesof(t_ij).keys()) == set([i, j])
    assert t_ij.shape == torch.Size([4])

    t_i = to_tensor(t_ij, [j])
    assert set(sizesof(t_i).keys()) == set([i])
    assert t_i.shape == torch.Size([3, 4])

    t_ = to_tensor(t_i, [i])
    assert set(sizesof(t_).keys()) == set([])
    assert t_.shape == torch.Size([2, 3, 4])
    assert torch.allclose(t_, t)

    t__ = to_tensor(t_, [])
    assert set(sizesof(t__).keys()) == set([])
    assert t__.shape == torch.Size([2, 3, 4])
    assert torch.allclose(t_, t__)
