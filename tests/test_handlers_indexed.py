import itertools
import logging

import pyro.distributions as dist
import pytest
import torch

from effectful.handlers.indexed import (
    IndexSet,
    cond,
    cond_n,
    gather,
    indices_of,
    name_to_sym,
    stack,
)
from effectful.handlers.torch import Indexable, sizesof, to_tensor
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import deffn

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
        i[d] = name_to_sym(n)()
    return Indexable(t)[tuple(i)]


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_indices_of_tensor(enum_shape, plate_shape, batch_shape, event_shape):
    batch_dim_names = {
        f"b{i}": -1 - i
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
        **{
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
        f"b{i}": -1 - i
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
        **{
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
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = torch.randn(full_batch_shape + event_shape)

    world = IndexSet(
        **{
            name: {max(full_batch_shape[dim] - 2, 0)}
            for name, dim in name_to_dim.items()
        }
    )

    ivalue = indexed_batch(value, len(full_batch_shape), name_to_dim)

    actual = gather(ivalue, world)

    # for each gathered index, check that the gathered value is equal to the
    # value at that index
    world_vars = []
    for name, inds in world.items():
        world_vars.append([(name_to_sym(name), i) for i in range(len(inds))])

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
    return deffn(value, *ordered_vars)


def test_stack():
    t1 = torch.randn(5, 3)
    t2 = torch.randn(5, 3)

    a, b, x = name_to_sym("a"), name_to_sym("b"), name_to_sym("x")
    l1 = Indexable(t1)[a(), b()]
    l2 = Indexable(t2)[a(), b()]
    l3 = stack([l1, l2], x.__name__)

    f = indexed_to_defun(l3, [x, a, b])

    for i in range(5):
        for j in range(3):
            assert f(0, i, j) == t1[i, j]
            assert f(1, i, j) == t2[i, j]


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_cond_tensor_associate(enum_shape, batch_shape, plate_shape, event_shape):
    cf_dim = -1 - len(plate_shape)
    new_dim = name_to_sym("new_dim")
    ind1, ind2, ind3 = (
        IndexSet(**{new_dim.__name__: {0}}),
        IndexSet(**{new_dim.__name__: {1}}),
        IndexSet(**{new_dim.__name__: {2}}),
    )
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

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

    vars = list(map(name_to_sym, name_to_dim.keys()))
    assert (to_tensor(actual_full, vars) == to_tensor(actual_left, vars)).all()
    assert (to_tensor(actual_left, vars) == to_tensor(actual_right, vars)).all()
