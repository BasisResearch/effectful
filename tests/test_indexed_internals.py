import contextlib
import itertools
import logging

import pyro.distributions as dist
import pytest
import torch

from effectful.indexed.handlers import IndexPlatesMessenger
from effectful.indexed.internals.handlers import add_indices
from effectful.indexed.ops import (
    IndexSet,
    cond,
    cond_n,
    gather,
    get_index_plates,
    indices_of,
    lift_tensor,
    name_to_sym,
    stack,
    to_tensor,
)
from effectful.internals.sugar import gensym, sizesof, torch_getitem
from effectful.ops.core import evaluate
from effectful.ops.function import defun
from effectful.ops.handler import handler

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


def test_lift_tensor():
    raw_value = torch.randn(2, 3, 4)
    name_to_dim = {"dim1": -2}
    lifted_value, vars_ = lift_tensor(raw_value, event_dim=1, name_to_dim=name_to_dim)

    f_lifted = defun(lifted_value, *vars_)
    assert (f_lifted(0) == raw_value[0]).all()


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_indices_of_tensor(enum_shape, plate_shape, batch_shape, event_shape):
    batch_dim_names = {
        f"b{i}": -1 - i
        for i in range(len(plate_shape), len(plate_shape) + len(batch_shape))
    }

    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = torch.randn(full_batch_shape + event_shape)
    actual_world = indices_of(
        value, event_dim=len(event_shape), name_to_dim=batch_dim_names
    )

    expected_world = IndexSet(
        **{
            name: set(range(full_batch_shape[dim]))
            for name, dim in batch_dim_names.items()
            if full_batch_shape[dim] > 1
        }
    )

    assert actual_world == expected_world


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_indices_of_distribution(enum_shape, plate_shape, batch_shape, event_shape):
    batch_dim_names = {
        f"b{i}": -1 - i
        for i in range(len(plate_shape), len(plate_shape) + len(batch_shape))
    }

    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = (
        dist.Normal(0, 1)
        .expand(full_batch_shape + event_shape)
        .to_event(len(event_shape))
    )
    actual_world = indices_of(value, name_to_dim=batch_dim_names)

    expected_world = IndexSet(
        **{
            name: set(range(full_batch_shape[dim]))
            for name, dim in batch_dim_names.items()
            if full_batch_shape[dim] > 1
        }
    )

    assert actual_world == expected_world


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
@pytest.mark.parametrize("use_effect", [True, False])
def test_gather_tensor(enum_shape, plate_shape, batch_shape, event_shape, use_effect):
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

    with contextlib.ExitStack() as stack:
        if use_effect:
            stack.enter_context(IndexPlatesMessenger(cf_dim))
            for name, dim in name_to_dim.items():
                add_indices(
                    IndexSet(**{name: set(range(max(2, full_batch_shape[dim])))})
                )
            _name_to_dim = None
        else:
            _name_to_dim = name_to_dim

        lifted_value, vars_ = lift_tensor(
            value, event_dim=len(event_shape), name_to_dim=_name_to_dim
        )

    actual = gather(lifted_value, world)

    # for each gathered index, check that the gathered value is equal to the
    # value at that index
    world_vars = []
    for name, inds in world.items():
        sym = name_to_sym(name)
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
    ordered_vars = [[v for v in vars_ if v is name_to_sym(n)][0] for n in names]
    return defun(value, *ordered_vars)


def test_stack():
    t1 = torch.randn(5, 3)
    t2 = torch.randn(5, 3)

    l1, _ = lift_tensor(t1, name_to_dim={"a": 0, "b": 1})
    l2, _ = lift_tensor(t2, name_to_dim={"a": 0, "b": 1})
    l3 = stack([l1, l2], "x")

    f = indexed_to_defun(l3, ["x", "a", "b"])

    for i in range(5):
        for j in range(3):
            assert f(0, i, j) == t1[i, j]
            assert f(1, i, j) == t2[i, j]


def test_index_incompatible():
    """Check that using the same index in two incompatible dimensions raises an error."""
    i = gensym(int)
    with pytest.raises(ValueError):
        torch_getitem(torch.randn(2, 3), (i(), i()))

    torch_getitem(torch.randn(2, 2), (i(), i()))


def test_simple_distribution():
    i = gensym(int)
    t = torch_getitem(torch.tensor([0.5, 0.2, 0.9]), (i(),))

    dist.Beta(t, t, validate_args=False)

    dist.Bernoulli(t, validate_args=False)


def test_index_plate_names():
    with IndexPlatesMessenger(-1):
        add_indices(IndexSet(a={0, 1}))
        index_plates = get_index_plates()
        x_ind = indices_of(torch.randn(2))

    assert "a" in x_ind
    assert len(index_plates) == 1
    for name, frame in index_plates.items():
        assert name != frame.name


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_cond_tensor_associate(enum_shape, batch_shape, plate_shape, event_shape):
    cf_dim = -1 - len(plate_shape)
    event_dim = len(event_shape)
    ind1, ind2, ind3 = (
        IndexSet(new_dim={0}),
        IndexSet(new_dim={1}),
        IndexSet(new_dim={2}),
    )
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    case = torch.randint(0, 3, enum_shape + batch_shape + plate_shape)
    value1 = torch.randn(batch_shape + plate_shape + event_shape)
    value2 = torch.randn(
        enum_shape + batch_shape + (1,) * len(plate_shape) + event_shape
    )
    value3 = torch.randn(enum_shape + batch_shape + plate_shape + event_shape)

    with IndexPlatesMessenger(cf_dim):
        for name, dim in name_to_dim.items():
            add_indices(
                IndexSet(**{name: set(range(max(3, (batch_shape + plate_shape)[dim])))})
            )

        actual_full = cond_n(
            {ind1: value1, ind2: value2, ind3: value3}, case, event_dim=event_dim
        )

        actual_left = cond(
            cond(value1, value2, case == 1, event_dim=event_dim),
            value3,
            case >= 2,
            event_dim=event_dim,
        )

        actual_right = cond(
            value1,
            cond(value2, value3, case == 2, event_dim=event_dim),
            case >= 1,
            event_dim=event_dim,
        )

        assert (
            indices_of(actual_full, event_dim=event_dim)
            == indices_of(actual_left, event_dim=event_dim)
            == indices_of(actual_right, event_dim=event_dim)
        )

    name_to_dim = list(name_to_dim.items())
    names = [i[0] for i in name_to_dim]

    f_actual_full = indexed_to_defun(actual_full, names)
    f_actual_left = indexed_to_defun(actual_left, names)
    f_actual_right = indexed_to_defun(actual_right, names)

    for idx in itertools.product(*[range(d[1]) for d in name_to_dim]):
        assert (f_actual_full(*idx) == f_actual_left(*idx)).all()
        assert (f_actual_left(*idx) == f_actual_right(*idx)).all()


def test_to_tensor():
    i = name_to_sym("i")
    j = name_to_sym("j")
    k = name_to_sym("k")

    # test that named dimensions can be removed and reordered
    t = torch.randn([2, 3, 4])
    t1 = to_tensor(torch_getitem(t, [i(), j(), k()]), [i, j, k])
    t2 = to_tensor(torch_getitem(t.permute((2, 0, 1)), [k(), i(), j()]), [i, j, k])
    t3 = to_tensor(torch_getitem(t.permute((1, 0, 2)), [j(), i(), k()]), [i, j, k])

    assert torch.allclose(t1, t2)
    assert torch.allclose(t1, t3)

    # test that to_tensor can remove some but not all named dimensions
    t_ijk = torch_getitem(t, [i(), j(), k()])
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
