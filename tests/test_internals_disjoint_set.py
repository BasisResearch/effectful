import random

import pytest

from effectful.internals.disjoint_set import DisjointSet


@pytest.fixture
def dsu():
    return DisjointSet(10)


def test_initial_state(dsu):
    for i in range(10):
        assert dsu.find(i) == i


def test_simple_union(dsu):
    assert dsu.union(1, 2) is True
    assert dsu.find(1) == dsu.find(2)


def test_union_idempotent(dsu):
    dsu.union(1, 2)
    assert dsu.union(1, 2) is False


def test_union_chain(dsu):
    dsu.union(1, 2)
    dsu.union(2, 3)
    assert dsu.find(1) == dsu.find(3)


def test_union_multiple_elements_all_connected(dsu):
    dsu.union(1, 2, 3, 4, 5)
    roots = {dsu.find(i) for i in [1, 2, 3, 4, 5]}
    assert len(roots) == 1


def test_union_multiple_elements_partial_overlap(dsu):
    dsu.union(1, 2)
    dsu.union(3, 4)
    dsu.union(2, 3, 5)

    roots = {dsu.find(i) for i in [1, 2, 3, 4, 5]}
    assert len(roots) == 1


def test_union_multiple_elements_with_existing_connections(dsu):
    dsu.union(1, 2)
    dsu.union(2, 3)
    dsu.union(3, 4, 5, 6)

    roots = {dsu.find(i) for i in [1, 2, 3, 4, 5, 6]}
    assert len(roots) == 1


def test_union_single_element(dsu):
    assert dsu.union(1) is False


def test_union_no_elements(dsu):
    assert dsu.union() is False


def test_union_self(dsu):
    assert dsu.union(3, 3) is False
    assert dsu.find(3) == 3


def test_transitivity(dsu):
    dsu.union(1, 2)
    dsu.union(2, 3)
    dsu.union(3, 4)
    assert dsu.find(1) == dsu.find(4)


def test_disjoint_sets_remain_separate(dsu):
    dsu.union(1, 2)
    dsu.union(3, 4)
    assert dsu.find(1) != dsu.find(3)


def test_randomized_unions():
    n = 50
    dsu = DisjointSet(n)

    groups = [{i} for i in range(n)]

    def find_group(x):
        for g in groups:
            if x in g:
                return g

    for _ in range(100):
        elems = random.sample(range(n), random.randint(2, 5))
        dsu.union(*elems)

        # merge ground-truth groups
        merged = set()
        for e in elems:
            merged |= find_group(e)

        groups = [g for g in groups if g.isdisjoint(merged)]
        groups.append(merged)

    # verify structure matches ground truth
    for g in groups:
        roots = {dsu.find(x) for x in g}
        assert len(roots) == 1


def test_path_compression_effect():
    dsu = DisjointSet(6)
    dsu.union(0, 1)
    dsu.union(1, 2)
    dsu.union(2, 3)
    dsu.union(3, 4)

    # Trigger compression
    root_before = dsu.find(4)
    root_after = dsu.find(4)

    assert root_before == root_after
