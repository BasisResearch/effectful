"""Tests for :mod:`effectful.internals.weak`.

Two complementary strategies:

* Hand-written invariant tests, parameterised over the dictionary flavours,
  each stating in its docstring the law it checks.
* A differential harness that replays a random operation sequence against both
  :class:`weakref.WeakKeyDictionary` and the dictionary under test, over the
  domain where the two are meant to be indistinguishable: weakly referenceable
  keys whose ``__eq__``/``__hash__`` are the inherited identity-based defaults.

Everything here is extensional: no test reads ``_dirty_len``, ``_iterating``,
``_pending_removals`` or ``data``.

The ports are from cpython's ``Lib/test/test_weakref.py`` (``MappingTestCase``)
and pytorch's ``test/test_weak.py``, which is where this module was forked from.

Beware of accidentally keeping a key alive: a ``for`` loop leaves its variable
bound after the loop, which is enough to make a "dead" key outlive a
``gc_collect()``. Build key lists with ``map`` and delete anything that lingers.
"""

import collections.abc
import contextlib
import copy
import dataclasses
import functools
import gc
import random
import re
import threading
import typing
import weakref

import pytest

from effectful.internals.weak import (
    AutoIdKeyDictionary,
    AutoIdRef,
    StrongIdKeyDictionary,
    StrongIdRef,
    WeakIdKeyDictionary,
    WeakIdRef,
    weak_memoize,
)

AnyDict = WeakIdKeyDictionary[typing.Any, typing.Any]


def gc_collect() -> None:
    """Force collection, twice, so cycles found in the first pass are freed."""
    gc.collect()
    gc.collect()


def weakrefable_ground_truth(value: typing.Any) -> bool:
    """Whether ``value`` can be weakly referenced, decided the expensive way.

    The reference for what ``AutoIdRef`` must agree with. Computed rather than
    tabulated, so the test cases cannot drift away from the truth.
    """
    try:
        weakref.ref(value)
    except TypeError:
        return False
    return True


class Obj:
    """A key with *equality* semantics -- cpython's ``test_weakref.Object``.

    Equal-but-distinct instances are what make the identity-keying laws
    meaningful: a stock ``WeakKeyDictionary`` collapses them into one entry.
    """

    def __init__(self, arg: int) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"<Obj {self.arg!r}>"

    def __eq__(self, other: object) -> typing.Any:
        return self.arg == other.arg if isinstance(other, Obj) else NotImplemented

    def __hash__(self) -> int:
        return hash(self.arg)


class Plain:
    """A key with the inherited identity semantics.

    This is the domain on which the dictionaries here and
    :class:`weakref.WeakKeyDictionary` must agree; see the differential tests.
    """

    def __init__(self, arg: int) -> None:
        self.arg = arg


class Hostile:
    """A key the standard library cannot hold: unhashable, and ``__eq__`` raises.

    Holding one of these is the whole reason this module exists -- upstream it is
    a Tensor, whose ``__eq__`` returns another Tensor rather than a bool.
    """

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        raise AssertionError("the key's __eq__ must never be called")


class Slotted:
    """Not weakly referenceable, despite being an ordinary instance."""

    __slots__ = ("x",)


class RefCycle:
    """Only collectable by the cycle collector, as in cpython's len tests."""

    def __init__(self) -> None:
        self.cycle = self


def big_int(i: int) -> int:
    """A non-weakrefable key that is *not* interned, so twins are distinguishable."""
    return 10**18 + i


def tup(i: int) -> tuple[int, ...]:
    return (i, i + 1)


@dataclasses.dataclass(frozen=True)
class Flavor:
    """A dictionary class paired with a kind of key it accepts."""

    cls: type[AnyDict]
    key: collections.abc.Callable[[int], typing.Any]
    weak: bool
    label: str

    def keys(self, n: int) -> list[typing.Any]:
        return [self.key(i) for i in range(n)]


FLAVORS = [
    Flavor(WeakIdKeyDictionary, Obj, weak=True, label="weak/obj"),
    Flavor(StrongIdKeyDictionary, Obj, weak=False, label="strong/obj"),
    Flavor(StrongIdKeyDictionary, big_int, weak=False, label="strong/int"),
    Flavor(AutoIdKeyDictionary, Obj, weak=True, label="auto/obj"),
    Flavor(AutoIdKeyDictionary, tup, weak=False, label="auto/tuple"),
]
WEAK_FLAVORS = [f for f in FLAVORS if f.weak]

DICT_CLASSES = [WeakIdKeyDictionary, StrongIdKeyDictionary, AutoIdKeyDictionary]


def flavors(cases: list[Flavor] = FLAVORS) -> typing.Any:
    return pytest.mark.parametrize("flavor", cases, ids=[f.label for f in cases])


classes = pytest.mark.parametrize(
    "cls", DICT_CLASSES, ids=[c.__name__ for c in DICT_CLASSES]
)


def build(flavor: Flavor, n: int = 3) -> tuple[AnyDict, list, dict[int, int]]:
    """A dictionary of ``n`` entries, its keys, and the ``id(key) -> value`` model."""
    d = flavor.cls()
    keys = flavor.keys(n)
    for i, k in enumerate(keys):
        d[k] = i
    return d, keys, {id(k): i for i, k in enumerate(keys)}


def assert_model(
    d: AnyDict,
    model: dict[int, typing.Any],
    present: collections.abc.Iterable = (),
    absent: collections.abc.Iterable = (),
) -> None:
    """Assert every observable of ``d`` agrees with ``model`` (``id(key) -> value``)."""
    assert len(d) == len(model)
    assert bool(d) == bool(model)
    assert {id(k): v for k, v in d.items()} == model
    assert sorted(id(k) for k in d.keys()) == sorted(model)
    assert sorted(id(k) for k in d) == sorted(model)
    assert sorted(d.values()) == sorted(model.values())
    # dict(d) goes through keys() and __getitem__; it collapses equal-but-distinct
    # keys, so compare it against a plain dict built the same way rather than the
    # identity-keyed model.
    assert dict(d) == dict(d.items())
    assert sorted(id(r()) for r in d.keyrefs() if r() is not None) == sorted(model)
    assert d == d.copy()
    for k in present:
        assert k in d
        assert d[k] == model[id(k)]
        assert d.get(k) == model[id(k)]
    for k in absent:
        assert k not in d
        assert d.get(k) is None
        assert d.get(k, "default") == "default"


###############################################################################
# Reference types: WeakIdRef, StrongIdRef, AutoIdRef
###############################################################################

REF_TYPES = [WeakIdRef, StrongIdRef, AutoIdRef]
ref_types = pytest.mark.parametrize(
    "ref_type", REF_TYPES, ids=[t.__name__ for t in REF_TYPES]
)


@ref_types
def test_ref_deref_and_hash(ref_type: typing.Any) -> None:
    """A live reference dereferences to its key and hashes as the key's id."""
    o = Obj(1)
    r = ref_type(o)
    assert r() is o
    assert hash(r) == id(o)


@ref_types
def test_ref_equal_to_itself_and_to_a_twin(ref_type: typing.Any) -> None:
    """A reference equals itself, and two to one object are equal and hash alike."""
    o = Obj(1)
    r, s = ref_type(o), ref_type(o)
    assert r == r
    assert r == s and s == r
    assert hash(r) == hash(s)


@ref_types
def test_ref_uses_identity_not_equality(ref_type: typing.Any) -> None:
    """References to equal-but-distinct objects are unequal."""
    a, b = Obj(1), Obj(1)
    assert a == b and hash(a) == hash(b)
    assert ref_type(a) != ref_type(b)
    assert ref_type(b) != ref_type(a)


@ref_types
def test_ref_never_touches_the_keys_protocols(ref_type: typing.Any) -> None:
    """The key's own ``__eq__`` and ``__hash__`` are never called.

    So an unhashable key whose ``__eq__`` raises still works as a referent.
    """
    a, b = Hostile(), Hostile()
    with pytest.raises(TypeError):
        hash(a)
    assert hash(ref_type(a)) == id(a)
    assert ref_type(a) == ref_type(a)
    assert ref_type(a) != ref_type(b)


def test_weak_ref_death() -> None:
    """A dead reference dereferences to None, still hashes, and equals only itself."""
    live = Obj(1)
    live_ref = WeakIdRef(live)
    dying = Obj(2)
    dead_ref = WeakIdRef(dying)
    dead_id = id(dying)
    del dying
    gc_collect()

    assert dead_ref() is None
    assert hash(dead_ref) == dead_id
    assert dead_ref == dead_ref
    assert dead_ref != live_ref
    assert live_ref != dead_ref


def test_weak_ref_aba() -> None:
    """A dead reference is not equal to a fresh one that reused its address.

    Deliberately no collection between the two allocations: ``Plain`` has no
    cycles, so refcounting frees it at the ``del``, and a ``gc.collect()`` in
    between only disturbs the free list that makes the address reuse happen.
    """
    for _ in range(100):
        first = Plain(1)
        recycled = id(first)
        dead_ref = WeakIdRef(first)
        del first
        second = Plain(2)
        if id(second) != recycled:
            del second
            continue
        live_ref = WeakIdRef(second)
        assert dead_ref() is None
        assert hash(dead_ref) == hash(live_ref)  # the trap: equal hashes
        assert dead_ref != live_ref
        assert live_ref != dead_ref
        return
    pytest.skip("no address reuse observed")


def test_ref_cross_type_symmetry() -> None:
    """A weak and a strong reference to one object are equal and hash alike.

    This is what lets :class:`AutoIdKeyDictionary` hold both kinds at once.
    """
    o = Obj(1)
    w, s = WeakIdRef(o), StrongIdRef(o)
    assert w == s and s == w
    assert hash(w) == hash(s) == id(o)


def test_ref_cross_type_death() -> None:
    """A dead weak reference is not equal to a strong one, in either direction."""
    live = Obj(1)
    strong = StrongIdRef(live)
    dying = Obj(2)
    dead = WeakIdRef(dying)
    del dying
    gc_collect()
    assert dead != strong
    assert strong != dead


def test_id_ref_is_strong() -> None:
    """A strong reference keeps its referent alive."""
    o = Obj(1)
    r = StrongIdRef(o)
    probe = weakref.ref(o)
    del o
    gc_collect()
    assert probe() is not None
    assert r() is probe()


NON_WEAKREFABLE = [
    ("int", 1),
    ("str", "s"),
    ("tuple", (1, 2)),
    ("list", [1]),
    ("dict", {1: 2}),
    ("float", 1.5),
    ("object", object()),
    ("slotted", Slotted()),
]


@pytest.mark.parametrize(
    "key", [v for _, v in NON_WEAKREFABLE], ids=[n for n, _ in NON_WEAKREFABLE]
)
def test_id_ref_accepts_non_weakrefable_keys(key: typing.Any) -> None:
    """A strong reference takes keys ``weakref.ref`` cannot."""
    assert not weakrefable_ground_truth(key)
    r = StrongIdRef(key)
    assert r() is key
    assert hash(r) == id(key)
    assert r == StrongIdRef(key)


def test_weak_id_ref_callback_fires() -> None:
    """A weak reference calls its callback, with itself, when the referent dies."""
    fired: list = []
    o = Obj(1)
    r = WeakIdRef(o, fired.append)
    del o
    gc_collect()
    assert fired == [r]


def test_id_ref_callback_never_fires() -> None:
    """A strong reference accepts the removal callback and never fires it."""
    fired: list = []
    o = Obj(1)
    r = StrongIdRef(o, fired.append)
    del o
    gc_collect()
    assert fired == []
    assert r() is not None


NON_REFS = [("int", 5), ("str", "x"), ("none", None), ("obj", Obj(1))]


@ref_types
@pytest.mark.parametrize(
    "other", [v for _, v in NON_REFS], ids=[n for n, _ in NON_REFS]
)
def test_ref_compared_to_a_non_ref(ref_type: typing.Any, other: typing.Any) -> None:
    """Comparing against a non-reference answers False rather than raising."""
    r = ref_type(Obj(99))
    assert (r == other) is False
    assert (other == r) is False
    assert (r != other) is True


###############################################################################
# Choosing a reference strength per key
#
# AutoIdRef._is_weakrefable is private, so it is tested through the only thing it
# decides: which reference AutoIdRef builds.
###############################################################################


WEAKREF_CASES = [
    ("int", 1),
    ("float", 1.5),
    ("bool", True),
    ("str", "s"),
    ("bytes", b"b"),
    ("tuple", (1,)),
    ("list", [1]),
    ("dict", {1: 2}),
    ("none", None),
    ("object", object()),
    ("slotted", Slotted()),
    ("set", set()),
    ("frozenset", frozenset()),
    ("obj", Obj(1)),
    ("class", Obj),
    ("type", int),
    ("function", weakrefable_ground_truth),
    ("module", weakref),
]


@pytest.mark.parametrize(
    "key", [v for _, v in WEAKREF_CASES], ids=[n for n, _ in WEAKREF_CASES]
)
def test_auto_id_ref_picks_by_weakrefability(key: typing.Any) -> None:
    """AutoIdRef builds a weak reference exactly when ``weakref.ref`` would work.

    Covers every kind of key the check can be asked about, including the ones it
    has to answer "no" for without raising.
    """
    r = AutoIdRef(key)
    assert isinstance(r, WeakIdRef) is weakrefable_ground_truth(key)
    assert isinstance(r, (WeakIdRef, StrongIdRef))
    assert not isinstance(r, AutoIdRef)
    assert r() is key
    assert hash(r) == id(key)


def test_auto_id_ref_type_cache_is_invisible() -> None:
    """Repeated keys and sibling instances of a cold type all get the same answer.

    Fresh classes, so the per-type cache starts empty for them and both the miss
    and the hit path are exercised.
    """

    class Fresh:
        pass

    class FreshSlotted:
        __slots__ = ()

    for _ in range(2):
        assert isinstance(AutoIdRef(Fresh()), WeakIdRef)
        assert isinstance(AutoIdRef(Fresh()), WeakIdRef)
        assert isinstance(AutoIdRef(FreshSlotted()), StrongIdRef)
        assert isinstance(AutoIdRef(FreshSlotted()), StrongIdRef)


###############################################################################
# Mapping laws, shared by every flavour
###############################################################################


@flavors()
def test_set_and_get(flavor: Flavor) -> None:
    """Assignment round-trips, and re-assigning a key overwrites it in place."""
    d = flavor.cls()
    k = flavor.key(0)
    value = ["a value"]
    d[k] = value
    assert d[k] is value
    assert d.get(k) is value
    assert len(d) == 1

    other = ["another"]
    d[k] = other
    assert d[k] is other
    assert len(d) == 1


@flavors()
def test_keys_are_compared_by_identity(flavor: Flavor) -> None:
    """Equal-but-distinct keys are distinct entries."""
    k1, k2 = flavor.key(0), flavor.key(0)
    assert k1 == k2 and k1 is not k2

    d = flavor.cls()
    d[k1] = "first"
    assert_model(d, {id(k1): "first"}, present=[k1], absent=[k2])

    d[k2] = "second"
    assert_model(d, {id(k1): "first", id(k2): "second"}, present=[k1, k2])
    assert d[k1] == "first"


@flavors()
def test_delitem(flavor: Flavor) -> None:
    """``del`` removes the entry; deleting it again raises KeyError."""
    d, keys, model = build(flavor)
    del d[keys[1]]
    del model[id(keys[1])]
    assert_model(d, model, present=[keys[0], keys[2]], absent=[keys[1]])

    with pytest.raises(KeyError):
        del d[keys[1]]


@flavors()
def test_pop(flavor: Flavor) -> None:
    """``pop`` removes and returns, honours a default, and raises without one.

    Ports ``WeakKeyDictionaryTestCase.test_pop``.
    """
    d, keys, model = build(flavor)
    assert d.pop(keys[1]) == model.pop(id(keys[1]))
    assert_model(d, model, absent=[keys[1]])

    with pytest.raises(KeyError):
        d.pop(keys[1])
    assert d.pop(keys[1], "default") == "default"
    assert_model(d, model, absent=[keys[1]])


@flavors()
def test_popitem(flavor: Flavor) -> None:
    """``popitem`` removes and returns a pair, then raises KeyError when empty.

    Ports ``check_popitem`` from pytorch's test_weak.py.
    """
    d, keys, model = build(flavor, n=2)
    for _ in range(2):
        k, v = d.popitem()
        assert model.pop(id(k)) == v
        assert_model(d, model, absent=[k])
    with pytest.raises(KeyError):
        d.popitem()


@flavors()
def test_setdefault(flavor: Flavor) -> None:
    """``setdefault`` inserts once, then returns the stored value without overwriting.

    Ports ``check_setdefault`` from pytorch's test_weak.py.
    """
    first, second = ["first"], ["second"]
    d = flavor.cls()
    k = flavor.key(0)

    assert d.setdefault(k, first) is first
    assert_model(d, {id(k): first}, present=[k])

    assert d.setdefault(k, second) is first
    assert_model(d, {id(k): first}, present=[k])


@flavors()
def test_update(flavor: Flavor) -> None:
    """``update`` and the constructor accept every form of source, or none.

    A mapping, another dictionary of the same class, an iterable of pairs, and no
    argument at all. Ports ``check_update``.
    """
    keys = flavor.keys(3)
    model = {id(k): i for i, k in enumerate(keys)}
    source = dict(zip(keys, range(3)))

    from_mapping = flavor.cls()
    from_mapping.update(source)
    assert_model(from_mapping, model, present=keys)

    from_pairs = flavor.cls()
    from_pairs.update(list(source.items()))
    assert_model(from_pairs, model, present=keys)

    from_same_class = flavor.cls()
    from_same_class.update(from_mapping)
    assert_model(from_same_class, model, present=keys)

    from_ctor = flavor.cls(source)
    assert_model(from_ctor, model, present=keys)

    from_mapping.update()
    assert_model(from_mapping, model, present=keys)


@flavors()
def test_update_from_pairs_preserves_distinct_identity_keys(flavor: Flavor) -> None:
    """A pair iterable must not collapse equal-but-distinct keys."""
    first, second = flavor.key(0), flavor.key(0)
    assert first == second and first is not second

    d = flavor.cls()
    d.update([(first, "first"), (second, "second")])

    assert_model(
        d,
        {id(first): "first", id(second): "second"},
        present=[first, second],
    )


@flavors()
def test_clear(flavor: Flavor) -> None:
    """``clear`` empties the dictionary."""
    d, keys, _ = build(flavor)
    d.clear()
    assert_model(d, {}, absent=keys)


@flavors()
def test_copy(flavor: Flavor) -> None:
    """``copy`` keeps the subclass, shares keys and values, and is independent."""
    d, keys, model = build(flavor)
    c = d.copy()

    assert type(c) is flavor.cls
    assert c == d
    assert_model(c, model, present=keys)
    for k in keys:
        assert c[k] is d[k]
    assert sorted(id(k) for k in c.keys()) == sorted(id(k) for k in d.keys())

    del c[keys[0]]
    assert keys[0] in d
    assert copy.copy(d) == d


@flavors()
def test_deepcopy(flavor: Flavor) -> None:
    """``deepcopy`` keeps the keys by identity but copies the values."""
    d = flavor.cls()
    keys = flavor.keys(2)
    for i, k in enumerate(keys):
        d[k] = [i]
    c = copy.deepcopy(d)

    assert type(c) is flavor.cls
    assert c == d
    for k in keys:
        assert c[k] == d[k]
        assert c[k] is not d[k]


@flavors()
def test_union_operators(flavor: Flavor) -> None:
    """``|``, ``|=`` and reflected ``|`` agree with the union, and keep the subclass.

    Ports ``test_weak_keyed_union_operators`` from both upstreams.
    """
    o1, o2, o3 = flavor.keys(3)
    wkd1 = flavor.cls({o1: 1, o2: 2})
    wkd2 = flavor.cls({o3: 3, o1: 4})
    wkd3 = wkd1.copy()
    d1 = {o2: "5", o3: "6"}
    pairs = [(o2, 7), (o3, 8)]

    def as_model(m: typing.Any) -> dict[int, typing.Any]:
        items = m.items() if hasattr(m, "items") else m
        return {id(k): v for k, v in items}

    tmp1 = wkd1 | wkd2
    assert as_model(tmp1) == as_model(wkd1) | as_model(wkd2)
    assert type(tmp1) is flavor.cls
    wkd1 |= wkd2
    assert wkd1 == tmp1

    tmp2 = wkd2 | d1
    assert as_model(tmp2) == as_model(wkd2) | as_model(d1)
    assert type(tmp2) is flavor.cls
    wkd2 |= d1
    assert wkd2 == tmp2

    tmp3 = wkd3.copy()
    tmp3 |= pairs
    assert as_model(tmp3) == as_model(wkd3) | as_model(pairs)
    assert type(tmp3) is flavor.cls

    tmp4 = d1 | wkd3
    assert as_model(tmp4) == as_model(d1) | as_model(wkd3)
    assert type(tmp4) is flavor.cls


@flavors()
def test_union_with_a_non_mapping(flavor: Flavor) -> None:
    """``|`` against a non-mapping raises TypeError."""
    d, _, _ = build(flavor)
    # The annotations now reject these statically too, which is the point.
    with pytest.raises(TypeError):
        d | 5  # type: ignore[operator]
    with pytest.raises(TypeError):
        5 | d  # type: ignore[operator]


@flavors()
def test_eq_is_identity_based(flavor: Flavor) -> None:
    """Equality compares key *identity*, not key equality."""
    d, keys, _ = build(flavor)

    assert d == d
    assert d == flavor.cls(dict(zip(keys, range(3))))
    assert d == {k: v for k, v in d.items()}
    assert d != flavor.cls()
    assert flavor.cls() == flavor.cls()

    twins = {flavor.key(i): i for i in range(3)}
    assert list(twins) == keys  # equal keys...
    assert d != twins  # ...but not the same objects

    assert d.__eq__(5) is NotImplemented
    assert (d == 5) is False
    assert (d != 5) is True


@flavors()
def test_repr(flavor: Flavor) -> None:
    """``repr`` names the actual subclass, and its address.

    Ports ``test_make_weak_keyed_dict_repr``.
    """
    assert re.fullmatch(rf"<{flavor.cls.__name__} at 0x[0-9a-f]+>", repr(flavor.cls()))


@flavors()
def test_keyrefs(flavor: Flavor) -> None:
    """``keyrefs`` dereference to exactly the live keys, and hash as their ids."""
    d, keys, _ = build(flavor)
    refs = d.keyrefs()
    assert sorted(id(r()) for r in refs) == sorted(id(k) for k in keys)
    assert all(hash(r) == id(r()) for r in refs)


@flavors()
def test_views_are_generators(flavor: Flavor) -> None:
    """``keys``, ``values`` and ``items`` are generators rather than views."""
    d, keys, _ = build(flavor)
    for view in (d.keys(), d.values(), d.items(), iter(d)):
        assert hasattr(view, "__iter__") and hasattr(view, "__next__")
        with pytest.raises(TypeError):
            len(view)  # type: ignore[arg-type]
    assert sorted(id(k) for k in iter(d)) == sorted(id(k) for k in d.keys())


@classes
def test_hostile_keys_are_supported(cls: type[AnyDict]) -> None:
    """Keys whose ``__eq__`` and ``__hash__`` are unusable work throughout.

    The fork's premise: a stock dict cannot even be built from one of these.
    """
    d = cls()
    a, b = Hostile(), Hostile()
    d[a] = "a"
    d[b] = "b"

    assert len(d) == 2
    assert d[a] == "a" and d[b] == "b"
    assert a in d and b in d
    assert sorted(id(k) for k in d.keys()) == sorted([id(a), id(b)])
    del d[a]
    assert a not in d and b in d

    with pytest.raises(TypeError):
        {a: 1}  # a stock dict cannot even build this


###############################################################################
# Weak keys: eviction, iteration guards, deferred removals
###############################################################################


def make_dict(flavor: Flavor, n: int = 10) -> tuple[AnyDict, list]:
    """cpython's ``make_weak_keyed_dict``, minus the lingering loop variable."""
    d = flavor.cls()
    objs = flavor.keys(n)
    for i, k in enumerate(objs):
        d[k] = i
    del k
    return d, objs


def test_weak_bad_key_types() -> None:
    """A key that cannot be weakly referenced raises, except for ``in``.

    Membership has to answer rather than raise, since ``x in d`` is legal for any
    ``x``. Ports ``test_weak_keyed_bad_delitem``.
    """
    d: typing.Any = WeakIdKeyDictionary()
    o = Obj(1)

    with pytest.raises(KeyError):
        del d[o]
    with pytest.raises(KeyError):
        d[o]

    for op in (
        lambda: d.__delitem__(13),
        lambda: d.__getitem__(13),
        lambda: d.__setitem__(13, 13),
        lambda: d.get(13),
        lambda: d.pop(13),
        lambda: d.setdefault(13, 13),
    ):
        with pytest.raises(TypeError):
            op()

    assert 13 not in d  # __contains__ answers rather than raising


@flavors(WEAK_FLAVORS)
def test_death_removes_the_entry(flavor: Flavor) -> None:
    """The entry goes when its key does, and the dictionary never keeps a key alive."""
    d = flavor.cls()
    keys = flavor.keys(3)
    for i, k in enumerate(keys):
        d[k] = i
    del k

    probe = weakref.ref(keys[1])
    model = {id(keys[0]): 0, id(keys[2]): 2}
    del keys[1]
    gc_collect()

    assert probe() is None
    assert_model(d, model, present=keys)
    assert len(d.keyrefs()) == 2


@flavors(WEAK_FLAVORS)
def test_dict_level_aba(flavor: Flavor) -> None:
    """A new object reusing a dead key's address is not in the dictionary.

    As in ``test_weak_ref_aba``, no collection between the two allocations: the
    key dies by refcount at the ``del``, and collecting in between makes the
    address reuse this depends on far less likely.
    """
    for _ in range(100):
        d = flavor.cls()
        first = flavor.key(0)
        recycled = id(first)
        d[first] = "gone"
        del first
        second = flavor.key(1)
        if id(second) != recycled:
            del second
            continue
        assert second not in d
        assert len(d) == 0
        with pytest.raises(KeyError):
            d[second]
        return
    pytest.skip("no address reuse observed")


@flavors(WEAK_FLAVORS)
def test_removal_is_deferred_while_iterating(flavor: Flavor) -> None:
    """A death mid-iteration is deferred, but never visible in ``len``.

    The entry stays in place until the iterator is dropped -- otherwise the walk
    would mutate the dictionary underneath itself -- so the dead reference is
    still among the ``keyrefs``. ``len`` discounts it regardless.
    """
    d, objs = make_dict(flavor)
    it = iter(d.items())
    next(it)

    del objs[-1]
    gc_collect()
    assert len(d) == 9
    assert len(d.keyrefs()) == 10
    assert sum(r() is None for r in d.keyrefs()) == 1

    del it
    gc_collect()
    assert len(d) == 9
    assert len(d.keyrefs()) == 9
    assert all(r() is not None for r in d.keyrefs())


@flavors(WEAK_FLAVORS)
@pytest.mark.parametrize("iter_name", ["keys", "items", "values", "keyrefs"])
def test_destroy_while_iterating(flavor: Flavor, iter_name: str) -> None:
    """A key dying mid-walk neither breaks the iterator nor loses the removal.

    Ports cpython's ``check_weak_destroy_while_iterating``.
    """
    d, objs = make_dict(flavor)
    n = len(d)
    it = iter(getattr(d, iter_name)())
    next(it)
    del objs[-1]
    gc_collect()
    assert len(list(it)) in (len(objs), len(objs) - 1)
    del it
    gc_collect()
    assert len(d) == n - 1


@contextlib.contextmanager
def killing_one_key(d: AnyDict, objs: list) -> collections.abc.Iterator[typing.Any]:
    """cpython's ``testcontext``: kill a key with an iterator held open."""
    it: typing.Any = iter(d.items())
    try:
        next(it)
        objs.pop()
        gc_collect()
        yield
    finally:
        it = None
        del it
        gc_collect()


@flavors(WEAK_FLAVORS)
def test_destroy_and_mutate_while_iterating(flavor: Flavor) -> None:
    """Every mutation still behaves while removals are pending.

    Ports cpython's ``check_weak_destroy_and_mutate_while_iterating``.
    """
    d, objs = make_dict(flavor)
    k, v = flavor.key(99), "v"

    with killing_one_key(d, objs):
        assert k not in d
    with killing_one_key(d, objs):
        with pytest.raises(KeyError):
            del d[k]
    assert k not in d
    with killing_one_key(d, objs):
        with pytest.raises(KeyError):
            d.pop(k)
    assert k not in d
    with killing_one_key(d, objs):
        d[k] = v
    assert d[k] == v

    ddict = copy.copy(d)
    with killing_one_key(d, objs):
        d.update(ddict)
    assert d == ddict
    with killing_one_key(d, objs):
        d.clear()
    assert len(d) == 0


@flavors(WEAK_FLAVORS)
def test_del_and_len_while_iterating(flavor: Flavor) -> None:
    """``len`` stays exact across pop, popitem, del and insert with removals pending.

    Ports cpython's ``check_weak_del_and_len_while_iterating``, the extensional
    exercise of the bookkeeping cpython issue #21173 added ``_scrub_removals`` for.
    """
    d, objs = make_dict(flavor)
    extra = flavor.key(123456)

    with killing_one_key(d, objs):
        n = len(d)
        d.pop(next(d.keys()))
        assert len(d) == n - 1
        d[extra] = extra
        assert len(d) == n
    with killing_one_key(d, objs):
        assert len(d) == n - 1
        d.popitem()
        assert len(d) == n - 2
    with killing_one_key(d, objs):
        assert len(d) == n - 3
        del d[next(d.keys())]
        assert len(d) == n - 4
    with killing_one_key(d, objs):
        assert len(d) == n - 5
        d.popitem()
        assert len(d) == n - 6
    with killing_one_key(d, objs):
        d.clear()
        assert len(d) == 0
    assert len(d) == 0


@flavors(WEAK_FLAVORS)
def test_explicit_delete_then_death_while_iterating(flavor: Flavor) -> None:
    """A key deleted by hand and *then* collected is not counted twice."""
    d, objs = make_dict(flavor)
    it = iter(d.items())
    next(it)

    doomed = objs.pop()
    del d[doomed]
    del doomed
    gc_collect()
    assert len(d) == 9

    del it
    gc_collect()
    assert len(d) == 9


@flavors(WEAK_FLAVORS)
def test_copy_skips_dead_keys(flavor: Flavor) -> None:
    """``copy`` and ``deepcopy`` skip dead keys rather than raising."""
    d, objs = make_dict(flavor, n=4)
    del objs[1:3]
    gc_collect()

    for c in (d.copy(), copy.deepcopy(d)):
        assert len(c) == 2
        assert sorted(id(k) for k in c.keys()) == sorted(id(k) for k in objs)


@flavors(WEAK_FLAVORS)
def test_len_with_cycles(flavor: Flavor) -> None:
    """``len`` settles to zero once cyclic keys are collected.

    Ports cpython's ``check_len_cycles``.
    """
    n = 20
    items = [RefCycle() for _ in range(n)]
    d = flavor.cls({o: 1 for o in items})
    it = d.items()
    with contextlib.suppress(StopIteration):
        next(it)
    del items
    gc.collect()
    n1 = len(d)
    del it
    gc.collect()
    n2 = len(d)
    assert n1 in (0, 1)
    assert n2 == 0


@flavors(WEAK_FLAVORS)
def test_len_race_against_the_collector(flavor: Flavor) -> None:
    """``len`` stays within bounds however the collector interleaves.

    Ports cpython's ``check_len_race``.
    """
    thresholds = gc.get_threshold()
    try:
        for th in range(1, 100, 7):
            n = 20
            gc.collect(0)
            gc.set_threshold(th, th, th)
            items = [RefCycle() for _ in range(n)]
            d = flavor.cls({o: 1 for o in items})
            del items
            it = d.items()
            with contextlib.suppress(StopIteration):
                next(it)
            n1 = len(d)
            del it
            n2 = len(d)
            assert 0 <= n1 <= n
            assert 0 <= n2 <= n1
    finally:
        gc.set_threshold(*thresholds)


@flavors(WEAK_FLAVORS)
def test_clear_with_only_dead_keys(flavor: Flavor) -> None:
    """``clear`` drives ``popitem``, which must survive a map of only dead keys."""
    d, objs = make_dict(flavor, n=3)
    del objs[:]
    gc_collect()
    d.clear()
    assert_model(d, {})


###############################################################################
# Strong keys: StrongIdKeyDictionary
###############################################################################


def test_id_key_dictionary_keeps_keys_alive() -> None:
    """A strong dictionary keeps its keys alive, and its entries with them."""
    d: StrongIdKeyDictionary = StrongIdKeyDictionary()
    k = Obj(1)
    probe = weakref.ref(k)
    d[k] = "v"
    key_id = id(k)
    del k
    gc_collect()

    assert probe() is not None
    assert len(d) == 1
    assert [id(x) for x in d.keys()] == [key_id]
    assert d[probe()] == "v"


@pytest.mark.parametrize(
    "key", [v for _, v in NON_WEAKREFABLE], ids=[n for n, _ in NON_WEAKREFABLE]
)
def test_id_key_dictionary_accepts_non_weakrefable_keys(key: typing.Any) -> None:
    """A strong dictionary takes any key, including unhashable ones."""
    d: StrongIdKeyDictionary = StrongIdKeyDictionary()
    d[key] = "v"
    assert d[key] == "v"
    assert key in d
    assert len(d) == 1
    del d[key]
    assert key not in d


def test_id_key_dictionary_is_stable_across_collection() -> None:
    """Nothing in a strong dictionary ever disappears on its own."""
    d, objs = make_dict(
        Flavor(StrongIdKeyDictionary, Obj, weak=False, label="strong/obj")
    )
    del objs[:]
    for _ in range(3):
        gc_collect()
        assert len(d) == 10
    assert len(list(d.items())) == 10


def test_id_key_dictionary_accepts_string_kwargs() -> None:
    """``update(**kwargs)`` works, since string keys need no weak reference."""
    d: StrongIdKeyDictionary = StrongIdKeyDictionary()
    d.update(a=1, b=2)
    assert sorted((k, v) for k, v in d.items()) == [("a", 1), ("b", 2)]


###############################################################################
# AutoIdKeyDictionary
###############################################################################


def test_auto_mixes_weak_and_strong_keys() -> None:
    """One dictionary, two reference strengths, chosen per key.

    Only the weakly referenceable entry is evictable.
    """
    d: AutoIdKeyDictionary = AutoIdKeyDictionary()
    weak_key, strong_key = Obj(1), (1, 2)
    d[weak_key] = "weak"
    d[strong_key] = "strong"

    assert len(d) == 2
    assert d[weak_key] == "weak" and d[strong_key] == "strong"

    probe = weakref.ref(weak_key)
    del weak_key
    gc_collect()

    assert probe() is None
    assert len(d) == 1
    assert d[strong_key] == "strong"  # the non-weakrefable entry is not evictable


def test_auto_accepts_string_kwargs() -> None:
    """``update(**kwargs)`` works here, where the weak dictionary raises."""
    d: AutoIdKeyDictionary = AutoIdKeyDictionary()
    d.update(a=1, b=2)
    assert sorted((k, v) for k, v in d.items()) == [("a", 1), ("b", 2)]

    weak: typing.Any = WeakIdKeyDictionary()
    with pytest.raises(TypeError):
        weak.update(a=1)


###############################################################################
# Differential equivalence with weakref.WeakKeyDictionary
###############################################################################


def snapshot(d: typing.Any) -> tuple:
    """Every observable that both implementations must agree on.

    Excludes ``repr``, ``type(d.copy())`` and the class of ``keyrefs()`` elements,
    which legitimately differ; the copy, repr and keyrefs tests pin those.
    """
    return (
        len(d),
        sorted((id(k), v) for k, v in d.items()),
        sorted(id(k) for k in d.keys()),
        sorted(id(k) for k in d),
        sorted(d.values()),
        sorted((id(k), v) for k, v in dict(d).items()),
        bool(d),
        sorted(id(r()) for r in d.keyrefs() if r() is not None),
    )


def attempt(fn: collections.abc.Callable[[], typing.Any]) -> typing.Any:
    """Run ``fn``, returning its result or the name of the exception it raised."""
    try:
        return ("ok", fn())
    except Exception as e:
        return ("raised", type(e).__name__)


OPS = [
    "set",
    "del",
    "pop",
    "popitem",
    "setdefault",
    "update",
    "clear",
    "copy",
    "deepcopy",
    "get",
    "contains",
    "eq",
    "kill",
    "new",
]


@pytest.mark.parametrize("seed", range(6))
@classes
def test_agrees_with_weak_key_dictionary(cls: type[AnyDict], seed: int) -> None:
    """Indistinguishable from the standard library on identity-semantics keys.

    ``StrongIdKeyDictionary`` agrees too, for a reason worth spelling out: it
    holds its keys strongly, which keeps the *reference* implementation's entries
    alive as well, so both dictionaries observe the same (empty) set of deaths.

    Iterators are never left open on one side only -- that would defer removals
    asymmetrically. Those cases are covered by the iteration tests above.
    """
    rng = random.Random(seed)
    ref: typing.Any = weakref.WeakKeyDictionary()
    test: typing.Any = cls()
    live = [Plain(i) for i in range(8)]

    for step in range(200):
        op = rng.choice(OPS)
        k: typing.Any = rng.choice(live)
        v = rng.randrange(1000)
        detail = f"class={cls.__name__} seed={seed} step={step} op={op}"

        if op == "set":
            ref[k] = v
            test[k] = v
        elif op == "del":
            assert attempt(lambda: ref.__delitem__(k)) == attempt(
                lambda: test.__delitem__(k)
            ), detail
        elif op == "pop":
            assert attempt(lambda: ref.pop(k, "miss")) == attempt(
                lambda: test.pop(k, "miss")
            ), detail
        elif op == "popitem":
            expected = attempt(lambda: ref.popitem())
            actual = attempt(lambda: test.popitem())
            if expected[0] == "ok":
                expected = ("ok", (id(expected[1][0]), expected[1][1]))
                actual = ("ok", (id(actual[1][0]), actual[1][1]))
            assert expected == actual, detail
        elif op == "setdefault":
            assert ref.setdefault(k, v) == test.setdefault(k, v), detail
        elif op == "update":
            source = {rng.choice(live): rng.randrange(1000) for _ in range(3)}
            ref.update(source)
            test.update(source)
        elif op == "clear":
            ref.clear()
            test.clear()
        elif op == "copy":
            assert snapshot(ref.copy()) == snapshot(test.copy()), detail
        elif op == "deepcopy":
            assert snapshot(copy.deepcopy(ref)) == snapshot(copy.deepcopy(test)), detail
        elif op == "get":
            assert ref.get(k) == test.get(k), detail
            assert ref.get(k, "default") == test.get(k, "default"), detail
        elif op == "contains":
            assert (k in ref) == (k in test), detail
        elif op == "eq":
            assert (ref == dict(ref.items())) == (test == dict(test.items())), detail
        elif op == "kill" and len(live) > 1:
            del live[rng.randrange(len(live))]
            gc.collect()  # Plain has no cycles, so one pass is enough
        elif op == "new":
            live.append(Plain(rng.randrange(1000)))

        # Drop the local reference so that a later "kill" can really collect it.
        k = None
        assert snapshot(ref) == snapshot(test), detail


def test_diverges_from_weak_key_dictionary_on_equality_keys() -> None:
    """The witness that identity keying is not equality keying."""
    k1, k2 = Obj(1), Obj(1)

    ref: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    test: WeakIdKeyDictionary = WeakIdKeyDictionary()
    for d in (ref, test):
        d[k1] = "first"
        d[k2] = "second"

    assert len(ref) == 1 and ref[k1] == "second"
    assert len(test) == 2 and test[k1] == "first" and test[k2] == "second"


def test_diverges_from_weak_key_dictionary_on_hostile_keys() -> None:
    """Keys the standard library cannot hold at all."""
    k = Hostile()

    ref: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    with pytest.raises(TypeError):
        ref[k] = "v"

    test: WeakIdKeyDictionary = WeakIdKeyDictionary()
    test[k] = "v"
    assert test[k] == "v"


###############################################################################
# weak_memoize
###############################################################################


def counted(fn: collections.abc.Callable) -> typing.Any:
    """``fn`` plus a record of the arguments it was actually called with.

    Records ``id(arg)`` rather than ``arg``: keeping the argument would defeat
    every test of when a cache entry is collected.
    """
    calls: list[int] = []

    @functools.wraps(fn)
    def wrapper(arg: typing.Any) -> typing.Any:
        calls.append(id(arg))
        return fn(arg)

    wrapper.calls = calls  # type: ignore[attr-defined]
    return wrapper


PURE_FUNCTIONS = [
    ("identity", lambda x: x),
    ("arg", lambda x: x.arg),
    ("none", lambda x: None),
    ("falsy", lambda x: 0),
    ("tuple", lambda x: (x.arg, x.arg)),
]


@pytest.mark.parametrize(
    "fn", [f for _, f in PURE_FUNCTIONS], ids=[n for n, _ in PURE_FUNCTIONS]
)
def test_memoize_agrees_with_the_bare_function(fn: typing.Any) -> None:
    """Memoizing does not change the answers, including None and other falsy ones."""
    memoized = weak_memoize(fn)
    keys = [Obj(i) for i in range(3)]
    for k in keys:
        assert memoized(k) == fn(k)
        assert memoized(k) == fn(k)


def test_memoize_calls_once_per_argument() -> None:
    """The wrapped function runs once per argument, however often it is called."""
    fn = counted(lambda x: None)
    memoized = weak_memoize(fn)
    a, b = Obj(1), Obj(2)

    for _ in range(3):
        assert memoized(a) is None
        assert memoized(b) is None

    assert fn.calls == [id(a), id(b)]


def test_memoize_keys_on_identity_not_equality() -> None:
    """Equal-but-distinct arguments are memoized separately."""
    fn = counted(lambda x: x.arg)
    memoized = weak_memoize(fn)
    a, b = Obj(1), Obj(1)
    assert a == b

    memoized(a)
    memoized(b)
    assert fn.calls == [id(a), id(b)]


def test_memoize_does_not_cache_exceptions() -> None:
    """A raising call is not cached, so the next call tries again."""
    calls: list = []

    @weak_memoize
    def boom(x: typing.Any) -> typing.Any:
        calls.append(x)
        raise ValueError("nope")

    k = Obj(1)
    for _ in range(2):
        with pytest.raises(ValueError):
            boom(k)
    assert len(calls) == 2


def test_memoize_decorator_forms_agree() -> None:
    """Bare, with arguments, and as a plain call all behave the same."""
    cache: AutoIdKeyDictionary = AutoIdKeyDictionary()
    bare = weak_memoize(counted(lambda x: x.arg))
    with_kwarg = weak_memoize(cache=cache)(counted(lambda x: x.arg))
    positional = weak_memoize(counted(lambda x: x.arg), cache=AutoIdKeyDictionary())

    k = Obj(7)
    assert bare(k) == with_kwarg(k) == positional(k) == 7
    assert bare(k) == with_kwarg(k) == positional(k) == 7
    for memoized in (bare, with_kwarg, positional):
        assert len(memoized.__wrapped__.calls) == 1


def test_memoize_uses_the_supplied_cache() -> None:
    """The supplied cache is populated, and pre-seeding it short-circuits the call."""
    cache: AutoIdKeyDictionary = AutoIdKeyDictionary()
    fn = counted(lambda x: x.arg)
    memoized = weak_memoize(fn, cache=cache)

    k = Obj(3)
    assert memoized(k) == 3
    assert k in cache and cache[k] == 3

    seeded = Obj(4)
    cache[seeded] = "pre-seeded"
    assert memoized(seeded) == "pre-seeded"
    assert fn.calls == [id(k)]


CACHE_FACTORIES = [
    ("WeakKeyDictionary", weakref.WeakKeyDictionary),
    ("WeakIdKeyDictionary", WeakIdKeyDictionary),
    ("StrongIdKeyDictionary", StrongIdKeyDictionary),
    ("AutoIdKeyDictionary", AutoIdKeyDictionary),
]


@pytest.mark.parametrize(
    "factory", [f for _, f in CACHE_FACTORIES], ids=[n for n, _ in CACHE_FACTORIES]
)
def test_memoize_works_with_every_cache_type(factory: typing.Any) -> None:
    """Every cache type the signature admits memoizes."""
    fn = counted(lambda x: x.arg)
    memoized = weak_memoize(fn, cache=factory())
    keys = [Obj(i) for i in range(3)]

    for _ in range(2):
        for k in keys:
            assert memoized(k) == k.arg
    assert len(fn.calls) == 3


def test_memoize_and_non_weakrefable_arguments() -> None:
    """A non-weakrefable argument: the default cache takes it, a weak cache does not."""
    fn = counted(lambda x: str(x))
    memoized = weak_memoize(fn)
    key = (1, 2)
    assert memoized(key) == memoized(key) == "(1, 2)"
    assert len(fn.calls) == 1

    weak_only: typing.Any = weak_memoize(lambda x: x, cache=weakref.WeakKeyDictionary())
    with pytest.raises(TypeError):
        weak_only((1, 2))


def test_memoize_is_scoped_to_the_arguments_lifetime() -> None:
    """An entry disappears when the argument it was computed from does."""
    cache: AutoIdKeyDictionary = AutoIdKeyDictionary()
    fn = counted(lambda x: x.arg)
    memoized = weak_memoize(fn, cache=cache)

    k = Obj(1)
    memoized(k)
    assert len(cache) == 1

    del k
    gc_collect()
    assert len(cache) == 0
    assert len(fn.calls) == 1


def test_memoize_keying_and_lifetime_follow_the_cache() -> None:
    """Neither the keying nor the entries' lifetime is fixed by ``weak_memoize``.

    Neither the identity keying nor the scoping to the argument's lifetime is a
    property of ``weak_memoize`` itself; both come from the cache, and the two
    non-default cache types in the signature give up one each.
    """
    # A stock WeakKeyDictionary keys on == , so equal-but-distinct arguments
    # share one entry.
    by_equality = counted(lambda x: x.arg)
    memoized = weak_memoize(by_equality, cache=weakref.WeakKeyDictionary())
    a, b = Obj(1), Obj(1)
    memoized(a)
    memoized(b)
    assert by_equality.calls == [id(a)]

    # A StrongIdKeyDictionary keeps the entry, and the key, alive.
    cache: StrongIdKeyDictionary = StrongIdKeyDictionary()
    kept = weak_memoize(lambda x: x.arg, cache=cache)
    k = Obj(2)
    probe = weakref.ref(k)
    kept(k)
    del k
    gc_collect()
    assert len(cache) == 1
    assert probe() is not None


def test_memoize_entry_survives_when_the_result_holds_the_argument() -> None:
    """An entry outlives its key if the value refers back to it.

    The entry can never be collected, because the value keeps the key alive. This
    is inherent to a weak-keyed cache; the test exists so that anyone changing the
    caching strategy sees it.
    """
    cache: AutoIdKeyDictionary = AutoIdKeyDictionary()

    @weak_memoize(cache=cache)
    def wrap(fn: typing.Any) -> typing.Any:
        @functools.wraps(fn)  # sets __wrapped__ = fn, a strong reference back
        def inner() -> typing.Any:
            return fn()

        return inner

    def target() -> int:
        return 1

    probe = weakref.ref(target)
    wrap(target)
    del target
    gc_collect()

    assert probe() is not None
    assert len(cache) == 1


def test_memoize_preserves_wrapper_metadata() -> None:
    """``__name__``, ``__doc__`` and ``__wrapped__`` come from the wrapped function."""

    def original(x: typing.Any) -> typing.Any:
        """A docstring."""
        return x

    memoized = weak_memoize(original)
    assert memoized.__name__ == "original"
    assert memoized.__doc__ == "A docstring."
    assert memoized.__wrapped__ is original  # type: ignore[attr-defined]


def test_memoize_agrees_with_functools_cache() -> None:
    """On identity-semantics arguments held alive, this is ``functools.cache``."""
    reference = counted(lambda x: x.arg * 2)
    test = counted(lambda x: x.arg * 2)
    cached = functools.cache(reference)
    memoized = weak_memoize(test)

    rng = random.Random(0)
    live = [Plain(i) for i in range(5)]
    for _ in range(100):
        k = rng.choice(live)
        assert cached(k) == memoized(k)
    assert reference.calls == test.calls


###############################################################################
# Concurrency
###############################################################################


@pytest.mark.timeout(30)
@classes
@pytest.mark.parametrize("deep", [False, True], ids=["copy", "deepcopy"])
def test_threaded_copy_under_collection(cls: type[AnyDict], deep: bool) -> None:
    """Ports pytorch's ``check_threaded_weak_dict_copy``, scaled down.

    Copying while another thread drops keys must not raise, however the removal
    callbacks interleave.
    """
    count = 2000
    exc: list[BaseException] = []

    class DummyKey:
        def __init__(self, ctr: int) -> None:
            self.ctr = ctr

    def dict_copy(d: AnyDict) -> None:
        try:
            copy.deepcopy(d) if deep else d.copy()
        except BaseException as e:  # noqa: BLE001
            exc.append(e)

    def pop_and_collect(lst: list) -> None:
        collected = 0
        while lst:
            lst.pop(random.Random(len(lst)).randrange(len(lst)))
            collected += 1
            if collected % 250 == 0:
                gc.collect()

    d = cls()
    keys = []
    for i in range(count):
        k = DummyKey(i)
        keys.append(k)
        d[k] = i
        del k

    copier = threading.Thread(target=dict_copy, args=(d,))
    collector = threading.Thread(target=pop_and_collect, args=(keys,))
    copier.start()
    collector.start()
    copier.join()
    collector.join()

    if exc:
        raise exc[0]
