"""Dictionaries that key on object identity rather than ``__eq__`` and ``__hash__``.

Adapted from pytorch's ``torch/utils/weak.py`` at 708f706, which is itself
adapted from cpython's ``Lib/weakref.py``:
https://github.com/pytorch/pytorch/blob/708f706e7ac19247a4d6f26e81c9c706ac14d50d/torch/utils/weak.py

A stock :class:`weakref.WeakKeyDictionary` compares its keys with ``==`` and
hashes them, which rules out most of what this package wants to cache on. A
term's ``__eq__`` is itself an operation, so comparing two of them builds
another term rather than answering the question; an interpretation is a plain
``dict``, so it cannot be hashed at all. Upstream the same problem arises for
Tensor keys, whose ``__eq__`` returns a Tensor of elementwise results. Keying on
``id`` sidesteps all of it, because the key's own protocols are never invoked.

The strategy is to wrap each key in a small object that hashes as ``id(key)``
and compares by identity, and to use that wrapper as the key of an ordinary
``dict``. Three wrappers are on offer, with a dictionary for each:

* :class:`WeakIdRef` and :class:`WeakIdKeyDictionary` hold the key weakly, so
  an entry disappears when its key is collected. Only some objects can be
  weakly referenced: ``int``, ``str``, ``tuple`` and the other leaves of an
  expression cannot.
* :class:`StrongIdRef` and :class:`StrongIdKeyDictionary` hold the key
  strongly, so they take any key at all, at the cost of bounding an entry's
  lifetime by the dictionary's rather than by the key's.
* :class:`AutoIdRef` and :class:`AutoIdKeyDictionary` decide per key, holding
  it weakly where that is possible.

Because the wrappers are stored in an ordinary ``dict``, nothing else refers to
them, and keeping only live ones in the map is the job of finalizers installed
on the original keys.
"""

import collections.abc
import copy
import functools
import types
import typing
import weakref


# TODO: make weakref properly thread safe following
# https://github.com/python/cpython/pull/125325
class _IterationGuard[K, V]:
    """Delays a weak container's removals until an iteration over it finishes.

    Registers itself in the container's set of current iterators on entry and
    commits the removals that piled up on exit, so that a key dying mid-walk
    does not mutate the dictionary out from under the walk. Relatively
    thread-safe, since sets are.
    """

    weakcontainer: weakref.ref["WeakIdKeyDictionary[K, V]"]

    def __init__(self, weakcontainer: "WeakIdKeyDictionary[K, V]") -> None:
        # Don't create cycles
        self.weakcontainer = weakref.ref(weakcontainer)

    def __enter__(self) -> typing.Self:
        w = self.weakcontainer()
        if w is not None:
            w._iterating.add(self)
        return self

    def __exit__(
        self,
        e: type[BaseException] | None,
        t: BaseException | None,
        b: types.TracebackType | None,
    ) -> None:
        w = self.weakcontainer()
        if w is not None:
            s = w._iterating
            s.remove(self)
            if not s:
                w._commit_removals()


class WeakIdRef[T](weakref.ref[T]):
    """A weak reference that hashes and compares by its referent's identity.

    Subclasses :class:`weakref.ref` rather than wrapping one. Composition would
    be simpler, but reusing weakref's callback mechanism requires the reference
    and the key to be exactly the same object, and reusing it keeps the
    divergence from ``Lib/weakref.py`` small.

    Prefer this over a bare ``weakref.ref`` whenever the reference will be used
    as a key; it handles a number of easy to get wrong cases transparently.
    """

    __slots__ = ["_id"]

    def __init__(
        self, key: T, callback: collections.abc.Callable[..., typing.Any] | None = None
    ) -> None:
        # Unlike stock weakref, which preserves hash semantics of the
        # original object but lazily defers hash calls until the first
        # time the user attempts to hash the weakref, we can eagerly
        # cache the id of the key as we know this is definitely the hash
        # method
        self._id = id(key)
        super().__init__(key, callback)  # type: ignore[call-arg]

    def __call__(self) -> T | None:
        r = super().__call__()
        # Special logic for Tensor PyObject resurrection
        if r is not None and hasattr(r, "_fix_weakref"):
            r._fix_weakref()
        return r

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other: object) -> bool:
        # Anything that is not a reference is the other operand's business:
        # dereferencing it below would raise TypeError rather than answer the
        # comparison. Stock weakref.ref defers here too.
        if not isinstance(other, (weakref.ref, StrongIdRef)):
            return NotImplemented

        # An attractive but wrong alternate implementation is to only test if
        # the stored _ids match.  This can lead to an ABA problem if you have:
        #
        #   a1 = A()
        #   w1 = WeakIdRef(a1)
        #   del a1
        #   a2 = A()  # suppose it gets the same ID as a1
        #   w2 = WeakIdRef(a2)
        #   print(w1 == w2)
        #
        # This should be False, as a1 and a2 are unrelated (and a1 is
        # dead anyway)
        a = self()
        b = other()
        if a is not None and b is not None:
            return a is b
        return self is other

    def __ne__(self, other: object) -> bool:
        # Not redundant with __eq__: weakref.ref implements == and != together
        # in C, and its != compares the referents with the equality operator.
        # Overriding __eq__ alone leaves != inherited from there, disagreeing
        # with == and calling the very __eq__ this class exists to route around.
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq


class StrongIdRef[T]:
    """A strong reference with the same interface as :class:`WeakIdRef`.

    Identity keying without weak references is what a cache scoped to a block
    wants: nothing can leak, because the whole dictionary is dropped when the
    block exits. It also accepts keys that cannot be weakly referenced at all,
    such as ``int``, ``str``, ``tuple``, ``list`` and ``dict``.

    Used by :class:`StrongIdKeyDictionary`, and by :class:`AutoIdRef` for the
    keys :class:`WeakIdRef` cannot take.
    """

    __slots__ = ["_id", "_obj"]

    def __init__(
        self, key: T, callback: collections.abc.Callable[..., typing.Any] | None = None
    ) -> None:
        # Accepts and ignores WeakIdKeyDictionary's removal callback: a strong
        # reference never dies while it is in the dictionary, so it never fires.
        self._id = id(key)
        self._obj = key

    def __call__(self) -> T | None:
        # Never actually None, but typed to match WeakIdRef so the two are
        # interchangeable as a dictionary's ref_type and inside AutoIdRef.
        return self._obj

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other: object) -> bool:
        # AutoIdRef puts both reference types in one dictionary, so this has to
        # stay symmetric with WeakIdRef.__eq__, which treats a dead referent as
        # equal only to itself.
        #
        # Against another strong reference, comparing ids is enough and is what
        # we want: neither referent can have died, so an id cannot have been
        # recycled, and a None key compares equal to itself rather than reading
        # as dead. Against a weak reference we have to dereference, because a
        # dead WeakIdRef keeps the id of an object that may since have been
        # replaced at that address -- the ABA case its own __eq__ guards.
        if isinstance(other, StrongIdRef):
            return self._id == other._id
        if not isinstance(other, weakref.ref):
            return NotImplemented
        return (b := other()) is not None and b is self._obj


class AutoIdRef[T]:
    """Builds a :class:`WeakIdRef`, or a :class:`StrongIdRef` where it must.

    Picking the reference type per key rather than per dictionary lets one
    dictionary accept any key at all. The cost is that entries for weakly
    referenceable keys are evictable while the rest are not, so the caller sees
    hit rates that depend on when the collector runs. Prefer
    :class:`WeakIdKeyDictionary` or :class:`StrongIdKeyDictionary` where the
    keys allow a single choice.

    A class rather than a factory function so that it can be assigned to
    :attr:`WeakIdKeyDictionary.ref_type`, where a plain function would be bound
    as a method and receive the dictionary as its first argument. Being a class
    also gives the weak-referenceability check below somewhere to live other
    than the module's public surface, which is where it belongs: choosing
    between the two reference types is the only thing that check is for.
    """

    # Weak-referenceability is a property of the type, so caching by type turns
    # the check into one lookup on the hot path. Weakly keyed, so that having
    # been asked about a dynamically created class does not pin it.
    _WEAKREFABLE: typing.ClassVar[weakref.WeakKeyDictionary[type, bool]] = (
        weakref.WeakKeyDictionary()
    )

    @classmethod
    def _is_weakrefable(cls, obj: object) -> bool:
        """Report whether ``obj`` can be the target of a weak reference.

        Answers without raising, so the caller can branch on it rather than
        catching ``TypeError`` -- worth doing where the answer is usually "no",
        as it is for the ``int``, ``str`` and ``tuple`` leaves of an expression.
        """
        ok = cls._WEAKREFABLE.get(t := type(obj))
        if ok is None:
            try:
                weakref.ref(obj)
                ok = True
            except TypeError:
                ok = False
            cls._WEAKREFABLE[t] = ok
        return ok

    def __new__(  # type: ignore[misc]  # deliberately returns another class
        cls, key: T, callback: collections.abc.Callable[..., typing.Any] | None = None
    ) -> "WeakIdRef[T] | StrongIdRef[T]":
        return (
            WeakIdRef(key, callback)
            if cls._is_weakrefable(key)
            else StrongIdRef(key, callback)
        )


class KeyRef[T](typing.Protocol):
    """The handle a :attr:`WeakIdKeyDictionary.ref_type` builds for a key.

    Hashable by the key's identity, and callable to get the key back -- or
    ``None`` where the reference was weak and the key has since died, which is
    why the result has to be checked before it is used.
    """

    def __call__(self) -> T | None: ...


class RefType(typing.Protocol):
    """What may be assigned to :attr:`WeakIdKeyDictionary.ref_type`.

    Both call shapes are here: bare, to look a key up, and with the
    dictionary's removal callback, to build an entry. Not parameterised by the
    key type, because a ``ClassVar`` cannot refer to its class's type variables.
    """

    def __call__(
        self,
        key: typing.Any,
        callback: collections.abc.Callable[..., typing.Any] | None = None,
        /,
    ) -> KeyRef[typing.Any]: ...


class KeysAndGetItem[K, V](typing.Protocol):
    """The part of a mapping that :meth:`WeakIdKeyDictionary.update` needs.

    This is ``_typeshed.SupportsKeysAndGetItem``, spelled out because that
    module does not exist at runtime. Anything narrower, a ``Mapping`` say,
    would be a signature that ``MutableMapping.update`` does not permit an
    override to have.
    """

    def keys(self) -> collections.abc.Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V: ...


# Distinguishes "no default given" from a default of None, in pop below.
_MISSING: typing.Final = object()


class WeakIdKeyDictionary[K, V](collections.abc.MutableMapping[K, V]):
    """A :class:`weakref.WeakKeyDictionary` keyed on ``id`` rather than ``==``.

    Directly adapted from cpython's ``Lib/weakref.py``. Keys are held weakly and
    their entries disappear once they are collected, which is also why ``keys``,
    ``values`` and ``items`` are generators rather than the views a ``Mapping``
    would normally return: they have to skip keys that died mid-walk.
    """

    # Reference strength is a property of the dictionary, so subclasses select
    # it by overriding this rather than callers passing it per instance.
    ref_type: typing.ClassVar[RefType] = WeakIdRef

    # Declared here rather than annotated in ``__init__``, where the ``dict``
    # parameter below shadows the builtin the annotations would need.
    data: dict[KeyRef[K], V]
    _remove: collections.abc.Callable[[KeyRef[K]], None]
    _pending_removals: list[KeyRef[K]]
    _iterating: set[_IterationGuard[K, V]]
    _dirty_len: bool

    def __init__(self, dict: collections.abc.Mapping[K, V] | None = None) -> None:
        self.data = {}

        def remove(
            k: KeyRef[K],
            selfref: weakref.ref["WeakIdKeyDictionary[K, V]"] = weakref.ref(self),
        ) -> None:
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(k)
                else:
                    try:
                        del self.data[k]
                    except KeyError:
                        pass

        self._remove = remove
        self._pending_removals = []
        self._iterating = set()
        self._dirty_len = False
        if dict is not None:
            self.update(dict)

    def _commit_removals(self) -> None:
        # NOTE: We don't need to call this method before mutating the dict,
        # because a dead weakref never compares equal to a live weakref,
        # even if they happened to refer to equal objects.
        # However, it means keys may already have been removed.
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return

            try:
                del d[key]
            except KeyError:
                pass

    def _scrub_removals(self) -> None:
        d = self.data
        self._pending_removals = [k for k in self._pending_removals if k in d]
        self._dirty_len = False

    def __delitem__(self, key: K) -> None:
        self._dirty_len = True
        del self.data[self.ref_type(key)]

    def __getitem__(self, key: K) -> V:
        return self.data[self.ref_type(key)]

    def __len__(self) -> int:
        if self._dirty_len and self._pending_removals:
            # self._pending_removals may still contain keys which were
            # explicitly removed, we have to scrub them (see issue #21173).
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} at {id(self):#x}>"

    def __setitem__(self, key: K, value: V) -> None:
        self.data[self.ref_type(key, self._remove)] = value

    def copy(self) -> typing.Self:
        new = self.__class__()  # not WeakIdKeyDictionary: keep the subclass's ref_type
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = value
        return new

    __copy__ = copy

    def __deepcopy__(self, memo: dict[int, typing.Any]) -> typing.Self:
        new = self.__class__()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = copy.deepcopy(value, memo)
        return new

    @typing.overload
    def get(self, key: K) -> V | None: ...

    @typing.overload
    def get(self, key: K, default: V) -> V: ...

    @typing.overload
    def get[D](self, key: K, default: D) -> V | D: ...

    def get(self, key: K, default: typing.Any = None) -> typing.Any:
        return self.data.get(self.ref_type(key), default)

    def __contains__(self, key: object) -> bool:
        try:
            wr = self.ref_type(key)
        except TypeError:
            # A key the ref_type cannot take is simply not in the dictionary.
            return False
        return wr in self.data

    def items(self) -> collections.abc.Iterator[tuple[K, V]]:  # type: ignore[override]
        with _IterationGuard(self):
            for wr, value in self.data.items():
                key = wr()
                if key is not None:
                    yield key, value

    def keys(self) -> collections.abc.Iterator[K]:  # type: ignore[override]
        with _IterationGuard(self):
            for wr in self.data:
                obj = wr()
                if obj is not None:
                    yield obj

    __iter__ = keys

    def values(self) -> collections.abc.Iterator[V]:  # type: ignore[override]
        with _IterationGuard(self):
            for wr, value in self.data.items():
                if wr() is not None:
                    yield value

    def keyrefs(self) -> list[KeyRef[K]]:
        """Return a list of references to the keys.

        The references are not guaranteed to be 'live' at the time they are
        used, so the result of calling one needs to be checked before being
        used. This can be used to avoid creating references that will cause the
        garbage collector to keep the keys around longer than needed.
        """
        return list(self.data)

    def popitem(self) -> tuple[K, V]:
        self._dirty_len = True
        while True:
            key, value = self.data.popitem()
            o = key()
            if o is not None:
                return o, value

    @typing.overload
    def pop(self, key: K) -> V: ...

    @typing.overload
    def pop(self, key: K, default: V) -> V: ...

    @typing.overload
    def pop[D](self, key: K, default: D) -> V | D: ...

    def pop(self, key: K, default: typing.Any = _MISSING) -> typing.Any:
        self._dirty_len = True

        if default is _MISSING:
            return self.data.pop(self.ref_type(key))
        return self.data.pop(self.ref_type(key), default)

    # Omitting the default is only meaningful when None is a value this
    # dictionary can hold, which is what the first overload's annotated self
    # says. Both are as MutableMapping.setdefault declares them.
    @typing.overload
    def setdefault(
        self: "WeakIdKeyDictionary[K, V | None]", key: K, default: None = None
    ) -> V | None: ...

    @typing.overload
    def setdefault(self, key: K, default: V) -> V: ...

    def setdefault(self, key: K, default: typing.Any = None) -> typing.Any:
        return self.data.setdefault(self.ref_type(key, self._remove), default)

    def update(
        self,
        dict: KeysAndGetItem[K, V]
        | collections.abc.Iterable[tuple[K, V]]
        | None = None,
        **kwargs: V,
    ) -> None:
        d = self.data
        if dict is not None:
            # Any because the fallback is whatever the builtin dict() accepts,
            # which covers both forms above and is not expressible as a type.
            source: typing.Any = dict
            if not hasattr(source, "items"):
                source = type({})(source)
            for key, value in source.items():
                d[self.ref_type(key, self._remove)] = value
        if kwargs:
            # Only well typed when K is str, which no annotation here can say.
            self.update(typing.cast(KeysAndGetItem[K, V], kwargs))

    def __ior__(
        self, other: KeysAndGetItem[K, V] | collections.abc.Iterable[tuple[K, V]]
    ) -> typing.Self:
        self.update(other)
        return self

    def __or__(
        self, other: collections.abc.Mapping[K, V]
    ) -> "typing.Self | types.NotImplementedType":
        if isinstance(other, collections.abc.Mapping):
            c = self.copy()
            c.update(other)
            return c
        return NotImplemented

    def __ror__(
        self, other: collections.abc.Mapping[K, V]
    ) -> "typing.Self | types.NotImplementedType":
        if isinstance(other, collections.abc.Mapping):
            c = self.__class__()
            c.update(other)
            c.update(self)
            return c
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        # Mapping's default equality tests the keys for equality, and this
        # dictionary's whole point is that its keys are compared by identity.
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        return {id(k): v for k, v in self.items()} == {
            id(k): v for k, v in other.items()
        }


class StrongIdKeyDictionary[K, V](WeakIdKeyDictionary[K, V]):
    """The strong counterpart of :class:`WeakIdKeyDictionary`.

    Keys are compared by identity and kept alive, so this accepts keys that
    cannot be weakly referenced and its entries never disappear on their own.
    Use it for a cache whose own lifetime already bounds its entries', such as
    one scoped to a block.
    """

    ref_type: typing.ClassVar[RefType] = StrongIdRef


class AutoIdKeyDictionary[K, V](WeakIdKeyDictionary[K, V]):
    """Accepts any key, holding it weakly where that is possible.

    The one to reach for when a single cache has to serve keys of both kinds;
    see :class:`AutoIdRef` for what it gives up relative to the two dictionaries
    above.
    """

    ref_type: typing.ClassVar[RefType] = AutoIdRef


type WeakKeyCache[S, T] = weakref.WeakKeyDictionary[S, T] | WeakIdKeyDictionary[S, T]


@typing.overload
def weak_memoize[S, T](
    fn: collections.abc.Callable[[S], T], *, cache: WeakKeyCache[S, T] | None = None
) -> collections.abc.Callable[[S], T]: ...


@typing.overload
def weak_memoize[S, T](
    *, cache: WeakKeyCache[S, T] | None = None
) -> collections.abc.Callable[
    [collections.abc.Callable[[S], T]], collections.abc.Callable[[S], T]
]: ...


def weak_memoize[S, T](
    fn: collections.abc.Callable[[S], T] | None = None,
    *,
    cache: WeakKeyCache[S, T] | None = None,
) -> typing.Any:
    """Memoize ``fn`` on its single argument.

    How the argument is keyed, and how long an entry lives, are both properties
    of ``cache`` rather than of this function. The default is an
    :class:`AutoIdKeyDictionary`, which keys on the argument's identity and
    scopes the entry to the argument's lifetime wherever the argument can be
    weakly referenced. A :class:`weakref.WeakKeyDictionary` keys on ``==``
    instead; a :class:`StrongIdKeyDictionary` keeps every entry, and every key,
    for as long as the cache itself lives.

    An entry outlives its key even in the weak cases if the result refers back
    to the argument, as a wrapper built by :func:`functools.wraps` does: the
    value keeps its own key alive.

    Usable bare, with arguments, or as a plain call: ``@weak_memoize``,
    ``@weak_memoize(cache=...)`` and ``weak_memoize(fn, cache=...)`` are all
    supported, and the first two are what the overloads above distinguish.
    """
    if fn is None:
        return functools.partial(weak_memoize, cache=cache)

    if cache is None:
        cache = AutoIdKeyDictionary()

    @functools.wraps(fn)
    def _memoized(arg: S) -> T:
        if arg in cache:
            return cache[arg]
        result = fn(arg)
        cache[arg] = result
        return result

    return _memoized
