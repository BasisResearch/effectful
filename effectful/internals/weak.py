# note: adapted from https://github.com/pytorch/pytorch/blob/708f706e7ac19247a4d6f26e81c9c706ac14d50d/torch/utils/weak.py
import collections.abc
import copy
import functools
import types
import typing
import weakref


# TODO: make weakref properly thread safe following
# https://github.com/python/cpython/pull/125325
class _IterationGuard[K, V]:
    # This context manager registers itself in the current iterators of the
    # weak container, such as to delay all removals until the context manager
    # exits.
    # This technique should be relatively thread-safe (since sets are).

    # CHANGED: parameterised by the container's key and value types, so that the
    # attribute below names the container rather than being an opaque weakref.
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


# This file defines a variant of WeakKeyDictionary that overrides the hashing
# behavior of the key to use object identity, rather than the builtin
# __eq__/__hash__ functions.  This is useful for Tensor weak keys, as their
# __eq__ implementation return a Tensor (elementwise equality), which means
# you can't use them directly with the WeakKeyDictionary in standard library.
#
# Our implementation strategy is to create a wrapper weak key object, which we
# use as a key in a stock Python dictionary.  This is similar to how weakref
# implements WeakKeyDictionary, but instead of using weakref.ref as the
# wrapper, we use a custom wrapper that has different __eq__ and __hash__
# behavior.  Note that we subsequently store this weak key directly in an
# ORDINARY dictionary, since the newly constructed WeakIdKey's only use would
# be a dictionary so it would have no strong references.  Ensuring that
# only live WeakIdKeys are in the map is handled by putting finalizers on the
# original key object.


# It is simpler to implement this with composition, but if we want to
# directly reuse the callback mechanism on weakref, we need the weakref
# and the key to be exactly the same object.  Reusing the callback mechanism
# minimizes the divergence between our implementation and Lib/weakref.py
#
# NB: Prefer using this when working with weakrefs of Tensors; e.g., do
# WeakIdRef(tensor) rather than weakref.ref(tensor); it handles a number of
# easy to get wrong cases transparently for you.
class WeakIdRef[T](weakref.ref[T]):
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

    def __eq__(self, other: typing.Any) -> bool:
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
        #
        # CHANGED: defer to the other operand for anything that is not a
        # reference, as stock weakref.ref does. Dereferencing it below would
        # raise TypeError instead of answering the comparison.
        if not isinstance(other, (weakref.ref, StrongIdRef)):
            return NotImplemented
        a = self()
        b = other()
        if a is not None and b is not None:
            return a is b
        return self is other

    def __ne__(self, other: typing.Any) -> bool:
        # CHANGED: weakref.ref implements == and != together in C, and its !=
        # compares the referents with the equality operator. Overriding __eq__
        # alone leaves != inherited from there, disagreeing with == and calling
        # the very __eq__ this class exists to route around -- for a Tensor key
        # that returns another Tensor rather than a bool.
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq


# This is a strong counterpart to WeakIdRef, for identity keying without weak
# references. That is what a cache scoped to a block wants: nothing can leak,
# because the whole dictionary is dropped when the block exits. It also accepts
# keys that cannot be weakly referenced at all, such as int, str, tuple, list
# and dict. Used by IdKeyDictionary, and by AutoIdRef for the keys WeakIdRef
# cannot take.
class StrongIdRef[T]:
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

    def __eq__(self, other: typing.Any) -> bool:
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


# Picks a reference type per key rather than per dictionary: weak where the key
# supports it, strong otherwise. This lets one dictionary accept any key at all,
# at the cost of making entries for weak-referenceable keys evictable, so the
# caller sees hit rates that depend on when the collector runs. Prefer
# WeakIdKeyDictionary or StrongIdKeyDictionary when the keys allow a single
# choice.
#
# This is a class rather than a factory function so that it can be assigned to
# WeakIdKeyDictionary.ref_type: a plain function there would be bound as a method
# and receive the dictionary as its first argument. Being a class also gives the
# weak-referenceability check somewhere to live other than the module's public
# surface, which is right: choosing between the two reference types is the only
# thing that check is for.
class AutoIdRef[T]:
    # CHANGED: weak-referenceability is a property of the type, so caching by
    # type turns the check into one lookup on the hot path. Weakly keyed, so that
    # having been asked about a dynamically created class does not pin it.
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


# CHANGED: the handle a ``ref_type`` builds for a key -- hashable by the key's
# identity, and callable to get the key back, or None where the reference was
# weak and the key has since died. Naming it lets ``data``, ``_pending_removals``
# and ``keyrefs`` say what they hold instead of falling back to Any, and puts the
# "check the result before using it" of keyrefs' docstring into the type.
class KeyRef[T](typing.Protocol):
    def __call__(self) -> T | None: ...


# CHANGED: what may be assigned to ``ref_type``. Both call shapes are here: bare
# to look a key up, and with the dictionary's removal callback to build an entry.
# Not parameterised by the key type: this is used as a ClassVar, which cannot
# refer to the class's type variables.
class RefType(typing.Protocol):
    def __call__(
        self,
        key: typing.Any,
        callback: collections.abc.Callable[..., typing.Any] | None = None,
        /,
    ) -> KeyRef[typing.Any]: ...


# CHANGED: the part of a mapping that ``update`` needs from its argument, which
# is what ``dict(...)`` accepts too. This is ``_typeshed.SupportsKeysAndGetItem``,
# spelled out here because that module does not exist at runtime; taking anything
# narrower, such as a Mapping, would be a signature that MutableMapping.update
# does not permit.
class KeysAndGetItem[K, V](typing.Protocol):
    def keys(self) -> collections.abc.Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V: ...


# Distinguishes "no default given" from a default of None, in pop below.
_MISSING: typing.Final = object()


# This is directly adapted from cpython/Lib/weakref.py
class WeakIdKeyDictionary[K, V](collections.abc.MutableMapping[K, V]):
    # CHANGED: reference strength is a property of the dictionary, so subclasses
    # select it by overriding this rather than callers passing it per instance.
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
        # A list of dead weakrefs (keys to be removed)
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
        del self.data[self.ref_type(key)]  # CHANGED

    def __getitem__(self, key: K) -> V:
        return self.data[self.ref_type(key)]  # CHANGED

    def __len__(self) -> int:
        if self._dirty_len and self._pending_removals:
            # self._pending_removals may still contain keys which were
            # explicitly removed, we have to scrub them (see issue #21173).
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} at {id(self):#x}>"

    def __setitem__(self, key: K, value: V) -> None:
        self.data[self.ref_type(key, self._remove)] = value  # CHANGED

    def copy(self) -> typing.Self:
        new = self.__class__()  # CHANGED: preserve the subclass's ref_type
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

    # CHANGED: the overloads Mapping.get declares, rather than one signature that
    # forced the default to be a V and had to be excused from the override check.
    @typing.overload
    def get(self, key: K) -> V | None: ...

    @typing.overload
    def get(self, key: K, default: V) -> V: ...

    @typing.overload
    def get[D](self, key: K, default: D) -> V | D: ...

    def get(self, key: K, default: typing.Any = None) -> typing.Any:
        return self.data.get(self.ref_type(key), default)  # CHANGED

    def __contains__(self, key: object) -> bool:
        try:
            wr = self.ref_type(key)  # CHANGED
        except TypeError:
            return False
        return wr in self.data

    # Generators rather than the views MutableMapping specifies: a view would
    # have to hold the dictionary, and these have to skip keys that died mid-walk.
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
        """Return a list of weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
        return list(self.data)

    def popitem(self) -> tuple[K, V]:
        self._dirty_len = True
        while True:
            key, value = self.data.popitem()
            o = key()
            if o is not None:
                return o, value

    # CHANGED: the overloads MutableMapping.pop declares, which is also what makes
    # the implementation's *args legible: the second argument is the default.
    @typing.overload
    def pop(self, key: K) -> V: ...

    @typing.overload
    def pop(self, key: K, default: V) -> V: ...

    @typing.overload
    def pop[D](self, key: K, default: D) -> V | D: ...

    def pop(self, key: K, default: typing.Any = _MISSING) -> typing.Any:
        self._dirty_len = True

        # CHANGED: a sentinel rather than *args, so that the signature says which
        # argument the overloads above are talking about.
        if default is _MISSING:
            return self.data.pop(self.ref_type(key))
        return self.data.pop(self.ref_type(key), default)

    # CHANGED: as MutableMapping.setdefault declares it. Omitting the default is
    # only meaningful when None is a value this dictionary can hold, which is what
    # the first overload's annotated self says.
    @typing.overload
    def setdefault(
        self: "WeakIdKeyDictionary[K, V | None]", key: K, default: None = None
    ) -> V | None: ...

    @typing.overload
    def setdefault(self, key: K, default: V) -> V: ...

    def setdefault(self, key: K, default: typing.Any = None) -> typing.Any:
        return self.data.setdefault(
            self.ref_type(key, self._remove), default
        )  # CHANGED

    def update(  # CHANGED: annotated
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
                d[self.ref_type(key, self._remove)] = value  # CHANGED
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

    # Default Mapping equality will tests keys for equality, but
    # we want to test ids for equality
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        return {id(k): v for k, v in self.items()} == {
            id(k): v for k, v in other.items()
        }


# CHANGED: the strong counterpart of WeakIdKeyDictionary. Keys are compared by
# identity and kept alive, so this accepts keys that cannot be weakly referenced
# and its entries never disappear on their own. Use it for a cache whose own
# lifetime already bounds the entries', such as one scoped to a block.
class StrongIdKeyDictionary[K, V](WeakIdKeyDictionary[K, V]):
    ref_type: typing.ClassVar[collections.abc.Callable[..., typing.Any]] = StrongIdRef


# CHANGED: accepts any key, holding it weakly where that is possible. This is
# the one to reach for when a single cache has to serve keys of both kinds; see
# AutoIdRef for what it gives up relative to the two dictionaries above.
class AutoIdKeyDictionary[K, V](WeakIdKeyDictionary[K, V]):
    ref_type: typing.ClassVar[collections.abc.Callable[..., typing.Any]] = AutoIdRef


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
    """Memoize ``fn`` using weak references to its argument and result.

    The memoization is scoped to the lifetime of the argument, so that when the
    argument is garbage collected, the memoized result is also discarded.

    Usable bare or with arguments: ``@weak_memoize`` and
    ``@weak_memoize(cache=...)`` are both decorators, which is what the two
    overloads above distinguish.
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
