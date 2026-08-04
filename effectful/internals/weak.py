# note: adapted from https://github.com/pytorch/pytorch/blob/708f706e7ac19247a4d6f26e81c9c706ac14d50d/torch/utils/weak.py
import collections.abc
import copy
import functools
import typing
import weakref


# TODO: make weakref properly thread safe following
# https://github.com/python/cpython/pull/125325
class _IterationGuard:
    # This context manager registers itself in the current iterators of the
    # weak container, such as to delay all removals until the context manager
    # exits.
    # This technique should be relatively thread-safe (since sets are).

    def __init__(self, weakcontainer) -> None:
        # Don't create cycles
        self.weakcontainer = weakref.ref(weakcontainer)

    def __enter__(self):
        w = self.weakcontainer()
        if w is not None:
            w._iterating.add(self)
        return self

    def __exit__(self, e, t, b):
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
        a = self()
        b = other()
        if a is not None and b is not None:
            return a is b
        return self is other


# This is a strong counterpart to WeakIdRef, for identity keying without weak
# references. That is what a cache scoped to a block wants: nothing can leak,
# because the whole dictionary is dropped when the block exits. It also accepts
# keys that cannot be weakly referenced at all, such as int, str, tuple, list
# and dict. Used by IdKeyDictionary, and by AutoIdRef for the keys WeakIdRef
# cannot take.
class IdRef[T]:
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
        if isinstance(other, IdRef):
            return self._id == other._id
        return (b := other()) is not None and b is self._obj


_WEAKREFABLE: dict[type, bool] = {}


def is_weakrefable(obj: object) -> bool:
    """Report whether ``obj`` can be the target of a weak reference.

    Answers without raising, so callers can branch on it rather than catching
    ``TypeError`` -- worth doing where the answer is usually "no", as it is for
    the ``int``, ``str`` and ``tuple`` leaves of an expression.
    """
    # Weak-referenceability is a property of the type, so caching by type turns
    # this into one dict lookup on the hot path.
    ok = _WEAKREFABLE.get(t := type(obj))
    if ok is None:
        try:
            weakref.ref(obj)
            ok = True
        except TypeError:
            ok = False
        _WEAKREFABLE[t] = ok
    return ok


# Picks a reference type per key rather than per dictionary: weak where the key
# supports it, strong otherwise. This lets one dictionary accept any key at all,
# at the cost of making entries for weak-referenceable keys evictable, so the
# caller sees hit rates that depend on when the collector runs. Prefer
# WeakIdKeyDictionary or IdKeyDictionary when the keys allow a single choice.
#
# This is a class rather than a factory function so that it can be assigned to
# WeakIdKeyDictionary.ref_type: a plain function there would be bound as a
# method and receive the dictionary as its first argument.
class AutoIdRef[T]:
    def __new__(  # type: ignore[misc]  # deliberately returns another class
        cls, key: T, callback: collections.abc.Callable[..., typing.Any] | None = None
    ) -> "WeakIdRef[T] | IdRef[T]":
        return WeakIdRef(key, callback) if is_weakrefable(key) else IdRef(key, callback)


# This is the same as WeakIdRef but equality is checked using hash() rather than id.
# This will be equivalent to the one above except for classes where hash is not their id.
class _WeakHashRef[T](weakref.ref[T]):
    __slots__ = ["_id"]

    def __init__(
        self, key: T, callback: collections.abc.Callable[..., typing.Any] | None = None
    ) -> None:
        # Unlike stock weakref, which preserves hash semantics of the
        # original object but lazily defers hash calls until the first
        # time the user attempts to hash the weakref, we can eagerly
        # cache the id of the key as we know this is definitely the hash
        # method
        self._id = hash(key)
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
        # Use hash equality to determine ref equality.
        # ScriptObject implements __hash__ to return the wrapped IValue's id, so
        # this is equivalent to doing an identity comparison.
        a = self()
        b = other()
        if a is not None and b is not None:
            return hash(a) == hash(b)
        return self is other


# This is directly adapted from cpython/Lib/weakref.py
class WeakIdKeyDictionary[K, V](collections.abc.MutableMapping[K, V]):
    # CHANGED: reference strength is a property of the dictionary, so subclasses
    # select it by overriding this rather than callers passing it per instance.
    # Not parameterised by K: a ClassVar cannot refer to the class's type
    # variables, so the reference type it builds is only known as a callable.
    ref_type: typing.ClassVar[collections.abc.Callable[..., typing.Any]] = WeakIdRef

    # Declared here rather than annotated in ``__init__``, where the ``dict``
    # parameter below shadows the builtin the annotations would need.
    data: dict[typing.Any, V]
    _pending_removals: list[typing.Any]
    _iterating: set[_IterationGuard]
    _dirty_len: bool

    def __init__(self, dict: collections.abc.Mapping[K, V] | None = None) -> None:
        self.data = {}

        def remove(k, selfref=weakref.ref(self)) -> None:
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

    def __deepcopy__(self, memo) -> typing.Self:
        new = self.__class__()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = copy.deepcopy(value, memo)
        return new

    def get(self, key: K, default: V | None = None) -> V | None:  # type: ignore[override]
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

    def keyrefs(self) -> list[typing.Any]:
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

    # pyrefly: ignore [bad-override]
    def pop(self, key: K, *args: typing.Any) -> typing.Any:
        self._dirty_len = True

        return self.data.pop(self.ref_type(key), *args)  # CHANGED

    def setdefault(self, key: K, default: typing.Any = None) -> typing.Any:
        return self.data.setdefault(
            self.ref_type(key, self._remove), default
        )  # CHANGED

    def update(self, dict=None, **kwargs) -> None:
        d = self.data
        if dict is not None:
            if not hasattr(dict, "items"):
                dict = type({})(dict)
            for key, value in dict.items():
                d[self.ref_type(key, self._remove)] = value  # CHANGED
        if kwargs:
            self.update(kwargs)

    def __ior__(self, other):
        self.update(other)
        return self

    def __or__(self, other):
        if isinstance(other, collections.abc.Mapping):
            c = self.copy()
            c.update(other)
            return c
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, collections.abc.Mapping):
            c = self.__class__()
            c.update(other)
            c.update(self)
            return c
        return NotImplemented

    # Default Mapping equality will tests keys for equality, but
    # we want to test ids for equality
    def __eq__(self, other):
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        return {id(k): v for k, v in self.items()} == {
            id(k): v for k, v in other.items()
        }


# CHANGED: the strong counterpart of WeakIdKeyDictionary. Keys are compared by
# identity and kept alive, so this accepts keys that cannot be weakly referenced
# and its entries never disappear on their own. Use it for a cache whose own
# lifetime already bounds the entries', such as one scoped to a block.
class IdKeyDictionary[K, V](WeakIdKeyDictionary[K, V]):
    ref_type: typing.ClassVar[collections.abc.Callable[..., typing.Any]] = IdRef


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

    def _memoized(arg: S) -> T:
        if arg in cache:
            return cache[arg]
        result = fn(arg)
        cache[arg] = result
        return result

    return _memoized
