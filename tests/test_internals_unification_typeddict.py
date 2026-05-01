"""TypedDict unification tests using Required/NotRequired/ReadOnly annotations.

Separated from test_internals_unification.py because mypy 1.19 cannot serialize
RequiredType instances to its cache, causing a crash.
"""

import collections.abc
import typing
from typing import ReadOnly

import pytest

from effectful.internals.unification import (
    canonicalize,
    unify,
)

if typing.TYPE_CHECKING:
    K = typing.Any
    V = typing.Any
else:
    K = typing.TypeVar("K")
    V = typing.TypeVar("V")


# --- NotRequired fields ---


def test_unify_typeddict_notrequired_present():
    """NotRequired field present in subtype unifies when subtype also has NotRequired."""

    class Pattern(typing.TypedDict):
        name: str
        nickname: typing.NotRequired[str]

    class Sub(typing.TypedDict):
        name: str
        nickname: typing.NotRequired[str]

    subs = unify(Pattern, Sub)
    assert subs == {}


def test_unify_typeddict_notrequired_absent():
    """NotRequired field absent from subtype is OK."""

    class Pattern(typing.TypedDict):
        name: str
        nickname: typing.NotRequired[str]

    class Sub(typing.TypedDict):
        name: str

    subs = unify(Pattern, Sub)
    assert subs == {}


def test_unify_typeddict_required_missing_raises():
    """Required field missing from subtype raises TypeError."""

    class Pattern(typing.TypedDict):
        name: str
        age: int

    class Sub(typing.TypedDict):
        name: str

    with pytest.raises(TypeError, match="required field 'age'"):
        unify(Pattern, Sub)


# --- total=False ---


def test_unify_typeddict_total_false_all_optional():
    """total=False makes all fields optional; absent fields OK."""

    class Pattern(typing.TypedDict, total=False):
        name: str
        age: int

    class Sub(typing.TypedDict, total=False):
        name: str

    subs = unify(Pattern, Sub)
    assert subs == {}


def test_unify_typeddict_total_false_with_required():
    """total=False with Required override: Required field must exist."""

    class Pattern(typing.TypedDict, total=False):
        name: typing.Required[str]
        age: int

    class Sub(typing.TypedDict):
        name: str

    # age is optional (total=False), name is Required -> OK
    subs = unify(Pattern, Sub)
    assert subs == {}


def test_unify_typeddict_total_false_required_missing_raises():
    """total=False with Required field missing in subtype raises TypeError."""

    class Pattern(typing.TypedDict, total=False):
        name: typing.Required[str]
        age: int

    class Sub(typing.TypedDict):
        age: int

    with pytest.raises(TypeError, match="required field 'name'"):
        unify(Pattern, Sub)


# --- TypedDict inheritance ---


def test_unify_typeddict_inheritance():
    """Derived TypedDict includes base fields."""

    class Base(typing.TypedDict):
        name: str

    class Derived(Base):
        age: int

    class Sub(typing.TypedDict):
        name: str
        age: int

    subs = unify(Derived, Sub)
    assert subs == {}


def test_unify_typeddict_inheritance_generic():
    """Generic inherited TypedDict with TypeVar extraction."""

    class Base[T](typing.TypedDict):
        value: T

    class Derived(Base[int]):
        label: str

    class Sub(typing.TypedDict):
        value: int
        label: str

    subs = unify(Derived, Sub)
    # T from Base is now resolved to int via _get_typeddict_hints
    assert subs == {}


# --- Edge cases ---


def test_unify_empty_typeddict():
    """Empty TypedDict unifies with any TypedDict (no fields to check)."""

    class Empty(typing.TypedDict):
        pass

    class NonEmpty(typing.TypedDict):
        name: str

    subs = unify(Empty, NonEmpty)
    assert subs == {}


def test_unify_mapping_empty_typeddict():
    """Mapping[K, V] with empty TypedDict binds K=str via Mapping-TypedDict handler."""

    class Empty(typing.TypedDict):
        pass

    subs = unify(collections.abc.Mapping[K, V], Empty)
    # New handler unifies K with str (TypedDict keys are always str)
    assert subs[K] == str


def test_canonicalize_typeddict_notrequired():
    """Canonicalize preserves NotRequired annotations."""

    class TD(typing.TypedDict):
        name: str
        items: typing.NotRequired[list[int]]

    result = canonicalize(TD)
    assert typing.is_typeddict(result)
    assert "items" in result.__optional_keys__


# --- Required/NotRequired soundness tests ---


def test_unify_typeddict_required_vs_notrequired_raises():
    """Required in typ + NotRequired in subtyp → TypeError."""

    class Pattern(typing.TypedDict):
        name: str  # Required

    class Sub(typing.TypedDict):
        name: typing.NotRequired[str]

    with pytest.raises(TypeError, match="Required in pattern but NotRequired"):
        unify(Pattern, Sub)


def test_unify_typeddict_notrequired_mutable_vs_required_raises():
    """NotRequired mutable field in typ + Required in subtyp → TypeError."""

    class Pattern(typing.TypedDict):
        name: typing.NotRequired[str]

    class Sub(typing.TypedDict):
        name: str  # Required

    with pytest.raises(TypeError, match="NotRequired in pattern but Required"):
        unify(Pattern, Sub)


# --- Invariance tests ---


def test_unify_typeddict_invariance_rejects_subtype():
    """Mutable fields are invariant: bool ≤ int covariantly but not invariantly."""

    class Pattern(typing.TypedDict):
        x: int

    class Sub(typing.TypedDict):
        x: bool

    with pytest.raises(TypeError):
        unify(Pattern, Sub)


def test_unify_typeddict_readonly_covariance():
    """ReadOnly field allows covariant subtyping."""
    Pattern = typing.TypedDict("Pattern", {"x": ReadOnly[int]})  # noqa: UP013
    Sub = typing.TypedDict("Sub", {"x": ReadOnly[bool]})  # noqa: UP013

    # bool is subtype of int, ReadOnly allows covariance
    subs = unify(Pattern, Sub)
    assert subs == {}


def test_unify_typeddict_readonly_notrequired_to_required():
    """ReadOnly NotRequired in typ, Required in subtyp → OK (promotion)."""
    Pattern = typing.TypedDict(  # noqa: UP013
        "Pattern",
        {"x": ReadOnly[typing.NotRequired[str]]},
    )
    Sub = typing.TypedDict("Sub", {"x": str})  # noqa: UP013

    subs = unify(Pattern, Sub)
    assert subs == {}


# --- Mapping[K, V] vs TypedDict tests ---
def test_unify_mapping_typeddict_homogeneous():
    """Mapping[str, V] with homogeneous TypedDict binds V."""

    class TD(typing.TypedDict):
        a: int
        b: int

    subs = unify(collections.abc.Mapping[str, V], TD)
    assert subs[V] == int


def test_unify_mapping_typeddict_heterogeneous_raises():
    """Mapping[str, V] with heterogeneous TypedDict raises TypeError."""

    class TD(typing.TypedDict):
        name: str
        age: int

    with pytest.raises(TypeError):
        unify(collections.abc.Mapping[str, V], TD)


def test_unify_mapping_typeddict_int_key_raises():
    """Mapping[int, V] vs TypedDict raises TypeError (keys must be str)."""

    class TD(typing.TypedDict):
        name: str

    with pytest.raises(TypeError):
        unify(collections.abc.Mapping[int, V], TD)


def test_unify_mutablemapping_typeddict_invariance():
    """MutableMapping[str, V] requires invariant field types."""

    class TD(typing.TypedDict):
        x: int
        y: int

    subs = unify(collections.abc.MutableMapping[str, V], TD)
    assert subs[V] == int
