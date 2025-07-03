import typing

import pytest

from effectful.internals.unification import (
    freetypevars,
    substitute,
    unify,
)

T = typing.TypeVar("T")
K = typing.TypeVar("K")
V = typing.TypeVar("V")
U = typing.TypeVar("U")


@pytest.mark.parametrize(
    "typ,fvs",
    [
        # Basic cases
        (T, {T}),
        (int, set()),
        (str, set()),
        # Single TypeVar in generic
        (list[T], {T}),
        (set[T], {T}),
        (tuple[T], {T}),
        # Multiple TypeVars
        (dict[K, V], {K, V}),
        (tuple[K, V], {K, V}),
        (dict[T, T], {T}),  # Same TypeVar used twice
        # Nested generics with TypeVars
        (list[dict[K, V]], {K, V}),
        (dict[K, list[V]], {K, V}),
        (list[tuple[T, U]], {T, U}),
        (tuple[list[T], dict[K, V]], {T, K, V}),
        # Concrete types in generics
        (list[int], set()),
        (dict[str, int], set()),
        (tuple[int, str, float], set()),
        # Mixed concrete and TypeVars
        (dict[str, T], {T}),
        (dict[K, int], {K}),
        (tuple[T, int, V], {T, V}),
        (list[tuple[int, T]], {T}),
        # Deeply nested
        (list[dict[K, list[tuple[V, T]]]], {K, V, T}),
        (dict[tuple[K, V], list[dict[U, T]]], {K, V, U, T}),
        # Union types (if supported)
        (list[T] | dict[K, V], {T, K, V}),
        (T | int, {T}),
    ],
    ids=str
)
def test_freetypevars(typ: type, fvs: set[typing.TypeVar]):
    assert freetypevars(typ) == fvs


@pytest.mark.parametrize(
    "typ,subs,expected",
    [
        # Basic substitution
        (T, {T: int}, int),
        (T, {T: str}, str),
        (T, {T: list[int]}, list[int]),
        # TypeVar not in mapping
        (T, {K: int}, T),
        (T, {}, T),
        # Non-TypeVar types
        (int, {T: str}, int),
        (str, {}, str),
        (list[int], {T: str}, list[int]),
        # Single TypeVar in generic
        (list[T], {T: int}, list[int]),
        (set[T], {T: str}, set[str]),
        (tuple[T], {T: float}, tuple[float]),
        # Multiple TypeVars
        (dict[K, V], {K: str, V: int}, dict[str, int]),
        (tuple[K, V], {K: int, V: str}, tuple[int, str]),
        (dict[K, V], {K: str}, dict[str, V]),  # Partial substitution
        # Same TypeVar used multiple times
        (dict[T, T], {T: int}, dict[int, int]),
        (tuple[T, T, T], {T: str}, tuple[str, str, str]),
        # Nested generics
        (list[dict[K, V]], {K: str, V: int}, list[dict[str, int]]),
        (dict[K, list[V]], {K: int, V: str}, dict[int, list[str]]),
        (list[tuple[T, U]], {T: int, U: str}, list[tuple[int, str]]),
        # Mixed concrete and TypeVars
        (dict[str, T], {T: int}, dict[str, int]),
        (tuple[int, T, str], {T: float}, tuple[int, float, str]),
        (list[tuple[int, T]], {T: str}, list[tuple[int, str]]),
        # Deeply nested substitution
        (list[dict[K, list[V]]], {K: str, V: int}, list[dict[str, list[int]]]),
        (dict[tuple[K, V], list[T]], {K: int, V: str, T: float}, dict[tuple[int, str], list[float]]),
        # Substituting with generic types
        (T, {T: list[int]}, list[int]),
        (list[T], {T: dict[str, int]}, list[dict[str, int]]),
        (dict[K, V], {K: list[int], V: dict[str, float]}, dict[list[int], dict[str, float]]),
        # Empty substitution
        (list[T], {}, list[T]),
        (dict[K, V], {}, dict[K, V]),
        # Union types (if supported)
        (T | int, {T: str}, str | int),
        (list[T] | dict[K, V], {T: int, K: str, V: float}, list[int] | dict[str, float]),
        # Irrelevant substitutions (TypeVars not in type)
        (list[T], {K: int, V: str}, list[T]),
        (int, {T: str, K: int}, int),
    ],
    ids=str
)
def test_substitute(
    typ: type, subs: typing.Mapping[typing.TypeVar, type], expected: type
):
    assert substitute(typ, subs) == expected


@pytest.mark.parametrize(
    "pattern,concrete,expected_subs",
    [
        (T, int, {T: int}),
        (list[T], list[int], {T: int}),
    ],
    ids=str
)
def test_unify(
    pattern: type,
    concrete: type,
    expected_subs: typing.Mapping[typing.TypeVar, type],
):
    assert unify(pattern, concrete, {}) == expected_subs


def test_infer_return_type():
    pass  # TODO fill this in
