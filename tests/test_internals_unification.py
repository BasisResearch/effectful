import typing

import pytest

from effectful.internals.unification import (
    freetypevars,
    substitute,
    unify,
)

T = typing.TypeVar("T")


@pytest.mark.parametrize(
    "typ,fvs",
    [
        (list[T], {T}),
        (dict[str, T], {T}),
        (int, set()),
        (list[int], set()),
        (dict[str, int], set()),
    ],
)
def test_freetypevars(typ: type, fvs: set[typing.TypeVar]):
    assert freetypevars(typ) == fvs


@pytest.mark.parametrize(
    "typ,subs,expected",
    [
        (list[T], {T: int}, list[int]),
        (dict[str, T], {T: int}, dict[str, int]),
        (int, {}, int),
        (list[int], {}, list[int]),
        (dict[str, int], {}, dict[str, int]),
    ],
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
)
def test_unify(
    pattern: type,
    concrete: type,
    expected_subs: typing.Mapping[typing.TypeVar, type],
):
    assert unify(pattern, concrete, {}) == expected_subs


def test_infer_return_type():
    pass  # TODO fill this in
