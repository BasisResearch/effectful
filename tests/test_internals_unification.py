import typing

import pytest

from effectful.internals.unification import (
    freetypevars,
    substitute,
    unify,
)


@pytest.mark.parametrize(
    "typ,fvs",
    [
        (list[typing.TypeVar("T")], {typing.TypeVar("T")}),
        (dict[str, typing.TypeVar("T")], {typing.TypeVar("T")}),
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
        (list[typing.TypeVar("T")], {typing.TypeVar("T"): int}, list[int]),
        (dict[str, typing.TypeVar("T")], {typing.TypeVar("T"): int}, dict[str, int]),
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
    "pattern,concrete,subs,expected",
    [
        (typing.TypeVar("T"), int, {}, {typing.TypeVar("T"): int}),
        (list[typing.TypeVar("T")], list[int], {typing.TypeVar("T"): int}),
    ],
)
def test_unify(
    pattern: type,
    concrete: type,
    subs: typing.Mapping[typing.TypeVar, type],
    expected: typing.Mapping[typing.TypeVar, type],
):
    assert unify(pattern, concrete, subs) == expected


def test_infer_return_type():
    pass  # TODO fill this in
