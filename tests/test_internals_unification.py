import typing

import pytest

from effectful.internals.unification import infer_return_type, unify, substitute, freetypevars


@pytest.mark.parametrize(
"typ,fvs", [
    (typing.List[typing.TypeVar("T")], {typing.TypeVar("T")}),
    (typing.Dict[str, typing.TypeVar("T")], {typing.TypeVar("T")}),
    (int, set()),
    (typing.List[int], set()),
    (typing.Dict[str, int], set()),
])
def test_freetypevars(typ: type, fvs: set[typing.TypeVar]):
    assert freetypevars(typ) == fvs


@pytest.mark.parametrize(
    "typ,subs,expected", [
        (typing.List[typing.TypeVar("T")], {typing.TypeVar("T"): int}, typing.List[int]),
        (typing.Dict[str, typing.TypeVar("T")], {typing.TypeVar("T"): int}, typing.Dict[str, int]),
        (int, {}, int),
        (typing.List[int], {}, typing.List[int]),
        (typing.Dict[str, int], {}, typing.Dict[str, int]),
    ]
)
def test_substitute(typ: type, subs: typing.Mapping[typing.TypeVar, type], expected: type):
    assert substitute(typ, subs) == expected


@pytest.mark.parametrize(
    "pattern,concrete,subs,expected", [
        (typing.TypeVar("T"), int, {}, {typing.TypeVar("T"): int}),
        (typing.List[typing.TypeVar("T")], typing.List[int], {typing.TypeVar("T"): int}),
    ]
)
def test_unify(pattern: type, concrete: type, subs: typing.Mapping[typing.TypeVar, type], expected: typing.Mapping[typing.TypeVar, type]):
    assert unify(pattern, concrete, subs) == expected


def test_infer_return_type():
    pass  # TODO fill this in