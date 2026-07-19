"""The effect-row engine (``ε``): ``usesof`` / ``Uses`` / ``Computation`` / ``Requires``.

Branches off ``master`` — the engine lives in ``ops/`` and does not depend on #694.
"""
from typing import Annotated, Callable

import pytest

from effectful.ops.effects import (
    Computation,
    Requires,
    UndeclaredCallable,
    Uses,
    check_requires,
    usesof,
)
from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled


@defop
def read() -> int:
    raise NotHandled


@defop
def write(v: int) -> None:
    raise NotHandled


@defop
def pure_add(a: int, b: int) -> Annotated[int, Uses[()]]:  # declared pure
    raise NotHandled


def test_usesof_is_the_op_row():
    assert usesof(write(pure_add(read(), read()))) == frozenset({read, write})


def test_uses_empty_is_pure():
    # a Uses[()] op contributes nothing itself
    assert pure_add not in usesof(pure_add(read(), read()))


def test_computation_callback_is_entered():
    @defop
    def apply_cb(fn: Annotated[Callable[[int], int], Computation]) -> int:
        raise NotHandled

    assert usesof(apply_cb(lambda x: read())) == frozenset({apply_cb, read})


def test_undeclared_callable_fails_loudly():
    @defop
    def bad(fn: Callable[[int], int]) -> int:  # callable arg, not Computation/Uses[()]
        raise NotHandled

    with pytest.raises(UndeclaredCallable):
        usesof(bad(lambda x: x))


def test_requires_provenance():
    @defop
    def sink(x: Annotated[int, Requires(read)]) -> None:
        raise NotHandled

    assert check_requires(sink(read())) == {}  # x came from read
    assert check_requires(sink(pure_add(1, 1))) == {sink: {"x": frozenset({read})}}
