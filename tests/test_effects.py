"""The effect-row engine (``ε``): ``usesof`` / ``Uses`` / ``Computation`` / ``Requires``.

Branches off ``master`` — the engine lives in ``ops/`` and does not depend on #694.
"""
from typing import Annotated, Callable

import pytest

from effectful.ops.effects import (
    Computation,
    Requires,
    UndeclaredCallable,
    UnsoundCallbackFold,
    Uses,
    check_requires,
    check_uses,
    effect_type,
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


def test_computation_callback_ignoring_or_passing_arg_folds():
    @defop
    def cb_pass(fn: Annotated[Callable[[int], None], Computation]) -> int:
        raise NotHandled

    # passes the arg straight to an op (no inspection) — folds soundly
    assert usesof(cb_pass(lambda x: write(x))) == frozenset({cb_pass, write})


def test_computation_callback_inspecting_arg_is_refused_not_silent():
    @defop
    def cb(fn: Annotated[Callable[[int], int], Computation]) -> int:
        raise NotHandled

    # branches on the arg -> would drop a branch if folded on a fake value -> refuse loudly
    with pytest.raises(UnsoundCallbackFold):
        usesof(cb(lambda x: write(x) if x else read()))  # type: ignore[func-returns-value]
    # destructures the arg -> refuse loudly (not silently swallowed)
    with pytest.raises(UnsoundCallbackFold):
        usesof(cb(lambda x: x.field))


def test_undeclared_callable_fails_loudly():
    @defop
    def bad(fn: Callable[[int], int]) -> int:  # callable arg, not Computation/Uses[()]
        raise NotHandled

    with pytest.raises(UndeclaredCallable):
        usesof(bad(lambda x: x))


def test_effect_type_pairs_tau_and_epsilon():
    tau, eps = effect_type(pure_add(read(), read()))
    assert tau is int and eps == frozenset({read})


def test_check_uses_flags_undeclared_effects():
    @defop
    def declared() -> Annotated[int, Uses[read]]:  # declares read only
        raise NotHandled

    assert check_uses(declared, read()) == frozenset()          # body ⊆ declared
    assert check_uses(declared, write(read())) == frozenset({write})  # write undeclared


def test_requires_provenance():
    @defop
    def sink(x: Annotated[int, Requires(read)]) -> None:
        raise NotHandled

    assert check_requires(sink(read())) == {}  # x came from read
    assert check_requires(sink(pure_add(1, 1))) == {sink: {"x": frozenset({read})}}
