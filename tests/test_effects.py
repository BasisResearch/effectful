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
    check_requires,
    check_uses,
    effect_type,
    usesof,
)
from effectful.ops.syntax import Uses, defop
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

    # Every way of *inspecting* the arg must refuse loudly — never fold on a fake value
    # (which would silently drop a branch) and never leak a raw TypeError. The refusal is
    # default-deny, so these cover the operator families, not a hand-picked few.
    inspecting = [
        lambda x: write(x) if x else read(),        # truthiness branch
        lambda x: write(x) if x == 0 else read(),   # __eq__ branch (identity would say False)
        lambda x: write(x) if x < 1 else read(),    # ordering branch
        lambda x: write(x + 1),                     # arithmetic
        lambda x: write(len(x)),                    # __len__
        lambda x: write(str(x)),                    # formatting/conversion
        lambda x: x(),                              # calls the arg
        lambda x: x.field,                          # attribute access
        lambda x: x[0],                             # indexing
    ]
    for cb_fn in inspecting:
        with pytest.raises(UnsoundCallbackFold):
            usesof(cb(cb_fn))  # type: ignore[func-returns-value]


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


def test_requires_provenance_holds_transitively():
    # provenance is the arg's whole effect row, so a required op reached *through* other
    # ops still satisfies Requires.
    @defop
    def sink(x: Annotated[int, Requires(read)]) -> None:
        raise NotHandled

    # read is under a pure combinator but still in x's row -> satisfied
    assert check_requires(sink(pure_add(read(), 1))) == {}


def test_requires_reports_only_the_missing_ops():
    # Requires(read, write) on an arg that provides only read -> report just write.
    @defop
    def sink(x: Annotated[int, Requires(read, write)]) -> None:
        raise NotHandled

    assert check_requires(sink(read())) == {sink: {"x": frozenset({write})}}
    assert check_requires(sink(write(read()))) == {}  # both present -> satisfied


def test_requires_is_per_argument_and_by_keyword():
    # multiple Requires on different params, passed by keyword; each checked independently.
    @defop
    def move(
        src: Annotated[int, Requires(read)],
        dst: Annotated[int, Requires(write)],
    ) -> None:
        raise NotHandled

    assert check_requires(move(src=read(), dst=write(1))) == {}
    # dst lacks write in its provenance -> only dst flagged
    assert check_requires(move(src=read(), dst=read())) == {move: {"dst": frozenset({write})}}


def test_requires_absent_annotation_is_unconstrained():
    # an argument with no Requires imposes no provenance obligation.
    @defop
    def sink(x: int) -> None:
        raise NotHandled

    assert check_requires(sink(pure_add(1, 1))) == {}
