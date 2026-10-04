"""A ty diagnostic about calling an operation names the operation and its signature.

ty reports a wrong call to an operation against ``Operation.__call__``, whose
parameters are the generic ``*args, **kwargs``; the model reading the diagnostic
needs the operation's own signature to repair the call.
"""

import pytest

from effectful.handlers.llm.harness.validation.hooks import type_check
from effectful.handlers.llm.harness.validation.ty import TyTypeChecker
from effectful.ops.semantics import handler

OPERATIONS = """from typing import reveal_type

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Operation


@defop
def refine(predicate: int, keyword: str) -> str:
    raise NotHandled


@defop
def other(x: int) -> str:
    raise NotHandled


def plain(predicate: int, keyword: str) -> str:
    return keyword


"""
SIGNATURE = "`refine` is called as refine(predicate: int, keyword: str) -> str"


def failure(body: str, *, checked_from: int | None = None) -> str:
    """The TypeError message for `body` after the operations, checking only the
    lines from `checked_from` on (the whole module when omitted)."""
    with handler(TyTypeChecker()), pytest.raises(TypeError) as raised:
        type_check(OPERATIONS + body, checked_from, None)
    return str(raised.value)


@pytest.mark.parametrize(
    "call",
    [
        "refine(1)",
        "refine('one', 'a')",
        "refine(1, 'a', colour='b')",
        "refine(1, 'a', 'b')",
    ],
    ids=["missing", "wrong-type", "unknown-keyword", "too-many"],
)
def test_operation_call_errors_name_the_operation_and_its_signature(call: str):
    message = failure(f"value = {call}\n")
    assert "operation `refine`" in message and SIGNATURE in message
    assert "Operation.__call__" not in message
    assert "ops/types.py" not in message


def test_positional_counts_exclude_the_bound_self():
    assert "expected 2, got 3" in failure("value = refine(1, 'a', 'b')\n")


def test_a_wrong_nested_result_blames_the_operation_receiving_it():
    message = failure("value = refine(other(1), 'a')\n")
    assert "operation `refine`" in message and SIGNATURE in message
    assert "operation `other`" not in message


@pytest.mark.parametrize(
    "body",
    [
        "values = [op('one', 'a') for op in (refine,)]\n",
        "def call(op: Operation[[int, str], str]) -> str:\n    return op('one', 'a')\n",
    ],
    ids=["comprehension", "parameter"],
)
def test_operations_reached_through_local_names_are_named(body: str):
    message = failure(body)
    assert "`op` is called as op(" in message
    assert "Operation.__call__" not in message


def test_plain_function_errors_keep_tys_wording():
    message = failure("value = plain(1)\n")
    assert "`plain`" in message and "is called as" not in message


def test_the_modules_own_reveal_type_calls_do_not_disturb_the_probe():
    body = "reveal_type(other)\nvalue = refine(1)\n"
    checked_from = len(OPERATIONS.splitlines()) + 2
    message = failure(body, checked_from=checked_from)
    assert SIGNATURE in message


def test_a_reveal_type_defined_by_the_checked_code_cannot_intercept_the_probe():
    body = (
        "def reveal_type(value: object) -> object:\n"
        "    return value\n"
        "\n"
        "\n"
        "value = refine(1)\n"
    )
    assert SIGNATURE in failure(body)


def test_a_binding_named_like_the_probe_cannot_intercept_it():
    body = (
        "def call(_effectful_ty_probe: int) -> str:\n"
        "    return refine(_effectful_ty_probe)\n"
    )
    assert SIGNATURE in failure(body)


def test_a_syntax_error_outside_the_checked_region_keeps_tys_wording():
    body = "value = refine(1)\n"
    checked_from = len(OPERATIONS.splitlines()) + 1
    source = OPERATIONS + body + "def broken(:\n"
    with handler(TyTypeChecker()), pytest.raises(TypeError) as raised:
        type_check(source, checked_from, checked_from)
    assert "missing-argument" in str(raised.value)


def test_a_module_opening_with_a_docstring_and_future_import_is_probed():
    source = '"""Generated."""\n\nfrom __future__ import annotations\n\n' + (
        OPERATIONS + "value = refine(1)\n"
    )
    with handler(TyTypeChecker()), pytest.raises(TypeError) as raised:
        type_check(source, None, None)
    assert SIGNATURE in str(raised.value)


def test_a_clean_operation_call_passes():
    with handler(TyTypeChecker()):
        type_check(OPERATIONS + "value = refine(1, 'a')\n")
