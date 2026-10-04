"""A type checker's error about calling an operation reads as one about calling a
function: it names the operation and carries its signature.

Both checkers report a wrong call to an operation against ``Operation.__call__``,
whose parameters are the generic ``*args, **kwargs``; the reader repairing the call
needs the operation's own signature.
"""

from collections.abc import Callable
from typing import Any

import pytest

from effectful.handlers.llm.harness.validation.hooks import type_check
from effectful.handlers.llm.harness.validation.mypy import MypyTypeChecker
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
SIGNATURE = "refine(predicate: int, keyword: str) -> str"


@pytest.fixture(params=[TyTypeChecker, MypyTypeChecker], ids=["ty", "mypy"])
def checker(request: pytest.FixtureRequest) -> Callable[[], Any]:
    return request.param


def failure(
    checker: Callable[[], Any], body: str, *, checked_from: int | None = None
) -> str:
    """The TypeError message for `body` after the operations, checking only the
    lines from `checked_from` on (the whole module when omitted)."""
    with handler(checker()), pytest.raises(TypeError) as raised:
        type_check(OPERATIONS + body, checked_from, None)
    return str(raised.value)


def names_operation(message: str, callee: str = "refine") -> bool:
    """Whether `message` names `callee` as a function and no longer mentions the
    generic ``Operation.__call__``."""
    return (
        (f"function `{callee}`" in message or f'\\"{callee}\\"' in message)
        and "__call__" not in message
        and "ops/types.py" not in message
    )


@pytest.mark.parametrize(
    "call",
    [
        "refine(1)",
        "refine('one', 'a')",
        "refine(1, keyword=2)",
        "refine(1, 'a', colour='b')",
        "refine(1, 'a', 'b')",
    ],
    ids=["missing", "wrong-type", "wrong-keyword-type", "unknown-keyword", "too-many"],
)
def test_operation_call_errors_name_the_operation_and_its_signature(
    checker: Callable[[], Any], call: str
):
    message = failure(checker, f"value = {call}\n")
    assert names_operation(message) and SIGNATURE in message


def test_ty_positional_counts_exclude_the_bound_self():
    message = failure(TyTypeChecker, "value = refine(1, 'a', 'b')\n")
    assert "expected 2, got 3" in message


def test_a_wrong_nested_result_blames_the_operation_receiving_it(
    checker: Callable[[], Any],
):
    message = failure(checker, "value = refine(other(1), 'a')\n")
    assert names_operation(message) and SIGNATURE in message
    assert "other(x: int)" not in message


@pytest.mark.parametrize(
    "body",
    [
        "values = [op('one', 'a') for op in (refine,)]\n",
        "def call(op: Operation[[int, str], str]) -> str:\n    return op('one', 'a')\n",
    ],
    ids=["comprehension", "parameter"],
)
def test_operations_reached_through_local_names_are_named(
    checker: Callable[[], Any], body: str
):
    message = failure(checker, body)
    assert names_operation(message, "op") and "op(" in message


def test_plain_function_errors_keep_the_checkers_wording(checker: Callable[[], Any]):
    message = failure(checker, "value = plain(1)\n")
    assert "plain" in message and "Function signature" not in message


@pytest.mark.parametrize(
    "body",
    [
        "reveal_type(other)\nvalue = refine(1)\n",
        "def reveal_type(value: object) -> object:\n    return value\n\n\n"
        "value = refine(1)\n",
        "def call(_effectful_type_probe: int) -> str:\n"
        "    return refine(_effectful_type_probe)\n",
    ],
    ids=["module-reveal", "defined-reveal", "alias-named-binding"],
)
def test_the_probe_is_not_disturbed_by_the_checked_code(
    checker: Callable[[], Any], body: str
):
    checked_from = len(OPERATIONS.splitlines()) + body.count("\n")
    assert SIGNATURE in failure(checker, body, checked_from=checked_from)


def test_a_syntax_error_outside_the_checked_region_keeps_the_wording():
    # mypy stops at a syntax error, so only ty reports past one.
    line = len(OPERATIONS.splitlines()) + 1
    source = OPERATIONS + "value = refine(1)\n" + "def broken(:\n"
    with handler(TyTypeChecker()), pytest.raises(TypeError) as raised:
        type_check(source, line, line)
    assert "missing-argument" in str(raised.value)


def test_a_module_opening_with_a_docstring_and_future_import_is_probed(
    checker: Callable[[], Any],
):
    source = '"""Generated."""\n\nfrom __future__ import annotations\n\n' + (
        OPERATIONS + "value = refine(1)\n"
    )
    with handler(checker()), pytest.raises(TypeError) as raised:
        type_check(source, None, None)
    assert SIGNATURE in str(raised.value)


def test_non_ascii_text_before_the_call_does_not_hide_the_operation_under_ty():
    message = failure(TyTypeChecker, "café = 1; value = refine('one', 'a')\n")
    assert names_operation(message) and SIGNATURE in message


def test_non_ascii_text_before_the_call_keeps_mypys_wording():
    # mypy's columns past non-ASCII text match neither characters nor bytes.
    message = failure(MypyTypeChecker, "café = 1; value = refine('one', 'a')\n")
    assert "__call__" in message and "Function signature" not in message


def test_a_clean_operation_call_passes(checker: Callable[[], Any]):
    with handler(checker()):
        type_check(OPERATIONS + "value = refine(1, 'a')\n", None, None)
