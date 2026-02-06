import os
from collections.abc import Callable

import pytest

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider, ResultDecodingError
from effectful.handlers.llm.encoding import Encodable, SynthesizedFunction
from effectful.handlers.llm.evaluation import DoctestHandler, UnsafeEvalProvider
from effectful.ops.semantics import NotHandled, handler

HAS_OPENAI_KEY = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]
requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY environment variable not set"
)


@Template.define
def synthesize_counter_with_doctest(char: str) -> Callable[[str], int]:
    """Generate a Python function named count_char that counts occurrences of the character '{char}'
    in a given input string.

    The function should be case-sensitive.

    Examples:
        >>> count_char("banana")
        4
    """
    raise NotHandled


@Template.define
def synthesize_inner_with_doctest(char: str) -> Callable[[str], int]:
    """Generate a Python function named count_char that counts occurrences of the character '{char}'
    in a given input string.

    The function should be case-sensitive.

    Examples:
        >>> count_char("orange")
        3
    """
    raise NotHandled


@Template.define
def synthesize_outer(char: str) -> Callable[[str], int]:
    """Use the synthesize_inner_with_doctest tool to produce the function and return it.
    Do not implement the function yourself.
    """
    raise NotHandled


class TestDoctestExecution:
    """Tests for doctest execution during callable synthesis."""

    def test_decode_runs_doctest(self):
        encodable = Encodable.define(Callable[[str], int], {})
        func_source = SynthesizedFunction(
            module_code="def count_char(input_string: str) -> int:\n"
            "    return input_string.count('a')"
        )
        doctest_handler = DoctestHandler()
        doctest_handler._doctest_stack.append(">>> count_char('banana')\n4\n")
        with (
            handler(UnsafeEvalProvider()),
            handler(doctest_handler),
        ):
            with pytest.raises(TypeError, match="doctest failed"):
                encodable.decode(func_source)

    @requires_openai
    def test_template_doctest_runs(self):
        provider = LiteLLMProvider(model="gpt-4o-mini")
        with (
            handler(provider),
            handler(UnsafeEvalProvider()),
            handler(DoctestHandler()),
        ):
            with pytest.raises(ResultDecodingError, match="doctest failed"):
                synthesize_counter_with_doctest("a")

    @requires_openai
    def test_nested_synthesis_doctest_runs(self):
        provider = LiteLLMProvider(model="gpt-4o-mini")
        with (
            handler(provider),
            handler(UnsafeEvalProvider()),
            handler(DoctestHandler()),
        ):
            with pytest.raises(ResultDecodingError, match="doctest failed"):
                synthesize_outer("o")
