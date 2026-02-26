import doctest as _doctest
import os
from collections.abc import Callable

import pytest

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    ResultDecodingError,
    call_user,
)
from effectful.handlers.llm.doctest import DoctestHandler
from effectful.handlers.llm.encoding import Encodable, SynthesizedFunction
from effectful.handlers.llm.evaluation import UnsafeEvalProvider
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

        >>> count_char("banana")
        4
    """
    raise NotHandled


@Template.define
def synthesize_inner_with_doctest(char: str) -> Callable[[str], int]:
    """Generate a Python function named count_char that counts occurrences of the character '{char}'
    in a given input string.

    The function should be case-sensitive.

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


# ---------------------------------------------------------------------------
# Unit tests: extract_doctests
# ---------------------------------------------------------------------------


class TestExtractDoctests:
    """Tests for the DoctestHandler.extract_doctests classmethod."""

    def test_strips_examples(self):
        docstring = (
            "Compute something.\n\n    >>> foo(1)\n    2\n    >>> foo(3)\n    4\n"
        )
        stripped, examples = DoctestHandler.extract_doctests(docstring)
        assert ">>>" not in stripped
        assert len(examples) == 2
        assert examples[0].source.strip() == "foo(1)"
        assert examples[0].want == "2\n"
        assert examples[1].source.strip() == "foo(3)"
        assert examples[1].want == "4\n"

    def test_no_examples(self):
        docstring = "Just a description.\nNo examples here.\n"
        stripped, examples = DoctestHandler.extract_doctests(docstring)
        assert stripped == docstring
        assert examples == []

    def test_preserves_non_example_text(self):
        docstring = "Title.\n\nSome details.\n\n    >>> f(1)\n    42\n\nMore text.\n"
        stripped, examples = DoctestHandler.extract_doctests(docstring)
        assert "Title." in stripped
        assert "Some details." in stripped
        assert "More text." in stripped
        assert ">>>" not in stripped
        assert len(examples) == 1


# ---------------------------------------------------------------------------
# Unit tests: Case 2 – prompt stripping
# ---------------------------------------------------------------------------


class TestCase2PromptStripping:
    """Verify that call_user receives a stripped template (no >>> examples)."""

    def test_call_user_receives_stripped_template(self):
        """The DoctestHandler should strip >>> from the template before fwd."""
        captured_templates: list[str] = []

        def spy_call_user(template, env):
            captured_templates.append(template)
            # Return a dummy message
            return {
                "role": "user",
                "content": template,
                "id": "test-id",
            }

        doctest_handler = DoctestHandler()
        # DoctestHandler must be inner (most recent) so _strip_prompt runs
        # first, then fwd() reaches the spy.
        with handler({call_user: spy_call_user}), handler(doctest_handler):
            # Directly invoke call_user with a template containing >>>
            template_str = "Generate function.\n\n    >>> foo(1)\n    42\n"
            call_user(template_str, {})

        assert len(captured_templates) == 1
        assert ">>>" not in captured_templates[0]
        assert "Generate function." in captured_templates[0]


# ---------------------------------------------------------------------------
# Unit tests: Case 2 – doctest execution (existing tests, updated)
# ---------------------------------------------------------------------------


class TestDoctestExecution:
    """Tests for doctest execution during callable synthesis (Case 2)."""

    def test_decode_runs_doctest(self):
        encodable = Encodable.define(Callable[[str], int], {})
        func_source = SynthesizedFunction(
            module_code="def count_char(input_string: str) -> int:\n"
            "    return input_string.count('a')"
        )
        doctest_handler = DoctestHandler()
        # Push cached Example objects (matching the new _doctest_stack type).
        doctest_handler._doctest_stack.append(
            [_doctest.Example("count_char('banana')\n", "4\n")]
        )
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


# ---------------------------------------------------------------------------
# Unit tests: Case 1 – calibration
# ---------------------------------------------------------------------------


@Template.define
def summarize(text: str) -> str:
    """Summarize the following text into a single short sentence: '{text}'

    >>> summarize("The quick brown fox jumps over the lazy dog.")
    'A fox jumps over a dog.'
    """
    raise NotHandled


class TestCase1Calibration:
    """Tests for Case 1 (tool-calling) calibration and prefix caching."""

    def test_callable_detection(self):
        """Templates returning Callable should be Case 2, others Case 1."""
        from effectful.handlers.llm.encoding import CallableEncodable, Encodable

        def is_callable_return(t):
            return isinstance(
                Encodable.define(t.__signature__.return_annotation),
                CallableEncodable,
            )

        assert is_callable_return(synthesize_counter_with_doctest)
        assert not is_callable_return(summarize)

    def test_extraction_cache_populated(self):
        """_get_doctests should populate the extraction cache."""
        dh = DoctestHandler()
        stripped, examples = dh._get_doctests(summarize)
        assert ">>>" not in stripped
        assert len(examples) == 1
        # Second call should return cached result
        stripped2, examples2 = dh._get_doctests(summarize)
        assert stripped2 is stripped
        assert examples2 is examples

    def test_bind_history_restores_state(self):
        """_bind_history should restore template.__history__ after use."""
        import collections

        dh = DoctestHandler()

        # Template starts without __history__
        assert not hasattr(summarize, "__history__")

        history = collections.OrderedDict()
        with dh._bind_history(summarize, history):
            assert summarize.__history__ is history  # type: ignore[attr-defined]

        # Cleaned up after context exit
        assert not hasattr(summarize, "__history__")

    @requires_openai
    def test_case1_calibration_integration(self):
        """End-to-end: calibration should cache a prefix for tool-calling."""
        provider = LiteLLMProvider(model="gpt-4o-mini")
        dh = DoctestHandler()
        with handler(provider), handler(dh):
            # This should trigger calibration for the summarize template
            result = summarize("The quick brown fox jumps over the lazy dog.")

        # After the call, summarize should have a cached prefix
        assert summarize in dh._prefix_cache
        assert isinstance(result, str)
        assert len(result) > 0

        # Calibration should clean up: no lingering __history__ on template
        assert not hasattr(summarize, "__history__")
