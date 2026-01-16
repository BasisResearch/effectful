"""Tests for LLM program synthesis functionality."""

import inspect
from collections import ChainMap
from collections.abc import Callable

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import LiteLLMProvider
from effectful.handlers.llm.synthesis import (
    EncodableSynthesizedFunction,
    ProgramSynthesis,
    SynthesisError,
    SynthesizedFunction,
)
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

from .test_handlers_llm_provider import (
    LimitLLMCallsHandler,
    requires_openai,
    retry_on_error,
)


@Template.define
def create_function(char: str) -> Callable[[str], int]:
    """Create a function that counts occurrences of the character '{char}' in a string.
    Do not use any tools and implement this function directly.

    Return as a code block with the last definition being the function.
    """
    raise NotHandled


class TestProgramSynthesis:
    """Tests for ProgramSynthesis handler functionality."""

    @requires_openai
    @retry_on_error(error=SynthesisError, n=3)
    def test_generates_callable(self):
        """Test ProgramSynthesis handler generates executable code."""
        with (
            handler(LiteLLMProvider(model_name="gpt-4o-mini")),
            handler(ProgramSynthesis()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            count_func = create_function("a")

            assert callable(count_func)
            # Test the generated function
            assert count_func("banana") == 3
            assert count_func("cherry") == 0
            assert count_func("aardvark") == 3

    @requires_openai
    @retry_on_error(error=SynthesisError, n=3)
    def test_inspect_getsource_works(self):
        """Test that inspect.getsource() works on synthesized functions."""
        with (
            handler(LiteLLMProvider(model_name="gpt-4o-mini")),
            handler(ProgramSynthesis()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            count_func = create_function("x")

            # inspect.getsource() should work on the synthesized function
            source = inspect.getsource(count_func)
            assert isinstance(source, str)
            assert len(source) > 0
            assert "def" in source

            # __source__ attribute should also be available with full module code
            assert hasattr(count_func, "__source__")
            assert "def" in count_func.__source__

    @requires_openai
    @retry_on_error(error=SynthesisError, n=3)
    def test_synthesized_attribute(self):
        """Test that __synthesized__ attribute is attached to generated functions."""
        from effectful.handlers.llm.synthesis import SynthesizedFunction

        with (
            handler(LiteLLMProvider(model_name="gpt-4o-mini")),
            handler(ProgramSynthesis()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            count_func = create_function("z")

            assert hasattr(count_func, "__synthesized__")
            assert isinstance(count_func.__synthesized__, SynthesizedFunction)
            # The synthesized module code should be present
            assert "def" in count_func.__synthesized__.module_code
            # The function should work correctly
            assert count_func("pizza") == 2  # two 'z's in "pizza"

    def test_immutable_context(self):
        """Test that synthesis does not pollute the original lexical context."""
        from effectful.handlers.llm.synthesis import SynthesisContextHandler

        # Create a context with known contents
        original_context: ChainMap[str, object] = ChainMap({"helper": lambda x: x * 2})
        original_keys = set(original_context.keys())

        # Synthesize a function that defines new names
        synth = SynthesizedFunction(
            function_name="my_func",
            module_code="""
def internal_helper(x):
    return x + 1

def my_func(n):
    return internal_helper(n) * 2
""",
        )

        # Decode with the context using the handler
        with handler(SynthesisContextHandler(original_context)):
            func = EncodableSynthesizedFunction.decode(synth)

        # Verify the function works
        assert func(5) == 12  # (5 + 1) * 2

        # Verify original context was NOT polluted with new definitions
        assert set(original_context.keys()) == original_keys
        assert "my_func" not in original_context
        assert "internal_helper" not in original_context
