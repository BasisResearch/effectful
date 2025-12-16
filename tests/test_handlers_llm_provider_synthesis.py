"""Tests for LLM program synthesis functionality."""

import functools
import inspect
import os
from collections.abc import Callable

import pytest

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import LiteLLMProvider, completion
from effectful.handlers.llm.synthesis import ProgramSynthesis, SynthesisError
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# Check for API keys
HAS_OPENAI_KEY = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]

requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY environment variable not set"
)


def retry_on_error(error: type[Exception], n: int):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(n):
                try:
                    return func(*args, **kwargs)
                except error as e:
                    if i < n - 1:
                        continue
                    raise e

        return wrapper

    return decorator


class LimitLLMCallsHandler(ObjectInterpretation):
    max_calls: int
    no_calls: int = 0

    def __init__(self, max_calls: int):
        self.max_calls = max_calls

    @implements(completion)
    def _completion(self, *args, **kwargs):
        if self.no_calls >= self.max_calls:
            raise RuntimeError(
                f"Test used too many requests (max_calls = {self.max_calls})"
            )
        self.no_calls += 1
        return fwd()


@Template.define
def create_function(char: str) -> Callable[[str], int]:
    """Create a function that counts occurrences of the character '{char}' in a string.
    Do not use any tools.

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
