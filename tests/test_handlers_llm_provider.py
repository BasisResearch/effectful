"""Tests for LLM handlers and providers.

This module tests the functionality from build/main.py and build/llm.py,
breaking down individual components like LiteLLMProvider, LLMLoggingHandler,
ProgramSynthesis, and sampling strategies.
"""

import logging
import os
from collections.abc import Callable
from enum import Enum

import pytest
from pydantic import Field
from pydantic.dataclasses import dataclass

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import (
    LiteLLMProvider,
    LLMLoggingHandler,
    completion,
)
from effectful.handlers.llm.synthesis import ProgramSynthesis, SynthesisError
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# Check for API keys
HAS_OPENAI_KEY = "OPENAI_API_KEY" in os.environ
HAS_ANTHROPIC_KEY = "ANTHROPIC_API_KEY" in os.environ

# Pytest markers for skipping tests based on API key availability
requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY environment variable not set"
)
requires_anthropic = pytest.mark.skipif(
    not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY environment variable not set"
)


# ============================================================================
# Test Fixtures and Mock Data
# ============================================================================
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


class MovieGenre(str, Enum):
    """Movie genre classifications."""

    ACTION = "action"
    COMEDY = "comedy"
    DRAMA = "drama"
    HORROR = "horror"
    SCIFI = "sci-fi"
    ROMANCE = "romance"


@dataclass(frozen=True)
class MovieClassification:
    """Classification result for a movie."""

    genre: MovieGenre
    explanation: str = Field(
        ..., description="explanation for the given movie classification"
    )


@Template.define
def classify_genre(plot: str) -> MovieClassification:
    """Classify the movie genre based on this plot: {plot}"""
    raise NotImplementedError


@Template.define
def simple_prompt(topic: str) -> str:
    """Write a short sentence about {topic}."""
    raise NotImplementedError


@Template.define
def generate_number(max_value: int) -> int:
    """Generate a random number between 1 and {max_value}. Return only the number."""
    raise NotImplementedError


@Template.define
def create_function(char: str) -> Callable[[str], int]:
    """Create a function that counts occurrences of the character '{char}' in a string.

    Return as a code block with the last definition being the function.
    """
    raise NotHandled


class TestLiteLLMProvider:
    """Tests for LiteLLMProvider basic functionality."""

    @requires_openai
    @pytest.mark.parametrize("model_name", ["gpt-4o-mini", "gpt-3.5-turbo"])
    def test_simple_prompt_multiple_models(self, model_name):
        """Test that LiteLLMProvider works with different model configurations."""
        with (
            handler(LiteLLMProvider(model_name=model_name)),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("testing")
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param("gpt-4o", marks=requires_openai),
            pytest.param("claude-haiku-4-5", marks=requires_anthropic),
        ],
    )
    def test_simple_prompt_cross_endpoint(self, model_name):
        """Test that LiteLLMProvider works across different API endpoints."""
        with (
            handler(LiteLLMProvider(model_name=model_name)),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("testing")
            assert isinstance(result, str)
            assert len(result) > 0

    @requires_openai
    def test_structured_output(self):
        """Test LiteLLMProvider with structured Pydantic output."""
        plot = "A rogue cop must stop a evil group from taking over a skyscraper."

        with (
            handler(LiteLLMProvider(model_name="gpt-4o")),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            classification = classify_genre(plot)

            assert isinstance(classification, MovieClassification)
            assert isinstance(classification.genre, MovieGenre)
            assert classification.genre == MovieGenre.ACTION
            assert isinstance(classification.explanation, str)
            assert len(classification.explanation) > 0

    @requires_openai
    def test_integer_return_type(self):
        """Test LiteLLMProvider with integer return type."""
        with (
            handler(LiteLLMProvider(model_name="gpt-4o")),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = generate_number(100)

            assert isinstance(result, int)
            assert 1 <= result <= 100

    @requires_openai
    def test_with_config_params(self):
        """Test LiteLLMProvider accepts and uses additional configuration parameters."""
        # Test with temperature parameter
        with (
            handler(LiteLLMProvider(model_name="gpt-4o-mini", temperature=0.1)),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("deterministic test")
            assert isinstance(result, str)


class TestLLMLoggingHandler:
    """Tests for LLMLoggingHandler functionality."""

    @requires_openai
    def test_logs_requests(self, caplog):
        """Test that LLMLoggingHandler properly logs LLM requests."""
        with caplog.at_level(logging.INFO):
            with (
                handler(LLMLoggingHandler()),
                handler(LiteLLMProvider(model_name="gpt-4o-mini")),
                handler(LimitLLMCallsHandler(max_calls=1)),
            ):
                result = simple_prompt("testing")
                assert isinstance(result, str)

        # Check that logging occurred
        assert any("llm.request" in record.message for record in caplog.records)

    @requires_openai
    def test_custom_logger(self, caplog):
        """Test LLMLoggingHandler with a custom logger."""
        custom_logger = logging.getLogger("test_custom_logger")

        with caplog.at_level(logging.INFO, logger="test_custom_logger"):
            with (
                handler(LLMLoggingHandler(logger=custom_logger)),
                handler(LiteLLMProvider(model_name="gpt-4o-mini")),
                handler(LimitLLMCallsHandler(max_calls=1)),
            ):
                result = simple_prompt("testing")
                assert isinstance(result, str)

        # Verify custom logger was used
        assert any(
            record.name == "test_custom_logger" and "llm.request" in record.message
            for record in caplog.records
        )


class TestProgramSynthesis:
    """Tests for ProgramSynthesis handler functionality."""

    @requires_openai
    def test_generates_callable(self):
        """Test ProgramSynthesis handler generates executable code."""
        for i in range(3):
            try:
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
            except SynthesisError as e:
                if i < 2:
                    continue
                raise e


# Global state for tool calling tests
evaluation_count = 0
evaluation_results = []


def reset_evaluation_state():
    """Reset global evaluation state for testing."""
    global evaluation_count, evaluation_results
    evaluation_count = 0
    evaluation_results = []


@dataclass
class Poem:
    """A poem with content and form."""

    content: str = Field(..., description="content of the poem")
    form: str = Field(..., description="name of the type of the poem")


class PoemQuality(str, Enum):
    """Quality rating for a poem."""

    GOOD = "GOOD"
    OKAY = "OKAY"
    BAD = "BAD"


def evaluate_poem_tool(poem: Poem, explanation: str) -> PoemQuality:
    """Evaluate the quality of a poem.

    Parameters:
    - poem: Poem object representing the poem
    - explanation: natural language explanation of the thought process
    """
    global evaluation_count, evaluation_results
    evaluation_count += 1

    # Simple heuristic: require at least 2 evaluations, then approve
    quality = PoemQuality.BAD if evaluation_count < 2 else PoemQuality.GOOD

    evaluation_results.append(
        {"poem": poem, "explanation": explanation, "quality": quality}
    )

    return quality


@Template.define(tools=[evaluate_poem_tool])
def generate_good_poem(topic: str) -> Poem:
    """Generate a good poem about {topic} returning your result following
    the provided json schema. Use the provided tools to evaluate the quality
    and you MUST make sure it is a good poem.
    """
    raise NotHandled


class TestToolCalling:
    """Tests for templates with tool calling functionality."""

    @requires_openai
    def test_tool_calling_openai(self):
        """Test that templates with tools work with openai."""
        reset_evaluation_state()
        with (
            handler(LiteLLMProvider(model_name="gpt-4o")),
            handler(LimitLLMCallsHandler(max_calls=3)),
        ):
            poem = generate_good_poem("Python")
            assert isinstance(poem, Poem)
            assert isinstance(poem.content, str)
            assert isinstance(poem.form, str)
            assert len(poem.content) > 0

            # Verify the tool was called at least once
            assert evaluation_count >= 1
            assert len(evaluation_results) >= 1

    @requires_anthropic
    def test_tool_calling_anthropic(self):
        """Test that templates with tools work across different providers."""
        reset_evaluation_state()
        with (
            handler(LiteLLMProvider(model_name="claude-sonnet-4-5-20250929")),
            handler(LimitLLMCallsHandler(max_calls=4)),
        ):
            poem = generate_good_poem("Python")
            assert isinstance(poem, Poem)
            assert isinstance(poem.content, str)
            assert isinstance(poem.form, str)
            assert len(poem.content) > 0

            # Verify the tool was called at least once
            assert evaluation_count >= 1
            assert len(evaluation_results) >= 1


class TestHandlerComposition:
    """Tests for composing multiple handlers together."""

    @requires_openai
    def test_logging_and_synthesis(self, caplog):
        """Test composing LLMLoggingHandler, LiteLLMProvider, and ProgramSynthesis."""
        with caplog.at_level(logging.INFO):
            with (
                handler(LLMLoggingHandler()),
                handler(LiteLLMProvider(model_name="gpt-4o")),
                handler(ProgramSynthesis()),
            ):
                count_func = create_function("x")

                assert callable(count_func)

        # Verify logging occurred
        assert any("llm.request" in record.message for record in caplog.records)
