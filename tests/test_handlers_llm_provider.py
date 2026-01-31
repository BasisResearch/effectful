"""Tests for LLM handlers and providers.
This module tests the functionality from build/main.py and build/llm.py,
breaking down individual components like LiteLLMProvider,
ProgramSynthesis, and sampling strategies.
"""

import functools
import json
import os
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import litellm
import pytest
from litellm.caching.caching import Cache
from litellm.files.main import ModelResponse
from PIL import Image
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    RetryHandler,
    Tool,
    call_assistant,
    completion,
)
from effectful.handlers.llm.encoding import Encodable
from effectful.handlers.llm.synthesis import ProgramSynthesis, SynthesisError
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# Check for API keys
HAS_OPENAI_KEY = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]
HAS_ANTHROPIC_KEY = (
    "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]
)

# Pytest markers for skipping tests based on API key availability
requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY environment variable not set"
)
requires_anthropic = pytest.mark.skipif(
    not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY environment variable not set"
)

REBUILD_FIXTURES = os.getenv("REBUILD_FIXTURES") == "true"

# ============================================================================


# Test Fixtures and Mock Data
# ============================================================================
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


class ReplayLiteLLMProvider(LiteLLMProvider):
    test_id: str
    call_count = 0

    def __init__(self, request: pytest.FixtureRequest, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_id = request.node.nodeid
        self.test_id = self.test_id.replace("/", "_").replace(":", "_")

    def call_id(self):
        call_id = f"_{self.call_count}" if self.call_count > 0 else ""
        self.call_count += 1
        return call_id

    @implements(completion)
    def _completion(self, *args, **kwargs):
        path = FIXTURE_DIR / f"{self.test_id}{self.call_id()}.json"
        if not REBUILD_FIXTURES:
            if not path.exists():
                raise RuntimeError(f"Missing replay fixture: {path}")
            with path.open() as f:
                result = ModelResponse.model_validate(json.load(f))
                return result
        result = fwd(*args, **kwargs)
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            f.write(result.model_dump_json(indent=2))
        return result


class LimitLLMCallsHandler(ObjectInterpretation):
    max_calls: int
    no_calls: int = 0

    def __init__(self, max_calls: int):
        self.max_calls = max_calls

    @implements(call_assistant)
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
    """Classify the movie genre based on this plot: {plot}."""
    raise NotImplementedError


@Template.define
def simple_prompt(topic: str) -> str:
    """Write a short sentence about {topic}."""
    raise NotImplementedError


@Template.define
def generate_number(max_value: int) -> int:
    """Generate a random number between 1 and {max_value}."""
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
    @pytest.mark.parametrize("model_name", ["gpt-4o-mini", "gpt-5-nano"])
    def test_simple_prompt_multiple_models(self, request, model_name):
        """Test that LiteLLMProvider works with different model configurations."""
        with (
            handler(ReplayLiteLLMProvider(request, model=model_name)),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("testing")
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param("gpt-4o-mini", marks=requires_openai),
            pytest.param("claude-haiku-4-5", marks=requires_anthropic),
        ],
    )
    def test_simple_prompt_cross_endpoint(self, request, model_name):
        """Test that ReplayLiteLLMProvider works across different API endpoints."""
        with (
            handler(ReplayLiteLLMProvider(request, model=model_name)),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("testing")
            assert isinstance(result, str)
            assert len(result) > 0

    @requires_openai
    def test_structured_output(self, request):
        """Test LiteLLMProvider with structured Pydantic output."""
        plot = "A rogue cop must stop a evil group from taking over a skyscraper."

        with (
            handler(ReplayLiteLLMProvider(request, model="gpt-5-nano")),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            classification = classify_genre(plot)

            assert isinstance(classification, MovieClassification)
            assert isinstance(classification.genre, MovieGenre)
            assert classification.genre == MovieGenre.ACTION
            assert isinstance(classification.explanation, str)
            assert len(classification.explanation) > 0

    @requires_openai
    def test_integer_return_type(self, request):
        """Test LiteLLMProvider with integer return type."""
        with (
            handler(ReplayLiteLLMProvider(request, model="gpt-5-nano")),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = generate_number(100)

            assert isinstance(result, int)
            assert 1 <= result <= 100

    @requires_openai
    def test_with_config_params(self, request):
        """Test LiteLLMProvider accepts and uses additional configuration parameters."""
        # Test with temperature parameter
        with (
            handler(
                ReplayLiteLLMProvider(request, model="gpt-4o-mini", temperature=0.1)
            ),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("deterministic test")
            assert isinstance(result, str)


@pytest.mark.xfail(reason="Program synthesis not implemented")
class TestProgramSynthesis:
    """Tests for ProgramSynthesis handler functionality."""

    @pytest.mark.xfail
    @requires_openai
    @retry_on_error(error=SynthesisError, n=3)
    def test_generates_callable(self, request):
        """Test ProgramSynthesis handler generates executable code."""
        with (
            handler(ReplayLiteLLMProvider(request, model="gpt-4o-mini")),
            handler(ProgramSynthesis()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            count_func = create_function("a")

            assert callable(count_func)
            # Test the generated function
            assert count_func("banana") == 3
            assert count_func("cherry") == 0
            assert count_func("aardvark") == 3


def smiley_face() -> Image.Image:
    bmp = [
        "00000000",
        "00100100",
        "00100100",
        "00000000",
        "01000010",
        "00111100",
        "00000000",
        "00000000",
    ]

    img = Image.new("1", (8, 8))
    for y, row in enumerate(bmp):
        for x, c in enumerate(row):
            img.putpixel((x, y), 1 if c == "1" else 0)
    return img


@Template.define
def categorise_image(image: Image.Image) -> str:
    """Return a description of the following image.
    {image}"""
    raise NotHandled


@requires_openai
def test_image_input(request):
    with (
        handler(ReplayLiteLLMProvider(request, model="gpt-4o")),
        handler(LimitLLMCallsHandler(max_calls=3)),
    ):
        assert any("smile" in categorise_image(smiley_face()) for _ in range(3))


class BookReview(BaseModel):
    """A book review with rating and summary."""

    title: str = Field(..., description="title of the book")
    rating: int = Field(..., description="rating from 1 to 5", ge=1, le=5)
    summary: str = Field(..., description="brief summary of the review")


@Template.define
def review_book(plot: str) -> BookReview:
    """Review a book based on this plot: {plot}."""
    raise NotImplementedError


class TestPydanticBaseModelReturn:
    @requires_openai
    def test_pydantic_basemodel_return(self, request):
        plot = "A young wizard discovers he has magical powers and goes to a school for wizards."

        with (
            handler(ReplayLiteLLMProvider(request, model="gpt-5-nano")),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            review = review_book(plot)

            assert isinstance(review, BookReview)
            assert isinstance(review.title, str)
            assert len(review.title) > 0
            assert isinstance(review.rating, int)
            assert 1 <= review.rating <= 5
            assert isinstance(review.summary, str)
            assert len(review.summary) > 0


def test_litellm_caching_integration(request):
    litellm.cache = Cache()
    with handler(ReplayLiteLLMProvider(request, model="gpt-4o")):
        p1 = simple_prompt("apples")
        p2 = simple_prompt("apples")
        p3 = simple_prompt("oranges")
        assert p1 == p2, (
            "when caching is enabled, LLM requests with the same parameters will produce the same outputs"
        )
        assert p3 != p2, "different inputs should still produce different outputs"


def test_litellm_caching_integration_disabled(request):
    litellm.cache = Cache()
    with handler(ReplayLiteLLMProvider(request, model="gpt-4o", caching=False)):
        p1 = simple_prompt("apples")
        p2 = simple_prompt("apples")
        assert p1 != p2, "if caching is not enabled, inputs produce different outputs"


def test_litellm_caching_selective(request):
    with handler(ReplayLiteLLMProvider(request, model="gpt-4o")):
        p1 = simple_prompt("apples")
        p2 = simple_prompt("apples")
        assert p1 != p2, "when caching is not enabled, llm outputs should be different"
        litellm.enable_cache()
        p1 = simple_prompt("apples")
        p2 = simple_prompt("apples")
        assert p1 == p2, (
            "when caching is enabled, LLM requests with the same parameters will produce the same outputs"
        )
        litellm.disable_cache()
        p1 = simple_prompt("apples")
        p2 = simple_prompt("apples")
        assert p1 != p2, "when caching is not enabled, llm outputs should be different"


# ============================================================================
# RetryHandler Tests
# ============================================================================


class MockCompletionHandler(ObjectInterpretation):
    """Mock handler that returns pre-configured completion responses."""

    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.call_count = 0
        self.received_messages: list = []

    @implements(completion)
    def _completion(self, model, messages=None, **kwargs):
        self.received_messages.append(list(messages) if messages else [])
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


def make_tool_call_response(
    tool_name: str, tool_args: str, tool_call_id: str = "call_1"
) -> ModelResponse:
    """Create a ModelResponse with a tool call."""
    return ModelResponse(
        id="test",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        model="test-model",
    )


def make_text_response(content: str) -> ModelResponse:
    """Create a ModelResponse with text content."""
    return ModelResponse(
        id="test",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        model="test-model",
    )


@Tool.define
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


class TestRetryHandler:
    """Tests for RetryHandler functionality."""

    def test_retry_handler_succeeds_on_first_attempt(self):
        """Test that RetryHandler passes through when no error occurs."""
        # Response with valid tool call
        responses = [make_text_response('{"value": "hello"}')]

        mock_handler = MockCompletionHandler(responses)

        with handler(RetryHandler(num_retries=3)), handler(mock_handler):
            message, tool_calls, result = call_assistant(
                messages=[{"role": "user", "content": "test"}],
                tools={},
                response_format=Encodable.define(str),
                model="test-model",
            )

        assert mock_handler.call_count == 1
        assert result == "hello"

    def test_retry_handler_retries_on_invalid_tool_call(self):
        """Test that RetryHandler retries when tool call decoding fails."""
        # First response has invalid tool args, second has valid response
        responses = [
            make_tool_call_response(
                "add_numbers", '{"a": "not_an_int", "b": 2}'
            ),  # Invalid
            make_text_response('{"value": "success"}'),  # Valid
        ]

        mock_handler = MockCompletionHandler(responses)

        with handler(RetryHandler(num_retries=3)), handler(mock_handler):
            message, tool_calls, result = call_assistant(
                messages=[{"role": "user", "content": "test"}],
                tools={"add_numbers": add_numbers},
                response_format=Encodable.define(str),
                model="test-model",
            )

        assert mock_handler.call_count == 2
        assert result == "success"
        # Check that the second call included error feedback
        assert len(mock_handler.received_messages[1]) > len(
            mock_handler.received_messages[0]
        )

    def test_retry_handler_retries_on_unknown_tool(self):
        """Test that RetryHandler retries when tool is not found."""
        # First response has unknown tool, second has valid response
        responses = [
            make_tool_call_response("unknown_tool", '{"x": 1}'),  # Unknown tool
            make_text_response('{"value": "success"}'),  # Valid
        ]

        mock_handler = MockCompletionHandler(responses)

        with handler(RetryHandler(num_retries=3)), handler(mock_handler):
            message, tool_calls, result = call_assistant(
                messages=[{"role": "user", "content": "test"}],
                tools={"add_numbers": add_numbers},
                response_format=Encodable.define(str),
                model="test-model",
            )

        assert mock_handler.call_count == 2
        assert result == "success"

    def test_retry_handler_exhausts_retries(self):
        """Test that RetryHandler raises after exhausting all retries."""
        # All responses have invalid tool calls
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": "bad"}'),
        ]

        mock_handler = MockCompletionHandler(responses)

        with pytest.raises(Exception):  # Will raise the underlying decoding error
            with handler(RetryHandler(num_retries=2)), handler(mock_handler):
                call_assistant(
                    messages=[{"role": "user", "content": "test"}],
                    tools={"add_numbers": add_numbers},
                    response_format=Encodable.define(str),
                    model="test-model",
                )

        # Should have attempted 3 times (1 initial + 2 retries)
        assert mock_handler.call_count == 3

    def test_retry_handler_with_zero_retries(self):
        """Test RetryHandler with num_retries=0 fails immediately on error."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": "bad"}'),
        ]

        mock_handler = MockCompletionHandler(responses)

        with pytest.raises(Exception):
            with handler(RetryHandler(num_retries=0)), handler(mock_handler):
                call_assistant(
                    messages=[{"role": "user", "content": "test"}],
                    tools={"add_numbers": add_numbers},
                    response_format=Encodable.define(str),
                    model="test-model",
                )

        assert mock_handler.call_count == 1

    def test_retry_handler_valid_tool_call_passes_through(self):
        """Test that valid tool calls are decoded and returned."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
        ]

        mock_handler = MockCompletionHandler(responses)

        with handler(RetryHandler(num_retries=3)), handler(mock_handler):
            message, tool_calls, result = call_assistant(
                messages=[{"role": "user", "content": "test"}],
                tools={"add_numbers": add_numbers},
                response_format=Encodable.define(str),
                model="test-model",
            )

        assert mock_handler.call_count == 1
        assert len(tool_calls) == 1
        assert tool_calls[0].tool == add_numbers
        assert result is None  # No result when there are tool calls

    def test_retry_handler_retries_on_invalid_result(self):
        """Test that RetryHandler retries when result decoding fails."""
        # First response has invalid JSON, second has valid response
        responses = [
            make_text_response('{"value": "not valid for int"}'),  # Invalid for int
            make_text_response('{"value": 42}'),  # Valid
        ]

        mock_handler = MockCompletionHandler(responses)

        with handler(RetryHandler(num_retries=3)), handler(mock_handler):
            message, tool_calls, result = call_assistant(
                messages=[{"role": "user", "content": "test"}],
                tools={},
                response_format=Encodable.define(int),
                model="test-model",
            )

        assert mock_handler.call_count == 2
        assert result == 42
        # Check that the second call included error feedback
        assert len(mock_handler.received_messages[1]) > len(
            mock_handler.received_messages[0]
        )

    def test_retry_handler_exhausts_retries_on_result_decoding(self):
        """Test that RetryHandler raises after exhausting retries on result decoding."""
        # All responses have invalid results for int type
        responses = [
            make_text_response('{"value": "not an int"}'),
        ]

        mock_handler = MockCompletionHandler(responses)

        with pytest.raises(Exception):  # Will raise the underlying decoding error
            with handler(RetryHandler(num_retries=2)), handler(mock_handler):
                call_assistant(
                    messages=[{"role": "user", "content": "test"}],
                    tools={},
                    response_format=Encodable.define(int),
                    model="test-model",
                )

        # Should have attempted 3 times (1 initial + 2 retries)
        assert mock_handler.call_count == 3
