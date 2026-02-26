"""Tests for LLM handlers and providers.
This module tests the functionality from build/main.py and build/llm.py,
<<<<<<< HEAD
breaking down individual components like LiteLLMProvider, LLMLoggingHandler,
ProgramSynthesis, and sampling strategies.
"""

import functools
import json
import logging
import os
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import pytest
=======
breaking down individual components like LiteLLMProvider,
ProgramSynthesis, and sampling strategies.
"""

import collections
import functools
import inspect
import json
import os
import re
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import litellm
import pytest
import tenacity
from litellm import ChatCompletionMessageToolCall
from litellm.caching.caching import Cache
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
from litellm.files.main import ModelResponse
from PIL import Image
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

<<<<<<< HEAD
from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import (
    LiteLLMProvider,
    LLMLoggingHandler,
    completion,
)
from effectful.handlers.llm.synthesis import ProgramSynthesis, SynthesisError
=======
from effectful.handlers.llm import Agent, Template
from effectful.handlers.llm.completions import (
    DecodedToolCall,
    LiteLLMProvider,
    ResultDecodingError,
    RetryLLMHandler,
    Tool,
    ToolCallDecodingError,
    ToolCallExecutionError,
    _get_history,
    call_assistant,
    call_tool,
    completion,
)
from effectful.handlers.llm.encoding import Encodable, SynthesizedFunction
from effectful.handlers.llm.evaluation import UnsafeEvalProvider
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
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

<<<<<<< HEAD
# ============================================================================


=======

# ============================================================================
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
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
<<<<<<< HEAD
=======
    call_count = 0
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8

    def __init__(self, request: pytest.FixtureRequest, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_id = request.node.nodeid
        self.test_id = self.test_id.replace("/", "_").replace(":", "_")

<<<<<<< HEAD
    @implements(completion)
    def _completion(self, *args, **kwargs):
        path = FIXTURE_DIR / f"{self.test_id}.json"
=======
    def call_id(self):
        call_id = f"_{self.call_count}" if self.call_count > 0 else ""
        self.call_count += 1
        return call_id

    @implements(completion)
    def _completion(self, *args, **kwargs):
        path = FIXTURE_DIR / f"{self.test_id}{self.call_id()}.json"
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
        if not REBUILD_FIXTURES:
            if not path.exists():
                raise RuntimeError(f"Missing replay fixture: {path}")
            with path.open() as f:
                result = ModelResponse.model_validate(json.load(f))
                return result
<<<<<<< HEAD
        result = fwd(self.model_name, *args, **(self.config | kwargs))
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(result.model_dump(), f, indent=2, sort_keys=True)
=======
        result = fwd(*args, **kwargs)
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            f.write(result.model_dump_json(indent=2))
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
        return result


class LimitLLMCallsHandler(ObjectInterpretation):
    max_calls: int
    no_calls: int = 0

    def __init__(self, max_calls: int):
        self.max_calls = max_calls

<<<<<<< HEAD
    @implements(completion)
=======
    @implements(call_assistant)
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
    def _completion(self, *args, **kwargs):
        if self.no_calls >= self.max_calls:
            raise RuntimeError(
                f"Test used too many requests (max_calls = {self.max_calls})"
            )
        self.no_calls += 1
        return fwd()


<<<<<<< HEAD
class MovieGenre(str, Enum):
=======
class MovieGenre(StrEnum):
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
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


<<<<<<< HEAD
=======
class _ToolNameAgent(Agent):
    @Template.define
    def helper(self) -> str:
        """Return the literal string 'ok'."""
        raise NotHandled

    @Template.define
    def ask(self, prompt: str) -> str:
        """Answer briefly: {prompt}"""
        raise NotHandled


>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
class TestLiteLLMProvider:
    """Tests for LiteLLMProvider basic functionality."""

    @requires_openai
    @pytest.mark.parametrize("model_name", ["gpt-4o-mini", "gpt-5-nano"])
    def test_simple_prompt_multiple_models(self, request, model_name):
        """Test that LiteLLMProvider works with different model configurations."""
        with (
<<<<<<< HEAD
            handler(ReplayLiteLLMProvider(request, model_name=model_name)),
=======
            handler(ReplayLiteLLMProvider(request, model=model_name)),
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
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
<<<<<<< HEAD
            handler(ReplayLiteLLMProvider(request, model_name=model_name)),
=======
            handler(ReplayLiteLLMProvider(request, model=model_name)),
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
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
<<<<<<< HEAD
            handler(ReplayLiteLLMProvider(request, model_name="gpt-5-nano")),
=======
            handler(ReplayLiteLLMProvider(request, model="gpt-5-nano")),
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
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
<<<<<<< HEAD
            handler(ReplayLiteLLMProvider(request, model_name="gpt-5-nano")),
=======
            handler(ReplayLiteLLMProvider(request, model="gpt-5-nano")),
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
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
<<<<<<< HEAD
                ReplayLiteLLMProvider(
                    request, model_name="gpt-4o-mini", temperature=0.1
                )
=======
                ReplayLiteLLMProvider(request, model="gpt-4o-mini", temperature=0.1)
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
            ),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("deterministic test")
            assert isinstance(result, str)


<<<<<<< HEAD
class TestLLMLoggingHandler:
    """Tests for LLMLoggingHandler functionality."""

    @requires_openai
    def test_logs_requests(self, request, caplog):
        """Test that LLMLoggingHandler properly logs LLM requests."""
        with caplog.at_level(logging.INFO):
            with (
                handler(ReplayLiteLLMProvider(request, model_name="gpt-4o-mini")),
                handler(LLMLoggingHandler()),
                handler(LimitLLMCallsHandler(max_calls=1)),
            ):
                result = simple_prompt("testing")
                assert isinstance(result, str)

        # Check that logging occurred
        assert any("llm.request" in record.message for record in caplog.records)

    @requires_openai
    def test_custom_logger(self, request, caplog):
        """Test LLMLoggingHandler with a custom logger."""
        custom_logger = logging.getLogger("test_custom_logger")

        with caplog.at_level(logging.INFO, logger="test_custom_logger"):
            with (
                handler(ReplayLiteLLMProvider(request, model_name="gpt-4o-mini")),
                handler(LLMLoggingHandler(logger=custom_logger)),
                handler(LimitLLMCallsHandler(max_calls=1)),
            ):
                result = simple_prompt("testing")
                assert isinstance(result, str)

        # Verify custom logger was used
        assert any(
            record.name == "test_custom_logger" and "llm.request" in record.message
            for record in caplog.records
        )


@pytest.mark.xfail(reason="Program synthesis not implemented")
class TestProgramSynthesis:
    """Tests for ProgramSynthesis handler functionality."""

    @pytest.mark.xfail
    @requires_openai
    @retry_on_error(error=SynthesisError, n=3)
    def test_generates_callable(self, request):
        """Test ProgramSynthesis handler generates executable code."""
        with (
            handler(ReplayLiteLLMProvider(request, model_name="gpt-4o-mini")),
            handler(ProgramSynthesis()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            count_func = create_function("a")

            assert callable(count_func)
            # Test the generated function
            assert count_func("banana") == 3
            assert count_func("cherry") == 0
            assert count_func("aardvark") == 3
=======
@requires_openai
def test_agent_tool_names_are_openai_compatible_integration():
    agent = _ToolNameAgent()
    template = agent.ask
    tools = template.tools
    expected_helper_tool_name = f"self__{agent.helper.__name__}"
    assert tools
    assert expected_helper_tool_name in tools
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in tools)

    # End-to-end provider call. If tool names violate OpenAI schema, this raises BadRequest.
    with (
        handler(
            LiteLLMProvider(model="gpt-4o-mini", tool_choice="none", max_tokens=16)
        ),
        handler(LimitLLMCallsHandler(max_calls=1)),
    ):
        result = agent.ask("Reply with exactly 'ok'. Do not call tools.")

    assert isinstance(result, str)
    assert result
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8


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
<<<<<<< HEAD
        handler(ReplayLiteLLMProvider(request, model_name="gpt-4o")),
=======
        handler(ReplayLiteLLMProvider(request, model="gpt-4o")),
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
        handler(LimitLLMCallsHandler(max_calls=3)),
    ):
        assert any("smile" in categorise_image(smiley_face()) for _ in range(3))


<<<<<<< HEAD
=======
class ImageDescription(BaseModel):
    """Description of a set of images."""

    description: str = Field(description="What you see in the images")
    count: int = Field(description="Number of images provided")


@Template.define
def describe_images(context: str, views: list[Image.Image]) -> ImageDescription:
    """You are a vision assistant. Describe what you see.

    <context>
    {context}
    </context>

    <views>
    {views}
    </views>

    Return JSON with a description of the images and the count of images provided.
    """
    raise NotHandled


@requires_openai
def test_list_image_input(request):
    """Regression test for GitHub issue #552: list[Image.Image] in templates."""
    img_red = Image.new("RGB", (64, 64), (255, 0, 0))
    img_blue = Image.new("RGB", (64, 64), (0, 0, 255))

    with (
        handler(ReplayLiteLLMProvider(request, model="gpt-4o")),
        handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(3))),
        handler(LimitLLMCallsHandler(max_calls=3)),
    ):
        result = describe_images(
            context="Two colored squares",
            views=[img_red, img_blue],
        )

    assert isinstance(result, ImageDescription)
    assert result.count == 2


>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
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
<<<<<<< HEAD
            handler(ReplayLiteLLMProvider(request, model_name="gpt-5-nano")),
=======
            handler(ReplayLiteLLMProvider(request, model="gpt-5-nano")),
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
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
<<<<<<< HEAD
=======


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
# RetryLLMHandler Tests
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


@pytest.fixture
def message_sequence_provider():
    message_sequence = collections.OrderedDict(
        id1={"id": "id1", "role": "user", "content": "test"},
    )
    return message_sequence, {_get_history: lambda: message_sequence}


@pytest.fixture
def mock_completion_handler_factory():
    def _factory(responses: list[ModelResponse]) -> MockCompletionHandler:
        return MockCompletionHandler(responses)

    return _factory


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


class TestRetryLLMHandler:
    """Tests for RetryLLMHandler functionality."""

    def test_retry_handler_succeeds_on_first_attempt(self):
        """Test that RetryLLMHandler passes through when no error occurs."""
        # Response with valid tool call
        responses = [make_text_response("hello")]

        mock_handler = MockCompletionHandler(responses)

        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with (
            handler(RetryLLMHandler()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                tools={},
                response_format=Encodable.define(str),
                model="test-model",
            )

        assert mock_handler.call_count == 1
        assert result == "hello"

    def test_retry_handler_retries_on_invalid_tool_call(self):
        """Test that RetryLLMHandler retries when tool call decoding fails."""
        # First response has invalid tool args, second has valid response
        responses = [
            make_tool_call_response(
                "add_numbers", '{"a": "not_an_int", "b": 2}'
            ),  # Invalid
            make_text_response("success"),  # Valid
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with (
            handler(RetryLLMHandler()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
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
        """Test that RetryLLMHandler retries when tool is not found."""
        # First response has unknown tool, second has valid response
        responses = [
            make_tool_call_response("unknown_tool", '{"x": 1}'),  # Unknown tool
            make_text_response("success"),  # Valid
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with (
            handler(RetryLLMHandler()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                tools={"add_numbers": add_numbers},
                response_format=Encodable.define(str),
                model="test-model",
            )

        assert mock_handler.call_count == 2
        assert result == "success"

    def test_retry_handler_exhausts_retries(self):
        """Test that RetryLLMHandler raises after exhausting all retries."""
        # All responses have invalid tool calls
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": "bad"}'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}
        with pytest.raises(ToolCallDecodingError):
            with (
                handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(3))),
                handler(mock_handler),
                handler(message_sequence_provider),
            ):
                call_assistant(
                    tools={"add_numbers": add_numbers},
                    response_format=Encodable.define(str),
                    model="test-model",
                )

        # Should have attempted 3 times (1 initial + 2 retries)
        assert mock_handler.call_count == 3

    def test_retry_handler_with_zero_retries(self):
        """Test RetryLLMHandler with stop_after_attempt(1) fails immediately on error."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": "bad"}'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with pytest.raises(ToolCallDecodingError):
            with (
                handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(1))),
                handler(mock_handler),
                handler(message_sequence_provider),
            ):
                call_assistant(
                    tools={"add_numbers": add_numbers},
                    response_format=Encodable.define(str),
                    model="test-model",
                )

    def test_retry_handler_valid_tool_call_passes_through(self):
        """Test that valid tool calls are decoded and returned."""
        responses = [
            make_tool_call_response(
                "add_numbers", '{"a": {"value": 1}, "b": {"value": 2}}'
            ),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with (
            handler(RetryLLMHandler()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                tools={"add_numbers": add_numbers},
                response_format=Encodable.define(str),
                model="test-model",
            )

        assert mock_handler.call_count == 1
        assert len(tool_calls) == 1
        assert tool_calls[0].tool == add_numbers
        assert result is None  # No result when there are tool calls

    @requires_openai
    def test_codeadapt_notebook_replay_fixture(self, request):
        """Replay fixture for codeadapt higher-order tool flow."""

        @Template.define
        def generate_paragraph() -> str:
            """Please generate a paragraph: with exactly 4 sentences ending with 'walk', 'tumbling', 'another', and 'lunatic'."""
            raise NotHandled

        @Template.define
        def codeact(
            template_name: str,
            args_json: str = "[]",
            kwargs_json: str = "{}",
        ) -> Callable[[], str]:
            """Generate a code that solve the following problem:
            {template_name}
            Args/kwargs are provided as JSON strings (args_json, kwargs_json).
            DO NOT USE codeadapt tool.
            """
            raise NotHandled

        @Template.define
        def codeadapt(
            template_name: str,
            args_json: str = "[]",
            kwargs_json: str = "{}",
        ) -> str:
            """Reason about the template, uses the codeact tool to generate a code that solve the problem.
            The template:
            {template_name}
            Args/kwargs are provided as JSON strings (args_json, kwargs_json).
            Generated program MUST use the name `solution` not `generate_paragraph`.
            """
            raise NotHandled

        with (
            handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(3))),
            handler(ReplayLiteLLMProvider(request, model="gpt-4o")),
            handler(UnsafeEvalProvider()),
        ):
            result = codeadapt("generate_paragraph")

        assert isinstance(result, str)

    def test_retry_handler_retries_on_invalid_result(self):
        """Test that RetryLLMHandler retries when result decoding fails."""
        # First response has invalid JSON, second has valid response
        responses = [
            make_text_response('"not valid for int"'),  # Invalid for int
            make_text_response('{"value": 42}'),  # Valid
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with (
            handler(RetryLLMHandler()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
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
        """Test that RetryLLMHandler raises after exhausting retries on result decoding."""
        # All responses have invalid results for int type
        responses = [
            make_text_response('"not an int"'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with pytest.raises(ResultDecodingError):
            with (
                handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(3))),
                handler(mock_handler),
                handler(message_sequence_provider),
            ):
                call_assistant(
                    tools={},
                    response_format=Encodable.define(int),
                    model="test-model",
                )

        # Should have attempted 3 times (1 initial + 2 retries)
        assert mock_handler.call_count == 3

    def test_retry_handler_raises_tool_call_decoding_error(self):
        """Test that RetryLLMHandler raises ToolCallDecodingError with correct attributes."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": "bad"}'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with pytest.raises(ToolCallDecodingError) as exc_info:
            with (
                handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(1))),
                handler(mock_handler),
                handler(message_sequence_provider),
            ):
                call_assistant(
                    tools={"add_numbers": add_numbers},
                    response_format=Encodable.define(str),
                    model="test-model",
                )

        error = exc_info.value
        assert error.raw_tool_call.function.name == "add_numbers"
        assert error.raw_tool_call.id == "call_1"
        assert error.raw_message is not None
        assert "add_numbers" in str(error)

    def test_retry_handler_raises_result_decoding_error(self):
        """Test that RetryLLMHandler raises ResultDecodingError with correct attributes."""
        responses = [
            make_text_response('"not an int"'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with pytest.raises(ResultDecodingError) as exc_info:
            with (
                handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(1))),
                handler(mock_handler),
                handler(message_sequence_provider),
            ):
                call_assistant(
                    tools={},
                    response_format=Encodable.define(int),
                    model="test-model",
                )

        error = exc_info.value
        assert error.raw_message is not None
        assert error.original_error is not None

    def test_retry_handler_error_feedback_contains_tool_name(self):
        """Test that error feedback messages contain the tool name."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": 2}'),
            make_text_response("success"),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with (
            handler(RetryLLMHandler()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            call_assistant(
                tools={"add_numbers": add_numbers},
                response_format=Encodable.define(str),
                model="test-model",
            )

        # Check that the error feedback in the second call mentions the tool name
        second_call_messages = mock_handler.received_messages[1]
        tool_feedback = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_feedback) == 1
        assert "add_numbers" in tool_feedback[0]["content"]

    def test_retry_handler_unknown_tool_error_contains_tool_name(self):
        """Test that unknown tool errors contain the tool name in the feedback."""
        responses = [
            make_tool_call_response("nonexistent_tool", '{"x": 1}'),
            make_text_response("success"),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with (
            handler(RetryLLMHandler()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            call_assistant(
                tools={"add_numbers": add_numbers},
                response_format=Encodable.define(str),
                model="test-model",
            )

        # Check that the error feedback mentions the unknown tool
        second_call_messages = mock_handler.received_messages[1]
        tool_feedback = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_feedback) == 1
        assert "nonexistent_tool" in tool_feedback[0]["content"]

    def test_retry_handler_include_traceback_in_error_feedback(self):
        """Test that include_traceback=True adds traceback to error messages."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": 2}'),
            make_text_response("success"),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with (
            handler(RetryLLMHandler(include_traceback=True)),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            call_assistant(
                tools={"add_numbers": add_numbers},
                response_format=Encodable.define(str),
                model="test-model",
            )

        # Check that the error feedback includes traceback
        second_call_messages = mock_handler.received_messages[1]
        tool_feedback = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_feedback) == 1
        assert "Traceback:" in tool_feedback[0]["content"]
        assert "```" in tool_feedback[0]["content"]

    def test_retry_handler_no_traceback_when_disabled(self):
        """Test that include_traceback=False doesn't add traceback."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": 2}'),
            make_text_response("success"),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        with (
            handler(RetryLLMHandler(include_traceback=False)),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            call_assistant(
                tools={"add_numbers": add_numbers},
                response_format=Encodable.define(str),
                model="test-model",
            )

        # Check that the error feedback does not include traceback
        second_call_messages = mock_handler.received_messages[1]
        tool_feedback = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_feedback) == 1
        assert "Traceback:" not in tool_feedback[0]["content"]


# ============================================================================
# Tool Execution Error Tests
# ============================================================================


@Tool.define
def failing_tool(x: int) -> int:
    """A tool that always raises an exception."""
    raise ValueError(f"Tool failed with input {x}")


@Tool.define
def divide_tool(a: int, b: int) -> int:
    """Divide a by b."""
    return a // b


class TestToolExecutionErrorHandling:
    """Tests for runtime tool execution error handling."""

    def test_retry_handler_catches_tool_runtime_error(self):
        """Test that RetryLLMHandler catches tool runtime errors and returns error message."""

        # Create a decoded tool call for failing_tool
        sig = inspect.signature(failing_tool)
        bound_args = sig.bind(x=42)
        tool_call = DecodedToolCall(failing_tool, bound_args, "call_1", "failing_tool")

        with handler(RetryLLMHandler()):
            result = call_tool(tool_call)

        # The result should be an error message, not an exception
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert "Tool execution failed" in result["content"]
        assert "failing_tool" in result["content"]
        assert "42" in result["content"]

    def test_retry_handler_catches_division_by_zero(self):
        """Test that RetryLLMHandler catches division by zero errors."""

        sig = inspect.signature(divide_tool)
        bound_args = sig.bind(a=10, b=0)
        tool_call = DecodedToolCall(divide_tool, bound_args, "call_div", "divide_tool")

        with handler(RetryLLMHandler()):
            result = call_tool(tool_call)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_div"
        assert "Tool execution failed" in result["content"]
        assert "divide_tool" in result["content"]

    def test_successful_tool_execution_returns_result(self):
        """Test that successful tool executions return normal results."""

        sig = inspect.signature(add_numbers)
        bound_args = sig.bind(a=3, b=4)
        tool_call = DecodedToolCall(add_numbers, bound_args, "call_add", "add_numbers")

        with handler(RetryLLMHandler()):
            result = call_tool(tool_call)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_add"
        # The result should be the serialized return value, not an error
        assert "Tool execution failed" not in result["content"]

    def test_tool_execution_error_not_pruned_from_messages(self):
        """Test that tool execution errors are NOT pruned (they're legitimate failures)."""
        # This test verifies the docstring claim that tool execution errors
        # should be kept in the message history, unlike decoding errors

        # First call: valid tool call that will fail at runtime
        # Second call: successful text response
        responses = [
            make_tool_call_response("failing_tool", '{"x": {"value": 42}}'),
            make_text_response("handled the error"),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = collections.OrderedDict(
            id1={"id": "id1", "role": "user", "content": "test"},
        )
        message_sequence_provider = {_get_history: lambda: message_sequence}

        # We need a custom provider that actually calls call_tool
        class TestProvider(ObjectInterpretation):
            @implements(call_assistant)
            def _call_assistant(self, tools, response_format, model, **kwargs):
                return fwd(tools, response_format, model, **kwargs)

        with (
            handler(RetryLLMHandler()),
            handler(TestProvider()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                tools={"failing_tool": failing_tool},
                response_format=Encodable.define(str),
                model="test-model",
            )

        # First call should succeed (tool call is valid)
        assert mock_handler.call_count == 1
        assert len(tool_calls) == 1


# ============================================================================
# Error Class Tests
# ============================================================================


class TestErrorClasses:
    """Tests for the error class definitions."""

    def test_tool_call_decoding_error_string_representation(self):
        """Test ToolCallDecodingError string includes relevant info."""
        original = ValueError("invalid value")
        raw_tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_abc",
                "function": {"name": "my_function", "arguments": "{}"},
            }
        )
        error = ToolCallDecodingError(
            original_error=original,
            raw_message={"role": "assistant"},
            raw_tool_call=raw_tool_call,
        )

        error_str = str(error)
        assert "my_function" in error_str
        assert "invalid value" in error_str

    def test_result_decoding_error_string_representation(self):
        """Test ResultDecodingError string includes relevant info."""
        original = ValueError("parse error")
        error = ResultDecodingError(original, raw_message={"role": "assistant"})

        error_str = str(error)
        assert "parse error" in error_str
        assert "decoding response" in error_str.lower()

    def test_error_classes_preserve_original_error(self):
        """Test that all error classes preserve the original exception."""
        original = TypeError("type mismatch")
        mock_message = {"role": "assistant", "content": "test"}
        raw_tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "id",
                "function": {"name": "fn", "arguments": "{}"},
            }
        )

        tool_decode_err = ToolCallDecodingError(
            original_error=original,
            raw_message=mock_message,
            raw_tool_call=raw_tool_call,
        )
        assert tool_decode_err.original_error is original

        result_decode_err = ResultDecodingError(original, mock_message)
        assert result_decode_err.original_error is original

    def test_tool_call_decoding_error_includes_raw_message(self):
        """Test that ToolCallDecodingError includes the raw message."""
        mock_message = {"role": "assistant", "content": "test"}
        raw_tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "id",
                "function": {"name": "fn", "arguments": "{}"},
            }
        )
        error = ToolCallDecodingError(
            original_error=ValueError("test"),
            raw_message=mock_message,
            raw_tool_call=raw_tool_call,
        )
        assert error.raw_message == mock_message


# ============================================================================
# Callable Synthesis Tests
# ============================================================================


@Template.define
def synthesize_adder() -> Callable[[int, int], int]:
    """Generate a Python function that adds two integers together.

    The function should take two integer parameters and return their sum.
    """
    raise NotHandled


@Template.define
def synthesize_string_processor() -> Callable[[str], str]:
    """Generate a Python function that converts a string to uppercase
    and adds exclamation marks at the end.
    """
    raise NotHandled


@Template.define
def synthesize_counter(char: str) -> Callable[[str], int]:
    """Generate a Python function that counts occurrences of the character '{char}'
    in a given input string.

    The function should be case-sensitive.
    """
    raise NotHandled


@Template.define
def synthesize_is_even() -> Callable[[int], bool]:
    """Generate a Python function that checks if a number is even.

    Return True if the number is divisible by 2, False otherwise.
    """
    raise NotHandled


@Template.define
def synthesize_three_param_func() -> Callable[[int, int, int], int]:
    """Generate a Python function that takes exactly three integer parameters
    and returns their product (multiplication).
    """
    raise NotHandled


class TestCallableSynthesis:
    """Tests for synthesizing callable functions via LLM."""

    @requires_openai
    def test_synthesize_adder_function(self, request):
        """Test that LLM can synthesize a simple addition function with correct signature."""
        with (
            handler(ReplayLiteLLMProvider(request, model="gpt-4o-mini")),
            handler(UnsafeEvalProvider()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            add_func = synthesize_adder()

            assert callable(add_func)
            assert add_func(2, 3) == 5
            assert add_func(0, 0) == 0
            assert add_func(-1, 1) == 0
            assert add_func(100, 200) == 300

    @requires_openai
    def test_synthesize_string_processor(self, request):
        """Test that LLM can synthesize a string processing function."""
        with (
            handler(ReplayLiteLLMProvider(request, model="gpt-4o-mini")),
            handler(UnsafeEvalProvider()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            process_func = synthesize_string_processor()

            assert callable(process_func)
            result = process_func("hello")
            assert isinstance(result, str)
            assert "HELLO" in result
            assert "!" in result

    @requires_openai
    def test_synthesize_counter_with_parameter(self, request):
        """Test that LLM can synthesize a parameterized counting function."""
        with (
            handler(ReplayLiteLLMProvider(request, model="gpt-4o-mini")),
            handler(UnsafeEvalProvider()),
            handler(LimitLLMCallsHandler(max_calls=3)),
        ):
            count_a = synthesize_counter("a")

            assert callable(count_a)
            assert count_a("banana") == 3
            assert count_a("cherry") == 0
            assert count_a("aardvark") == 3
            assert count_a("AAA") == 0  # case-sensitive

    @requires_openai
    def test_synthesized_function_roundtrip(self, request):
        """Test that a synthesized function can be encoded and decoded."""

        with (
            handler(ReplayLiteLLMProvider(request, model="gpt-4o-mini")),
            handler(UnsafeEvalProvider()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            # Synthesize a function
            add_func = synthesize_adder()
            assert callable(add_func)

            # Encode it back to SynthesizedFunction
            encodable = Encodable.define(Callable[[int, int], int], {})
            encoded = encodable.encode(add_func)
            assert isinstance(encoded, SynthesizedFunction)
            assert "def " in encoded.module_code

            # Decode it again and verify it still works
            decoded = encodable.decode(encoded)
            assert callable(decoded)
            assert decoded(5, 7) == 12

    @requires_openai
    def test_synthesize_bool_return_type(self, request):
        """Test that LLM respects bool return type in signature."""

        with (
            handler(ReplayLiteLLMProvider(request, model="gpt-4o-mini")),
            handler(UnsafeEvalProvider()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            is_even = synthesize_is_even()

            assert callable(is_even)
            # Verify return type annotation
            sig = inspect.signature(is_even)
            assert sig.return_annotation == bool

            # Verify behavior
            assert is_even(2) is True
            assert is_even(3) is False
            assert is_even(0) is True
            assert is_even(-4) is True

    @requires_openai
    def test_synthesize_three_params(self, request):
        """Test that LLM respects the exact number of parameters in signature."""

        with (
            handler(ReplayLiteLLMProvider(request, model="gpt-4o-mini")),
            handler(UnsafeEvalProvider()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            multiply_three = synthesize_three_param_func()

            assert callable(multiply_three)
            # Verify parameter count
            sig = inspect.signature(multiply_three)
            assert len(sig.parameters) == 3

            # Verify behavior
            assert multiply_three(2, 3, 4) == 24
            assert multiply_three(1, 1, 1) == 1
            assert multiply_three(5, 0, 10) == 0


class TestMessageSequence:
    """Tests for MessageSequence message sequence tracking."""

    def test_call_tool_sees_outer_message_sequence(self):
        """call_tool should not isolate; the tool sees the outer message sequence."""
        message_sequence = collections.OrderedDict()

        # Pre-populate the current frame with existing messages
        message_sequence["msg_1"] = {
            "id": "msg_1",
            "role": "user",
            "content": "hello",
        }
        message_sequence["msg_2"] = {
            "id": "msg_2",
            "role": "assistant",
            "content": "hi",
        }

        captured_frame = None

        class InnerToolHandler(ObjectInterpretation):
            @implements(add_numbers)
            def _add_numbers(self, *args, **kwargs):
                # Capture the state of the message sequence during execution
                nonlocal captured_frame
                captured_frame = dict(_get_history())
                return 42

        mock_tool_call = DecodedToolCall(
            tool=add_numbers,
            bound_args=inspect.signature(add_numbers).bind(1, 2),
            id="tc_1",
            name="add_numbers",
        )

        with (
            handler(InnerToolHandler()),
            handler({_get_history: lambda: message_sequence}),
        ):
            call_tool(mock_tool_call)

        # Tool sees the outer message sequence (2 pre-populated messages)
        assert len(captured_frame) == 2
        # Tool response is appended to the same outer sequence
        assert len(message_sequence) == 3

    def test_call_assistant_no_duplicate_messages(self):
        """call_assistant should prepend unseen frame messages without duplicating those already in input."""
        message_sequence = collections.OrderedDict()

        # Pre-populate frame with two messages
        msg_a = {"id": "msg_a", "role": "user", "content": "hello"}
        msg_b = {"id": "msg_b", "role": "assistant", "content": "hi"}
        message_sequence["msg_a"] = msg_a
        message_sequence["msg_b"] = msg_b

        captured_messages = []

        class InnerAssistantHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self_, model, messages, *args, **kwargs):
                captured_messages.extend(list(messages))
                response = {
                    "id": "response_1",
                    "role": "assistant",
                    "content": json.dumps({"value": "result"}),
                }
                return ModelResponse(
                    choices=[{"role": "assistant", "message": response}]
                )

        # Call with msg_b already in the input — it should not appear twice
        with (
            handler(InnerAssistantHandler()),
            handler({_get_history: lambda: message_sequence}),
        ):
            call_assistant(
                tools={},
                response_format=Encodable.define(str),
                model="test-model",
            )

        # Forwarded messages should be [msg_a (prefix), msg_b (input)] — no duplicates
        ids = [m["id"] for m in captured_messages]
        assert ids == ["msg_a", "msg_b"]
        assert len(ids) == len(set(ids))

    def test_call_assistant_no_duplicates_across_multiple_calls(self):
        """Calling call_assistant multiple times should never produce duplicate messages."""

        msg_user = {"id": "msg_user", "role": "user", "content": "hello"}
        message_sequence = collections.OrderedDict(msg_user=msg_user)

        call_log = []

        class InnerAssistantHandler(ObjectInterpretation):
            call_count = 0

            @implements(completion)
            def _completion(self_, model, messages, *args, **kwargs):
                call_log.append([m["id"] for m in messages])
                self_.call_count += 1
                response = {
                    "id": "response_1",
                    "role": "assistant",
                    "content": json.dumps({"value": "result"}),
                }
                return ModelResponse(
                    choices=[{"role": "assistant", "message": response}]
                )

        inner = InnerAssistantHandler()

        with (
            handler(inner),
            handler({_get_history: lambda: message_sequence}),
        ):
            # First call: input is the latest message (msg_user)
            resp1, _, _ = call_assistant(
                tools={},
                response_format=Encodable.define(str),
                model="test-model",
            )
            # Second call: input is the first response
            resp2, _, _ = call_assistant(
                tools={},
                response_format=Encodable.define(str),
                model="test-model",
            )

        # First call: prefix=[] + input=[msg_user]
        assert call_log[0] == ["msg_user"]

        # Second call: prefix=[msg_user] + input=[resp_1]
        # msg_user is in the frame but not in the input, so it appears as prefix
        # resp_1 is in both the frame and the input, so it's NOT duplicated
        assert call_log[1] == ["msg_user", "response_1"]
        assert len(call_log[1]) == len(set(call_log[1]))

    def test_call_assistant_saves_only_on_successful_fwd(self):
        """call_assistant should only save the response message to the frame when fwd() succeeds."""
        message_sequence = collections.OrderedDict()

        class FailingAssistantHandler(ObjectInterpretation):
            @implements(call_assistant)
            def _call_assistant(self_, messages, *args, **kwargs):
                raise RuntimeError("LLM call failed")

        msg = {"id": "input_msg", "role": "user", "content": "hello"}
        frame_snapshot = dict(message_sequence)

        with pytest.raises(RuntimeError, match="LLM call failed"):
            with (
                handler(FailingAssistantHandler()),
                handler({_get_history: lambda: message_sequence}),
            ):
                call_assistant(
                    messages=[msg],
                    tools={},
                    response_format=Encodable.define(str),
                    model="test-model",
                )

        # Frame should be unchanged — no response message was saved
        assert dict(message_sequence) == frame_snapshot


@Template.define
def compute_sum(a: int, b: int) -> int:
    """Compute the sum of {a} and {b}.

    You MUST use the add_numbers tool to compute the result.
    Do NOT compute the sum yourself.
    After getting the result from add_numbers, return it.
    """
    raise NotHandled


class MessageSequenceTracker(ObjectInterpretation):
    """Intercepts call_assistant to record message IDs forwarded by the provider."""

    def __init__(self):
        self.call_log: list[list[str]] = []

    @implements(call_assistant)
    def _call_assistant(self, *args, **kwargs):
        self.call_log.append([m["id"] for m in _get_history().values()])
        return fwd()


class TestMessageSequenceReplay:
    """Fixture-based tests verifying message sequence invariants through the full provider stack."""

    @requires_openai
    def test_simple_prompt_unique_message_ids(self, request):
        """A no-tool prompt should produce a single call_assistant with unique message IDs."""
        tracker = MessageSequenceTracker()
        with (
            handler(tracker),
            handler(ReplayLiteLLMProvider(request, model="gpt-4o-mini")),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("testing")

        assert isinstance(result, str)
        assert len(tracker.call_log) == 1
        ids = tracker.call_log[0]
        assert len(ids) == len(set(ids)), "message IDs should be unique"

    @requires_openai
    def test_tool_calling_no_duplicate_message_ids(self, request):
        """Tool-calling prompts should accumulate messages without duplicates across calls."""
        tracker = MessageSequenceTracker()

        with (
            handler(tracker),
            handler(ReplayLiteLLMProvider(request, model="gpt-4o-mini")),
            handler(LimitLLMCallsHandler(max_calls=4)),
        ):
            result = compute_sum(3, 5)

        assert result == 8
        assert len(tracker.call_log) >= 2  # at least: tool call round + final answer

        for i, ids in enumerate(tracker.call_log):
            assert len(ids) == len(set(ids)), (
                f"call_assistant invocation {i} has duplicate message IDs: {ids}"
            )

        # Each successive call should include all previous messages plus new ones
        for i in range(1, len(tracker.call_log)):
            prev_set = set(tracker.call_log[i - 1])
            curr_set = set(tracker.call_log[i])
            assert prev_set < curr_set, (
                f"call {i} messages should be a strict superset of call {i - 1}"
            )


# ============================================================================
# Issue #558: Agent recovery from erroneous tool calls
# ============================================================================


@Tool.define
def flaky_tool(x: int) -> str:
    """A tool that raises ConnectionError."""
    raise ConnectionError(f"transient failure for {x}")


@Tool.define
def type_error_tool(x: int) -> str:
    """A tool that raises TypeError."""
    raise TypeError(f"bad type for {x}")


class TestCallToolWrapsExecutionError:
    """call_tool should wrap runtime tool errors in ToolCallExecutionError."""

    def test_call_tool_raises_tool_call_execution_error(self):
        """call_tool wraps tool runtime errors in ToolCallExecutionError."""
        sig = inspect.signature(failing_tool)
        bound_args = sig.bind(x=7)
        tc = DecodedToolCall(failing_tool, bound_args, "call_wrap_1", "failing_tool")

        with pytest.raises(ToolCallExecutionError) as exc_info:
            call_tool(tc)

        err = exc_info.value
        assert err.raw_tool_call.name == "failing_tool"
        assert err.raw_tool_call.id == "call_wrap_1"
        assert isinstance(err.original_error, ValueError)

    def test_call_tool_preserves_cause_chain(self):
        """ToolCallExecutionError should chain from the original exception."""
        sig = inspect.signature(failing_tool)
        bound_args = sig.bind(x=1)
        tc = DecodedToolCall(failing_tool, bound_args, "call_chain", "failing_tool")

        with pytest.raises(ToolCallExecutionError) as exc_info:
            call_tool(tc)

        assert exc_info.value.__cause__ is exc_info.value.original_error

    def test_call_tool_success_does_not_raise(self):
        """Successful tool calls should not raise ToolCallExecutionError."""
        sig = inspect.signature(add_numbers)
        bound_args = sig.bind(a=3, b=4)
        tc = DecodedToolCall(add_numbers, bound_args, "call_ok", "add_numbers")

        result = call_tool(tc)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_ok"


class TestRetryHandlerCatchToolErrorsFiltering:
    """RetryLLMHandler should only catch tool errors matching catch_tool_errors."""

    def test_matching_error_returns_feedback_message(self):
        """When original_error matches catch_tool_errors, return error feedback."""
        sig = inspect.signature(flaky_tool)
        bound_args = sig.bind(x=1)
        tc = DecodedToolCall(flaky_tool, bound_args, "call_match", "flaky_tool")

        with handler(RetryLLMHandler(catch_tool_errors=ConnectionError)):
            result = call_tool(tc)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_match"
        assert "Tool execution failed" in result["content"]
        assert "flaky_tool" in result["content"]

    def test_non_matching_error_propagates_as_execution_error(self):
        """When original_error doesn't match catch_tool_errors, re-raise ToolCallExecutionError."""
        sig = inspect.signature(flaky_tool)
        bound_args = sig.bind(x=1)
        tc = DecodedToolCall(flaky_tool, bound_args, "call_no_match", "flaky_tool")

        # catch_tool_errors=TypeError, but tool raises ConnectionError
        with pytest.raises(ToolCallExecutionError) as exc_info:
            with handler(RetryLLMHandler(catch_tool_errors=TypeError)):
                call_tool(tc)

        assert isinstance(exc_info.value.original_error, ConnectionError)

    def test_default_catch_all_catches_everything(self):
        """Default catch_tool_errors=Exception catches all standard exceptions."""
        sig = inspect.signature(type_error_tool)
        bound_args = sig.bind(x=5)
        tc = DecodedToolCall(
            type_error_tool, bound_args, "call_default", "type_error_tool"
        )

        with handler(RetryLLMHandler()):
            result = call_tool(tc)

        assert result["role"] == "tool"
        assert "Tool execution failed" in result["content"]

    def test_tuple_of_error_types(self):
        """catch_tool_errors accepts a tuple of exception types."""
        sig = inspect.signature(flaky_tool)
        bound_args = sig.bind(x=1)
        tc = DecodedToolCall(flaky_tool, bound_args, "call_tuple", "flaky_tool")

        with handler(
            RetryLLMHandler(
                catch_tool_errors=(ConnectionError, ValueError),
            )
        ):
            result = call_tool(tc)

        assert result["role"] == "tool"
        assert "Tool execution failed" in result["content"]

    def test_no_retry_handler_propagates_execution_error(self):
        """Without RetryLLMHandler, ToolCallExecutionError propagates directly."""
        sig = inspect.signature(failing_tool)
        bound_args = sig.bind(x=1)
        tc = DecodedToolCall(failing_tool, bound_args, "call_no_retry", "failing_tool")

        with pytest.raises(ToolCallExecutionError):
            call_tool(tc)


class TestLiteLLMProviderMessagePruning:
    """LiteLLMProvider should prune messages added during a failed template call."""

    def test_messages_pruned_on_tool_execution_error(self):
        """When a tool error propagates, all messages from that call are pruned."""
        # LLM says "call flaky_tool", then tool raises unhandled error
        responses = [
            make_tool_call_response("flaky_tool", '{"x": {"value": 1}}'),
        ]
        mock_handler = MockCompletionHandler(responses)

        message_sequence = collections.OrderedDict()

        @Template.define
        def task_with_flaky_tool(instruction: str) -> str:
            """Do: {instruction}"""
            raise NotHandled

        with pytest.raises(ToolCallExecutionError):
            with (
                handler(LiteLLMProvider(model="test")),
                handler(mock_handler),
                handler({_get_history: lambda: message_sequence}),
            ):
                task_with_flaky_tool("go")

        # All messages from the failed call should be pruned
        assert len(message_sequence) == 0

    def test_messages_pruned_on_unhandled_decoding_error(self):
        """When a decoding error propagates (no retry handler), messages are pruned."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": "bad"}'),
        ]
        mock_handler = MockCompletionHandler(responses)

        message_sequence = collections.OrderedDict()

        @Template.define
        def task_with_tools(instruction: str) -> str:
            """Do: {instruction}"""
            raise NotHandled

        with pytest.raises(ToolCallDecodingError):
            with (
                handler(LiteLLMProvider(model="test")),
                handler(mock_handler),
                handler({_get_history: lambda: message_sequence}),
            ):
                task_with_tools("go")

        assert len(message_sequence) == 0

    def test_pre_existing_messages_preserved_on_error(self):
        """Pre-existing messages in the sequence are not pruned when a call fails."""
        responses = [
            make_tool_call_response("flaky_tool", '{"x": {"value": 1}}'),
        ]
        mock_handler = MockCompletionHandler(responses)

        message_sequence = collections.OrderedDict(
            existing={"id": "existing", "role": "user", "content": "hello"},
        )

        @Template.define
        def task_with_flaky_tool(instruction: str) -> str:
            """Do: {instruction}"""
            raise NotHandled

        with pytest.raises(ToolCallExecutionError):
            with (
                handler(LiteLLMProvider(model="test")),
                handler(mock_handler),
                handler({_get_history: lambda: message_sequence}),
            ):
                task_with_flaky_tool("go")

        # Pre-existing message should still be there
        assert len(message_sequence) == 1
        assert "existing" in message_sequence

    def test_successful_call_preserves_messages(self):
        """A successful top-level template call should write messages back to Agent history."""
        responses = [make_text_response("done")]
        mock_handler = MockCompletionHandler(responses)

        class SimpleAgent(Agent):
            """You are a persistence-check test agent.
            Your goal is to complete `simple_task` and persist successful history.
            """

            @Template.define
            def simple_task(self, instruction: str) -> str:
                """Do: {instruction}"""
                raise NotHandled

        agent = SimpleAgent()

        # No outer _get_history handler: LiteLLMProvider._call detects this is the
        # outermost template and writes back to the agent's __history__.
        with (
            handler(LiteLLMProvider(model="test")),
            handler(mock_handler),
        ):
            result = agent.simple_task("go")

        assert result == "done"
        # Agent's __history__ should have messages written back (system + user + assistant)
        assert len(agent.__history__) >= 2


class TestAgentCrossTemplateRecovery:
    """Issue #558: Agent should recover from errored tool calls across template methods.

    When a tool call fails and the error propagates (not caught by RetryLLMHandler),
    the agent's message history must be cleaned up so subsequent template calls
    don't fail due to orphaned assistant tool_calls messages.
    """

    def test_agent_second_call_succeeds_after_tool_error(self):
        """After a tool error in one template, another template on the same agent works."""

        @Tool.define
        def bad_service() -> str:
            """Fetch from a broken service."""
            raise ConnectionError("service down")

        import dataclasses

        @dataclasses.dataclass
        class TestAgent(Agent):
            """You are a cross-template recovery test agent.
            Your goal is to recover from failed tool calls across template methods.
            """

            @Template.define
            def step_with_tool(self, task: str) -> str:
                """Use bad_service for: {task}"""
                raise NotHandled

            @Template.define
            def step_no_tool(self, topic: str) -> str:
                """Summarize: {topic}. Do not use any tools."""
                raise NotHandled

        # Step 1: LLM calls bad_service → tool error propagates
        tool_call_response = make_tool_call_response("bad_service", "{}")
        # Step 2: Simple text response for the second template
        text_response = make_text_response("summary result")

        call_count = 0

        class TwoPhaseCompletionHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self, model, messages=None, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return tool_call_response
                return text_response

        agent = TestAgent()

        with handler(TwoPhaseCompletionHandler()):
            with handler(LiteLLMProvider(model="test")):
                # First call should fail with tool execution error
                with pytest.raises(ToolCallExecutionError):
                    agent.step_with_tool("stage 1")

                # History should be clean — no orphaned tool_calls
                # Second call should succeed without BadRequestError
                result = agent.step_no_tool("stage 2")

        assert result == "summary result"
        # Verify history doesn't contain messages from the failed call
        history = agent.__history__
        for msg in history.values():
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # If there's an assistant message with tool_calls, there must be
                # corresponding tool responses
                for tc in tool_calls:
                    tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                    has_response = any(
                        m.get("tool_call_id") == tc_id
                        for m in history.values()
                        if m.get("role") == "tool"
                    )
                    assert has_response, (
                        f"Orphaned tool_call {tc_id} in history without response"
                    )

    def test_agent_history_clean_after_error_pruning(self):
        """After an error, the agent history should contain no messages from the failed call."""

        @Tool.define
        def exploding_tool() -> str:
            """A tool that explodes."""
            raise RuntimeError("boom")

        import dataclasses

        @dataclasses.dataclass
        class CleanupAgent(Agent):
            """You are an error-cleanup test agent.
            Your goal is to ensure failed calls do not persist message history.
            """

            @Template.define
            def do_work(self, task: str) -> str:
                """Do: {task}"""
                raise NotHandled

        responses = [make_tool_call_response("exploding_tool", "{}")]
        mock = MockCompletionHandler(responses)
        agent = CleanupAgent()

        with pytest.raises(ToolCallExecutionError):
            with handler(LiteLLMProvider(model="test")), handler(mock):
                agent.do_work("go")

        # Agent history should be empty — all messages from failed call pruned
        assert len(agent.__history__) == 0

    def test_agent_history_preserved_for_successful_calls(self):
        """Successful calls should leave messages in agent history."""

        import dataclasses

        @dataclasses.dataclass
        class SuccessAgent(Agent):
            """You are a success-history test agent.
            Your goal is to preserve message history for successful calls.
            """

            @Template.define
            def greet(self, name: str) -> str:
                """Say hello to {name}."""
                raise NotHandled

        responses = [make_text_response("Hello!")]
        mock = MockCompletionHandler(responses)
        agent = SuccessAgent()

        with handler(LiteLLMProvider(model="test")), handler(mock):
            result = agent.greet("world")

        assert result == "Hello!"
        # History should contain messages from the successful call
        assert len(agent.__history__) >= 2  # user + assistant at minimum

    def test_agent_multiple_successful_calls_accumulate_history(self):
        """Multiple successful calls should accumulate in agent history."""

        import dataclasses

        @dataclasses.dataclass
        class ChatAgent(Agent):
            """You are a multi-call history test agent.
            Your goal is to accumulate conversation history across successful calls.
            """

            @Template.define
            def chat(self, msg: str) -> str:
                """Respond to: {msg}"""
                raise NotHandled

        call_count = 0

        class MultiResponseHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self, model, messages=None, **kwargs):
                nonlocal call_count
                call_count += 1
                return make_text_response(f"reply {call_count}")

        agent = ChatAgent()

        with handler(LiteLLMProvider(model="test")), handler(MultiResponseHandler()):
            r1 = agent.chat("first")
            r2 = agent.chat("second")

        assert r1 == "reply 1"
        assert r2 == "reply 2"
        # History should have messages from both calls
        assert len(agent.__history__) >= 4  # 2 * (user + assistant)

    def test_agent_error_then_success_accumulates_only_success(self):
        """After a failed call, only the subsequent successful call's messages remain."""

        @Tool.define
        def broken_tool() -> str:
            """Tool that breaks."""
            raise ValueError("broken")

        import dataclasses

        @dataclasses.dataclass
        class RecoveryAgent(Agent):
            """You are a failure-recovery test agent.
            Your goal is to recover after a failed call and retain only successful history.
            """

            @Template.define
            def risky(self, task: str) -> str:
                """Do risky: {task}"""
                raise NotHandled

            @Template.define
            def safe(self, task: str) -> str:
                """Do safe: {task}. Do not use tools."""
                raise NotHandled

        call_count = 0

        class PhaseHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self, model, messages=None, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return make_tool_call_response("broken_tool", "{}")
                return make_text_response("safe result")

        agent = RecoveryAgent()

        with handler(LiteLLMProvider(model="test")), handler(PhaseHandler()):
            with pytest.raises(ToolCallExecutionError):
                agent.risky("step 1")

            history_after_error = len(agent.__history__)
            assert history_after_error == 0

            result = agent.safe("step 2")

        assert result == "safe result"
        # Only messages from the successful call should be in history
        assert len(agent.__history__) >= 2
        assert len(agent.__history__) > history_after_error
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
