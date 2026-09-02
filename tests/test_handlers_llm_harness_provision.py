"""Tests for LLM handlers and providers.
This module tests the functionality from build/main.py and build/llm.py,
breaking down individual components like LiteLLMConfigurer,
ProgramSynthesis, and sampling strategies.
"""

import contextlib
import functools
import inspect
import json
import os
import re
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from types import CodeType

import litellm
import pydantic
import pytest
import tenacity
from litellm import ChatCompletionMessageToolCall
from litellm.caching.caching import Cache
from litellm.files.main import ModelResponse
from PIL import Image
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from effectful.handlers.llm import Agent, Skill
from effectful.handlers.llm.harness.durability.retrying import TenacityRetryer
from effectful.handlers.llm.harness.durability.transaction import (
    CompactionScope,
    HistoryBuilder,
)
from effectful.handlers.llm.harness.execution.builtin import BuiltinExecutor
from effectful.handlers.llm.harness.hooks import (
    AgentLoop,
    DecodedToolCall,
    ResultDecodingError,
    Tool,
    ToolCallDecodingError,
    ToolCallExecutionError,
    call_agent,
    call_assistant,
    call_tool,
    completion,
)
from effectful.handlers.llm.harness.legibility.lexical import (
    LexicalToolExtractor,
    _tools_in_scope,
)
from effectful.handlers.llm.harness.observability.rich import RichTerminalRenderer
from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
from effectful.handlers.llm.harness.serialization import (
    _is_empty_text_block,
    _NameAndTool,
    format_as_content_blocks,
    to_content_blocks,
)
from effectful.handlers.llm.harness.synthesis.body import (
    FinalBodySynthesizer,
)
from effectful.handlers.llm.harness.synthesis.snippet import StatefulReplSynthesizer
from effectful.handlers.llm.harness.validation.ty import TyTypeChecker
from effectful.handlers.llm.types import Encodable
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled
from tests.conftest import (
    EFFECTFUL_LLM_MODEL,
    MockCompletionHandler,
    add_numbers,
    make_text_response,
    make_tool_call_response,
    requires_llm,
    requires_vision,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

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


class ReplayLiteLLMProvider(LiteLLMConfigurer, AgentLoop):
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


class MovieGenre(StrEnum):
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


@Skill.define
def classify_genre(plot: str) -> MovieClassification:
    """Classify the movie genre based on this plot: {plot}."""
    raise NotImplementedError


@Skill.define
def simple_prompt(topic: str) -> str:
    """Write a short sentence about {topic}."""
    raise NotImplementedError


@Skill.define
def generate_number(max_value: int) -> int:
    """Generate a random number between 1 and {max_value}."""
    raise NotImplementedError


@Skill.define
def create_function(char: str) -> Callable[[str], int]:
    """Create a function that counts occurrences of the character '{char}' in a string.

    Return as a code block with the last definition being the function.
    """
    raise NotHandled


class _ToolNameAgent(Agent):
    @Skill.define
    def helper(self) -> str:
        """Return the literal string 'ok'."""
        raise NotHandled

    @Skill.define
    def ask(self, prompt: str) -> str:
        """Answer briefly: {prompt}"""
        raise NotHandled


class TestLiteLLMProvider:
    """Tests for the LiteLLM-backed agent loop's basic functionality."""

    def test_simple_prompt(self, request):
        """Test that a LiteLLM-backed skill call returns a non-empty string."""
        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("testing")
            assert isinstance(result, str)
            assert len(result) > 0

    def test_structured_output(self, request):
        """Test a LiteLLM-backed skill call with structured Pydantic output."""
        plot = "A rogue cop must stop a evil group from taking over a skyscraper."

        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            classification = classify_genre(plot)

            assert isinstance(classification, MovieClassification)
            assert isinstance(classification.genre, MovieGenre)
            assert classification.genre == MovieGenre.ACTION
            assert isinstance(classification.explanation, str)
            assert len(classification.explanation) > 0

    def test_integer_return_type(self, request):
        """Test a LiteLLM-backed skill call with integer return type."""
        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = generate_number(100)

            assert isinstance(result, int)
            assert 1 <= result <= 100

    def test_with_config_params(self, request):
        """Test LiteLLMConfigurer accepts and uses additional configuration parameters."""
        # Test with temperature parameter
        with (
            handler(
                ReplayLiteLLMProvider(
                    request, model=EFFECTFUL_LLM_MODEL, temperature=0.1
                )
            ),
            handler(LexicalToolExtractor()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("deterministic test")
            assert isinstance(result, str)


@requires_llm
def test_agent_tool_names_are_valid_integration():
    agent = _ToolNameAgent()
    skill = agent.ask
    tools = _tools_in_scope(skill.__context__)
    names = {t.__name__ for t in tools}
    assert tools
    assert agent.helper.__name__ in names
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in names)

    # End-to-end provider call. If tool names violate the schema, this raises BadRequest.
    # `max_tokens` only has to be big enough for the model to say something: a reply
    # truncated to empty content is decoded as "no final response" and fails the call
    # for a reason that has nothing to do with tool names.
    with (
        handler(AgentLoop()),
        handler(LexicalToolExtractor()),
        handler(
            LiteLLMConfigurer(
                model=EFFECTFUL_LLM_MODEL, tool_choice="none", max_tokens=64
            )
        ),
        handler(HistoryBuilder()),
        handler(LimitLLMCallsHandler(max_calls=1)),
    ):
        result = agent.ask("Reply with exactly 'ok'. Do not call tools.")

    assert isinstance(result, str)
    assert result


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


@Skill.define
def categorise_image(image: Image.Image) -> str:
    """Return a description of the following image.
    {image}"""
    raise NotHandled


@requires_vision
def test_image_input(request):
    with (
        handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
        handler(LexicalToolExtractor()),
        handler(LimitLLMCallsHandler(max_calls=3)),
    ):
        assert any("smile" in categorise_image(smiley_face()) for _ in range(3))


class ImageDescription(BaseModel):
    """Description of a set of images."""

    description: str = Field(description="What you see in the images")
    count: int = Field(description="Number of images provided")


@Skill.define
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


@requires_vision
def test_list_image_input(request):
    """Regression test for GitHub issue #552: list[Image.Image] in skills."""
    img_red = Image.new("RGB", (64, 64), (255, 0, 0))
    img_blue = Image.new("RGB", (64, 64), (0, 0, 255))

    with (
        handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
        handler(LexicalToolExtractor()),
        handler(TenacityRetryer(stop=tenacity.stop_after_attempt(3))),
        handler(LimitLLMCallsHandler(max_calls=3)),
    ):
        result = describe_images(
            context="Two colored squares",
            views=[img_red, img_blue],
        )

    assert isinstance(result, ImageDescription)
    assert result.count == 2


class BookReview(BaseModel):
    """A book review with rating and summary."""

    title: str = Field(..., description="title of the book")
    rating: int = Field(..., description="rating from 1 to 5", ge=1, le=5)
    summary: str = Field(..., description="brief summary of the review")


@Skill.define
def review_book(plot: str) -> BookReview:
    """Review a book based on this plot: {plot}."""
    raise NotImplementedError


class TestPydanticBaseModelReturn:
    def test_pydantic_basemodel_return(self, request):
        plot = "A young wizard discovers he has magical powers and goes to a school for wizards."

        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
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
    with (
        handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
        handler(LexicalToolExtractor()),
    ):
        p1 = simple_prompt("apples")
        p2 = simple_prompt("apples")
        p3 = simple_prompt("oranges")
        assert p1 == p2, (
            "when caching is enabled, LLM requests with the same parameters will produce the same outputs"
        )
        assert p3 != p2, "different inputs should still produce different outputs"


def test_litellm_caching_integration_disabled(request):
    litellm.cache = Cache()
    with (
        handler(
            ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL, caching=False)
        ),
        handler(LexicalToolExtractor()),
    ):
        p1 = simple_prompt("apples")
        p2 = simple_prompt("apples")
        assert p1 != p2, "if caching is not enabled, inputs produce different outputs"


def test_litellm_caching_selective(request):
    with (
        handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
        handler(LexicalToolExtractor()),
    ):
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
# TenacityRetryer Tests
# ============================================================================


@pytest.fixture
def message_sequence_provider():
    message_sequence = [{"role": "user", "content": "test"}]
    return message_sequence, {HistoryBuilder.get_history: lambda: message_sequence}


@pytest.fixture
def mock_completion_handler_factory():
    def _factory(responses: list[ModelResponse]) -> MockCompletionHandler:
        return MockCompletionHandler(responses)

    return _factory


class TestTenacityRetryer:
    """Tests for TenacityRetryer functionality."""

    def test_retry_handler_succeeds_on_first_attempt(self):
        """Test that TenacityRetryer passes through when no error occurs."""
        # Response with valid tool call
        responses = [make_text_response("hello")]

        mock_handler = MockCompletionHandler(responses)

        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with (
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                list(message_sequence),
                response_type=str,
                env={},
            )

        assert mock_handler.call_count == 1
        assert result == "hello"

    def test_retry_handler_retries_on_invalid_tool_call(self):
        """Test that TenacityRetryer retries when tool call decoding fails."""
        # First response has invalid tool args, second has valid response
        responses = [
            make_tool_call_response(
                "add_numbers", '{"a": "not_an_int", "b": 2}'
            ),  # Invalid
            make_text_response("success"),  # Valid
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with (
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                list(message_sequence),
                response_type=str,
                env={"add_numbers": add_numbers},
                tools={add_numbers},
            )

        assert mock_handler.call_count == 2
        assert result == "success"
        # Check that the second call included error feedback
        assert len(mock_handler.received_messages[1]) > len(
            mock_handler.received_messages[0]
        )

    def test_retry_handler_retries_on_unknown_tool(self):
        """Test that TenacityRetryer retries when tool is not found."""
        # First response has unknown tool, second has valid response
        responses = [
            make_tool_call_response("unknown_tool", '{"x": 1}'),  # Unknown tool
            make_text_response("success"),  # Valid
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with (
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                list(message_sequence),
                response_type=str,
                env={"add_numbers": add_numbers},
                tools={add_numbers},
            )

        assert mock_handler.call_count == 2
        assert result == "success"

    def test_retry_after_result_decode_failure_is_resendable(self):
        """The request a retry sends must not end with the model's own words.

        A malformed answer is the one failure whose feedback has no tool call to
        attach to, so it is a user message; were it a second assistant message,
        the retry would be asking Anthropic to prefill the reply rather than to
        write one (see `ResultDecodingError.to_feedback_message`).
        """
        responses = [
            make_text_response("about tree fiddy"),  # not JSON: fails to decode
            make_text_response(json.dumps({"value": 350})),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with (
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                list(message_sequence),
                response_type=int,
                env={},
            )

        assert result == 350
        retried = mock_handler.received_messages[1]
        assert [m["role"] for m in retried] == ["user", "assistant", "user"]

    def test_retry_handler_exhausts_retries(self):
        """Test that TenacityRetryer raises after exhausting all retries."""
        # All responses have invalid tool calls
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": "bad"}'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }
        with pytest.raises(ToolCallDecodingError):
            with (
                handler(HistoryBuilder()),
                handler(TenacityRetryer(stop=tenacity.stop_after_attempt(3))),
                handler(mock_handler),
                handler(message_sequence_provider),
            ):
                call_assistant(
                    list(message_sequence),
                    response_type=str,
                    env={"add_numbers": add_numbers},
                    tools={add_numbers},
                )

        # Should have attempted 3 times (1 initial + 2 retries)
        assert mock_handler.call_count == 3

    def test_retry_handler_with_zero_retries(self):
        """Test TenacityRetryer with stop_after_attempt(1) fails immediately on error."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": "bad"}'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with pytest.raises(ToolCallDecodingError):
            with (
                handler(HistoryBuilder()),
                handler(TenacityRetryer(stop=tenacity.stop_after_attempt(1))),
                handler(mock_handler),
                handler(message_sequence_provider),
            ):
                call_assistant(
                    list(message_sequence),
                    response_type=str,
                    env={"add_numbers": add_numbers},
                    tools={add_numbers},
                )

    def test_retry_handler_valid_tool_call_passes_through(self):
        """Test that valid tool calls are decoded and returned."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with (
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                list(message_sequence),
                response_type=str,
                env={"add_numbers": add_numbers},
                tools={add_numbers},
            )

        assert mock_handler.call_count == 1
        assert len(tool_calls) == 1
        assert tool_calls[0].tool == add_numbers
        assert result is None  # No result when there are tool calls

    def test_retry_answers_every_tool_call_when_one_fails_to_decode(self):
        """A partly-undecodable turn is still retried as a well-formed request.

        Decoding abandons a turn's remaining tool calls at the first failure, but
        the assistant message recorded for the retry advertised all of them and
        the failure's feedback answers only one. Both OpenAI APIs require exactly
        one output per advertised call, so an unanswered sibling turns the
        recoverable decode error into a `BadRequestError` -- which is not in this
        retryer's retry set, and so kills the call instead of informing it.
        """
        two_calls = ModelResponse(
            id="test",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_good",
                                "type": "function",
                                "function": {
                                    "name": "add_numbers",
                                    "arguments": '{"a": 1, "b": 2}',
                                },
                            },
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "add_numbers",
                                    "arguments": '{"a": "not", "b": "ints"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            model="test-model",
        )
        mock_handler = MockCompletionHandler(
            [two_calls, make_text_response("recovered")]
        )
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with (
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            _message, _tool_calls, result = call_assistant(
                list(message_sequence),
                response_type=str,
                env={"add_numbers": add_numbers},
                tools={add_numbers},
            )

        assert mock_handler.call_count == 2
        assert result == "recovered"

        retried = mock_handler.received_messages[1]
        advertised = [
            call["id"]
            for message in retried
            if message["role"] == "assistant"
            for call in message.get("tool_calls") or []
        ]
        answered = [m["tool_call_id"] for m in retried if m["role"] == "tool"]
        assert advertised == ["call_good", "call_bad"]
        assert sorted(answered) == sorted(advertised), (
            f"every advertised tool call must be answered, but "
            f"advertised={advertised} and answered={answered}"
        )

    def test_codeadapt_notebook_replay_fixture(self, request):
        """Replay fixture for codeadapt higher-order tool flow."""

        @Skill.define
        def generate_paragraph() -> str:
            """Please generate a paragraph: with exactly 4 sentences ending with 'walk', 'tumbling', 'another', and 'lunatic'."""
            raise NotHandled

        @Skill.define
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

        @Skill.define
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
            handler(TenacityRetryer(stop=tenacity.stop_after_attempt(3))),
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
        ):
            result = codeadapt("generate_paragraph")

        assert isinstance(result, str)

    def test_retry_handler_retries_on_invalid_result(self):
        """Test that TenacityRetryer retries when result decoding fails."""
        # First response has invalid JSON, second has valid response
        responses = [
            make_text_response('"not valid for int"'),  # Invalid for int
            make_text_response('{"value": 42}'),  # Valid
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with (
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                list(message_sequence),
                response_type=int,
                env={},
            )

        assert mock_handler.call_count == 2
        assert result == 42
        # Check that the second call included error feedback
        assert len(mock_handler.received_messages[1]) > len(
            mock_handler.received_messages[0]
        )

    def test_retry_handler_exhausts_retries_on_result_decoding(self):
        """Test that TenacityRetryer raises after exhausting retries on result decoding."""
        # All responses have invalid results for int type
        responses = [
            make_text_response('"not an int"'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with pytest.raises(ResultDecodingError):
            with (
                handler(HistoryBuilder()),
                handler(TenacityRetryer(stop=tenacity.stop_after_attempt(3))),
                handler(mock_handler),
                handler(message_sequence_provider),
            ):
                call_assistant(
                    list(message_sequence),
                    response_type=int,
                    env={},
                )

        # Should have attempted 3 times (1 initial + 2 retries)
        assert mock_handler.call_count == 3

    def test_retry_handler_raises_tool_call_decoding_error(self):
        """Test that TenacityRetryer raises ToolCallDecodingError with correct attributes."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": "bad"}'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with pytest.raises(ToolCallDecodingError) as exc_info:
            with (
                handler(HistoryBuilder()),
                handler(TenacityRetryer(stop=tenacity.stop_after_attempt(1))),
                handler(mock_handler),
                handler(message_sequence_provider),
            ):
                call_assistant(
                    list(message_sequence),
                    response_type=str,
                    env={"add_numbers": add_numbers},
                    tools={add_numbers},
                )

        error = exc_info.value
        assert error.raw_tool_call.function.name == "add_numbers"
        assert error.raw_tool_call.id == "call_1"
        assert error.raw_message is not None
        assert "add_numbers" in str(error)

    def test_retry_handler_raises_result_decoding_error(self):
        """Test that TenacityRetryer raises ResultDecodingError with correct attributes."""
        responses = [
            make_text_response('"not an int"'),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with pytest.raises(ResultDecodingError) as exc_info:
            with (
                handler(HistoryBuilder()),
                handler(TenacityRetryer(stop=tenacity.stop_after_attempt(1))),
                handler(mock_handler),
                handler(message_sequence_provider),
            ):
                call_assistant(
                    list(message_sequence),
                    response_type=int,
                    env={},
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
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with (
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            call_assistant(
                list(message_sequence),
                response_type=str,
                env={"add_numbers": add_numbers},
                tools={add_numbers},
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
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with (
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            call_assistant(
                list(message_sequence),
                response_type=str,
                env={"add_numbers": add_numbers},
                tools={add_numbers},
            )

        # Check that the error feedback mentions the unknown tool
        second_call_messages = mock_handler.received_messages[1]
        tool_feedback = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_feedback) == 1
        assert "nonexistent_tool" in tool_feedback[0]["content"]

    def test_retry_handler_include_traceback_in_error_feedback(self):
        """Test that error feedback carries the traceback of the failed decode."""
        responses = [
            make_tool_call_response("add_numbers", '{"a": "bad", "b": 2}'),
            make_text_response("success"),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        with (
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            call_assistant(
                list(message_sequence),
                response_type=str,
                env={"add_numbers": add_numbers},
                tools={add_numbers},
            )

        # Check that the error feedback includes traceback
        second_call_messages = mock_handler.received_messages[1]
        tool_feedback = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_feedback) == 1
        assert "Traceback:" in tool_feedback[0]["content"]
        assert "```" in tool_feedback[0]["content"]


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
        """Test that TenacityRetryer catches tool runtime errors and returns error message."""

        # Create a decoded tool call for failing_tool
        sig = inspect.signature(failing_tool)
        bound_args = sig.bind(x=42)
        tool_call = DecodedToolCall(failing_tool, bound_args, "call_1", "failing_tool")

        with handler(TenacityRetryer()):
            result, _, _ = call_tool(tool_call)

        # The result should be an error message, not an exception
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert "Tool execution failed" in result["content"]
        assert "failing_tool" in result["content"]
        assert "42" in result["content"]

    def test_retry_handler_catches_division_by_zero(self):
        """Test that TenacityRetryer catches division by zero errors."""

        sig = inspect.signature(divide_tool)
        bound_args = sig.bind(a=10, b=0)
        tool_call = DecodedToolCall(divide_tool, bound_args, "call_div", "divide_tool")

        with handler(TenacityRetryer()):
            result, _, _ = call_tool(tool_call)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_div"
        assert "Tool execution failed" in result["content"]
        assert "divide_tool" in result["content"]

    def test_successful_tool_execution_returns_result(self):
        """Test that successful tool executions return normal results."""

        sig = inspect.signature(add_numbers)
        bound_args = sig.bind(a=3, b=4)
        tool_call = DecodedToolCall(add_numbers, bound_args, "call_add", "add_numbers")

        with handler(TenacityRetryer()):
            result, _, _ = call_tool(tool_call)

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
            make_tool_call_response("failing_tool", '{"x": 42}'),
            make_text_response("handled the error"),
        ]

        mock_handler = MockCompletionHandler(responses)
        message_sequence = [{"role": "user", "content": "test"}]
        message_sequence_provider = {
            HistoryBuilder.get_history: lambda: message_sequence
        }

        # We need a custom provider that actually calls call_tool
        class TestProvider(ObjectInterpretation):
            @implements(call_assistant)
            def _call_assistant(
                self, messages, response_type, env, tools=frozenset(), **kwargs
            ):
                return fwd(messages, response_type, env, tools, **kwargs)

        with (
            handler(TenacityRetryer()),
            handler(TestProvider()),
            handler(mock_handler),
            handler(message_sequence_provider),
        ):
            message, tool_calls, result = call_assistant(
                list(message_sequence),
                response_type=str,
                env={"failing_tool": failing_tool},
                tools={failing_tool},
            )

        # First call should succeed (tool call is valid)
        assert mock_handler.call_count == 1
        assert len(tool_calls) == 1


class _StreamingMockCompletionHandler(ObjectInterpretation):
    """Streams a fixed text answer back, one character per chunk.

    `RichTerminalRenderer` forces ``stream=True`` and reassembles the chunks with
    ``litellm.stream_chunk_builder``, so exercising it needs a mock that answers
    in deltas rather than with a finished `ModelResponse`.
    """

    def __init__(self, content: str):
        self.content = content

    @implements(completion)
    def _completion(self, *args, **kwargs):
        assert kwargs.get("stream"), "expected the renderer to force streaming"
        return iter(
            [
                litellm.types.utils.ModelResponseStream(
                    id="test",
                    model="test-model",
                    choices=[
                        litellm.types.utils.StreamingChoices(
                            index=0,
                            delta=litellm.types.utils.Delta(
                                role="assistant", content=char
                            ),
                        )
                    ],
                )
                for char in self.content
            ]
        )


class TestForcedToolChoice:
    """`tool_choice` is provider configuration, so the provider is what enforces
    it: a response that disobeys it anyway -- some OpenAI-compatible servers
    treat it as advisory -- is reported as the protocol violation it is, rather
    than being decoded as a bare result (`required`) or executed (`none`)."""

    def test_prose_answer_is_rejected(self):
        with (
            handler(MockCompletionHandler([make_text_response("just prose")])),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model", tool_choice="required")),
            pytest.raises(ResultDecodingError, match="YOU MUST GENERATE A TOOL CALL"),
        ):
            call_assistant([], response_type=str, env={"add_numbers": add_numbers})

    def test_tool_call_is_accepted(self):
        response = make_tool_call_response("add_numbers", '{"a": 1, "b": 2}')
        with (
            handler(MockCompletionHandler([response])),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model", tool_choice="required")),
        ):
            _, tool_calls, _ = call_assistant(
                [],
                response_type=str,
                env={"add_numbers": add_numbers},
                tools={add_numbers},
            )
        assert len(tool_calls) == 1

    def test_tool_call_is_rejected_when_disabled(self):
        response = make_tool_call_response("add_numbers", '{"a": 1, "b": 2}')
        with (
            handler(MockCompletionHandler([response])),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model", tool_choice="none")),
            pytest.raises(ResultDecodingError, match="YOU MUST ANSWER DIRECTLY"),
        ):
            call_assistant(
                [],
                response_type=str,
                env={"add_numbers": add_numbers},
                tools={add_numbers},
            )

    def test_prose_answer_is_accepted_when_tools_are_disabled(self):
        with (
            handler(MockCompletionHandler([make_text_response("just prose")])),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model", tool_choice="none")),
        ):
            _, tool_calls, result = call_assistant(
                [], response_type=str, env={"add_numbers": add_numbers}
            )
        assert not tool_calls
        assert result == "just prose"

    def test_enclosed_tool_choice_wins(self):
        """Only the `tool_choice` litellm is actually sent is enforced.

        `LiteLLMConfigurer.completion` merges as ``{**self.config, **kwargs}``, so the enclosed
        provider's ``auto`` is what leaves for the model -- and an enclosing
        ``required`` that never made it into the request must not be held
        against the response.
        """
        with (
            handler(MockCompletionHandler([make_text_response("just prose")])),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model", tool_choice="required")),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model", tool_choice="auto")),
        ):
            _, _, result = call_assistant([], response_type=str, env={})
        assert result == "just prose"

    def test_enclosed_required_is_enforced(self):
        """The mirror image: the enclosed ``required`` is the value sent, so both
        configurers agree it is the one to enforce."""
        with (
            handler(MockCompletionHandler([make_text_response("just prose")])),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model", tool_choice="auto")),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model", tool_choice="required")),
            pytest.raises(ResultDecodingError, match="YOU MUST GENERATE A TOOL CALL"),
        ):
            call_assistant([], response_type=str, env={"add_numbers": add_numbers})

    def test_prose_answer_is_rejected_while_streaming(self):
        """The check must survive `RichTerminalRenderer` sitting between the
        provider and the model: the renderer forces streaming, so all the
        provider gets back is chunks, and the violation is only visible once
        they are reassembled."""
        with (
            handler(_StreamingMockCompletionHandler("just prose")),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model", tool_choice="required")),
            handler(RichTerminalRenderer()),
            pytest.raises(ResultDecodingError, match="YOU MUST GENERATE A TOOL CALL"),
        ):
            call_assistant([], response_type=str, env={"add_numbers": add_numbers})


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

    def test_feedback_message_omits_traceback_when_disabled(self):
        """`include_traceback=False` yields feedback with no traceback block."""
        raw_tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_abc",
                "function": {"name": "my_function", "arguments": "{}"},
            }
        )
        error = ToolCallDecodingError(
            original_error=ValueError("invalid value"),
            raw_message={"role": "assistant"},
            raw_tool_call=raw_tool_call,
        )

        with_tb = error.to_feedback_message(include_traceback=True)["content"]
        without_tb = error.to_feedback_message(include_traceback=False)["content"]

        assert "Traceback:" in with_tb
        assert "Traceback:" not in without_tb
        assert "my_function" in without_tb


# ============================================================================
# Callable Synthesis Tests
# ============================================================================


@Skill.define
def synthesize_adder() -> Callable[[int, int], int]:
    """Generate a Python function that adds two integers together.

    The function should take two integer parameters and return their sum.
    """
    raise NotHandled


@Skill.define
def synthesize_string_processor() -> Callable[[str], str]:
    """Generate a Python function that converts a string to uppercase
    and adds exclamation marks at the end.
    """
    raise NotHandled


@Skill.define
def synthesize_counter(char: str) -> Callable[[str], int]:
    """Generate a Python function that counts occurrences of the character '{char}'
    in a given input string.

    The function should be case-sensitive.
    """
    raise NotHandled


@Skill.define
def synthesize_is_even() -> Callable[[int], bool]:
    """Generate a Python function that checks if a number is even.

    Return True if the number is divisible by 2, False otherwise.
    """
    raise NotHandled


@Skill.define
def synthesize_three_param_func() -> Callable[[int, int, int], int]:
    """Generate a Python function that takes exactly three integer parameters
    and returns their product (multiplication).
    """
    raise NotHandled


class _MethodSynthesizer:
    """A class whose *method* is a Skill returning a Callable, so its synthesis
    anchor is a bound-method ``__default__`` (eb8680: bound-method Skills)."""

    @Skill.define
    def make_adder(self) -> Callable[[int, int], int]:
        """Generate a Python function that adds two integers together.

        The function should take two integer parameters and return their sum.
        Return a code block whose last definition is the function.
        """
        raise NotHandled


class TestCallableSynthesis:
    """Tests for synthesizing callable functions via LLM."""

    def test_synthesize_adder_function(self, request):
        """Test that LLM can synthesize a simple addition function with correct signature."""
        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            add_func = synthesize_adder()

            assert callable(add_func)
            assert add_func(2, 3) == 5
            assert add_func(0, 0) == 0
            assert add_func(-1, 1) == 0
            assert add_func(100, 200) == 300

    def test_synthesize_via_bound_method(self, request):
        """A *method* Skill synthesizes a callable end-to-end -- exercising the
        bound-method `__default__` anchor through the real splice type-check
        (TenacityRetryer lets the model recover from a malformed first draft)."""
        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(TenacityRetryer(stop=tenacity.stop_after_attempt(4))),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(LimitLLMCallsHandler(max_calls=4)),
        ):
            add_func = _MethodSynthesizer().make_adder()

            assert callable(add_func)
            assert add_func(2, 3) == 5
            assert add_func(-1, 1) == 0
            assert add_func(100, 200) == 300

    def test_synthesize_string_processor(self, request):
        """Test that LLM can synthesize a string processing function."""
        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            process_func = synthesize_string_processor()

            assert callable(process_func)
            result = process_func("hello")
            assert isinstance(result, str)
            assert "HELLO" in result
            assert "!" in result

    def test_synthesize_counter_with_parameter(self, request):
        """Test that LLM can synthesize a parameterized counting function."""
        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(LimitLLMCallsHandler(max_calls=3)),
        ):
            count_a = synthesize_counter("a")

            assert callable(count_a)
            assert count_a("banana") == 3
            assert count_a("cherry") == 0
            assert count_a("aardvark") == 3
            assert count_a("AAA") == 0  # case-sensitive

    def test_synthesized_function_roundtrip(self, request):
        """Test that a synthesized function can be encoded and decoded."""

        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            # Synthesize a function
            add_func = synthesize_adder()
            assert callable(add_func)

            # Encode it back to source
            adapter = pydantic.TypeAdapter(Encodable[Callable[[int, int], int]])
            encoded = adapter.dump_python(add_func, mode="json")
            assert isinstance(encoded, str)
            assert "def " in encoded

            # Decode it again and verify it still works
            decoded = adapter.validate_python(encoded)
            assert callable(decoded)
            assert decoded(5, 7) == 12

    def test_synthesize_bool_return_type(self, request):
        """Test that LLM respects bool return type in signature."""

        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
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

    def test_synthesize_three_params(self, request):
        """Test that LLM respects the exact number of parameters in signature."""

        with (
            handler(ReplayLiteLLMProvider(request, model=EFFECTFUL_LLM_MODEL)),
            handler(LexicalToolExtractor()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
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


def make_write_and_run_body_response(
    code: str,
    tool_call_id: str = "call_1",
    compact: CompactionScope = CompactionScope.NONE,
) -> ModelResponse:
    """A tool-call response in which the model finalizes by calling the
    synthesis ``write_and_run_body`` tool with a function it wrote."""
    return make_tool_call_response(
        "write_and_run_body",
        json.dumps({"implementation": code, "compact": compact}),
        tool_call_id=tool_call_id,
    )


@Skill.define
def double_it(x: int) -> int:
    """Return double the integer {x}."""
    raise NotHandled


class _Doubler(Agent):
    @Skill.define
    def double(self, x: int) -> int:
        """Return double the integer {x}."""
        raise NotHandled


class TestSynthesizeAndCall:
    """Tests for the SynthesizeAndCall handler, which answers a Skill by
    exposing a ``write_and_run_body`` tool that the model calls with a synthesized
    function; the function is applied to the original arguments, its value is the
    result, and the handler's `call_tool` rule marks the call final."""

    def test_returns_called_result(self):
        """The Skill result is the value of applying the synthesized function
        to the original arguments, not the function itself."""
        mock = MockCompletionHandler(
            [
                make_write_and_run_body_response(
                    "def double_it(x: int) -> int:\n    return x * 2\n"
                )
            ]
        )
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
        ):
            result = double_it(21)

        assert result == 42
        assert mock.call_count == 1

    def test_default_clear_still_finalizes(self):
        """A successful submission ends the call whatever its ``compact`` says.

        Finalizing and compacting are separate decisions, and only the first may
        gate ``is_final``. Testing the default scope specifically because it is
        the one a conflated condition drops: ``compact="none"`` compacts nothing,
        so a rule that finalizes only when it compacted leaves every ordinary
        submission unfinalized.

        The second response is unreachable and exists to make that failure
        *fail*: `MockCompletionHandler` repeats its last response forever, so a
        loop that will not terminate hangs the suite rather than reporting
        anything. With a second response the loop consumes it, answers "99", and
        the assertions below say so.
        """
        mock = MockCompletionHandler(
            [
                make_write_and_run_body_response(
                    "def double_it(x: int) -> int:\n    return x * 2\n",
                    compact=CompactionScope.NONE,
                ),
                make_text_response(json.dumps({"value": 99})),
            ]
        )
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(HistoryBuilder()),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
        ):
            result = double_it(21)

        assert result == 42
        assert mock.call_count == 1

    def test_value_recorded_as_tool_message(self):
        """The computed value enters history as a tool result, and is never
        fabricated as assistant content."""
        agent = _Doubler()
        mock = MockCompletionHandler(
            [
                make_write_and_run_body_response(
                    "def double(self, x: int) -> int:\n    return x * 2\n"
                )
            ]
        )
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(HistoryBuilder()),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
        ):
            result = agent.double(21)

        assert result == 42
        messages = list(agent.__history__)
        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert tool_messages, "computed value should be recorded as a tool result"
        assert "42" in str(tool_messages[-1]["content"])
        # The model never generated the value itself.
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert all("42" not in str(m.get("content") or "") for m in assistant_messages)

    def test_clear_keeps_the_finalizing_round(self):
        """``write_and_run_body(compact="conversation")`` compacts and still
        finalizes, leaving a history that ends on an answer.

        A finalizing call ends the loop the moment it succeeds, so there is no
        later turn to append anything: a compaction that dropped this round would
        leave the durable history ending on a request nobody answered, and the
        next call would put a second request straight after it. Keeping the round
        also keeps the submitted source, which is where a model is told to leave
        notes for its later self.
        """
        agent = _Doubler()
        mock = MockCompletionHandler(
            [
                make_write_and_run_body_response(
                    "def double(self, x: int) -> int:\n    return x * 2\n"
                ),
                make_write_and_run_body_response(
                    "def double(self, x: int) -> int:\n"
                    "    # NOTE-TO-SELF: doubling is just x * 2\n"
                    "    return x * 2\n",
                    tool_call_id="call_2",
                    compact=CompactionScope.CONVERSATION,
                ),
                make_write_and_run_body_response(
                    "def double(self, x: int) -> int:\n    return x * 2\n",
                    tool_call_id="call_3",
                ),
            ]
        )
        stack = (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(HistoryBuilder()),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
        )
        with contextlib.ExitStack() as ctx:
            for h in stack:
                ctx.enter_context(h)
            assert agent.double(1) == 2
            assert agent.double(21) == 42
            # A third call still assembles a well-formed request over the
            # compacted history rather than stacking a second user message on an
            # unanswered one.
            assert agent.double(5) == 10

        history = list(agent.__history__)
        # The first call is gone; the second's round survived whole, and the
        # third appended on top of it.
        assert [m["role"] for m in history] == [
            "system",
            "user",
            "assistant",
            "tool",
            "user",
            "assistant",
            "tool",
        ]
        assert "NOTE-TO-SELF" in json.dumps(history[2])
        assert "42" in str(history[3]["content"])

    def test_direct_structured_answer_is_allowed(self):
        """The synthesis tool is offered alongside, not instead of, direct
        structured output: the model may answer the return type directly."""
        mock = MockCompletionHandler([make_text_response(json.dumps({"value": 99}))])
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
        ):
            result = double_it(21)

        assert result == 99
        assert mock.call_count == 1

    def test_retries_on_runtime_error(self):
        """A synthesized function that raises when applied to the inputs surfaces
        as a ToolCallExecutionError; TenacityRetryer feeds the error back and the
        loop continues so the model can revise."""
        mock = MockCompletionHandler(
            [
                make_write_and_run_body_response(
                    "def double_it(x: int) -> int:\n    return x // 0\n",
                    tool_call_id="call_bad",
                ),
                make_write_and_run_body_response(
                    "def double_it(x: int) -> int:\n    return x * 2\n",
                    tool_call_id="call_good",
                ),
            ]
        )
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
            handler(TenacityRetryer()),
        ):
            result = double_it(21)

        assert result == 42
        assert mock.call_count == 2

    def test_normal_tool_calls_do_not_terminate(self):
        """A non-final tool call is fed back and the loop continues; only the
        ``write_and_run_body`` call terminates."""
        mock = MockCompletionHandler(
            [
                make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
                make_write_and_run_body_response(
                    "def double_it(x: int) -> int:\n    return x * 2\n"
                ),
            ]
        )
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
        ):
            # add_numbers is in scope as a lexical tool
            result = double_it(21)

        assert result == 42
        assert mock.call_count == 2

    def test_write_and_run_body_mixed_with_normal_call_is_rejected(self):
        """A finalizing call must be the only call in its turn: which call in a
        mixed turn is the answer is ambiguous, so the completion loop asserts
        rather than letting the trailing call overwrite the answer."""
        mixed = ModelResponse(
            id="test",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_submit",
                                "type": "function",
                                "function": {
                                    "name": "write_and_run_body",
                                    "arguments": json.dumps(
                                        {
                                            "implementation": "def double_it(x: int) -> int:\n    return x * 2\n"
                                        }
                                    ),
                                },
                            },
                            {
                                "id": "call_add",
                                "type": "function",
                                "function": {
                                    "name": "add_numbers",
                                    "arguments": '{"a": 1, "b": 2}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            model="test-model",
        )
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(MockCompletionHandler([mixed])),
        ):
            with pytest.raises(AssertionError, match="only call in its turn"):
                double_it(21)

    def test_rejects_variadic_parameters(self):
        """A signature with *args/**kwargs cannot be expressed as a Callable type,
        so building the synthesis tool for it is rejected."""

        @Skill.define
        def variadic(*args: int) -> int:
            """Sum the arguments."""
            raise NotHandled

        with pytest.raises(TypeError, match="variadic"):
            FinalBodySynthesizer._SubmitSolutionTool.define(
                variadic, variadic.__signature__.bind()
            )

    def test_implementation_is_advertised_as_a_bare_string(self):
        """The tool the model actually sees takes source as a JSON string, with no
        object to assemble and no `$ref` to resolve (#775)."""

        @Skill.define
        def add(a: int, b: int) -> int:
            """Add {a} and {b}."""
            raise NotHandled

        tool = FinalBodySynthesizer._SubmitSolutionTool.define(
            add, add.__signature__.bind(1, 2)
        )
        advertised = pydantic.TypeAdapter(Encodable[_NameAndTool]).dump_python(
            _NameAndTool("write_and_run_body", tool), mode="json", context={}
        )
        implementation = advertised["function"]["parameters"]["properties"][
            "implementation"
        ]
        assert implementation["type"] == "string"
        assert "$ref" not in json.dumps(implementation)


class TestSynthesizeAndCallDoctests:
    """SynthesizeAndCall validates the synthesized function against the
    Skill's own docstring doctests (#433), rerouting Skill calls in the
    doctests to the synthesized function so they never re-synthesize.

    The doctest-bearing Skills are defined *inside* each test rather than at
    module scope so pytest's ``--doctest-modules`` collection does not try to run
    them (they reference a Skill that needs an LLM to resolve)."""

    def test_passes_when_synthesized_function_meets_skill_doctests(self):
        # The Skill's docstring carries the doctests; the synthesized
        # function's OWN docstring is deliberately wrong to prove the Skill's
        # docstring is what gets run.
        @Skill.define
        def triple_it(x: int) -> int:
            """Return triple the integer {x}.

            >>> triple_it(2)
            6
            >>> triple_it(0)
            0
            """
            raise NotHandled

        good = (
            "def impl(x: int) -> int:\n"
            '    """>>> triple_it(2)\n'
            "    999\n"
            '    """\n'
            "    return x * 3\n"
        )
        mock = MockCompletionHandler([make_write_and_run_body_response(good)])
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(HistoryBuilder()),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
        ):
            result = triple_it(2)

        assert result == 6
        # A single completion: the doctest's `triple_it(...)` calls dispatched to
        # the synthesized function, never re-entering synthesis.
        assert mock.call_count == 1

    def test_rejects_then_retries_when_doctests_fail(self):
        @Skill.define
        def triple_it(x: int) -> int:
            """Return triple the integer {x}.

            >>> triple_it(2)
            6
            """
            raise NotHandled

        bad = "def impl(x: int) -> int:\n    return x * 2\n"  # doubles, not triples
        good = "def impl(x: int) -> int:\n    return x * 3\n"
        mock = MockCompletionHandler(
            [
                make_write_and_run_body_response(bad, tool_call_id="bad"),
                make_write_and_run_body_response(good, tool_call_id="good"),
            ]
        )
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(HistoryBuilder()),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
            handler(TenacityRetryer()),
        ):
            result = triple_it(2)

        assert result == 6
        assert mock.call_count == 2

    def test_skill_without_doctests_synthesizes_normally(self):
        @Skill.define
        def triple_it(x: int) -> int:
            """Return triple the integer {x}."""
            raise NotHandled

        good = "def impl(x: int) -> int:\n    return x * 3\n"
        mock = MockCompletionHandler([make_write_and_run_body_response(good)])
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(HistoryBuilder()),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
        ):
            result = triple_it(2)

        assert result == 6
        assert mock.call_count == 1

    def test_agent_method_doctests_route_to_synthesized_function(self):
        """An Agent-method Skill's doctests build their own instances
        (``agent = Doubler()``), so each ``agent.double(...)`` call dispatches a
        *fresh* per-instance op -- distinct from the one that triggered
        synthesis.  Matching on the shared class-level skill reroutes every
        such call to the synthesized function (with the instance passed as
        ``self``), so the doctests validate the synthesized code instead of
        re-synthesizing or hitting the LLM."""

        class Doubler(Agent):
            @Skill.define
            def double(self, x: int) -> int:
                """Return double the integer {x}.

                >>> agent = Doubler()
                >>> agent.double(2)
                4
                >>> agent.double(0)
                0
                """
                raise NotHandled

        # A drop-in syntactic replacement for the method body keeps `self` in the
        # signature; the synthesized function's OWN docstring is deliberately
        # wrong to prove the Skill's docstring is what gets run.
        good = (
            "def double(self, x: int) -> int:\n"
            '    """>>> never\n'
            "    run\n"
            '    """\n'
            "    return x * 2\n"
        )
        agent = Doubler()
        mock = MockCompletionHandler([make_write_and_run_body_response(good)])
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test-model")),
            handler(HistoryBuilder()),
            handler(FinalBodySynthesizer()),
            handler(TyTypeChecker()),
            handler(BuiltinExecutor()),
            handler(mock),
        ):
            result = agent.double(21)

        assert result == 42
        # A single completion: the doctest's `agent.double(...)` calls on a fresh
        # instance were answered by the synthesized function, never re-entering
        # synthesis.
        assert mock.call_count == 1


class TestMessageSequence:
    """Tests for MessageSequence message sequence tracking."""

    def test_append_message_rejects_consecutive_assistant_messages(self):
        """The model may not speak twice in a row.

        The pair is legal to *build* and illegal to send: a provider that merges
        consecutive assistant messages reads the result as a prefill, so the
        complaint arrives a request later and names nothing. `append_message` is
        the choke point every recorded message passes through, so the check is
        there.
        """
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        with handler({HistoryBuilder.get_history: lambda: history}):
            with pytest.raises(AssertionError):
                HistoryBuilder.append_message(
                    {"role": "assistant", "content": "hi again"}
                )
            HistoryBuilder.append_message({"role": "user", "content": "still there?"})
            HistoryBuilder.append_message({"role": "assistant", "content": "hi again"})

        assert [m["role"] for m in history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    def test_call_tool_sees_outer_message_sequence(self):
        """call_tool should not isolate; the tool sees the outer message sequence."""
        # Pre-populate the current frame with existing messages
        message_sequence = [
            {"role": "user", "content": "hello"},
            # The assistant turn must actually request `tc_1`: HistoryBuilder
            # rejects a tool message that answers no outstanding tool call.
            {
                "role": "assistant",
                "content": "hi",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "add_numbers", "arguments": "{}"},
                    }
                ],
            },
        ]

        captured_frame = None

        class InnerToolHandler(ObjectInterpretation):
            @implements(add_numbers)
            def _add_numbers(self, *args, **kwargs):
                # Capture the state of the message sequence during execution
                nonlocal captured_frame
                captured_frame = list(HistoryBuilder.get_history())
                return 42

        mock_tool_call = DecodedToolCall(
            tool=add_numbers,
            bound_args=inspect.signature(add_numbers).bind(1, 2),
            id="tc_1",
            name="add_numbers",
        )

        with (
            handler(HistoryBuilder()),
            handler(InnerToolHandler()),
            handler({HistoryBuilder.get_history: lambda: message_sequence}),
        ):
            call_tool(mock_tool_call)

        # Tool sees the outer message sequence (2 pre-populated messages)
        assert len(captured_frame) == 2
        # Tool response is appended to the same outer sequence
        assert len(message_sequence) == 3

    def test_call_assistant_no_duplicate_messages(self):
        """call_assistant should send its `messages` argument on verbatim."""
        # Pre-populate frame with two messages
        message_sequence = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        captured_messages = []

        class InnerAssistantHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self_, messages=None, *args, **kwargs):
                captured_messages.extend(list(messages))
                response = {
                    "role": "assistant",
                    "content": json.dumps({"value": "result"}),
                }
                return ModelResponse(
                    choices=[{"role": "assistant", "message": response}]
                )

        with (
            handler(InnerAssistantHandler()),
            handler({HistoryBuilder.get_history: lambda: message_sequence}),
        ):
            call_assistant(
                list(message_sequence),
                response_type=str,
                env={},
            )

        # Forwarded messages are exactly the ones passed in — no duplicates
        assert captured_messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_call_assistant_no_duplicates_across_multiple_calls(self):
        """Calling call_assistant multiple times should never produce duplicate messages."""

        message_sequence = [{"role": "user", "content": "hello"}]

        call_log = []

        class InnerAssistantHandler(ObjectInterpretation):
            call_count = 0

            @implements(completion)
            def _completion(self_, messages=None, *args, **kwargs):
                call_log.append([m["content"] for m in messages])
                self_.call_count += 1
                response = {
                    "role": "assistant",
                    "content": json.dumps({"value": "result"}),
                }
                return ModelResponse(
                    choices=[{"role": "assistant", "message": response}]
                )

        inner = InnerAssistantHandler()

        with (
            handler(HistoryBuilder()),
            handler(inner),
            handler({HistoryBuilder.get_history: lambda: message_sequence}),
        ):
            resp1, _, _ = call_assistant(
                list(message_sequence),
                response_type=str,
                env={},
            )
            # A second reply has to be a reply to something, since HistoryBuilder
            # rejects an assistant message that follows another.
            HistoryBuilder.append_message({"role": "user", "content": "again"})
            # HistoryBuilder appended the first response, so the second call
            # sends it along with the messages that preceded it.
            resp2, _, _ = call_assistant(
                list(message_sequence),
                response_type=str,
                env={},
            )

        answer = json.dumps({"value": "result"})
        assert call_log[0] == ["hello"]
        assert call_log[1] == ["hello", answer, "again"]

    def test_call_assistant_saves_only_on_successful_fwd(self):
        """call_assistant should only save the response message to the frame when fwd() succeeds."""
        message_sequence = []

        class FailingAssistantHandler(ObjectInterpretation):
            @implements(call_assistant)
            def _call_assistant(self_, *args, **kwargs):
                raise RuntimeError("LLM call failed")

        frame_snapshot = list(message_sequence)

        with pytest.raises(RuntimeError, match="LLM call failed"):
            with (
                handler(HistoryBuilder()),
                handler(FailingAssistantHandler()),
                handler({HistoryBuilder.get_history: lambda: message_sequence}),
            ):
                call_assistant(
                    list(message_sequence),
                    response_type=str,
                    env={},
                )

        # Frame should be unchanged — no response message was saved
        assert list(message_sequence) == frame_snapshot


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


def _drive_repl(body):
    """Run ``body(exec_code)`` inside one `PythonRepl`-scoped Skill call.

    A tiny `call_agent` handler stands in for the LLM loop, handing
    `body` the call's `exec_code` tool so it runs against one REPL session (the
    supported way to reach a session).  Install any outer handlers (e.g.
    `TenacityRetryer`) around the call.  Returns `body`'s result.
    """
    box = []
    repl = StatefulReplSynthesizer()

    class _Loop(ObjectInterpretation):
        @implements(call_agent)
        def _call(self, *_a, **_k):
            box.append(body(repl.exec_code))
            return None

    @Skill.define
    def _t() -> None:
        """Drive one REPL-scoped call."""
        raise NotImplementedError

    with (
        handler(TyTypeChecker()),
        handler(_Loop()),
        handler(BuiltinExecutor()),
        handler(repl),
    ):
        _t()
    return box[0]


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

        result, _, _ = call_tool(tc)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_ok"

    def test_call_tool_wraps_exec_code_error_and_retryer_feeds_it_back(self):
        """An `exec_code` runtime error propagates out of the session and is
        wrapped like any other failing tool -- and the REPL experience of
        "traceback returned, loop continues" is supplied by composition:
        `TenacityRetryer` converts the raise into a tool feedback message. The
        session is not the resilient layer; the stack is."""

        def bare(exec_code):
            bound_args = inspect.signature(exec_code).bind(
                pydantic.TypeAdapter(Encodable[CodeType]).validate_python("1 / 0")
            )
            tc = DecodedToolCall(exec_code, bound_args, "call_exec", "exec_code")
            with pytest.raises(ToolCallExecutionError) as exc_info:
                call_tool(tc)
            return exc_info.value

        err = _drive_repl(bare)
        assert isinstance(err.original_error, ZeroDivisionError)

        def with_retryer(exec_code):
            bound_args = inspect.signature(exec_code).bind(
                pydantic.TypeAdapter(Encodable[CodeType]).validate_python("1 / 0")
            )
            tc = DecodedToolCall(exec_code, bound_args, "call_exec", "exec_code")
            with handler(TenacityRetryer()):
                return call_tool(tc)[0]

        msg = _drive_repl(with_retryer)
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_exec"
        assert "ZeroDivisionError" in str(msg["content"])


class TestRetryHandlerCatchToolErrorsFiltering:
    """TenacityRetryer should only catch tool errors matching catch_tool_errors."""

    def test_matching_error_returns_feedback_message(self):
        """When original_error matches catch_tool_errors, return error feedback."""
        sig = inspect.signature(flaky_tool)
        bound_args = sig.bind(x=1)
        tc = DecodedToolCall(flaky_tool, bound_args, "call_match", "flaky_tool")

        with handler(TenacityRetryer(catch_tool_errors=ConnectionError)):
            result, _, _ = call_tool(tc)

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
            with handler(TenacityRetryer(catch_tool_errors=TypeError)):
                call_tool(tc)

        assert isinstance(exc_info.value.original_error, ConnectionError)

    def test_default_catch_all_catches_everything(self):
        """Default catch_tool_errors=Exception catches all standard exceptions."""
        sig = inspect.signature(type_error_tool)
        bound_args = sig.bind(x=5)
        tc = DecodedToolCall(
            type_error_tool, bound_args, "call_default", "type_error_tool"
        )

        with handler(TenacityRetryer()):
            result, _, _ = call_tool(tc)

        assert result["role"] == "tool"
        assert "Tool execution failed" in result["content"]

    def test_tuple_of_error_types(self):
        """catch_tool_errors accepts a tuple of exception types."""
        sig = inspect.signature(flaky_tool)
        bound_args = sig.bind(x=1)
        tc = DecodedToolCall(flaky_tool, bound_args, "call_tuple", "flaky_tool")

        with handler(
            TenacityRetryer(
                catch_tool_errors=(ConnectionError, ValueError),
            )
        ):
            result, _, _ = call_tool(tc)

        assert result["role"] == "tool"
        assert "Tool execution failed" in result["content"]

    def test_no_retry_handler_propagates_execution_error(self):
        """Without TenacityRetryer, ToolCallExecutionError propagates directly."""
        sig = inspect.signature(failing_tool)
        bound_args = sig.bind(x=1)
        tc = DecodedToolCall(failing_tool, bound_args, "call_no_retry", "failing_tool")

        with pytest.raises(ToolCallExecutionError):
            call_tool(tc)


class TestLiteLLMProviderMessagePruning:
    """`AgentLoop` should prune messages added during a failed skill call."""

    def test_messages_pruned_on_tool_execution_error(self):
        """When a tool error propagates, all messages from that call are pruned."""
        # LLM says "call flaky_tool", then tool raises unhandled error
        responses = [
            make_tool_call_response("flaky_tool", '{"x": 1}'),
        ]
        mock_handler = MockCompletionHandler(responses)

        message_sequence = []

        @Skill.define
        def task_with_flaky_tool(instruction: str) -> str:
            """Do: {instruction}"""
            raise NotHandled

        with pytest.raises(ToolCallExecutionError):
            with (
                handler(AgentLoop()),
                handler(LexicalToolExtractor()),
                handler(LiteLLMConfigurer(model="test")),
                handler(mock_handler),
                handler({HistoryBuilder.get_history: lambda: message_sequence}),
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

        message_sequence = []

        @Skill.define
        def task_with_tools(instruction: str) -> str:
            """Do: {instruction}"""
            raise NotHandled

        with pytest.raises(ToolCallDecodingError):
            with (
                handler(AgentLoop()),
                handler(LexicalToolExtractor()),
                handler(LiteLLMConfigurer(model="test")),
                handler(mock_handler),
                handler({HistoryBuilder.get_history: lambda: message_sequence}),
            ):
                task_with_tools("go")

        assert len(message_sequence) == 0

    def test_pre_existing_messages_preserved_on_error(self):
        """Pre-existing messages in the sequence are not pruned when a call fails."""
        responses = [
            make_tool_call_response("flaky_tool", '{"x": 1}'),
        ]
        mock_handler = MockCompletionHandler(responses)

        message_sequence = [{"role": "user", "content": "hello"}]

        @Skill.define
        def task_with_flaky_tool(instruction: str) -> str:
            """Do: {instruction}"""
            raise NotHandled

        with pytest.raises(ToolCallExecutionError):
            with (
                handler(AgentLoop()),
                handler(LexicalToolExtractor()),
                handler(LiteLLMConfigurer(model="test")),
                handler(mock_handler),
                handler({HistoryBuilder.get_history: lambda: message_sequence}),
            ):
                task_with_flaky_tool("go")

        # Pre-existing message should still be there
        assert message_sequence == [{"role": "user", "content": "hello"}]

    def test_successful_call_preserves_messages(self):
        """A successful top-level skill call should write messages back to Agent history."""
        responses = [make_text_response("done")]
        mock_handler = MockCompletionHandler(responses)

        class SimpleAgent(Agent):
            """You are a persistence-check test agent.
            Your goal is to complete `simple_task` and persist successful history.
            """

            @Skill.define
            def simple_task(self, instruction: str) -> str:
                """Do: {instruction}"""
                raise NotHandled

        agent = SimpleAgent()

        # No enclosing transaction: HistoryBuilder detects this is the outermost
        # call for the agent and writes back to its __history__.
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test")),
            handler(HistoryBuilder()),
            handler(mock_handler),
        ):
            result = agent.simple_task("go")

        assert result == "done"
        # Agent's __history__ should have messages written back (system + user + assistant)
        assert len(agent.__history__) >= 2


class TestAgentCrossSkillRecovery:
    """Issue #558: Agent should recover from errored tool calls across skill methods.

    When a tool call fails and the error propagates (not caught by TenacityRetryer),
    the agent's message history must be cleaned up so subsequent skill calls
    don't fail due to orphaned assistant tool_calls messages.
    """

    def test_agent_second_call_succeeds_after_tool_error(self):
        """After a tool error in one skill, another skill on the same agent works."""

        @Tool.define
        def bad_service() -> str:
            """Fetch from a broken service."""
            raise ConnectionError("service down")

        import dataclasses

        @dataclasses.dataclass
        class TestAgent(Agent):
            """You are a cross-skill recovery test agent.
            Your goal is to recover from failed tool calls across skill methods.
            """

            @Skill.define
            def step_with_tool(self, task: str) -> str:
                """Use bad_service for: {task}"""
                raise NotHandled

            @Skill.define
            def step_no_tool(self, topic: str) -> str:
                """Summarize: {topic}. Do not use any tools."""
                raise NotHandled

        # Step 1: LLM calls bad_service → tool error propagates
        tool_call_response = make_tool_call_response("bad_service", "{}")
        # Step 2: Simple text response for the second skill
        text_response = make_text_response("summary result")

        call_count = 0

        class TwoPhaseCompletionHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self, messages=None, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return tool_call_response
                return text_response

        agent = TestAgent()

        with handler(TwoPhaseCompletionHandler()):
            with (
                handler(AgentLoop()),
                handler(LexicalToolExtractor()),
                handler(LiteLLMConfigurer(model="test")),
                handler(HistoryBuilder()),
            ):
                # First call should fail with tool execution error
                with pytest.raises(ToolCallExecutionError):
                    agent.step_with_tool("stage 1")

                # History should be clean — no orphaned tool_calls
                # Second call should succeed without BadRequestError
                result = agent.step_no_tool("stage 2")

        assert result == "summary result"
        # Verify history doesn't contain messages from the failed call
        history = agent.__history__
        for msg in history:
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # If there's an assistant message with tool_calls, there must be
                # corresponding tool responses
                for tc in tool_calls:
                    tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                    has_response = any(
                        m.get("tool_call_id") == tc_id
                        for m in history
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

            @Skill.define
            def do_work(self, task: str) -> str:
                """Do: {task}"""
                raise NotHandled

        responses = [make_tool_call_response("exploding_tool", "{}")]
        mock = MockCompletionHandler(responses)
        agent = CleanupAgent()

        with pytest.raises(ToolCallExecutionError):
            with (
                handler(AgentLoop()),
                handler(LexicalToolExtractor()),
                handler(LiteLLMConfigurer(model="test")),
                handler(HistoryBuilder()),
                handler(mock),
            ):
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

            @Skill.define
            def greet(self, name: str) -> str:
                """Say hello to {name}."""
                raise NotHandled

        responses = [make_text_response("Hello!")]
        mock = MockCompletionHandler(responses)
        agent = SuccessAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test")),
            handler(HistoryBuilder()),
            handler(mock),
        ):
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

            @Skill.define
            def chat(self, msg: str) -> str:
                """Respond to: {msg}"""
                raise NotHandled

        call_count = 0

        class MultiResponseHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self, messages=None, **kwargs):
                nonlocal call_count
                call_count += 1
                return make_text_response(f"reply {call_count}")

        agent = ChatAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test")),
            handler(HistoryBuilder()),
            handler(MultiResponseHandler()),
        ):
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

            @Skill.define
            def risky(self, task: str) -> str:
                """Do risky: {task}"""
                raise NotHandled

            @Skill.define
            def safe(self, task: str) -> str:
                """Do safe: {task}. Do not use tools."""
                raise NotHandled

        call_count = 0

        class PhaseHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self, messages=None, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return make_tool_call_response("broken_tool", "{}")
                return make_text_response("safe result")

        agent = RecoveryAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test")),
            handler(HistoryBuilder()),
            handler(PhaseHandler()),
        ):
            with pytest.raises(ToolCallExecutionError):
                agent.risky("step 1")

            history_after_error = len(agent.__history__)
            assert history_after_error == 0

            result = agent.safe("step 2")

        assert result == "safe result"
        # Only messages from the successful call should be in history
        assert len(agent.__history__) >= 2
        assert len(agent.__history__) > history_after_error


class TestAgentSystemMessageDeduplication:
    """Regression tests for system message duplication bug.

    When AgentLoop.call_agent copies the history, call_system replaces the
    system message in the copy. Previously, history.update(history_copy) was
    used to merge back, which is additive — it didn't remove the stale system
    message key deleted from the copy. This caused multiple system messages to
    accumulate, triggering an assertion on the 3rd+ call.

    The fix is history.clear() before history.update(history_copy).
    """

    def test_three_consecutive_calls_no_system_message_duplication(self):
        """Three consecutive agent calls should not fail with duplicate system messages."""
        import dataclasses

        @dataclasses.dataclass
        class ThreeCallAgent(Agent):
            """You are a test agent for system message deduplication."""

            @Skill.define
            def ask(self, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        call_count = 0

        class CountingHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self, messages=None, **kwargs):
                nonlocal call_count
                call_count += 1
                return make_text_response(f"answer {call_count}")

        agent = ThreeCallAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test")),
            handler(HistoryBuilder()),
            handler(CountingHandler()),
        ):
            r1 = agent.ask("q1")
            r2 = agent.ask("q2")
            r3 = agent.ask("q3")

        assert r1 == "answer 1"
        assert r2 == "answer 2"
        assert r3 == "answer 3"

    def test_history_has_exactly_one_system_message_after_multiple_calls(self):
        """After multiple calls, the agent history should contain exactly one system message."""
        import dataclasses

        @dataclasses.dataclass
        class SystemMsgAgent(Agent):
            """You are a system message count test agent."""

            @Skill.define
            def do(self, task: str) -> str:
                """Do: {task}"""
                raise NotHandled

        call_count = 0

        class MultiHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self, messages=None, **kwargs):
                nonlocal call_count
                call_count += 1
                return make_text_response(f"done {call_count}")

        agent = SystemMsgAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test")),
            handler(HistoryBuilder()),
            handler(MultiHandler()),
        ):
            agent.do("a")
            agent.do("b")
            agent.do("c")
            agent.do("d")

        system_msgs = [m for m in agent.__history__ if m["role"] == "system"]
        assert len(system_msgs) == 1, (
            f"Expected exactly 1 system message, got {len(system_msgs)}"
        )

    def test_conversation_history_preserved_across_calls(self):
        """Earlier user/assistant messages should persist across multiple calls."""
        import dataclasses

        @dataclasses.dataclass
        class MemoryAgent(Agent):
            """You are a memory test agent."""

            @Skill.define
            def chat(self, msg: str) -> str:
                """User says: {msg}"""
                raise NotHandled

        call_count = 0

        class MemoryHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self, messages=None, **kwargs):
                nonlocal call_count
                call_count += 1
                # Verify that previous messages are visible to later calls
                if call_count == 3:
                    # Third call should see messages from calls 1 and 2
                    user_msgs = [m for m in messages if m["role"] == "user"]
                    assert len(user_msgs) == 3, (
                        f"Third call should see 3 user messages, got {len(user_msgs)}"
                    )
                return make_text_response(f"reply {call_count}")

        agent = MemoryAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test")),
            handler(HistoryBuilder()),
            handler(MemoryHandler()),
        ):
            agent.chat("first")
            agent.chat("second")
            agent.chat("third")

        # History should have: 1 system + 3 user + 3 assistant = 7
        assert len(agent.__history__) == 7
        roles = [m["role"] for m in agent.__history__]
        assert roles.count("system") == 1
        assert roles.count("user") == 3
        assert roles.count("assistant") == 3

    def test_system_message_is_always_first(self):
        """The system message should remain the first message after multiple calls."""
        import dataclasses

        @dataclasses.dataclass
        class OrderAgent(Agent):
            """You are a message order test agent."""

            @Skill.define
            def step(self, n: int) -> str:
                """Step {n}"""
                raise NotHandled

        call_count = 0

        class OrderHandler(ObjectInterpretation):
            @implements(completion)
            def _completion(self, messages=None, **kwargs):
                nonlocal call_count
                call_count += 1
                return make_text_response(f"step {call_count}")

        agent = OrderAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="test")),
            handler(HistoryBuilder()),
            handler(OrderHandler()),
        ):
            agent.step(1)
            agent.step(2)
            agent.step(3)

        messages = list(agent.__history__)
        assert messages[0]["role"] == "system", (
            "System message should be the first message in history"
        )


# ============================================================================
# Prompt Caching Tests
# ============================================================================


def _has_cache_control(msg: dict) -> bool:
    """Check if a message is marked for prompt caching.

    Either form litellm accepts counts: a `cache_control` key on a content block,
    or one on the message itself -- which is how a message whose content is a
    plain string (the assembled Markdown system prompt) carries it. See
    `test_anthropic_receives_cache_control_from_message_level_key` for the
    litellm transformation that consumes the message-level form.
    """
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and "cache_control" in b for b in content)
    return "cache_control" in msg


def _has_block_cache_control(msg: dict) -> bool:
    """Check for a breakpoint on a *content block* specifically.

    `LiteLLMConfigurer._add_cache_control` only ever produces this form, so it
    distinguishes what the provider added from the message-level key
    `call_system` puts on the system prompt.
    """
    content = msg.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and "cache_control" in b for b in content
    )


def _assert_valid_anthropic_request(msgs) -> None:
    """Assert `msgs` survives litellm's Anthropic transform as a legal request.

    The block assertions are upstream 400s: ``messages: text content blocks must
    be non-empty`` and ``cache_control cannot be set for empty text blocks``.
    Roles are not required to alternate; Anthropic accepts consecutive same-role
    turns. A turn must not be dropped, which is how a message with nothing in it
    used to disappear from a request.
    """
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    # A deep copy, because the transform rewrites messages in place and `msgs`
    # is the captured request the caller goes on to assert against.
    transformed = AnthropicConfig().transform_request(
        model="claude-sonnet-4-5",
        messages=json.loads(json.dumps(list(msgs))),
        optional_params={},
        litellm_params={},
        headers={},
    )

    def walk(blocks):
        """Every text block in `blocks`, including those nested in a tool_result."""
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                yield block
            elif isinstance(block.get("content"), list):
                yield from walk(block["content"])

    breakpoints = 0
    for message in transformed["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in walk(content):
            assert block.get("text"), (
                f"empty text block sent to Anthropic: {block} in {message}"
            )
        breakpoints += sum(1 for b in content if "cache_control" in b)

    for block in transformed.get("system") or []:
        assert block.get("text"), f"empty system block: {block}"
        breakpoints += "cache_control" in block

    assert breakpoints <= 4, (
        f"Anthropic allows four cache breakpoints per request; got {breakpoints}"
    )

    # Consecutive same-role turns merge, so compare role runs, not counts.
    def runs(roles):
        out = []
        for role in roles:
            if not out or out[-1] != role:
                out.append(role)
        return out

    sent = runs(
        "user" if m["role"] in ("user", "tool") else m["role"]
        for m in msgs
        if m["role"] != "system"
    )
    assert runs(m["role"] for m in transformed["messages"]) == sent, (
        f"a turn was dropped by the Anthropic transform: sent {sent}, "
        f"got {[m['role'] for m in transformed['messages']]}"
    )


def _empty_text_blocks(msgs) -> list:
    """Every empty text block in `msgs`, with whether it carries a breakpoint."""
    return [
        (msg["role"], "cache_control" in block)
        for msg in msgs
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if isinstance(block, dict)
        and block.get("type") == "text"
        and not block.get("text")
    ]


class CachingAgent(Agent):
    """A test agent with persistent history."""

    @Skill.define
    def ask(self, question: str) -> str:
        """You are a helpful assistant. Answer concisely: {question}"""
        raise NotHandled


class TestPromptCaching:
    """Tests that cache_control is present in messages sent to litellm."""

    def test_system_message_has_cache_control(self):
        """System message should include cache_control for prompt caching."""
        capture = MockCompletionHandler([make_text_response("42")])
        provider = LiteLLMConfigurer(model="test")

        # `capture` is terminal (it never forwards), so it goes *below* the
        # provider: LiteLLMConfigurer.completion has to run and forward into
        # it, or the request never gets the provider's config or breakpoint.
        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(provider),
            handler(HistoryBuilder()),
        ):
            simple_prompt("test")

        msgs = capture.received_messages[0]
        system_msgs = [m for m in msgs if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert _has_cache_control(system_msgs[0]), (
            f"System message should have cache_control. Got: {system_msgs[0]}"
        )

    def test_agent_user_message_has_cache_control(self):
        """Agent calls should add cache_control to the last user message."""
        capture = MockCompletionHandler([make_text_response("42")])
        provider = LiteLLMConfigurer(model="test")
        agent = CachingAgent()

        # `capture` is terminal (it never forwards), so it goes *below* the
        # provider: LiteLLMConfigurer.completion has to run and forward into
        # it, or the request never gets the provider's config or breakpoint.
        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(provider),
            handler(HistoryBuilder()),
        ):
            agent.ask("What is 2+2?")

        msgs = capture.received_messages[0]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 1
        content = user_msgs[0]["content"]
        assert isinstance(content, list)
        assert "cache_control" in content[-1], (
            f"Agent user message should have cache_control. Got: {content[-1]}"
        )

    def test_non_agent_user_message_has_cache_control(self):
        """Non-agent calls are marked too: the breakpoint is a transport-level
        concern applied to every request, so a plain Skill's tool-use rounds
        get the same cached prefix an Agent's turns do."""
        capture = MockCompletionHandler([make_text_response("42")])
        provider = LiteLLMConfigurer(model="test")

        # `capture` is terminal (it never forwards), so it goes *below* the
        # provider: LiteLLMConfigurer.completion has to run and forward into
        # it, or the request never gets the provider's config or breakpoint.
        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(provider),
            handler(HistoryBuilder()),
        ):
            simple_prompt("test")

        msgs = capture.received_messages[0]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        content = user_msgs[0]["content"]
        assert isinstance(content, list)
        assert "cache_control" in content[-1], (
            f"User message should have cache_control. Got: {content[-1]}"
        )

    def test_exactly_one_breakpoint_beyond_the_system_message(self):
        """Providers cap cache breakpoints per request (Anthropic allows four),
        so a long exchange must not accumulate one per turn."""
        capture = MockCompletionHandler(
            [
                make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
                make_tool_call_response("add_numbers", '{"a": 3, "b": 4}', "call_2"),
                make_text_response("42"),
            ]
        )
        provider = LiteLLMConfigurer(model="test")
        agent = CachingAgent()

        # `capture` is terminal (it never forwards), so it goes *below* the
        # provider: LiteLLMConfigurer.completion has to run and forward into
        # it, or the request never gets the provider's config or breakpoint.
        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(provider),
            handler(HistoryBuilder()),
        ):
            agent.ask("What is 2+2?")

        # The final request carries the longest history; count its breakpoints.
        msgs = capture.received_messages[-1]
        marked = [m for m in msgs if _has_cache_control(m)]
        assert [m["role"] for m in marked] == ["system", "tool"], (
            f"Expected the system message plus the last input message. Got: {marked}"
        )
        _assert_valid_anthropic_request(msgs)

    def test_breakpoint_advances_to_the_newest_message(self):
        """The breakpoint tracks the end of the conversation across turns, so
        each request extends the cached prefix instead of re-reading a stale one."""
        capture = MockCompletionHandler(
            [make_text_response("first"), make_text_response("second")]
        )
        provider = LiteLLMConfigurer(model="test")
        agent = CachingAgent()

        # `capture` is terminal (it never forwards), so it goes *below* the
        # provider: LiteLLMConfigurer.completion has to run and forward into
        # it, or the request never gets the provider's config or breakpoint.
        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(provider),
            handler(HistoryBuilder()),
        ):
            agent.ask("first question")
            agent.ask("second question")

        first, second = capture.received_messages
        assert len(second) > len(first)
        # In the second request only its own (newest) user message is marked.
        marked = [i for i, m in enumerate(second) if _has_cache_control(m)]
        assert marked == [0, len(second) - 1], (
            f"Expected the system message and the last message only. Got: {marked}"
        )
        _assert_valid_anthropic_request(second)

    def test_cache_control_never_enters_stored_history(self):
        """The annotation is added to the outgoing request, not the transcript,
        so it never reaches `__history__` (or an Agent's checkpoint)."""
        capture = MockCompletionHandler([make_text_response("42")])
        provider = LiteLLMConfigurer(model="test")
        agent = CachingAgent()

        # `capture` is terminal (it never forwards), so it goes *below* the
        # provider: LiteLLMConfigurer.completion has to run and forward into
        # it, or the request never gets the provider's config or breakpoint.
        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(provider),
            handler(HistoryBuilder()),
        ):
            agent.ask("What is 2+2?")

        # Every breakpoint in the request -- on the system message and on the
        # last input message alike -- is added by `_add_cache_control`, so the
        # transcript should carry none of them, in either form.
        sent = capture.received_messages[0]
        assert any(_has_cache_control(m) for m in sent), (
            "sanity: the outgoing request should carry a breakpoint"
        )
        assert not [m for m in agent.__history__ if _has_cache_control(m)], (
            f"cache_control leaked into stored history: {list(agent.__history__)}"
        )

    def test_cache_control_format_is_ephemeral(self):
        """cache_control should use the ephemeral type."""
        capture = MockCompletionHandler([make_text_response("42")])
        provider = LiteLLMConfigurer(model="test")

        # `capture` is terminal (it never forwards), so it goes *below* the
        # provider: LiteLLMConfigurer.completion has to run and forward into
        # it, or the request never gets the provider's config or breakpoint.
        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(provider),
            handler(HistoryBuilder()),
        ):
            simple_prompt("test")

        for msg in capture.received_messages[0]:
            if "cache_control" in msg:
                assert msg["cache_control"] == {"type": "ephemeral"}
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        assert block["cache_control"] == {"type": "ephemeral"}

    def test_anthropic_receives_cache_control_from_last_system_block(self):
        """The system message's breakpoint sits on its last content block, since
        `call_system` renders the assembled prompt as a list of blocks rather than one
        Markdown string. Verify litellm forwards it to Anthropic as a cached system
        block -- for list content the block-level form is the *only* one it reads, so
        a message-level key would silently stop caching the system prompt."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        msgs = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Hi.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        ]
        transformed = AnthropicConfig().transform_request(
            model="claude-sonnet-4-5",
            messages=msgs,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert transformed["system"] == [
            {"type": "text", "text": "Hi.", "cache_control": {"type": "ephemeral"}}
        ]

    def test_litellm_strips_cache_control_for_openai(self):
        """Verify litellm strips cache_control when transforming for OpenAI."""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        msgs = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Hi.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hi",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ]
        config = OpenAIGPTConfig()
        transformed = config.transform_request(
            model="gpt-4o",
            messages=msgs,
            optional_params={},
            litellm_params={},
            headers={},
        )
        for msg in transformed["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    assert "cache_control" not in block


# ============================================================================
# Empty content blocks
#
# Regression tests for GitHub issue #762.
# ============================================================================


@Tool.define
def silent_tool() -> str:
    """A tool whose output is empty -- an empty file, a search with no hits."""
    return ""


@Skill.define
def use_silent_tool() -> str:
    """Consult the note and report what it said."""
    raise NotHandled


class TestEmptyContentBlocks:
    """No message the harness builds may carry an empty text block."""

    @staticmethod
    def _consult_the_note():
        capture = MockCompletionHandler(
            [
                make_tool_call_response("silent_tool", "{}"),
                make_text_response("it was empty"),
            ]
        )
        # `capture` is terminal, so it goes below the provider: the breakpoint is
        # only added if `LiteLLMConfigurer.completion` runs and forwards into it.
        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="claude-sonnet-4-5")),
            handler(HistoryBuilder()),
        ):
            use_silent_tool()
        return capture

    def test_empty_tool_result_carries_no_empty_block(self):
        """A tool returning ``""`` used to encode to a single empty text block,
        in the one place `_add_cache_control` puts its second breakpoint."""
        sent = self._consult_the_note().received_messages[-1]

        assert _empty_text_blocks(sent) == [], (
            f"empty text block(s) in the request: {_empty_text_blocks(sent)}"
        )
        _assert_valid_anthropic_request(sent)

    def test_breakpoint_moves_off_an_empty_tool_result(self):
        """With no block to mark, the breakpoint falls back to the previous
        message rather than being dropped."""
        sent = self._consult_the_note().received_messages[-1]

        marked = [m["role"] for m in sent if _has_cache_control(m)]
        assert marked == ["system", "user"], (
            f"expected the breakpoint to fall back to the user message. Got: {marked}"
        )

    def test_a_value_that_merely_contains_an_empty_string_is_unchanged(self):
        """`to_content_blocks` still satisfies its linearization law: an empty
        string inside an encoded value keeps its JSON quotes."""
        assert to_content_blocks("") == []
        assert to_content_blocks({"a": ""}) == [{"type": "text", "text": '{"a": ""}'}]
        assert to_content_blocks([]) == [{"type": "text", "text": "[]"}]

    @pytest.mark.parametrize(
        "value", ["", "   ", "x", {}, [], {"a": ""}, {"a": [1, ""]}, 0, None]
    )
    def test_to_content_blocks_agrees_with_is_empty_text_block(self, value):
        """The invariant `HistoryBuilder.append_message` asserts."""
        assert not any(_is_empty_text_block(b) for b in to_content_blocks(value))

    @pytest.mark.parametrize(
        ("template", "expected"),
        [
            ("a{x}b", "ab"),
            ("a{x!r}b", "a''b"),
            ("a{x:>5}b", "a     b"),
            ("a{x!r:>6}b", "a    ''b"),
        ],
    )
    def test_a_conversion_still_runs_on_an_empty_value(self, template, expected):
        """A conversion or format spec can turn an empty value into something."""
        assert format_as_content_blocks(template, {"x": ""}) == [
            {"type": "text", "text": expected}
        ]

    def test_a_hole_that_formats_to_nothing_produces_no_block(self):
        assert format_as_content_blocks("{x}", {"x": ""}) == []

    def test_empty_block_never_enters_stored_history(self):
        """`SQLitePersister` checkpoints the transcript, so a block repaired
        only on the way out would still be durable."""
        capture = MockCompletionHandler(
            [
                make_tool_call_response("silent_tool", "{}"),
                make_text_response("it was empty"),
            ]
        )
        agent = CachingAgent()

        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="claude-sonnet-4-5")),
            handler(HistoryBuilder()),
        ):
            agent.ask("what does the note say?")

        assert _empty_text_blocks(agent.__history__) == []

    def test_exec_code_with_no_output_is_sendable(self):
        """`exec_code` returns the empty string for a snippet that printed
        nothing, and is in the default `harness()` stack."""

        def run_silent(exec_code):
            bound_args = inspect.signature(exec_code).bind(
                pydantic.TypeAdapter(Encodable[CodeType]).validate_python("x = 1 + 1")
            )
            tc = DecodedToolCall(exec_code, bound_args, "call_exec", "exec_code")
            return call_tool(tc)[0]

        msg = _drive_repl(run_silent)
        assert msg["role"] == "tool"
        assert _empty_text_blocks([msg]) == [], (
            f"exec_code produced an empty block: {msg}"
        )

    def test_call_user_never_emits_an_empty_block(self):
        """`call_user` and `call_system` render through `_render_prompt_section`,
        which drops empty text."""

        @Skill.define
        def ask_about(topic: str, note: str) -> str:
            """Say something about {topic}. {note}"""
            raise NotHandled

        capture = MockCompletionHandler([make_text_response("ok")])
        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="claude-sonnet-4-5")),
            handler(HistoryBuilder()),
        ):
            # Both holes encode to "", so the rendered prompt ends on one.
            ask_about("", "")

        sent = capture.received_messages[0]
        assert [m["role"] for m in sent] == ["system", "user"]
        assert _empty_text_blocks(sent) == []
        _assert_valid_anthropic_request(sent)

    def test_breakpoint_skips_an_unmarkable_message(self):
        """The same fallback, over messages the harness did not build."""
        provider = LiteLLMConfigurer(model="claude-sonnet-4-5")
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "user", "content": [{"type": "text", "text": ""}]},
        ]

        marked = provider._add_cache_control(msgs)

        assert [_has_block_cache_control(m) for m in marked] == [True, True, False], (
            f"breakpoint should fall back to the previous message. Got: {marked}"
        )

    def test_whitespace_is_content(self):
        """A block of spaces is not empty; Anthropic accepts one."""
        provider = LiteLLMConfigurer(model="claude-sonnet-4-5")
        blank = {"role": "user", "content": [{"type": "text", "text": "   "}]}

        assert to_content_blocks("   ") == [{"type": "text", "text": "   "}]
        assert _empty_text_blocks([blank]) == []
        assert _has_block_cache_control(provider._mark(blank))


class TestEmptyReply:
    """A reply with nothing to decode is retried, and reports the finish_reason."""

    @staticmethod
    def _response(content, finish_reason="stop", **extra):
        return ModelResponse(
            id="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, **extra},
                    "finish_reason": finish_reason,
                }
            ],
            model="test-model",
        )

    @pytest.mark.parametrize("content", [None, ""])
    def test_contentless_reply_names_the_finish_reason(self, content):
        """The finish_reason is the only thing that says why the reply is empty."""
        capture = MockCompletionHandler(
            [self._response(content, finish_reason="length")]
        )

        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="claude-sonnet-4-5")),
            handler(HistoryBuilder()),
        ):
            with pytest.raises(ResultDecodingError, match="finish_reason='length'"):
                simple_prompt("test")

    @pytest.mark.parametrize("content", [None, "", "   "])
    def test_an_empty_reply_is_retried_like_any_other_bad_output(self, content):
        capture = MockCompletionHandler(
            [self._response(content), make_text_response(json.dumps({"value": 4}))]
        )

        # `TenacityRetryer` goes inside `HistoryBuilder`: it appends the reply
        # that finally succeeded itself, so wrapping the other way records it
        # twice.
        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="claude-sonnet-4-5")),
            handler(HistoryBuilder()),
            handler(TenacityRetryer(stop=tenacity.stop_after_attempt(3))),
        ):
            assert generate_number(10) == 4

        assert capture.call_count == 2, "the empty reply should have been retried"

    def test_a_reply_carrying_only_reasoning_content_still_decodes(self):
        """The `content or reasoning_content` fallback is untouched."""
        capture = MockCompletionHandler(
            [self._response("", reasoning_content=json.dumps({"value": 7}))]
        )

        with (
            handler(capture),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="claude-sonnet-4-5")),
            handler(HistoryBuilder()),
        ):
            assert generate_number(10) == 7


# ============================================================================
# Scoping a call to a different model
# ============================================================================


class TestScopedModelOverride:
    """A nested `LiteLLMConfigurer` re-points one call at another model.

    This is the shape the optimization examples use to score an artifact against
    a cheap "worker" model while a stronger model drives the search. It is worth
    pinning because it is the *only* handler an example installs for itself, and
    because getting it wrong is quiet: a nested handler that answers `call_agent`
    without forwarding leaves `HistoryBuilder.get_history` unbound, and the
    request then goes out with no messages at all rather than raising.
    """

    @staticmethod
    def _stack(mock):
        return (
            handler(mock),
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer(model="outer-model")),
            handler(HistoryBuilder()),
        )

    def test_scoped_override_switches_model_and_keeps_history(self):
        mock = MockCompletionHandler([make_text_response("hi")])

        @Skill.define
        def greet(name: str) -> str:
            """Greet {name}."""

        with contextlib.ExitStack() as stack:
            for h in self._stack(mock):
                stack.enter_context(h)
            greet("world")
            with handler(LiteLLMConfigurer(model="worker-model")):
                greet("world")

        outer, inner = mock.received_messages
        assert [m["role"] for m in outer] == ["system", "user"]
        # The scoped call is an ordinary harness call: same history, new model.
        assert [m["role"] for m in inner] == ["system", "user"]

    def test_scoped_override_still_retries_a_malformed_answer(self):
        """The nested configurer implements `completion`, the lowest hook, so
        every handler above it -- including `TenacityRetryer` on `call_assistant`
        -- keeps applying to the scoped call."""
        # A non-`str` return type is decoded out of a `{"value": ...}` box, so the
        # first answer is malformed and the second is well-formed.
        mock = MockCompletionHandler(
            [make_text_response("not a number"), make_text_response('{"value": 7}')]
        )

        @Skill.define
        def count() -> int:
            """Return a number."""

        with contextlib.ExitStack() as stack:
            for h in self._stack(mock):
                stack.enter_context(h)
            stack.enter_context(
                handler(TenacityRetryer(stop=tenacity.stop_after_attempt(3)))
            )
            with handler(LiteLLMConfigurer(model="worker-model")):
                assert count() == 7

        assert mock.call_count == 2


# ============================================================================
# What the assembled stack offers the model
# ============================================================================


class _CaptureTools(ObjectInterpretation):
    """Answer any request with `text`, recording the tools it was sent."""

    def __init__(self, text: str = "ok"):
        self.text = text
        self.tools: list[dict] = []

    @implements(completion)
    def _completion(self, *args, **kwargs):
        self.tools = list(kwargs.get("tools") or [])
        return make_text_response(self.text)


def _offered_tool_names(**harness_kwargs) -> set[str]:
    """The tool names a `Skill` call under ``harness(**harness_kwargs)`` sends."""
    from effectful.handlers.llm.harness import harness

    @Skill.define
    def ask(q: str) -> str:
        """Answer {q}."""

    capture = _CaptureTools()
    with handler(harness(model="test", **harness_kwargs)), handler(capture):
        assert ask("hi") == "ok"
    return {t["function"]["name"] for t in capture.tools}


@pytest.mark.parametrize("eval_provider", ["builtin", "restricted"])
def test_synthesis_tools_are_offered_when_an_executor_can_run_them(eval_provider):
    offered = _offered_tool_names(eval_provider=eval_provider, tool_calling="json")
    assert {"exec_code", "write_and_run_body"} <= offered


def test_no_synthesis_tools_are_offered_without_an_executor():
    """``eval_provider="none"`` must not advertise a tool nothing can decode.

    `StatefulReplSynthesizer` and `FinalBodySynthesizer` each offer one --
    ``exec_code`` and ``write_and_run_body`` -- and each needs an executor to
    turn the model's source into something runnable. Offered without one, a
    model that calls either (which it may do however firmly the prompt tells it
    not to) has its own well-formed tool call fail to decode with
    ``NotImplementedError: An eval provider must be installed in order to parse
    code``. That is what made ``test_live_post_condition_is_repaired_on_retry``
    fail on one Python version and pass on the other in the same CI run: the
    difference was which way the model went, not anything about the stack.
    """
    offered = _offered_tool_names(eval_provider="none", tool_calling="json")
    assert not {"exec_code", "write_and_run_body"} & offered


@pytest.mark.parametrize("tool_calling", ["auto", "code"])
def test_code_tool_calling_without_an_executor_is_rejected(tool_calling):
    """A code pathway with nothing to run the code is refused at construction.

    `ValueError` rather than an assertion, so the guard survives ``python -O``:
    with assertions stripped, this configuration would build a stack whose tool
    caller offers the model a code pathway that nothing underneath can execute,
    and the failure would surface much later and further away.
    """
    from effectful.handlers.llm.harness import harness

    with pytest.raises(ValueError, match="needs an eval provider"):
        harness(model="test", tool_calling=tool_calling, eval_provider="none")


@pytest.mark.parametrize(
    "tool_calling,eval_provider",
    [
        ("json", "none"),
        ("json", "builtin"),
        ("auto", "builtin"),
        ("code", "restricted"),
    ],
)
def test_supported_tool_calling_and_executor_pairs_build(tool_calling, eval_provider):
    """The guard rejects one combination, not the neighbouring ones."""
    from effectful.handlers.llm.harness import harness

    assert harness(model="test", tool_calling=tool_calling, eval_provider=eval_provider)


@pytest.mark.parametrize(
    "tool_collection,declared,implicit",
    [
        ("none", False, False),
        ("explicit", True, False),
        ("auto", True, True),
    ],
)
def test_tool_collection_selects_the_lexical_extractor(
    tool_collection, declared, implicit
):
    """``tool_collection`` decides what is *collected* from the Skill's scope:
    nothing (``"none"`` -- only the harness's own tools are offered), the
    declared `Tool`/`Skill` values (``"explicit"``), or additionally the
    qualifying plain functions (``"auto"``)."""
    from effectful.handlers.llm import Tool
    from effectful.handlers.llm.harness import harness

    @Tool.define
    def declared_helper(x: int) -> int:
        """Add one."""
        return x + 1

    def implicit_helper(x: int) -> int:
        """Double x."""
        return x * 2

    @Skill.define
    def ask(q: str) -> str:
        """Answer {q}."""

    capture = _CaptureTools()
    with (
        handler(harness(model="test", tool_collection=tool_collection)),
        handler(capture),
    ):
        assert ask("hi") == "ok"
    offered = {t["function"]["name"] for t in capture.tools}

    assert ("declared_helper" in offered) == declared
    assert ("implicit_helper" in offered) == implicit
    # The harness's own tools are offered regardless of collection.
    assert {"exec_code", "write_and_run_body"} <= offered
