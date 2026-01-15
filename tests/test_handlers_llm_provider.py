"""Tests for LLM handlers and providers.
This module tests the functionality from build/main.py and build/llm.py,
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
from litellm.files.main import ModelResponse
from PIL import Image
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    LLMLoggingHandler,
    completion,
)
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

    def __init__(self, request: pytest.FixtureRequest, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_id = request.node.nodeid
        self.test_id = self.test_id.replace("/", "_").replace(":", "_")

    @implements(completion)
    def _completion(self, *args, **kwargs):
        path = FIXTURE_DIR / f"{self.test_id}.json"
        if not REBUILD_FIXTURES:
            if not path.exists():
                raise RuntimeError(f"Missing replay fixture: {path}")
            with path.open() as f:
                result = ModelResponse.model_validate(json.load(f))
                return result
        result = fwd(self.model_name, *args, **(self.config | kwargs))
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(result.model_dump(), f, indent=2, sort_keys=True)
        return result


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
            handler(ReplayLiteLLMProvider(request, model_name=model_name)),
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
            handler(ReplayLiteLLMProvider(request, model_name=model_name)),
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
            handler(ReplayLiteLLMProvider(request, model_name="gpt-5-nano")),
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
            handler(ReplayLiteLLMProvider(request, model_name="gpt-5-nano")),
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
                ReplayLiteLLMProvider(
                    request, model_name="gpt-4o-mini", temperature=0.1
                )
            ),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            result = simple_prompt("deterministic test")
            assert isinstance(result, str)


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
        handler(ReplayLiteLLMProvider(request, model_name="gpt-4o")),
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
            handler(ReplayLiteLLMProvider(request, model_name="gpt-5-nano")),
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
