"""Tests for LLM tool calling functionality - Book recommendation.

This module is separate to avoid lexical context pollution from other templates.
"""

import os
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, Field

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    RetryLLMHandler,
    call_assistant,
)
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# Check for API keys
HAS_OPENAI_KEY = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]
HAS_ANTHROPIC_KEY = (
    "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]
)

try:
    from effectful.handlers.llm.anthropic import (
        AnthropicProvider,
        _get_oauth_token_from_keychain,
    )

    HAS_ANTHROPIC_SDK = True
except ImportError:
    HAS_ANTHROPIC_SDK = False
    _get_oauth_token_from_keychain = lambda: None  # noqa: E731

HAS_CLAUDE_MAX = HAS_ANTHROPIC_SDK and _get_oauth_token_from_keychain() is not None
HAS_ANTHROPIC_AUTH = HAS_ANTHROPIC_KEY or HAS_CLAUDE_MAX

requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY environment variable not set"
)
requires_anthropic = pytest.mark.skipif(
    not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY environment variable not set"
)
requires_anthropic_sdk = pytest.mark.skipif(
    not HAS_ANTHROPIC_SDK or not HAS_ANTHROPIC_AUTH,
    reason="anthropic package not installed or no auth available",
)

_ANTHROPIC_SDK_MODEL = (
    "claude-haiku-4-5"
    if (HAS_CLAUDE_MAX and not HAS_ANTHROPIC_KEY)
    else "claude-haiku-4-5-20250514"
)
_USE_MAX = HAS_CLAUDE_MAX and not HAS_ANTHROPIC_KEY


@dataclass
class LimitLLMCallsHandler(ObjectInterpretation):
    """Handler that limits the number of LLM calls."""

    max_calls: int = 10
    call_count: int = 0

    @implements(call_assistant)
    def _completion(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count > self.max_calls:
            raise RuntimeError(
                f"Test used too many requests (max_calls = {self.max_calls})"
            )
        return fwd()


class BookRecommendation(BaseModel):
    """A book recommendation."""

    title: str = Field(..., description="The title of the book")
    reason: str = Field(..., description="Why this book is recommended")


@Tool.define
def recommend_book_tool(genre: str, mood: str) -> BookRecommendation:
    """Recommend a book based on genre and mood.

    Parameters:
    - genre: The genre of book to recommend
    - mood: The mood or feeling the reader is looking for
    """
    raise NotHandled


class LoggingBookRecommendationInterpretation(ObjectInterpretation):
    """Provides an interpretation for `recommend_book_tool` that tracks calls."""

    recommendation_count: int = 0
    recommendation_results: list[dict] = []

    @implements(recommend_book_tool)
    def _recommend_book_tool(self, genre: str, mood: str) -> BookRecommendation:
        self.recommendation_count += 1

        recommendation = BookRecommendation(
            title=f"The {mood.title()} {genre.title()} Adventure",
            reason=f"A perfect {genre} book for when you're feeling {mood}",
        )

        self.recommendation_results.append(
            {"genre": genre, "mood": mood, "recommendation": recommendation}
        )

        return recommendation


@Template.define
def get_book_recommendation(user_preference: str) -> BookRecommendation:
    """Get a book recommendation based on user preference: {user_preference}.

    You MUST use recommend_book_tool to get the recommendation.
    Return the recommendation as JSON with 'title' and 'reason' fields.
    """
    raise NotHandled


class TestPydanticBaseModelToolCalls:
    @pytest.mark.parametrize(
        "provider",
        [
            pytest.param(
                lambda: LiteLLMProvider(model="gpt-5-nano"),
                marks=requires_openai,
                id="litellm-gpt-5-nano",
            ),
            pytest.param(
                lambda: LiteLLMProvider(model="claude-sonnet-4-5-20250929"),
                marks=requires_anthropic,
                id="litellm-claude-sonnet",
            ),
            pytest.param(
                lambda: AnthropicProvider(
                    model=_ANTHROPIC_SDK_MODEL, max_subscription=_USE_MAX
                ),
                marks=requires_anthropic_sdk,
                id="anthropic-claude-haiku",
            ),
        ],
    )
    def test_pydantic_basemodel_tool_calling(self, provider):
        """Test that templates with tools work with Pydantic BaseModel."""
        book_rec_ctx = LoggingBookRecommendationInterpretation()
        with (
            handler(provider()),
            handler(RetryLLMHandler()),
            handler(LimitLLMCallsHandler(max_calls=6)),
            handler(book_rec_ctx),
        ):
            recommendation = get_book_recommendation("I love fantasy novels")

            assert isinstance(recommendation, BookRecommendation)
            assert isinstance(recommendation.title, str)
            assert len(recommendation.title) > 0
            assert isinstance(recommendation.reason, str)
            assert len(recommendation.reason) > 0

        # Verify the tool was called at least once
        assert book_rec_ctx.recommendation_count >= 1
        assert len(book_rec_ctx.recommendation_results) >= 1
