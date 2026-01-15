"""Tests for LLM tool calling functionality - Book recommendation.

This module is separate to avoid lexical context pollution from other templates.
"""

import os
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, Field

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, completion
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# Check for API keys
HAS_OPENAI_KEY = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]
HAS_ANTHROPIC_KEY = (
    "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]
)

requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY environment variable not set"
)
requires_anthropic = pytest.mark.skipif(
    not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY environment variable not set"
)


@dataclass
class LimitLLMCallsHandler(ObjectInterpretation):
    """Handler that limits the number of LLM calls."""

    max_calls: int = 10
    call_count: int = 0

    @implements(completion)
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
        "model_name",
        [
            pytest.param("gpt-5-nano", marks=requires_openai),
            pytest.param("claude-sonnet-4-5-20250929", marks=requires_anthropic),
        ],
    )
    def test_pydantic_basemodel_tool_calling(self, model_name):
        """Test that templates with tools work with Pydantic BaseModel."""
        book_rec_ctx = LoggingBookRecommendationInterpretation()
        with (
            handler(LiteLLMProvider(model_name=model_name)),
            handler(LimitLLMCallsHandler(max_calls=4)),
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
