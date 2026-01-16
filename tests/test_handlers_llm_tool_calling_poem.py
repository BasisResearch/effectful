"""Tests for LLM tool calling functionality - Poem evaluation.

This module is separate to avoid lexical context pollution from other templates.
"""

import os
from dataclasses import dataclass
from enum import Enum

import pytest
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    completion,
)
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


@pydantic_dataclass
class Poem:
    """A poem with content and form."""

    content: str = Field(..., description="content of the poem")
    form: str = Field(..., description="name of the type of the poem")


class PoemQuality(str, Enum):
    """Quality rating for a poem."""

    GOOD = "GOOD"
    OKAY = "OKAY"
    BAD = "BAD"


@Tool.define
def evaluate_poem_tool(poem: Poem, explanation: str) -> PoemQuality:
    """Evaluate the quality of a poem.

    Parameters:
    - poem: Poem object representing the poem
    - explanation: natural language explanation of the thought process
    """
    raise NotHandled


class LoggingPoemEvaluationInterpretation(ObjectInterpretation):
    """Provides an interpretation for `evaluate_poem_tool` that tracks evaluation counts."""

    evaluation_count: int = 0
    evaluation_results: list[dict] = []

    @implements(evaluate_poem_tool)
    def _evaluate_poem_tool(self, poem: Poem, explanation: str) -> PoemQuality:
        self.evaluation_count += 1

        # Simple heuristic: require at least 2 evaluations, then approve
        quality = PoemQuality.BAD if self.evaluation_count < 2 else PoemQuality.GOOD

        self.evaluation_results.append(
            {"poem": poem, "explanation": explanation, "quality": quality}
        )

        return quality


@Template.define
def generate_good_poem(topic: str) -> Poem:
    """Generate a good poem about {topic}.

    You MUST use the evaluate_poem_tool to check poem quality.
    Keep iterating until evaluate_poem_tool returns GOOD.
    Return your final poem as JSON with 'content' and 'form' fields.

    Do not call the 'generate_good_poem' tool.
    """
    raise NotHandled


class TestToolCalling:
    """Tests for templates with tool calling functionality."""

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param("gpt-5-nano", marks=requires_openai),
            pytest.param("claude-sonnet-4-5-20250929", marks=requires_anthropic),
        ],
    )
    def test_tool_calling(self, model_name):
        """Test that templates with tools work with openai."""
        poem_eval_ctx = LoggingPoemEvaluationInterpretation()
        with (
            handler(LiteLLMProvider(model_name=model_name)),
            handler(LimitLLMCallsHandler(max_calls=4)),
            handler(poem_eval_ctx),
        ):
            poem = generate_good_poem("Python")
            assert isinstance(poem, Poem)
            assert isinstance(poem.content, str)
            assert isinstance(poem.form, str)

        # Verify the tool was called at least once
        assert poem_eval_ctx.evaluation_count >= 1
        assert len(poem_eval_ctx.evaluation_results) >= 1
