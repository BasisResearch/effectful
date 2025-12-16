"""Tests for LLM image input functionality."""

from __future__ import annotations

import os

import pytest
from PIL import Image

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import LiteLLMProvider, completion
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# Check for API keys
HAS_OPENAI_KEY = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]

requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY environment variable not set"
)


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
    """Return a description of the following image. Do not use any tools.
    {image}"""
    raise NotHandled


@requires_openai
def test_image_input():
    with (
        handler(LiteLLMProvider(model_name="gpt-4o")),
        handler(LimitLLMCallsHandler(max_calls=3)),
    ):
        assert any("smile" in categorise_image(smiley_face()).lower() for _ in range(3))

