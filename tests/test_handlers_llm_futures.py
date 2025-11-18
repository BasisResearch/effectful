"""
Tests for the LLM handler Future support.

This module tests that LLM templates with Future[T] return types
correctly submit work concurrently and decode using the inner type.
"""

import time
from concurrent.futures import Future
from inspect import BoundArguments
from typing import Any, override

from effectful.handlers.futures import ThreadPoolFuturesInterpretation
from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import OpenAIAPIProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled


class SlowMockLLMProvider(OpenAIAPIProvider):
    """Mock provider that simulates slow LLM responses for testing concurrency."""

    def __init__(self, response, delay: float = 0.05, mapping={}):
        self.response = response
        self.delay = delay
        self.calls: list[tuple[Any, tuple[Any], type]] = []
        self.mapping = mapping

    @override
    def _openai_api_call[T](
        self, template: Any, args: BoundArguments, retty: type[T]
    ) -> T:
        self.calls.append((template, args.args, retty))
        time.sleep(self.delay)

        return self.mapping.get((template, tuple(args.args)), self.response)


@Template.define
def hiaku(topic: str) -> Future[str]:
    """Return a hiaku about {topic}."""
    raise NotHandled


# synchronous template for comparison
@Template.define
def hiaku_s(topic: str) -> str:
    """Return a hiaku about {topic}."""
    raise NotImplementedError


def test_future_return_type_decodes_inner_type():
    """Test that Future[int] templates correctly decode to int."""
    ref_hiaku = "apples to oranges, oranges to pears, I don't know what a hiaku is"
    mock_provider = SlowMockLLMProvider(ref_hiaku, delay=0.001)

    with handler(ThreadPoolFuturesInterpretation()), handler(mock_provider):
        future = hiaku("apples")
        assert isinstance(future, Future)
        result = future.result()
        assert result == ref_hiaku
