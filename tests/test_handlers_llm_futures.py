"""
Tests for the LLM handler Future support.

This module tests that LLM templates with Future[T] return types
correctly submit work concurrently and decode using the inner type.
"""

import time
from collections.abc import Callable
from concurrent.futures import Future
from inspect import BoundArguments
from typing import Any, override

import effectful.handlers.futures as futures
from effectful.handlers.futures import Executor, ThreadPoolFuturesInterpretation
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
        self, template: Template, args: BoundArguments, retty: type[T]
    ) -> T:
        self.calls.append((template, args.args, retty))
        time.sleep(self.delay)
        return self.mapping.get(template, {}).get(tuple(args.args), self.response)


@Template.define
def hiaku(topic: str) -> str:
    """Return a hiaku about {topic}."""
    raise NotHandled


def test_future_return_type_decodes_inner_type():
    """Test that llm templates correctly decode to int, even wrapped in a future."""
    ref_hiaku = "apples to oranges, oranges to pears, I don't know what a hiaku is"
    mock_provider = SlowMockLLMProvider(ref_hiaku, delay=0.001)

    with handler(ThreadPoolFuturesInterpretation()), handler(mock_provider):
        future = Executor.submit(hiaku, "apples")
        assert isinstance(future, Future)
        result = future.result()
        assert result == ref_hiaku


@Template.define
def generate_program(task: str) -> Callable[[int], int]:
    """Generate a Python program that {task}."""
    raise NotHandled


def test_concurrent_program_generation():
    """Simulate concurrent LLM calls to generate Python programs and pick the best one."""
    # Mock responses for different approaches to the same task
    responses = {
        generate_program: {
            ("implement fibonacci algorithm 0",): "def fib(n: int) -> int: return n",
            (
                "implement fibonacci algorithm 1",
            ): "def fib(n: int) -> int: return n * fib(n - 1)",
            (
                "implement fibonacci algorithm 2",
            ): "def fib(n: int) -> int: return fib(n - 2) + fib(n - 1) if n > 1 else 0",
        }
    }

    mock_provider = SlowMockLLMProvider(
        response="print('Default')", delay=0.01, mapping=responses
    )

    user_request: str = "implement fibonacci algorithm"

    with handler(ThreadPoolFuturesInterpretation()), handler(mock_provider):
        # Launch multiple LLM calls concurrently
        tasks = [
            Executor.submit(generate_program, (user_request + f" {i}"))
            for i in range(3)
        ]

        # Collect all results as they finish
        results_as_completed = (f.result() for f in futures.as_completed(tasks))

        valid_results = [(result, len(result)) for result in results_as_completed]

        # Pick the "best" result (here: the shortest program, as a naive heuristic)
        best_program = max(valid_results, key=lambda pair: pair[1])[0]

    # Assertions
    assert len(valid_results) == 3
    assert best_program in set(responses[generate_program].values())
