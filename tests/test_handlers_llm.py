import json
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple, TypedDict

import pytest
from litellm.types.utils import ModelResponse

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import (
    LiteLLMProvider,
    RetryLLMHandler,
    completion,
)
from effectful.handlers.llm.synthesis import ProgramSynthesis
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements


class MockLLMProvider[T](ObjectInterpretation):
    """Mock provider for testing.

    Initialized with prompts and responses. Raises if an unexpected prompt is given.
    """

    def __init__(self, prompt_responses: dict[str, T]):
        """Initialize with a dictionary mapping prompts to expected responses.

        Args:
            prompt_responses: Dict mapping prompt strings to their expected responses
        """
        self.prompt_responses = prompt_responses

    @implements(Template.__call__)
    def _call[**P](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        prompt = template.__prompt_template__.format(**bound_args.arguments)

        if prompt not in self.prompt_responses:
            raise ValueError(f"Unexpected prompt: {prompt!r}")

        response = self.prompt_responses[prompt]
        return response


class SingleResponseLLMProvider[T](LiteLLMProvider):
    """Simplified mock provider that returns a single response for any prompt."""

    def __init__(self, response: T):
        """Initialize with a single response string.

        Args:
            response: The response to return for any template call
        """
        if not isinstance(response, str):
            response_str = json.dumps({"value": response})
        else:
            response_str = response

        self.response: ModelResponse = ModelResponse(
            choices=[{"message": {"content": response_str}}]
        )

    @implements(completion)
    def _completion(self, *args, **kwargs) -> Any:
        return self.response


class RawStringLLMProvider(LiteLLMProvider):
    """Mock provider that returns a raw JSON string response (as returned by LLM)."""

    def __init__(self, raw_json_string: str):
        """Initialize with a raw JSON string response.

        Args:
            raw_json_string: The raw JSON string response from LLM (before decoding)
        """
        self.response: ModelResponse = ModelResponse(
            choices=[{"message": {"content": raw_json_string}}]
        )

    @implements(completion)
    def _completion(self, *args, **kwargs) -> Any:
        return self.response


# Test templates from the notebook examples
@Template.define
def limerick(theme: str) -> str:
    """Write a limerick on the theme of {theme}."""
    raise NotImplementedError


@Template.define
def haiku(theme: str) -> str:
    """Write a haiku on the theme of {theme}."""
    raise NotImplementedError


@Template.define()
def primes(first_digit: int) -> int:
    """Give exactly one prime number with {first_digit} as the first digit. Respond with only the number."""
    raise NotImplementedError


@Template.define
def count_char(char: str) -> Callable[[str], int]:
    """Write a function which takes a string and counts the occurrances of '{char}'."""
    raise NotImplementedError


# Unit tests
def test_limerick():
    """Test the limerick template returns a string."""
    mock_response = "There once was a fish from the sea"
    mock_provider = MockLLMProvider(
        {"Write a limerick on the theme of fish.": mock_response}
    )

    with handler(mock_provider):
        result = limerick("fish")
        assert result == mock_response
        assert isinstance(result, str)


def test_primes_decode_int():
    """Test the primes template correctly decodes integer response."""
    mock_provider = SingleResponseLLMProvider(61)

    with handler(mock_provider):
        result = primes(6)
        assert result == 61
        assert isinstance(result, int)


def test_count_char_with_program_synthesis():
    """Test the count_char template with program synthesis."""
    mock_code = """<code>
def count_occurrences(s):
    return s.count('a')
</code>"""
    mock_provider = SingleResponseLLMProvider(mock_code)

    with handler(mock_provider), handler(ProgramSynthesis()):
        count_a = count_char("a")
        assert callable(count_a)
        assert count_a("banana") == 3
        assert count_a("cherry") == 0


class FailingThenSucceedingProvider[T](ObjectInterpretation):
    """Mock provider that fails a specified number of times before succeeding."""

    def __init__(
        self,
        fail_count: int,
        success_response: T,
        exception_factory: Callable[[], Exception],
    ):
        """Initialize the provider.

        Args:
            fail_count: Number of times to fail before succeeding
            success_response: Response to return after failures
            exception_factory: Factory function that creates exceptions to raise
        """
        self.fail_count = fail_count
        self.success_response = success_response
        self.exception_factory = exception_factory
        self.call_count = 0

    @implements(Template.__call__)
    def _call[**P](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise self.exception_factory()
        return self.success_response


def test_retry_handler_succeeds_after_failures():
    """Test that RetryLLMHandler retries and eventually succeeds."""
    provider = FailingThenSucceedingProvider(
        fail_count=2,
        success_response="Success after retries!",
        exception_factory=lambda: ValueError("Temporary failure"),
    )
    retry_handler = RetryLLMHandler(max_retries=3, exception_cls=ValueError)

    with handler(provider), handler(retry_handler):
        result = limerick("test")
        assert result == "Success after retries!"
        assert provider.call_count == 3  # 2 failures + 1 success


def test_retry_handler_exhausts_retries():
    """Test that RetryLLMHandler raises after max retries exhausted."""
    provider = FailingThenSucceedingProvider(
        fail_count=5,  # More failures than retries
        success_response="Never reached",
        exception_factory=lambda: ValueError("Persistent failure"),
    )
    retry_handler = RetryLLMHandler(max_retries=3, exception_cls=ValueError)

    with pytest.raises(ValueError, match="Persistent failure"):
        with handler(provider), handler(retry_handler):
            limerick("test")

    assert provider.call_count == 3  # Should have tried 3 times


def test_retry_handler_only_catches_specified_exception():
    """Test that RetryLLMHandler only catches the specified exception class."""
    provider = FailingThenSucceedingProvider(
        fail_count=1,
        success_response="Success",
        exception_factory=lambda: TypeError("Wrong type"),  # Different exception type
    )
    retry_handler = RetryLLMHandler(max_retries=3, exception_cls=ValueError)

    # TypeError should not be caught, should propagate immediately
    with pytest.raises(TypeError, match="Wrong type"):
        with handler(provider), handler(retry_handler):
            limerick("test")

    assert provider.call_count == 1  # Should have only tried once


def test_retry_handler_with_error_feedback():
    """Test that RetryLLMHandler includes error feedback when enabled."""
    call_prompts: list[str] = []

    class PromptCapturingProvider(ObjectInterpretation):
        """Provider that captures prompts and fails once."""

        def __init__(self):
            self.call_count = 0

        @implements(Template.__call__)
        def _call(self, template: Template, *args, **kwargs):
            self.call_count += 1
            call_prompts.append(template.__prompt_template__)
            if self.call_count == 1:
                raise ValueError("First attempt failed")
            return "Success on retry"

    provider = PromptCapturingProvider()
    retry_handler = RetryLLMHandler(
        max_retries=2, add_error_feedback=True, exception_cls=ValueError
    )

    with handler(provider), handler(retry_handler):
        result = limerick("test")
        assert result == "Success on retry"

    assert len(call_prompts) == 2
    # First call has original prompt
    assert "Write a limerick on the theme of {theme}." in call_prompts[0]
    # Second call should include error feedback with traceback
    assert "Retry generating" in call_prompts[1]
    assert "First attempt failed" in call_prompts[1]


class Color(str, Enum):
    """Color enum for testing."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclass
class Person:
    """Person dataclass for testing."""

    name: str
    age: int
    height: float


class PointDict(TypedDict):
    """Point TypedDict for testing."""

    x: int
    y: int


class PointTuple(NamedTuple):
    """Point NamedTuple for testing."""

    x: int
    y: int


@Template.define
def generate_number_value() -> numbers.Number:
    """Generate a number value."""
    raise NotImplementedError


@Template.define
def generate_bool_value() -> bool:
    """Generate a boolean value."""
    raise NotImplementedError


@Template.define
def generate_color_enum() -> Color:
    """Generate a Color enum value."""
    raise NotImplementedError


@Template.define
def generate_person() -> Person:
    """Generate a Person dataclass."""
    raise NotImplementedError


@Template.define
def generate_point_dict() -> PointDict:
    """Generate a PointDict TypedDict."""
    raise NotImplementedError


@Template.define
def generate_plain_tuple() -> tuple[int, bool]:
    """Generate a pair of an int and a bool."""
    raise NotImplementedError


@Template.define
def generate_point_tuple() -> PointTuple:
    """Generate a PointTuple NamedTuple."""
    raise NotImplementedError


class TestBaseTypeDecoding:
    """Tests for decoding various base types from mocked LLM responses using golden values."""

    def test_numbers_number(self):
        """Test decoding numbers.Number with float value."""
        raw_response = '{\n  "value": 42.0\n}'
        mock_provider = RawStringLLMProvider(raw_response)

        with handler(mock_provider):
            result = generate_number_value()
            assert isinstance(result, numbers.Number)
            assert result == 42.0
            assert isinstance(result, float)

    def test_bool_true(self):
        """Test decoding bool with True value."""
        raw_response = '{\n  "value": true\n}'
        mock_provider = RawStringLLMProvider(raw_response)

        with handler(mock_provider):
            result = generate_bool_value()
            assert isinstance(result, bool)
            assert result is True

    def test_enum_enum(self):
        """Test decoding enum.Enum value."""
        raw_response = '{"value": "red"}'
        mock_provider = RawStringLLMProvider(raw_response)

        with handler(mock_provider):
            result = generate_color_enum()
            assert isinstance(result, Color)
            assert result == Color.RED

    def test_dataclass(self):
        """Test decoding dataclass value."""
        raw_response = '{\n  "value": {\n    "name": "John Doe",\n    "age": 30,\n    "height": 5.9\n  }\n}'
        mock_provider = RawStringLLMProvider(raw_response)

        with handler(mock_provider):
            result = generate_person()
            assert isinstance(result, Person)
            assert result.name == "John Doe"
            assert result.age == 30
            assert result.height == 5.9

    def test_typeddict(self):
        """Test decoding TypedDict value."""
        raw_response = '{\n  "value": {\n    "x": 5,\n    "y": 10\n  }\n}'
        mock_provider = RawStringLLMProvider(raw_response)

        with handler(mock_provider):
            result = generate_point_dict()
            assert isinstance(result, dict)
            assert result["x"] == 5
            assert result["y"] == 10
            # TypedDict is just a dict at runtime
            assert type(result) is dict

    def test_plain_tuple(self):
        """Test decoding plain tuple value."""
        raw_response = '{\n  "value": {\n    "0": 5,\n    "1": false\n  }\n}'
        mock_provider = RawStringLLMProvider(raw_response)

        with handler(mock_provider):
            result = generate_plain_tuple()
            assert result[0] == 5
            assert result[1] == False
            assert isinstance(result, tuple)

    def test_namedtuple(self):
        """Test decoding NamedTuple value."""
        raw_response = '{\n  "value": {\n    "x": 5,\n    "y": 10\n  }\n}'
        mock_provider = RawStringLLMProvider(raw_response)

        with handler(mock_provider):
            result = generate_point_tuple()
            assert isinstance(result, PointTuple)
            assert result.x == 5
            assert result.y == 10
            assert isinstance(result, tuple)
