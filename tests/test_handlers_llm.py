from collections.abc import Callable

import pytest

from effectful.handlers.llm import Template
from effectful.handlers.llm.structure import DecodeError, decode
from effectful.handlers.llm.synthesis import ProgramSynthesis
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements


class MockLLMProvider(ObjectInterpretation):
    """Mock provider for testing.

    Initialized with prompts and responses. Raises if an unexpected prompt is given.
    """

    def __init__(self, prompt_responses: dict[str, str]):
        """Initialize with a dictionary mapping prompts to expected responses.

        Args:
            prompt_responses: Dict mapping prompt strings to their expected responses
        """
        self.prompt_responses = prompt_responses

    @implements(Template.__call__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        prompt = template.__prompt_template__.format(**bound_args.arguments)

        if prompt not in self.prompt_responses:
            raise ValueError(f"Unexpected prompt: {prompt!r}")

        response = self.prompt_responses[prompt]

        ret_type = template.__signature__.return_annotation
        return decode(ret_type, response)


class SingleResponseLLMProvider(ObjectInterpretation):
    """Simplified mock provider that returns a single response for any prompt."""

    def __init__(self, response: str):
        """Initialize with a single response string.

        Args:
            response: The response to return for any template call
        """
        self.response = response

    @implements(Template.__call__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        ret_type = template.__signature__.return_annotation
        return decode(ret_type, self.response)


# Test templates from the notebook examples
@Template.define()
def limerick(theme: str) -> str:
    """Write a limerick on the theme of {theme}."""
    raise NotImplementedError


@Template.define()
def haiku(theme: str) -> str:
    """Write a haiku on the theme of {theme}."""
    raise NotImplementedError


@Template.define()
def primes(first_digit: int) -> int:
    """Give exactly one prime number with {first_digit} as the first digit. Respond with only the number."""
    raise NotImplementedError


@Template.define()
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
    mock_provider = SingleResponseLLMProvider("61")

    with handler(mock_provider):
        result = primes(6)
        assert result == 61
        assert isinstance(result, int)


def test_primes_decode_error():
    """Test that non-numeric responses raise DecodeError."""
    mock_provider = SingleResponseLLMProvider("not a number")

    with handler(mock_provider):
        with pytest.raises(DecodeError) as exc_info:
            primes(7)
        assert exc_info.value.type_ == int
        assert exc_info.value.response == "not a number"


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


def test_decode_primitives():
    """Test decode function with primitive types."""
    assert decode(str, "hello") == "hello"
    assert decode(int, "42") == 42
    assert decode(float, "3.14") == 3.14
    assert decode(bool, "true") == True
    assert decode(bool, "false") == False

    with pytest.raises(DecodeError):
        decode(int, "not a number")

    with pytest.raises(DecodeError):
        decode(bool, "maybe")
