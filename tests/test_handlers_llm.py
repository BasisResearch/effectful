import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from effectful.handlers.llm import Template
from effectful.handlers.llm.synthesis import (
    ProgramSynthesis,
    SynthesisError,
    SynthesizedFunction,
    collect_type_sources,
    format_type_context,
)
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


class SingleResponseLLMProvider[T](ObjectInterpretation):
    """Simplified mock provider that returns a single response for any prompt."""

    def __init__(self, response: T):
        """Initialize with a single response string.

        Args:
            response: The response to return for any template call
        """
        self.response = response

    @implements(Template.__call__)
    def _call[**P](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
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


@dataclass
class Person:
    name: str
    age: int


@Template.define
def make_greeter(style: str) -> Callable[[Person], str]:
    """Create a greeting function for a person with the given style."""
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
    # Structured response: LLM provides function name, param names, and body
    mock_response = SynthesizedFunction(
        function_name="count_occurrences",
        param_names=["text"],
        body="    return text.count('a')",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with handler(mock_provider), handler(ProgramSynthesis()):
        count_a = count_char("a")
        assert callable(count_a)
        assert count_a("banana") == 3
        assert count_a("cherry") == 0


def test_collect_type_sources():
    """Test the collect_type_sources function."""
    type_sources = collect_type_sources(Person)
    assert type_sources == {Person: inspect.getsource(Person)}


def test_format_type_context():
    """Test the format_type_context function."""
    type_sources = {Person: inspect.getsource(Person)}
    assert (
        format_type_context(type_sources)
        == textwrap.dedent(inspect.getsource(Person)).strip()
    )


def test_count_char_with_typed_body():
    """Test program synthesis constructs function with correct prescribed types."""
    # The body uses the LLM-provided parameter name
    mock_response = SynthesizedFunction(
        function_name="count_chars",
        param_names=["s"],
        body="    return s.count('x')",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with handler(mock_provider), handler(ProgramSynthesis()):
        count_x = count_char("x")
        assert callable(count_x)
        # Verify the function works
        assert count_x("xylophone") == 1
        assert count_x("xxx") == 3


def test_make_greeter_with_program_synthesis():
    """Test program synthesis with custom type (Person) in the signature."""
    mock_response = SynthesizedFunction(
        function_name="greet_person",
        param_names=["person"],
        body='    return f"Hello, {person.name}!"',
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with handler(mock_provider), handler(ProgramSynthesis()):
        greeter = make_greeter("formal")
        assert callable(greeter)
        # Test the generated function works with the custom type
        person = Person(name="Alice", age=30)
        assert greeter(person) == "Hello, Alice!"


def test_program_synthesis_invalid_body():
    """Test that synthesis fails when body has syntax errors."""
    mock_response = SynthesizedFunction(
        function_name="bad_func",
        param_names=["x"],
        body="    return this is not valid python",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with pytest.raises(SynthesisError, match="evaluation failed"):
        with handler(mock_provider), handler(ProgramSynthesis()):
            count_char("a")


def test_program_synthesis_runtime_error():
    """Test that synthesis fails when body raises runtime error on compile."""
    mock_response = SynthesizedFunction(
        function_name="bad_func",
        param_names=["x"],
        body="    return undefined_variable",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    # This should compile fine but fail at runtime when called
    with handler(mock_provider), handler(ProgramSynthesis()):
        func = count_char("a")
        # The function is created, but calling it will fail
        with pytest.raises(NameError):
            func("test")


def test_program_synthesis_with_type_check():
    """Test program synthesis with optional mypy type checking enabled."""
    mock_response = SynthesizedFunction(
        function_name="count_chars",
        param_names=["text"],
        body="    return text.count('a')",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    # With type_check=True, mypy verifies the generated code
    with handler(mock_provider), handler(ProgramSynthesis(type_check=True)):
        count_a = count_char("a")
        assert callable(count_a)
        assert count_a("banana") == 3


def test_program_synthesis_type_check_catches_body_errors():
    """Test that type checking catches type errors in the function body."""
    # Body returns wrong type (str instead of int)
    mock_response = SynthesizedFunction(
        function_name="bad_return",
        param_names=["text"],
        body='    return "not an int"',
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with pytest.raises(SynthesisError, match="Type check failed"):
        with handler(mock_provider), handler(ProgramSynthesis(type_check=True)):
            count_char("a")
