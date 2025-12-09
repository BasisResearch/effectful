from collections.abc import Callable
from dataclasses import dataclass

import pytest

from effectful.handlers.llm import Template
from effectful.handlers.llm.synthesis import (
    ProgramSynthesis,
    SynthesisError,
    SynthesizedFunction,
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
    mock_response = SynthesizedFunction(
        function_name="count_occurrences",
        module_code="""
def count_occurrences(text: str) -> int:
    return text.count('a')
""",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with handler(mock_provider), handler(ProgramSynthesis()):
        count_a = count_char("a")
        assert callable(count_a)
        assert count_a("banana") == 3
        assert count_a("cherry") == 0


def test_count_char_with_untyped_function():
    """Test program synthesis works even when LLM omits type annotations."""
    mock_response = SynthesizedFunction(
        function_name="count_chars",
        module_code="""
def count_chars(s):
    return s.count('x')
""",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with handler(mock_provider), handler(ProgramSynthesis()):
        count_x = count_char("x")
        assert callable(count_x)
        assert count_x("xylophone") == 1
        assert count_x("xxx") == 3


def test_make_greeter_with_program_synthesis():
    """Test program synthesis with custom type (Person) in the signature."""
    mock_response = SynthesizedFunction(
        function_name="greet_person",
        module_code="""
def greet_person(person: Person) -> str:
    return f"Hello, {person.name}!"
""",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with handler(mock_provider), handler(ProgramSynthesis()):
        greeter = make_greeter("formal")
        assert callable(greeter)
        person = Person(name="Alice", age=30)
        assert greeter(person) == "Hello, Alice!"


def test_program_synthesis_invalid_code():
    """Test that synthesis fails when module has syntax errors."""
    mock_response = SynthesizedFunction(
        function_name="bad_func",
        module_code="""
def bad_func(x):
    return this is not valid python
""",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with pytest.raises(SynthesisError, match="Syntax error"):
        with handler(mock_provider), handler(ProgramSynthesis()):
            count_char("a")


def test_program_synthesis_runtime_error():
    """Test that runtime errors propagate when calling the function."""
    mock_response = SynthesizedFunction(
        function_name="bad_func",
        module_code="""
def bad_func(x):
    return undefined_variable
""",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with handler(mock_provider), handler(ProgramSynthesis()):
        func = count_char("a")
        with pytest.raises(NameError):
            func("test")


def test_program_synthesis_with_type_check():
    """Test program synthesis with mypy type checking via type assertion."""
    mock_response = SynthesizedFunction(
        function_name="count_chars",
        module_code="""
def count_chars(text: str) -> int:
    return text.count('a')
""",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with handler(mock_provider), handler(ProgramSynthesis(type_check=True)):
        count_a = count_char("a")
        assert callable(count_a)
        assert count_a("banana") == 3


def test_program_synthesis_type_check_catches_signature_mismatch():
    """Test that type checking catches when function signature doesn't match."""
    # Function returns str but expected int - type assertion will fail
    mock_response = SynthesizedFunction(
        function_name="bad_return",
        module_code="""
def bad_return(text: str) -> str:
    return "not an int"
""",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with pytest.raises(SynthesisError, match="Type check failed"):
        with handler(mock_provider), handler(ProgramSynthesis(type_check=True)):
            count_char("a")


# Helper function for lexical scope test - defined at module level
def double_count(text: str, char: str) -> int:
    """Count occurrences of a character and double it."""
    return text.count(char) * 2


# Template that captures the lexical function above
@Template.define
def make_double_counter(char: str) -> Callable[[str], int]:
    """Create a function that counts occurrences of '{char}' and doubles the result.
    Use the double_count helper function."""
    raise NotImplementedError


def test_program_synthesis_with_lexical_function():
    """Test that synthesized code can use functions from the lexical scope."""
    mock_response = SynthesizedFunction(
        function_name="count_and_double",
        module_code="""
def count_and_double(text: str) -> int:
    return double_count(text, 'a')
""",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with handler(mock_provider), handler(ProgramSynthesis()):
        counter = make_double_counter("a")
        assert callable(counter)
        assert counter("banana") == 6  # 3 'a's doubled
        assert counter("cherry") == 0


def test_program_synthesis_lexical_function_in_prompt():
    """Test that lexical functions are included in the template's context."""
    assert "double_count" in make_double_counter.lexical_context
    source, func = make_double_counter.lexical_context["double_count"]
    assert "Count occurrences of a character and double it" in source
    assert func is double_count


def test_program_synthesis_with_helper_in_module():
    """Test that module can include helper functions."""
    mock_response = SynthesizedFunction(
        function_name="count_and_triple",
        module_code="""
def multiply_by_three(n: int) -> int:
    return n * 3

def count_and_triple(text: str) -> int:
    return multiply_by_three(text.count('a'))
""",
    )
    mock_provider = SingleResponseLLMProvider(mock_response)

    with handler(mock_provider), handler(ProgramSynthesis()):
        counter = count_char("a")
        assert callable(counter)
        assert counter("banana") == 9  # 3 'a's tripled
        assert counter("aardvark") == 9
