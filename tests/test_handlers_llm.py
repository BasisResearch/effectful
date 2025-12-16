from collections.abc import Callable
from dataclasses import dataclass

import pytest

import pytest

from effectful.handlers.llm import Template
<<<<<<< HEAD
from effectful.handlers.llm.synthesis import (
    ProgramSynthesis,
    SynthesisError,
    SynthesizedFunction,
)
from effectful.ops.semantics import handler
=======
from effectful.handlers.llm.providers import RetryLLMHandler
from effectful.handlers.llm.synthesis import ProgramSynthesis
from effectful.ops.semantics import NotHandled, handler
>>>>>>> 931d5071d3f386a224cf46c103ca1905fa3c12df
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
    raise NotHandled


@Template.define
def haiku(theme: str) -> str:
    """Write a haiku on the theme of {theme}."""
    raise NotHandled


@Template.define()
def primes(first_digit: int) -> int:
    """Give exactly one prime number with {first_digit} as the first digit. Respond with only the number."""
    raise NotHandled


@Template.define
def count_char(char: str) -> Callable[[str], int]:
    """Write a function which takes a string and counts the occurrances of '{char}'."""
    raise NotHandled


# Mutually recursive templates (module-level for live globals)
@Template.define
def mutual_a() -> str:
    """Use mutual_a and mutual_b as tools to do task A."""
    raise NotHandled


@Template.define
def mutual_b() -> str:
    """Use mutual_a and mutual_b as tools to do task B."""
    raise NotHandled


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


<<<<<<< HEAD
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
=======
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


def test_template_captures_other_templates_in_lexical_context():
    """Test that Templates defined in lexical scope are captured (orchestrator pattern)."""

    # Define sub-templates first
    @Template.define
    def story_with_moral(topic: str) -> str:
        """Write a story about {topic} with a moral lesson. Do not use any tools at all for this."""
        raise NotHandled

    @Template.define
    def story_funny(topic: str) -> str:
        """Write a funny story about {topic}. Do not use any tools at all for this."""
        raise NotHandled

    # Main orchestrator template has access to sub-templates
    @Template.define
    def write_story(topic: str, style: str) -> str:
        """Write a story about {topic} in style {style}."""
        raise NotHandled

    # __context__ is a ChainMap(locals, globals) - locals shadow globals
    # Sub-templates should be visible in lexical context
    assert "story_with_moral" in write_story.__context__
    assert "story_funny" in write_story.__context__
    assert write_story.__context__["story_with_moral"] is story_with_moral
    assert write_story.__context__["story_funny"] is story_funny

    # Templates in lexical context are exposed as callable tools
    assert story_with_moral in write_story.tools
    assert story_funny in write_story.tools


def test_template_composition_with_chained_calls():
    """Test calling one template and passing result to another."""

    @Template.define
    def generate_topic() -> str:
        """Generate an interesting topic for a story. Do not try to use any tools for this beside from write_story."""
        raise NotHandled

    @Template.define
    def write_story(topic: str) -> str:
        """Write a short story about {topic}."""
        raise NotHandled

    # Verify generate_topic is in write_story's lexical context
    assert "generate_topic" in write_story.__context__

    # Test chained template calls
    mock_provider = SingleResponseLLMProvider("A magical forest")

    with handler(mock_provider):
        topic = generate_topic()
        assert topic == "A magical forest"

    # Now use that topic in the next template
    mock_provider2 = SingleResponseLLMProvider(
        "Once upon a time in a magical forest..."
    )

    with handler(mock_provider2):
        story = write_story(topic)
        assert story == "Once upon a time in a magical forest..."


def test_mutually_recursive_templates():
    """Test that module-level templates can see each other (mutual recursion)."""
    # Both mutual_a and mutual_b should see each other via ChainMap (globals visible)
    assert "mutual_a" in mutual_a.__context__
    assert "mutual_b" in mutual_a.__context__
    assert "mutual_a" in mutual_b.__context__
    assert "mutual_b" in mutual_b.__context__

    # They should also be in each other's tools
    assert mutual_a in mutual_b.tools
    assert mutual_b in mutual_a.tools
    # And themselves (self-recursion)
    assert mutual_a in mutual_a.tools
    assert mutual_b in mutual_b.tools


# Module-level variable for shadowing test
shadow_test_value = "global"


def test_lexical_context_shadowing():
    """Test that local variables shadow global variables in lexical context."""
    # Local shadows global
    shadow_test_value = "local"  # noqa: F841 - intentional shadowing

    @Template.define
    def template_with_shadowed_var() -> str:
        """Test template."""
        raise NotHandled

    # The lexical context should see the LOCAL value, not global
    assert "shadow_test_value" in template_with_shadowed_var.__context__
    assert (
        template_with_shadowed_var.__context__["shadow_test_value"] == shadow_test_value
    )


def test_lexical_context_sees_globals_when_no_local():
    """Test that globals are visible when there's no local shadow."""

    @Template.define
    def template_sees_global() -> str:
        """Test template."""
        raise NotHandled

    # Should see the global value (no local shadow in this scope)
    assert "shadow_test_value" in template_sees_global.__context__
    assert template_sees_global.__context__["shadow_test_value"] == "global"
>>>>>>> 931d5071d3f386a224cf46c103ca1905fa3c12df
