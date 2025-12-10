from collections.abc import Callable

import pytest

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import RetryLLMHandler
from effectful.handlers.llm.synthesis import ProgramSynthesis
from effectful.ops.semantics import NotHandled, handler
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


def test_template_captures_other_templates_in_lexical_context():
    """Test that Templates defined in lexical scope are captured (orchestrator pattern)."""

    # Define sub-templates first
    @Template.define
    def story_with_moral(topic: str) -> str:
        """Write a story about {topic} with a moral lesson."""
        raise NotHandled

    @Template.define
    def story_funny(topic: str) -> str:
        """Write a funny story about {topic}."""
        raise NotHandled

    # Main orchestrator template has access to sub-templates
    @Template.define
    def write_story(topic: str, style: str) -> str:
        """Write a story about {topic} in style {style}."""
        raise NotHandled

    # Sub-templates should be captured in orchestrator's lexical context
    assert "story_with_moral" in write_story.lexical_context
    assert "story_funny" in write_story.lexical_context
    assert write_story.lexical_context["story_with_moral"] is story_with_moral
    assert write_story.lexical_context["story_funny"] is story_funny

    # Templates in lexical context are now exposed as callable tools
    assert story_with_moral in write_story.tools
    assert story_funny in write_story.tools


def test_template_composition_with_chained_calls():
    """Test calling one template and passing result to another."""

    @Template.define
    def generate_topic() -> str:
        """Generate an interesting topic for a story."""
        raise NotHandled

    @Template.define
    def write_story(topic: str) -> str:
        """Write a short story about {topic}."""
        raise NotHandled

    # Verify generate_topic is in write_story's lexical context
    assert "generate_topic" in write_story.lexical_context

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
