"""Tests for LLM type/class synthesis functionality."""

import functools
import os

import pytest

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import LiteLLMProvider, completion
from effectful.handlers.llm.synthesis import SynthesisError
from effectful.handlers.llm.type_synthesis import TypeSynthesis
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# Check for API keys
HAS_OPENAI_KEY = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]

requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY environment variable not set"
)


def retry_on_error(error: type[Exception], n: int):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(n):
                try:
                    return func(*args, **kwargs)
                except error as e:
                    if i < n - 1:
                        continue
                    raise e

        return wrapper

    return decorator


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


# Base class for type synthesis tests
class Animal:
    """Base class for animals."""

    def speak(self) -> str:
        raise NotImplementedError

    def move(self) -> str:
        raise NotImplementedError


@Template.define
def create_animal(behavior: str) -> type[Animal]:
    """Create an Animal subclass with the specified behavior: {behavior}

    The class should implement speak() and move() methods.
    Do not use any tools.
    """
    raise NotHandled


class TestTypeSynthesis:
    """Tests for type (class) synthesis functionality."""

    @requires_openai
    @retry_on_error(error=SynthesisError, n=3)
    def test_generates_subtype(self):
        """Test that TypeSynthesis can generate a subtype of a base class."""
        with (
            handler(LiteLLMProvider(model_name="gpt-4o-mini")),
            handler(TypeSynthesis()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            DogClass = create_animal("a dog that barks and walks")

            # Verify it's a type
            assert isinstance(DogClass, type)
            # Verify it inherits from Animal
            assert issubclass(DogClass, Animal)

            # Create an instance and test methods
            dog = DogClass()
            speak_result = dog.speak()
            move_result = dog.move()

            assert isinstance(speak_result, str)
            assert isinstance(move_result, str)

    @requires_openai
    @retry_on_error(error=SynthesisError, n=3)
    def test_synthesized_type_has_source(self):
        """Test that synthesized types have __source__ attribute."""
        with (
            handler(LiteLLMProvider(model_name="gpt-4o-mini")),
            handler(TypeSynthesis()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            CatClass = create_animal("a cat that meows and prowls")

            assert hasattr(CatClass, "__source__")
            assert "class" in CatClass.__source__
            assert hasattr(CatClass, "__synthesized__")

    def test_type_synthesis_requires_base_in_context(self):
        """Test that type synthesis fails if base type is not in lexical context."""
        import dataclasses

        from effectful.handlers.llm import LexicalContext

        # Create a template with Animal in return type but NOT in context
        @Template.define
        def create_orphan() -> type[Animal]:
            """Create an animal."""
            raise NotHandled

        # Create a modified template with empty context
        orphan_template = dataclasses.replace(
            create_orphan,
            __context__=LexicalContext({}),  # Empty context - no Animal
        )

        with pytest.raises(
            SynthesisError, match="must be in the template's lexical context"
        ):
            with (
                handler(LiteLLMProvider(model_name="gpt-4o-mini")),
                handler(TypeSynthesis()),
            ):
                orphan_template()
