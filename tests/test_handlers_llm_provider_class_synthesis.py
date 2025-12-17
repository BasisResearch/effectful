"""Tests for LLM type/class synthesis functionality."""

import inspect
import logging
import sys

import pytest

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import LiteLLMProvider, LLMLoggingHandler
from effectful.handlers.llm.synthesis import SynthesisError
from effectful.handlers.llm.type_synthesis import TypeSynthesis
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

from .test_handlers_llm_provider import (
    LimitLLMCallsHandler,
    requires_openai,
    retry_on_error,
)


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
        logger = logging.getLogger("effectful.llm")
        logger.setLevel(logging.INFO)
        log_handler = logging.StreamHandler(sys.stdout)
        log_handler.setFormatter(logging.Formatter("%(levelname)s %(payload)s"))
        logger.addHandler(log_handler)
        llm_logger = LLMLoggingHandler(logger=logger)
        with (
            handler(LiteLLMProvider(model_name="gpt-4o-mini")),
            handler(TypeSynthesis()),
            handler(LimitLLMCallsHandler(max_calls=1)),
            handler(llm_logger),
        ):
            CatClass = create_animal("a cat that meows and prowls")

            source = inspect.getsource(CatClass)
            assert hasattr(CatClass, "__synthesized__")
            assert hasattr(CatClass, "__source__")
            assert source == CatClass.__source__

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
