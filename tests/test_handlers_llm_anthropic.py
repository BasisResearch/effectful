"""Integration tests for AnthropicProvider-specific behavior.

Tests that are shared with LiteLLMProvider (basic prompts, structured output,
tool calling) live in the existing test modules via parametrization. This file
covers AnthropicProvider-only concerns: agent history, handler composition, and
Claude Max configuration.
"""

import dataclasses
import os

import pytest

from effectful.handlers.llm import Agent, Template
from effectful.handlers.llm.completions import (
    RetryLLMHandler,
    _get_history,
    call_assistant,
)
from effectful.ops.semantics import coproduct, fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# Reuse helpers from existing test modules
from tests.test_handlers_llm_provider import (
    LimitLLMCallsHandler,
    simple_prompt,
)

try:
    from effectful.handlers.llm.anthropic import (
        AnthropicProvider,
        _get_oauth_token_from_keychain,
    )

    HAS_ANTHROPIC_SDK = True
except ImportError:
    HAS_ANTHROPIC_SDK = False
    _get_oauth_token_from_keychain = lambda: None  # noqa: E731

HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_CLAUDE_MAX = HAS_ANTHROPIC_SDK and _get_oauth_token_from_keychain() is not None
HAS_ANTHROPIC_AUTH = HAS_ANTHROPIC_KEY or HAS_CLAUDE_MAX

requires_anthropic_sdk = pytest.mark.skipif(
    not HAS_ANTHROPIC_SDK or not HAS_ANTHROPIC_AUTH,
    reason="anthropic package not installed or no auth available (API key or Claude Max)",
)

# Claude Max uses short model IDs; API key uses full dated IDs
ANTHROPIC_MODEL = "claude-haiku-4-5" if (HAS_CLAUDE_MAX and not HAS_ANTHROPIC_KEY) else "claude-haiku-4-5-20250514"
_USE_MAX = HAS_CLAUDE_MAX and not HAS_ANTHROPIC_KEY


def _provider(**overrides):
    kwargs = {"model": ANTHROPIC_MODEL, "max_tokens": 1024, "max_subscription": _USE_MAX}
    kwargs.update(overrides)
    return AnthropicProvider(**kwargs)


# ============================================================================
# Agent conversation history (requires its own Template definitions for
# lexical scope isolation)
# ============================================================================


@dataclasses.dataclass
class ChatBot(Agent):
    """A simple chat bot for testing conversation history."""

    bot_name: str = "TestBot"

    @Template.define
    def send(self, user_input: str) -> str:
        """You are {self.bot_name}. Respond briefly. User says: {user_input}"""
        raise NotHandled


@requires_anthropic_sdk
class TestAnthropicAgent:

    def test_agent_remembers_conversation_history(self):
        with (
            handler(_provider()),
            handler(RetryLLMHandler()),
            handler(LimitLLMCallsHandler(max_calls=4)),
        ):
            bot = ChatBot()
            bot.send("My name is Alice and I live in Paris.")
            response2 = bot.send("What is my name?")
            assert "alice" in response2.lower()


# ============================================================================
# Handler composition
# ============================================================================


@requires_anthropic_sdk
class TestAnthropicHandlerComposition:

    def test_coproduct_with_retry(self):
        with handler(coproduct(_provider(), RetryLLMHandler())):
            result = simple_prompt("hello")
            assert isinstance(result, str)
            assert len(result) > 0

    def test_logging_handler_sees_call_assistant(self):
        call_log: list[str] = []

        class LoggingHandler(ObjectInterpretation):
            @implements(call_assistant)
            def _log(self, *args, **kwargs):
                call_log.append("call_assistant")
                return fwd()

        with (
            handler(_provider()),
            handler(LoggingHandler()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            simple_prompt("test")

        assert call_log == ["call_assistant"]

    def test_message_history_populated(self):
        captured_history = None

        class HistoryCapture(ObjectInterpretation):
            @implements(call_assistant)
            def _capture(self, *args, **kwargs):
                nonlocal captured_history
                captured_history = dict(_get_history())
                return fwd()

        with (
            handler(_provider()),
            handler(HistoryCapture()),
            handler(LimitLLMCallsHandler(max_calls=1)),
        ):
            simple_prompt("test")

        assert captured_history is not None
        roles = [m["role"] for m in captured_history.values()]
        assert "system" in roles
        assert "user" in roles
