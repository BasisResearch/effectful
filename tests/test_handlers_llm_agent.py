"""Tests for Agent mixin message sequence semantics."""

import collections
import dataclasses

from litellm import ModelResponse

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    RetryLLMHandler,
    completion,
)
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Helpers (same pattern as test_handlers_llm_provider.py)
# ---------------------------------------------------------------------------


def make_text_response(content: str) -> ModelResponse:
    return ModelResponse(
        id="test",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        model="test-model",
    )


def make_tool_call_response(
    tool_name: str, tool_args: str, tool_call_id: str = "call_1"
) -> ModelResponse:
    return ModelResponse(
        id="test",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        model="test-model",
    )


class MockCompletionHandler(ObjectInterpretation):
    """Returns pre-configured responses and captures messages sent to the LLM."""

    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.call_count = 0
        self.received_messages: list[list] = []

    @implements(completion)
    def _completion(self, model, messages=None, **kwargs):
        self.received_messages.append(list(messages) if messages else [])
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


# ---------------------------------------------------------------------------
# Agent subclass used by most tests
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ChatBot(Agent):
    """Simple chat agent for testing history accumulation."""

    bot_name: str = dataclasses.field(default="ChatBot")

    @Template.define
    def send(self, user_input: str) -> str:
        """A friendly bot named {self.bot_name}. User writes: {user_input}"""
        raise NotHandled


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentHistoryAccumulation:
    """History accumulates across sequential calls on the same instance."""

    def test_second_call_sees_prior_messages(self):
        mock = MockCompletionHandler(
            [
                make_text_response('{"value": "hi"}'),
                make_text_response('{"value": "good"}'),
            ]
        )
        bot = ChatBot()

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("hello")
            bot.send("how are you")

        # First call: system + user → 2 messages
        assert len(mock.received_messages[0]) > 0

        # Second call: previous system + user + assistant, PLUS new system + user → 5
        assert len(mock.received_messages[1]) > len(mock.received_messages[0])

        # Verify roles in second call
        roles = [m["role"] for m in mock.received_messages[1]]
        assert roles.count("assistant") >= 1
        assert roles.count("user") >= 2

    def test_history_contains_all_messages_after_two_calls(self):
        mock = MockCompletionHandler(
            [
                make_text_response('{"value": "r1"}'),
                make_text_response('{"value": "r2"}'),
            ]
        )
        bot = ChatBot()

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("a")
            bot.send("b")

        # After two complete calls the history should have:
        #   call 1: system, user, assistant  (3)
        #   call 2: system, user, assistant  (3)
        assert len(bot.__history__) >= 4

    def test_message_ids_are_unique(self):
        mock = MockCompletionHandler(
            [
                make_text_response('{"value": "r1"}'),
                make_text_response('{"value": "r2"}'),
            ]
        )
        bot = ChatBot()

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("a")
            bot.send("b")

        ids = list(bot.__history__.keys())
        assert len(ids) == len(set(ids)), "message IDs must be unique"


class TestAgentIsolation:
    """Each agent instance has independent history; non-agent templates are unaffected."""

    def test_two_agents_have_independent_histories(self):
        mock = MockCompletionHandler(
            [
                make_text_response('{"value": "from bot1"}'),
                make_text_response('{"value": "from bot2"}'),
            ]
        )
        bot1 = ChatBot()
        bot2 = ChatBot()

        with handler(LiteLLMProvider()), handler(mock):
            bot1.send("msg for bot1")
            bot2.send("msg for bot2")

        # bot2's call should NOT contain bot1's messages
        assert len(mock.received_messages[1]) >= 1  # system + user only

        # Each bot has its own history
        assert len(bot1.__history__) >= 2  # system, user, assistant
        assert len(bot2.__history__) >= 2

        # Histories share no message IDs
        assert set(bot1.__history__.keys()).isdisjoint(set(bot2.__history__.keys()))

    def test_non_agent_template_gets_fresh_sequence(self):
        @Template.define
        def standalone(topic: str) -> str:
            """Write about {topic}."""
            raise NotHandled

        mock = MockCompletionHandler(
            [
                make_text_response('{"value": "agent reply"}'),
                make_text_response('{"value": "standalone reply"}'),
                make_text_response('{"value": "agent reply 2"}'),
            ]
        )
        bot = ChatBot()

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("hello")
            standalone("fish")
            bot.send("bye")

        # standalone (call index 1) should see only system + user (fresh sequence)
        assert len(mock.received_messages[1]) >= 1

        # bot's third call (call index 2) should see its accumulated history
        # but NOT the standalone messages
        assert len(mock.received_messages[2]) >= 3


class TestAgentCachedProperty:
    """__history__ is lazily created per instance without requiring __init__."""

    def test_no_init_required(self):
        class MinimalAgent(Agent):
            @Template.define
            def greet(self, name: str) -> str:
                """Hello {name}."""
                raise NotHandled

        agent = MinimalAgent()
        # Should be an OrderedDict, created on first access
        assert isinstance(agent.__history__, collections.OrderedDict)
        assert len(agent.__history__) == 0

    def test_subclass_with_own_init(self):
        class CustomAgent(Agent):
            def __init__(self, name: str):
                self.name = name

            @Template.define
            def greet(self) -> str:
                """Say hello."""
                raise NotHandled

        agent = CustomAgent("Alice")
        assert agent.name == "Alice"
        assert isinstance(agent.__history__, collections.OrderedDict)

    def test_history_is_per_instance(self):
        a = ChatBot()
        b = ChatBot()
        a.__history__["fake"] = {"id": "fake", "role": "user", "content": "x"}
        assert "fake" not in b.__history__


class TestAgentWithToolCalls:
    """Agent methods that trigger tool calls maintain correct history."""

    def test_tool_call_results_appear_in_history(self):
        @Tool.define
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        class MathAgent(Agent):
            @Template.define
            def compute(self, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        mock = MockCompletionHandler(
            [
                make_tool_call_response("add", '{"a": 2, "b": 3}'),
                make_text_response('{"value": "The answer is 5"}'),
            ]
        )
        agent = MathAgent()

        with handler(LiteLLMProvider()), handler(mock):
            result = agent.compute("what is 2+3?")

        assert result == "The answer is 5"

        # History should contain: system, user, assistant (tool_call),
        # tool (result), assistant (final)
        roles = [m["role"] for m in agent.__history__.values()]
        assert "tool" in roles
        assert roles.count("assistant") == 2


class TestAgentWithRetryHandler:
    """RetryLLMHandler composes correctly with Agent history."""

    def test_failed_retries_dont_pollute_history(self):
        mock = MockCompletionHandler(
            [
                # First attempt: invalid result for int
                make_text_response('{"value": "not_an_int"}'),
                # Retry: valid
                make_text_response('{"value": 42}'),
            ]
        )

        class NumberAgent(Agent):
            @Template.define
            def pick_number(self) -> int:
                """Pick a number."""
                raise NotHandled

        agent = NumberAgent()

        with (
            handler(LiteLLMProvider()),
            handler(RetryLLMHandler(num_retries=3)),
            handler(mock),
        ):
            result = agent.pick_number()

        assert result == 42

        # The malformed assistant message and error feedback from the retry
        # should NOT appear in the agent's history. Only the final successful
        # assistant message should be there.
        roles = {m["role"] for m in agent.__history__.values()}
        assert {"user", "assistant"} == roles - {"system"}
