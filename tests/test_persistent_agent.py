"""Tests for PersistentAgent + PersistenceHandler + CompactionHandler.

Checkpointing, compaction, crash recovery, nested calls, subclass state
persistence, and system prompt augmentation.
"""

import dataclasses
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
from litellm import ModelResponse

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import (
    AgentHistoryHandler,
    LiteLLMProvider,
    RetryLLMHandler,
    ToolCallExecutionError,
    completion,
    get_agent_history,
)
from effectful.handlers.llm.persistence import (
    CompactionHandler,
    PersistenceHandler,
    PersistentAgent,
)
from effectful.handlers.llm.template import get_bound_agent
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Helpers
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
# Test agents
# ---------------------------------------------------------------------------


class ChatBot(PersistentAgent):
    """You are a persistent chat bot for testing."""

    @Template.define
    def send(self, user_input: str) -> str:
        """User says: {user_input}"""
        raise NotHandled


@dataclasses.dataclass
class StatefulBot(PersistentAgent):
    """You are a stateful bot that tracks learned patterns."""

    __agent_id__ = "StatefulBot"

    learned_patterns: list[str] = dataclasses.field(default_factory=list)
    call_count: int = 0

    @Template.define
    def send(self, user_input: str) -> str:
        """User says: {user_input}"""
        raise NotHandled


class NestedBot(PersistentAgent):
    """You are a nested-call test bot."""

    @Template.define
    def inner_check(self, payload: str) -> str:
        """Check: {payload}. Do not use tools."""
        raise NotHandled

    @Tool.define
    def check_tool(self, payload: str) -> str:
        """Check payload by calling an inner template."""
        return self.inner_check(payload)

    @Template.define
    def outer(self, payload: str) -> str:
        """Call `check_tool` for: {payload}, then return final answer."""
        raise NotHandled


# ---------------------------------------------------------------------------
# Tests: Agent.__agent_id__
# ---------------------------------------------------------------------------


class TestAgentId:
    """All Agent subclasses get agent_id."""

    def test_plain_agent_defaults_to_id(self):
        class PlainAgent(Agent):
            """Plain."""

            @Template.define
            def ask(self, q: str) -> str:
                """Q: {q}"""
                raise NotHandled

        agent = PlainAgent()
        assert agent.__agent_id__ == str(id(agent))

    def test_persistent_agent_requires_agent_id(self):
        bot = ChatBot(agent_id="my-chatbot")
        assert bot.__agent_id__ == "my-chatbot"

    def test_dataclass_class_level_id(self):
        """Dataclass subclasses can set __agent_id__ as a class attribute."""
        bot = StatefulBot()
        assert bot.__agent_id__ == "StatefulBot"

    def test_bound_template_has_agent_via_context(self):
        bot = ChatBot(agent_id="ChatBot")
        bound = bot.send
        assert get_bound_agent(bound) is bot


# ---------------------------------------------------------------------------
# Tests: basic persistence (PersistenceHandler)
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """PersistenceHandler save/load round-trip correctly."""

    def test_save_creates_file(self, tmp_path: Path):
        persist = PersistenceHandler(tmp_path)
        bot = ChatBot(agent_id="ChatBot")
        with handler(AgentHistoryHandler()):
            path = persist.save(bot)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_load_round_trip_empty(self, tmp_path: Path):
        persist = PersistenceHandler(tmp_path)
        bot = ChatBot(agent_id="ChatBot")
        with handler(AgentHistoryHandler()):
            persist.save(bot)

        hist = AgentHistoryHandler()
        bot2 = ChatBot(agent_id="ChatBot")
        persist2 = PersistenceHandler(tmp_path)
        with handler(hist):
            persist2.ensure_loaded(bot2)
        assert len(hist._histories.get("ChatBot", {})) == 0
        assert persist2.get_handoff("ChatBot") == ""

    def test_save_load_round_trip_with_history(self, tmp_path: Path):
        hist = AgentHistoryHandler()
        persist = PersistenceHandler(tmp_path)
        bot = ChatBot(agent_id="ChatBot")
        with handler(hist):
            history = get_agent_history(bot.__agent_id__)
            history["msg1"] = {
                "id": "msg1",
                "role": "user",
                "content": "hello",
            }
            persist._handoffs["ChatBot"] = "working on X"
            persist.save(bot)

        hist2 = AgentHistoryHandler()
        bot2 = ChatBot(agent_id="ChatBot")
        persist2 = PersistenceHandler(tmp_path)
        with handler(hist2):
            persist2.ensure_loaded(bot2)
            loaded_history = get_agent_history(bot2.__agent_id__)
        assert persist2.get_handoff("ChatBot") == "working on X"
        assert len(loaded_history) == 1
        assert loaded_history["msg1"]["content"] == "hello"

    def test_load_returns_false_when_no_checkpoint(self, tmp_path: Path):
        persist = PersistenceHandler(tmp_path)
        bot = ChatBot(agent_id="ChatBot")
        with handler(AgentHistoryHandler()):
            assert not persist.ensure_loaded(bot)

    def test_load_returns_true_when_checkpoint_exists(self, tmp_path: Path):
        persist = PersistenceHandler(tmp_path)
        bot = ChatBot(agent_id="ChatBot")
        with handler(AgentHistoryHandler()):
            persist.save(bot)
            # Reset loaded state
            persist._loaded.clear()
            assert persist.ensure_loaded(bot)

    def test_atomic_write(self, tmp_path: Path):
        """Checkpoint write uses tmp + rename for atomicity."""
        persist = PersistenceHandler(tmp_path)
        bot = ChatBot(agent_id="ChatBot")
        with handler(AgentHistoryHandler()):
            path = persist.save(bot)
        assert not path.with_suffix(".tmp").exists()

    def test_custom_agent_id(self, tmp_path: Path):
        persist = PersistenceHandler(tmp_path)

        class CustomBot(PersistentAgent):
            """Custom."""

            @Template.define
            def ask(self, q: str) -> str:
                """Q: {q}"""
                raise NotHandled

        bot = CustomBot(agent_id="custom-bot")
        with handler(AgentHistoryHandler()):
            persist.save(bot)
        assert (tmp_path / "custom-bot.json").exists()

    def test_persist_dir_created_automatically(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        persist = PersistenceHandler(nested)
        bot = ChatBot(agent_id="ChatBot")
        with handler(AgentHistoryHandler()):
            persist.save(bot)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Tests: subclass state persistence (checkpoint_state / restore_state)
# ---------------------------------------------------------------------------


class TestSubclassStatePersistence:
    """Dataclass fields on subclasses are automatically persisted."""

    def test_dataclass_fields_round_trip(self, tmp_path: Path):
        persist = PersistenceHandler(tmp_path)
        bot = StatefulBot()
        bot.learned_patterns = ["pattern A", "pattern B"]
        bot.call_count = 5
        with handler(AgentHistoryHandler()):
            persist.save(bot)

        bot2 = StatefulBot()
        persist2 = PersistenceHandler(tmp_path)
        with handler(AgentHistoryHandler()):
            persist2.ensure_loaded(bot2)
        assert bot2.learned_patterns == ["pattern A", "pattern B"]
        assert bot2.call_count == 5

    def test_non_dataclass_has_empty_state(self):
        """Non-dataclass subclass returns empty state dict."""
        bot = ChatBot(agent_id="ChatBot")
        assert bot.checkpoint_state() == {}

    def test_non_serializable_fields_skipped(self, tmp_path: Path):
        @dataclasses.dataclass
        class WeirdBot(PersistentAgent):
            """Bot with a non-serializable field."""

            __agent_id__ = "WeirdBot"

            callback: object = dataclasses.field(default=None)
            name: str = "test"

            @Template.define
            def send(self, msg: str) -> str:
                """Say: {msg}"""
                raise NotHandled

        persist = PersistenceHandler(tmp_path)
        bot = WeirdBot()
        bot.callback = lambda x: x  # not JSON serializable
        bot.name = "Alice"
        with handler(AgentHistoryHandler()):
            persist.save(bot)

        bot2 = WeirdBot()
        persist2 = PersistenceHandler(tmp_path)
        with handler(AgentHistoryHandler()):
            persist2.ensure_loaded(bot2)
        assert bot2.name == "Alice"
        assert bot2.callback is None

    def test_custom_checkpoint_restore(self, tmp_path: Path):
        """Users can override checkpoint_state / restore_state."""

        class CustomBot(PersistentAgent):
            """Custom serialisation bot."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.data = {"counter": 0}

            def checkpoint_state(self):
                return {"data": self.data}

            def restore_state(self, state):
                self.data = state.get("data", {"counter": 0})

            @Template.define
            def send(self, msg: str) -> str:
                """Say: {msg}"""
                raise NotHandled

        persist = PersistenceHandler(tmp_path)
        bot = CustomBot(agent_id="CustomBot")
        bot.data["counter"] = 42
        with handler(AgentHistoryHandler()):
            persist.save(bot)

        bot2 = CustomBot(agent_id="CustomBot")
        persist2 = PersistenceHandler(tmp_path)
        with handler(AgentHistoryHandler()):
            persist2.ensure_loaded(bot2)
        assert bot2.data["counter"] == 42

    def test_state_saved_in_checkpoint_file(self, tmp_path: Path):
        """The checkpoint JSON contains a 'state' key with subclass fields."""
        persist = PersistenceHandler(tmp_path)
        bot = StatefulBot()
        bot.learned_patterns = ["X"]
        bot.call_count = 3
        with handler(AgentHistoryHandler()):
            persist.save(bot)

        data = json.loads((tmp_path / "StatefulBot.json").read_text())
        assert "state" in data
        assert data["state"]["learned_patterns"] == ["X"]
        assert data["state"]["call_count"] == 3


# ---------------------------------------------------------------------------
# Tests: automatic checkpointing around template calls
# ---------------------------------------------------------------------------


class TestAutomaticCheckpointing:
    """Template calls on PersistentAgent trigger auto-checkpointing."""

    def test_checkpoint_saved_after_successful_call(self, tmp_path: Path):
        mock = MockCompletionHandler([make_text_response("hello")])
        persist = PersistenceHandler(tmp_path)
        bot = ChatBot(agent_id="ChatBot")

        with (
            handler(LiteLLMProvider()),
            handler(mock),
            handler(persist),
        ):
            bot.send("hi")

        cp = tmp_path / "ChatBot.json"
        assert cp.exists()
        data = json.loads(cp.read_text())
        assert len(data["history"]) > 0
        assert data["handoff"] == ""

    def test_checkpoint_saved_on_exception(self, tmp_path: Path):
        class FailingMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self, *args, **kwargs):
                raise RuntimeError("boom")

        persist = PersistenceHandler(tmp_path)
        bot = ChatBot(agent_id="ChatBot")
        with pytest.raises(RuntimeError, match="boom"):
            with (
                handler(LiteLLMProvider()),
                handler(FailingMock()),
                handler(persist),
            ):
                bot.send("hi")

        data = json.loads((tmp_path / "ChatBot.json").read_text())
        assert "Executing send" in data["handoff"]

    def test_handoff_describes_current_call(self, tmp_path: Path):
        """Before the template runs, handoff records what's in progress."""
        handoff_during_call = []

        class SpyMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self_, model, messages=None, **kwargs):
                data = json.loads((tmp_path / "ChatBot.json").read_text())
                handoff_during_call.append(data["handoff"])
                return make_text_response("ok")

        persist = PersistenceHandler(tmp_path)
        bot = ChatBot(agent_id="ChatBot")
        with (
            handler(LiteLLMProvider()),
            handler(SpyMock()),
            handler(persist),
        ):
            bot.send("hello")

        assert len(handoff_during_call) == 1
        assert "Executing send" in handoff_during_call[0]

    def test_history_persists_across_sessions(self, tmp_path: Path):
        mock = MockCompletionHandler([make_text_response("reply1")])
        bot = ChatBot(agent_id="ChatBot")
        provider1 = LiteLLMProvider()

        with (
            handler(provider1),
            handler(mock),
            handler(PersistenceHandler(tmp_path)),
        ):
            bot.send("first message")

        history_after_first = len(provider1._histories.get("ChatBot", {}))

        # "Restart" — new handler + new agent instance
        mock2 = MockCompletionHandler([make_text_response("reply after restart")])
        bot2 = ChatBot(agent_id="ChatBot")
        provider2 = LiteLLMProvider()

        with (
            handler(provider2),
            handler(mock2),
            handler(PersistenceHandler(tmp_path)),
        ):
            bot2.send("second message")

        assert len(provider2._histories.get("ChatBot", {})) > history_after_first

    def test_second_call_sees_prior_history(self, tmp_path: Path):
        mock = MockCompletionHandler(
            [make_text_response("r1"), make_text_response("r2")]
        )
        bot = ChatBot(agent_id="ChatBot")

        with (
            handler(LiteLLMProvider()),
            handler(mock),
            handler(PersistenceHandler(tmp_path)),
        ):
            bot.send("a")
            bot.send("b")

        assert len(mock.received_messages[1]) > len(mock.received_messages[0])

    def test_dataclass_state_saved_around_template_calls(self, tmp_path: Path):
        mock = MockCompletionHandler([make_text_response("ok")])
        bot = StatefulBot()
        bot.call_count = 7

        with (
            handler(LiteLLMProvider()),
            handler(mock),
            handler(PersistenceHandler(tmp_path)),
        ):
            bot.send("test")

        data = json.loads((tmp_path / "StatefulBot.json").read_text())
        assert data["state"]["call_count"] == 7


# ---------------------------------------------------------------------------
# Tests: crash recovery
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    """Handoff notes enable resumption after crashes."""

    def test_handoff_survives_crash(self, tmp_path: Path):
        class CrashMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self, *args, **kwargs):
                raise RuntimeError("process killed")

        bot = ChatBot(agent_id="ChatBot")
        with pytest.raises(RuntimeError):
            with (
                handler(LiteLLMProvider()),
                handler(CrashMock()),
                handler(PersistenceHandler(tmp_path)),
            ):
                bot.send("important task")

        # New handler instance loads handoff from disk
        persist2 = PersistenceHandler(tmp_path)
        bot2 = ChatBot(agent_id="ChatBot")
        with handler(AgentHistoryHandler()):
            persist2.ensure_loaded(bot2)
        assert "Executing send" in persist2.get_handoff("ChatBot")

    def test_system_prompt_includes_handoff(self, tmp_path: Path):
        """After a crash, the next call's system prompt includes the handoff."""

        # Simulate crash
        class CrashMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self, *args, **kwargs):
                raise RuntimeError("crash")

        bot = ChatBot(agent_id="ChatBot")
        with pytest.raises(RuntimeError):
            with (
                handler(LiteLLMProvider()),
                handler(CrashMock()),
                handler(PersistenceHandler(tmp_path)),
            ):
                bot.send("important task")

        # Next session: spy on system prompt
        system_prompts = []

        class SpyMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self_, model, messages=None, **kwargs):
                system_prompts.extend(
                    m.get("content", "")
                    for m in (messages or [])
                    if m.get("role") == "system"
                )
                return make_text_response("resumed")

        bot2 = ChatBot(agent_id="ChatBot")
        with (
            handler(LiteLLMProvider()),
            handler(SpyMock()),
            handler(PersistenceHandler(tmp_path)),
        ):
            bot2.send("resume")

        assert any("[HANDOFF FROM PRIOR SESSION]" in p for p in system_prompts)

    def test_handoff_cleared_on_success(self, tmp_path: Path):
        # Create crash checkpoint
        class CrashMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self, *a, **kw):
                raise RuntimeError("crash")

        bot = ChatBot(agent_id="ChatBot")
        with pytest.raises(RuntimeError):
            with (
                handler(LiteLLMProvider()),
                handler(CrashMock()),
                handler(PersistenceHandler(tmp_path)),
            ):
                bot.send("crash task")

        # Successful run clears handoff
        mock = MockCompletionHandler([make_text_response("done")])
        bot2 = ChatBot(agent_id="ChatBot")
        persist = PersistenceHandler(tmp_path)
        with (
            handler(LiteLLMProvider()),
            handler(mock),
            handler(persist),
        ):
            bot2.send("new task")

        assert persist.get_handoff("ChatBot") == ""
        data = json.loads((tmp_path / "ChatBot.json").read_text())
        assert data["handoff"] == ""

    def test_dataclass_state_survives_crash(self, tmp_path: Path):
        class CrashMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self, *a, **kw):
                raise RuntimeError("crash")

        bot = StatefulBot()
        bot.learned_patterns = ["important insight"]
        bot.call_count = 3

        with pytest.raises(RuntimeError):
            with (
                handler(LiteLLMProvider()),
                handler(CrashMock()),
                handler(PersistenceHandler(tmp_path)),
            ):
                bot.send("boom")

        bot2 = StatefulBot()
        persist2 = PersistenceHandler(tmp_path)
        with handler(AgentHistoryHandler()):
            persist2.ensure_loaded(bot2)
        assert bot2.learned_patterns == ["important insight"]
        assert bot2.call_count == 3


# ---------------------------------------------------------------------------
# Tests: nested template calls
# ---------------------------------------------------------------------------


class TestNestedCalls:
    """Only outermost template call triggers checkpointing."""

    def test_nested_template_via_tool_completes(self, tmp_path: Path):
        mock = MockCompletionHandler(
            [
                make_tool_call_response("self__check_tool", '{"payload": "demo"}'),
                make_text_response("inner result"),
                make_text_response("outer result"),
            ]
        )
        bot = NestedBot(agent_id="NestedBot")

        with (
            handler(LiteLLMProvider()),
            handler(mock),
            handler(PersistenceHandler(tmp_path)),
        ):
            result = bot.outer("demo")

        assert result == "outer result"

    def test_nested_call_does_not_double_checkpoint(self, tmp_path: Path):
        save_count = 0
        persist = PersistenceHandler(tmp_path)
        original_save = PersistenceHandler.save

        def counting_save(self, agent):
            nonlocal save_count
            save_count += 1
            return original_save(self, agent)

        mock = MockCompletionHandler(
            [
                make_tool_call_response("self__check_tool", '{"payload": "demo"}'),
                make_text_response("inner"),
                make_text_response("outer"),
            ]
        )
        bot = NestedBot(agent_id="NestedBot")
        PersistenceHandler.save = counting_save  # type: ignore[method-assign]
        try:
            with (
                handler(LiteLLMProvider()),
                handler(mock),
                handler(persist),
            ):
                bot.outer("demo")
        finally:
            PersistenceHandler.save = original_save  # type: ignore[method-assign]

        # Should be exactly 2: one before call, one after
        assert save_count == 2

    def test_handoff_cleared_after_nested_success(self, tmp_path: Path):
        mock = MockCompletionHandler(
            [
                make_tool_call_response("self__check_tool", '{"payload": "demo"}'),
                make_text_response("inner"),
                make_text_response("outer"),
            ]
        )
        persist = PersistenceHandler(tmp_path)
        bot = NestedBot(agent_id="NestedBot")

        with (
            handler(LiteLLMProvider()),
            handler(mock),
            handler(persist),
        ):
            bot.outer("demo")

        assert persist.get_handoff("NestedBot") == ""


# ---------------------------------------------------------------------------
# Tests: context compaction (CompactionHandler)
# ---------------------------------------------------------------------------


class TestContextCompaction:
    """CompactionHandler compacts agent history after template calls."""

    def test_compact_reduces_history(self):
        history: OrderedDict[str, Any] = OrderedDict()
        for i in range(10):
            history[f"msg{i}"] = {
                "id": f"msg{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
            }

        compaction = CompactionHandler(max_history_len=6)
        mock = MockCompletionHandler(
            [make_text_response("Summary of prior conversation.")]
        )
        provider = LiteLLMProvider()
        with handler(provider), handler(mock):
            stored = get_agent_history("PlainBot")
            stored.update(history)
            compaction._compact("PlainBot", stored)

        result = provider._histories["PlainBot"]
        assert len(result) < 10
        first_msg = next(iter(result.values()))
        assert "CONTEXT SUMMARY" in first_msg["content"]

    def test_compaction_preserves_recent_messages(self):
        history: OrderedDict[str, Any] = OrderedDict()
        for i in range(10):
            history[f"msg{i}"] = {
                "id": f"msg{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
            }

        compaction = CompactionHandler(max_history_len=6)
        keep_recent = max(6 // 2, 4)
        mock = MockCompletionHandler([make_text_response("Summary.")])
        provider = LiteLLMProvider()
        with handler(provider), handler(mock):
            stored = get_agent_history("ChatBot")
            stored.update(history)
            compaction._compact("ChatBot", stored)

        result = provider._histories["ChatBot"]
        remaining_ids = list(result.keys())
        for i in range(10 - keep_recent, 10):
            assert f"msg{i}" in remaining_ids

    def test_compaction_triggered_by_template_call(self, tmp_path: Path):
        bot = ChatBot(agent_id="ChatBot")
        provider = LiteLLMProvider()

        with handler(provider):
            history = get_agent_history(bot.__agent_id__)
            for i in range(6):
                history[f"old{i}"] = {
                    "id": f"old{i}",
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Old message {i}",
                }

        mock = MockCompletionHandler(
            [
                make_text_response("new reply"),
                make_text_response("Summary of old conversation."),
            ]
        )
        with (
            handler(provider),
            handler(mock),
            handler(CompactionHandler(max_history_len=4)),
            handler(PersistenceHandler(tmp_path)),
        ):
            bot.send("trigger compaction")

        result = provider._histories.get("ChatBot", {})
        assert len(result) <= 4 + 4

    def test_compaction_works_on_plain_agent(self):
        """CompactionHandler works on any Agent, not just PersistentAgent."""

        class PlainBot(Agent):
            """Plain bot."""

            @Template.define
            def send(self, msg: str) -> str:
                """Say: {msg}"""
                raise NotHandled

        bot = PlainBot()
        provider = LiteLLMProvider()

        with handler(provider):
            history = get_agent_history(bot.__agent_id__)
            for i in range(10):
                history[f"msg{i}"] = {
                    "id": f"msg{i}",
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Message {i}",
                }

        mock = MockCompletionHandler(
            [make_text_response("reply"), make_text_response("Summary.")]
        )
        with (
            handler(provider),
            handler(mock),
            handler(CompactionHandler(max_history_len=4)),
        ):
            bot.send("trigger")

        result = provider._histories.get(bot.__agent_id__, {})
        assert len(result) > 0, "history should not be empty after compaction"
        first_msg = next(iter(result.values()))
        assert "CONTEXT SUMMARY" in first_msg["content"]
        assert len(result) <= 4 + 4

    def test_compaction_triggered_naturally_on_plain_agent(self):
        """CompactionHandler compacts after enough calls accumulate history.

        Makes multiple template calls on a plain Agent so that history
        exceeds max_history_len, then verifies compaction fires and
        produces a summary message.
        """

        class PlainBot(Agent):
            """Plain bot."""

            @Template.define
            def send(self, msg: str) -> str:
                """Say: {msg}"""
                raise NotHandled

        # 4 calls × ~3 msgs each (system+user+assistant) = ~12 msgs
        # Compaction threshold is 6, so it should trigger.
        responses = [make_text_response(f"reply-{i}") for i in range(4)]
        # Extra response for the summarize_context call during compaction
        responses.append(make_text_response("Summary of conversation."))
        mock = MockCompletionHandler(responses)

        bot = PlainBot()
        provider = LiteLLMProvider()
        with (
            handler(provider),
            handler(mock),
            handler(CompactionHandler(max_history_len=6)),
        ):
            for i in range(4):
                bot.send(f"message-{i}")

        history = provider._histories.get(bot.__agent_id__, {})
        # Should have been compacted: summary + recent messages
        first_msg = next(iter(history.values()))
        assert "CONTEXT SUMMARY" in first_msg["content"]

    def test_compaction_on_plain_agent_preserves_functionality(self):
        """After compaction, the plain Agent still works for subsequent calls."""

        class PlainBot(Agent):
            """Plain bot."""

            @Template.define
            def send(self, msg: str) -> str:
                """Say: {msg}"""
                raise NotHandled

        responses = [make_text_response(f"reply-{i}") for i in range(4)]
        # Compaction summary call fires after the 4th reply
        responses.append(make_text_response("Summary."))
        # Then the 5th send() call
        responses.append(make_text_response("reply-4"))
        mock = MockCompletionHandler(responses)

        bot = PlainBot()
        provider = LiteLLMProvider()
        with (
            handler(provider),
            handler(mock),
            handler(CompactionHandler(max_history_len=6)),
        ):
            for i in range(4):
                bot.send(f"msg-{i}")
            # This call happens after compaction
            result = bot.send("after-compaction")

        assert result == "reply-4"


# ---------------------------------------------------------------------------
# Tests: system prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """System prompt of PersistentAgent includes class docstring."""

    def test_base_docstring_used(self):
        bot = ChatBot(agent_id="ChatBot")
        assert "persistent chat bot" in bot.__system_prompt__

    def test_no_handoff_initially(self):
        bot = ChatBot(agent_id="ChatBot")
        assert "[HANDOFF" not in bot.__system_prompt__


# ---------------------------------------------------------------------------
# Tests: agent isolation
# ---------------------------------------------------------------------------


class TestAgentIsolation:
    """Multiple PersistentAgent instances are independent in the handler."""

    def test_two_agents_independent(self, tmp_path: Path):
        bot1 = ChatBot(agent_id="bot1")
        bot2 = ChatBot(agent_id="bot2")

        persist = PersistenceHandler(tmp_path)
        with handler(AgentHistoryHandler()):
            persist._handoffs["bot1"] = "bot1 work"
            persist.save(bot1)

        persist2 = PersistenceHandler(tmp_path)
        with handler(AgentHistoryHandler()):
            persist2.ensure_loaded(bot2)
        assert persist2.get_handoff("bot2") == ""

    def test_same_dir_different_agent_id(self, tmp_path: Path):
        class Bot(PersistentAgent):
            """A bot."""

            @Template.define
            def ask(self, q: str) -> str:
                """Q: {q}"""
                raise NotHandled

        persist = PersistenceHandler(tmp_path)
        bot_a = Bot(agent_id="alpha")
        bot_b = Bot(agent_id="beta")

        with handler(AgentHistoryHandler()):
            persist._handoffs["alpha"] = "alpha work"
            persist.save(bot_a)
            persist._handoffs["beta"] = "beta work"
            persist.save(bot_b)

        persist2 = PersistenceHandler(tmp_path)
        a2 = Bot(agent_id="alpha")
        b2 = Bot(agent_id="beta")
        with handler(AgentHistoryHandler()):
            persist2.ensure_loaded(a2)
            persist2.ensure_loaded(b2)
        assert persist2.get_handoff("alpha") == "alpha work"
        assert persist2.get_handoff("beta") == "beta work"


# ---------------------------------------------------------------------------
# Tests: compatibility with RetryLLMHandler
# ---------------------------------------------------------------------------


class TestRetryCompatibility:
    """PersistentAgent works with RetryLLMHandler and PersistenceHandler."""

    def test_retry_then_success(self, tmp_path: Path):
        mock = MockCompletionHandler(
            [
                make_text_response('"not_an_int"'),
                make_text_response('{"value": 42}'),
            ]
        )

        class NumberBot(PersistentAgent):
            """You are a number bot."""

            @Template.define
            def pick(self) -> int:
                """Pick a number."""
                raise NotHandled

        persist = PersistenceHandler(tmp_path)
        bot = NumberBot(agent_id="NumberBot")
        with (
            handler(LiteLLMProvider()),
            handler(RetryLLMHandler()),
            handler(mock),
            handler(persist),
        ):
            result = bot.pick()

        assert result == 42
        assert persist.get_handoff("NumberBot") == ""
        assert (tmp_path / "NumberBot.json").exists()


# ---------------------------------------------------------------------------
# Tests: PersistenceHandler is optional
# ---------------------------------------------------------------------------


class TestWithoutHandler:
    """PersistentAgent works without PersistenceHandler — no auto-checkpointing."""

    def test_agent_works_without_persistence_handler(self):
        mock = MockCompletionHandler([make_text_response("hello")])
        bot = ChatBot(agent_id="ChatBot")

        with handler(LiteLLMProvider()), handler(mock):
            result = bot.send("hi")

        assert result == "hello"


# ---------------------------------------------------------------------------
# Tests: nested calls with failures + persistence
# ---------------------------------------------------------------------------


class TestNestedCallFailuresWithPersistence:
    """Nested tool calls that fail should not corrupt persistence state."""

    def test_nested_tool_failure_still_checkpoints(self, tmp_path: Path):
        """If a nested tool raises, the outermost handler saves a crash checkpoint."""

        class FailingBot(PersistentAgent):
            """Bot whose tool always fails."""

            @Template.define
            def inner(self, payload: str) -> str:
                """Check: {payload}"""
                raise NotHandled

            @Tool.define
            def failing_tool(self, payload: str) -> str:
                """Check payload — always raises."""
                raise RuntimeError("tool exploded")

            @Template.define
            def outer(self, payload: str) -> str:
                """Call `failing_tool` for: {payload}."""
                raise NotHandled

        mock = MockCompletionHandler(
            [
                make_tool_call_response("self__failing_tool", '{"payload": "boom"}'),
            ]
        )
        bot = FailingBot(agent_id="FailingBot")
        persist = PersistenceHandler(tmp_path)

        with pytest.raises(ToolCallExecutionError, match="tool exploded"):
            with (
                handler(LiteLLMProvider()),
                handler(mock),
                handler(persist),
            ):
                bot.outer("go")

        # Crash checkpoint should exist with handoff
        data = json.loads((tmp_path / "FailingBot.json").read_text())
        assert "Executing outer" in data["handoff"]

    def test_nested_tool_failure_then_recovery(self, tmp_path: Path):
        """After a nested tool failure, next session resumes with handoff."""
        mock_crash = MockCompletionHandler(
            [
                make_tool_call_response("self__check_tool", '{"payload": "crash"}'),
            ]
        )

        class CrashInnerBot(PersistentAgent):
            """Bot with crashing inner tool."""

            call_count = 0

            @Template.define
            def inner_check(self, payload: str) -> str:
                """Check: {payload}"""
                raise NotHandled

            @Tool.define
            def check_tool(self, payload: str) -> str:
                """Check payload."""
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("first call fails")
                return self.inner_check(payload)

            @Template.define
            def outer(self, payload: str) -> str:
                """Call `check_tool` for: {payload}, then return answer."""
                raise NotHandled

        bot = CrashInnerBot(agent_id="CrashInnerBot")

        # Session 1: crash
        with pytest.raises(ToolCallExecutionError, match="first call fails"):
            with (
                handler(LiteLLMProvider()),
                handler(mock_crash),
                handler(PersistenceHandler(tmp_path)),
            ):
                bot.outer("task")

        # Session 2: successful recovery
        system_prompts: list[str] = []

        class SpyMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self_, model, messages=None, **kwargs):
                system_prompts.extend(
                    m.get("content", "")
                    for m in (messages or [])
                    if m.get("role") == "system"
                )
                return make_text_response("recovered")

        bot2 = CrashInnerBot(agent_id="CrashInnerBot")
        with (
            handler(LiteLLMProvider()),
            handler(SpyMock()),
            handler(PersistenceHandler(tmp_path)),
        ):
            result = bot2.outer("retry")

        assert result == "recovered"
        assert any("[HANDOFF FROM PRIOR SESSION]" in p for p in system_prompts)


# ---------------------------------------------------------------------------
# Tests: Agent and PersistentAgent coexistence
# ---------------------------------------------------------------------------


class TestAgentPersistentAgentCoexistence:
    """Plain Agent and PersistentAgent work side-by-side."""

    def test_plain_and_persistent_agent_in_same_handler(self, tmp_path: Path):
        """Both agent types work under the same LiteLLMProvider."""

        class PlainBot(Agent):
            """Plain bot."""

            @Template.define
            def ask(self, q: str) -> str:
                """Q: {q}"""
                raise NotHandled

        mock = MockCompletionHandler(
            [make_text_response("plain-reply"), make_text_response("persist-reply")]
        )
        plain = PlainBot()
        persistent = ChatBot(agent_id="ChatBot")
        persist = PersistenceHandler(tmp_path)

        with handler(LiteLLMProvider()), handler(mock), handler(persist):
            r1 = plain.ask("hello")
            r2 = persistent.send("hello")

        assert r1 == "plain-reply"
        assert r2 == "persist-reply"
        # Only the PersistentAgent gets a checkpoint file
        assert (tmp_path / "ChatBot.json").exists()
        assert not list(tmp_path.glob(f"{plain.__agent_id__}*"))

    def test_persistent_agent_tool_calls_plain_agent(self, tmp_path: Path):
        """A PersistentAgent's tool can delegate to a plain Agent.

        The mock returns a tool_call for the outer PersistentAgent, whose
        tool implementation calls the inner plain Agent. The inner Agent
        makes one LLM call. After the tool result is returned, the outer
        Agent's final LLM call produces the result.

        Mock response sequence:
          0: outer → tool_call(self__delegate, {"q": "sub-task"})
          1: inner plain agent → "inner-answer"
          2: outer → "final-answer" (after getting tool result)
        """

        class InnerPlainAgent(Agent):
            """Inner helper agent."""

            @Template.define
            def answer(self, q: str) -> str:
                """Answer: {q}"""
                raise NotHandled

        inner = InnerPlainAgent()

        class OuterPersistent(PersistentAgent):
            """Outer persistent agent that delegates via tool."""

            @Tool.define
            def delegate(self, q: str) -> str:
                """Delegate a sub-question to an inner agent."""
                return inner.answer(q)

            @Template.define
            def process(self, task: str) -> str:
                """Process: {task}. Use `delegate` for sub-questions."""
                raise NotHandled

        mock = MockCompletionHandler(
            [
                make_tool_call_response("self__delegate", '{"q": "sub-task"}'),
                make_text_response("inner-answer"),
                make_text_response("final-answer"),
            ]
        )
        outer = OuterPersistent(agent_id="outer")
        persist = PersistenceHandler(tmp_path)

        with handler(LiteLLMProvider()), handler(mock), handler(persist):
            result = outer.process("do it")

        assert result == "final-answer"
        assert (tmp_path / "outer.json").exists()

    def test_plain_agent_tool_calls_persistent_agent(self, tmp_path: Path):
        """A plain Agent's tool can delegate to a PersistentAgent.

        Mock response sequence:
          0: outer plain → tool_call(self__delegate, {"q": "sub"})
          1: inner persistent → "persisted-answer"
          2: outer plain → "done" (after getting tool result)
        """

        inner = ChatBot(agent_id="inner-bot")

        class OuterPlain(Agent):
            """Plain agent that delegates to a persistent agent."""

            @Tool.define
            def delegate(self, q: str) -> str:
                """Delegate to persistent bot."""
                return inner.send(q)

            @Template.define
            def run(self, task: str) -> str:
                """Run: {task}. Use `delegate` for sub-tasks."""
                raise NotHandled

        mock = MockCompletionHandler(
            [
                make_tool_call_response("self__delegate", '{"q": "sub"}'),
                make_text_response("persisted-answer"),
                make_text_response("done"),
            ]
        )
        outer = OuterPlain()
        persist = PersistenceHandler(tmp_path)

        with handler(LiteLLMProvider()), handler(mock), handler(persist):
            result = outer.run("go")

        assert result == "done"
        # The inner PersistentAgent should NOT get checkpointed by
        # PersistenceHandler because it was called nested (not outermost)
        # But its history should still be tracked by LiteLLMProvider

    def test_two_persistent_agents_cooperate(self, tmp_path: Path):
        """Two PersistentAgents with different IDs work independently.

        Mock response sequence:
          0: planner → "the plan"
          1: executor → "executed"
        """
        mock = MockCompletionHandler(
            [make_text_response("the plan"), make_text_response("executed")]
        )

        planner = ChatBot(agent_id="planner")
        executor = ChatBot(agent_id="executor")
        persist = PersistenceHandler(tmp_path)

        with handler(LiteLLMProvider()), handler(mock), handler(persist):
            plan = planner.send("make a plan")
            result = executor.send(f"execute: {plan}")

        assert plan == "the plan"
        assert result == "executed"
        assert (tmp_path / "planner.json").exists()
        assert (tmp_path / "executor.json").exists()

        # Each has independent history
        planner_data = json.loads((tmp_path / "planner.json").read_text())
        executor_data = json.loads((tmp_path / "executor.json").read_text())
        assert len(planner_data["history"]) > 0
        assert len(executor_data["history"]) > 0
