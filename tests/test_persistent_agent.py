"""Tests for PersistentAgent — checkpointing, compaction, crash recovery,
nested calls, subclass state persistence, and system prompt augmentation.
"""

import dataclasses
import json
from pathlib import Path

import pytest
from litellm import ModelResponse

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    RetryLLMHandler,
    completion,
)
from effectful.handlers.llm.persistence import PersistentAgent
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

    persist_dir: Path = dataclasses.field(default=Path("."))
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
# Tests: basic persistence
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """save_checkpoint / load_checkpoint round-trip correctly."""

    def test_save_creates_file(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path)
        path = bot.save_checkpoint()
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_load_round_trip_empty(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path)
        bot.save_checkpoint()
        bot2 = ChatBot(persist_dir=tmp_path)
        assert len(bot2.__history__) == 0
        assert bot2.handoff == ""

    def test_save_load_round_trip_with_history(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path)
        bot.set_handoff("working on X")
        bot.__history__["msg1"] = {
            "id": "msg1",
            "role": "user",
            "content": "hello",
        }
        bot.save_checkpoint()

        bot2 = ChatBot(persist_dir=tmp_path)
        assert bot2.handoff == "working on X"
        assert len(bot2.__history__) == 1
        assert bot2.__history__["msg1"]["content"] == "hello"

    def test_load_returns_false_when_no_checkpoint(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path)
        assert not bot.load_checkpoint()

    def test_load_returns_true_when_checkpoint_exists(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path)
        bot.save_checkpoint()
        bot2 = ChatBot(persist_dir=tmp_path)
        # Constructor calls load_checkpoint; call again to test return value
        assert bot2.load_checkpoint()

    def test_atomic_write(self, tmp_path: Path):
        """Checkpoint write uses tmp + rename for atomicity."""
        bot = ChatBot(persist_dir=tmp_path)
        bot.save_checkpoint()
        # No .tmp file should remain
        assert not bot._checkpoint_path.with_suffix(".tmp").exists()

    def test_custom_agent_id(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path, agent_id="custom-bot")
        bot.save_checkpoint()
        assert (tmp_path / "custom-bot.json").exists()

    def test_persist_dir_created_automatically(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        bot = ChatBot(persist_dir=nested)
        bot.save_checkpoint()
        assert nested.exists()


# ---------------------------------------------------------------------------
# Tests: subclass state persistence (checkpoint_state / restore_state)
# ---------------------------------------------------------------------------


class TestSubclassStatePersistence:
    """Dataclass fields on subclasses are automatically persisted."""

    def test_dataclass_fields_round_trip(self, tmp_path: Path):
        bot = StatefulBot(persist_dir=tmp_path)
        bot.learned_patterns = ["pattern A", "pattern B"]
        bot.call_count = 5
        bot.save_checkpoint()

        bot2 = StatefulBot(persist_dir=tmp_path)
        assert bot2.learned_patterns == ["pattern A", "pattern B"]
        assert bot2.call_count == 5

    def test_non_dataclass_has_empty_state(self, tmp_path: Path):
        """Non-dataclass subclass returns empty state dict."""
        bot = ChatBot(persist_dir=tmp_path)
        assert bot.checkpoint_state() == {}

    def test_non_serializable_fields_skipped(self, tmp_path: Path):
        @dataclasses.dataclass
        class WeirdBot(PersistentAgent):
            """Bot with a non-serializable field."""

            persist_dir: Path = dataclasses.field(default=Path("."))
            callback: object = dataclasses.field(default=None)
            name: str = "test"

            @Template.define
            def send(self, msg: str) -> str:
                """Say: {msg}"""
                raise NotHandled

        bot = WeirdBot(persist_dir=tmp_path)
        bot.callback = lambda x: x  # not JSON serializable
        bot.name = "Alice"
        bot.save_checkpoint()

        bot2 = WeirdBot(persist_dir=tmp_path)
        # name was serializable → restored; callback was not → stays default
        assert bot2.name == "Alice"
        assert bot2.callback is None

    def test_custom_checkpoint_restore(self, tmp_path: Path):
        """Users can override checkpoint_state / restore_state."""

        class CustomBot(PersistentAgent):
            """Custom serialisation bot."""

            def __init__(self, persist_dir, **kwargs):
                self.data = {"counter": 0}
                super().__init__(persist_dir, **kwargs)

            def checkpoint_state(self):
                return {"data": self.data}

            def restore_state(self, state):
                self.data = state.get("data", {"counter": 0})

            @Template.define
            def send(self, msg: str) -> str:
                """Say: {msg}"""
                raise NotHandled

        bot = CustomBot(persist_dir=tmp_path)
        bot.data["counter"] = 42
        bot.save_checkpoint()

        bot2 = CustomBot(persist_dir=tmp_path)
        assert bot2.data["counter"] == 42

    def test_state_saved_in_checkpoint_file(self, tmp_path: Path):
        """The checkpoint JSON contains a 'state' key with subclass fields."""
        bot = StatefulBot(persist_dir=tmp_path)
        bot.learned_patterns = ["X"]
        bot.call_count = 3
        bot.save_checkpoint()

        data = json.loads(bot._checkpoint_path.read_text())
        assert "state" in data
        assert data["state"]["learned_patterns"] == ["X"]
        assert data["state"]["call_count"] == 3


# ---------------------------------------------------------------------------
# Tests: automatic checkpointing around template calls
# ---------------------------------------------------------------------------


class TestAutomaticCheckpointing:
    """Template calls on PersistentAgent trigger save_checkpoint."""

    def test_checkpoint_saved_after_successful_call(self, tmp_path: Path):
        mock = MockCompletionHandler([make_text_response("hello")])
        bot = ChatBot(persist_dir=tmp_path)

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("hi")

        assert bot._checkpoint_path.exists()
        data = json.loads(bot._checkpoint_path.read_text())
        assert len(data["history"]) > 0
        # Handoff should be cleared after successful call
        assert data["handoff"] == ""

    def test_checkpoint_saved_on_exception(self, tmp_path: Path):
        """Even if the LLM call fails, checkpoint with handoff is saved."""

        class FailingMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self, *args, **kwargs):
                raise RuntimeError("boom")

        bot = ChatBot(persist_dir=tmp_path)
        with pytest.raises(RuntimeError, match="boom"):
            with handler(LiteLLMProvider()), handler(FailingMock()):
                bot.send("hi")

        data = json.loads(bot._checkpoint_path.read_text())
        assert "Executing send" in data["handoff"]

    def test_handoff_describes_current_call(self, tmp_path: Path):
        """Before the template runs, handoff records what's in progress."""
        handoff_during_call = []

        class SpyMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self_, model, messages=None, **kwargs):
                # Read the checkpoint mid-call
                data = json.loads(bot._checkpoint_path.read_text())
                handoff_during_call.append(data["handoff"])
                return make_text_response("ok")

        bot = ChatBot(persist_dir=tmp_path)
        with handler(LiteLLMProvider()), handler(SpyMock()):
            bot.send("hello")

        assert len(handoff_during_call) == 1
        assert "Executing send" in handoff_during_call[0]

    def test_history_persists_across_sessions(self, tmp_path: Path):
        mock = MockCompletionHandler(
            [make_text_response("reply1"), make_text_response("reply2")]
        )
        bot = ChatBot(persist_dir=tmp_path)

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("first message")

        # "Restart" — new instance, same persist_dir
        mock2 = MockCompletionHandler([make_text_response("reply after restart")])
        bot2 = ChatBot(persist_dir=tmp_path)

        with handler(LiteLLMProvider()), handler(mock2):
            bot2.send("second message")

        # Second bot should see prior history + new messages
        assert len(bot2.__history__) > len(bot.__history__)

    def test_second_call_sees_prior_history(self, tmp_path: Path):
        mock = MockCompletionHandler(
            [make_text_response("r1"), make_text_response("r2")]
        )
        bot = ChatBot(persist_dir=tmp_path)

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("a")
            bot.send("b")

        # Second call should have seen the first call's messages
        assert len(mock.received_messages[1]) > len(mock.received_messages[0])

    def test_dataclass_state_saved_around_template_calls(self, tmp_path: Path):
        """Dataclass fields are included in the automatic checkpoint."""
        mock = MockCompletionHandler([make_text_response("ok")])
        bot = StatefulBot(persist_dir=tmp_path)
        bot.call_count = 7

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("test")

        data = json.loads(bot._checkpoint_path.read_text())
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

        bot = ChatBot(persist_dir=tmp_path)
        with pytest.raises(RuntimeError):
            with handler(LiteLLMProvider()), handler(CrashMock()):
                bot.send("important task")

        # New instance should see the handoff
        bot2 = ChatBot(persist_dir=tmp_path)
        assert "Executing send" in bot2.handoff
        assert "[HANDOFF" in bot2.__system_prompt__

    def test_system_prompt_includes_handoff(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path)
        bot.set_handoff("Was processing query about France")
        assert "[HANDOFF FROM PRIOR SESSION]" in bot.__system_prompt__
        assert "France" in bot.__system_prompt__

    def test_handoff_cleared_on_success(self, tmp_path: Path):
        mock = MockCompletionHandler([make_text_response("done")])
        bot = ChatBot(persist_dir=tmp_path)
        bot.set_handoff("leftover from last crash")

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("new task")

        assert bot.handoff == ""
        data = json.loads(bot._checkpoint_path.read_text())
        assert data["handoff"] == ""

    def test_dataclass_state_survives_crash(self, tmp_path: Path):
        """Subclass state is preserved in the crash checkpoint."""

        class CrashMock(ObjectInterpretation):
            @implements(completion)
            def _completion(self, *args, **kwargs):
                raise RuntimeError("crash")

        bot = StatefulBot(persist_dir=tmp_path)
        bot.learned_patterns = ["important insight"]
        bot.call_count = 3

        with pytest.raises(RuntimeError):
            with handler(LiteLLMProvider()), handler(CrashMock()):
                bot.send("boom")

        # Restore and check subclass state survived
        bot2 = StatefulBot(persist_dir=tmp_path)
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
        bot = NestedBot(persist_dir=tmp_path)

        with handler(LiteLLMProvider()), handler(mock):
            result = bot.outer("demo")

        assert result == "outer result"

    def test_nested_call_does_not_double_checkpoint(self, tmp_path: Path):
        """Inner template call should not trigger extra checkpointing."""
        save_count = 0
        original_save = PersistentAgent.save_checkpoint

        def counting_save(self):
            nonlocal save_count
            save_count += 1
            return original_save(self)

        mock = MockCompletionHandler(
            [
                make_tool_call_response("self__check_tool", '{"payload": "demo"}'),
                make_text_response("inner"),
                make_text_response("outer"),
            ]
        )
        bot = NestedBot(persist_dir=tmp_path)
        PersistentAgent.save_checkpoint = counting_save  # type: ignore[method-assign]
        try:
            with handler(LiteLLMProvider()), handler(mock):
                bot.outer("demo")
        finally:
            PersistentAgent.save_checkpoint = original_save  # type: ignore[method-assign]

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
        bot = NestedBot(persist_dir=tmp_path)

        with handler(LiteLLMProvider()), handler(mock):
            bot.outer("demo")

        assert bot.handoff == ""


# ---------------------------------------------------------------------------
# Tests: context compaction
# ---------------------------------------------------------------------------


class TestContextCompaction:
    """History compaction replaces old messages with a summary."""

    def test_compact_reduces_history(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path, max_history_len=6)

        # Manually populate history with more messages than max_history_len
        for i in range(10):
            bot.__history__[f"msg{i}"] = {
                "id": f"msg{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
            }

        # Mock the compaction LLM call
        mock = MockCompletionHandler(
            [make_text_response("Summary of prior conversation.")]
        )
        with handler(LiteLLMProvider()), handler(mock):
            bot._compact_history()

        # History should be reduced
        assert len(bot.__history__) < 10
        # Should contain a compaction summary message
        first_msg = next(iter(bot.__history__.values()))
        assert "CONTEXT SUMMARY" in first_msg["content"]

    def test_compaction_preserves_recent_messages(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path, max_history_len=6)
        keep_recent = max(6 // 2, 4)

        for i in range(10):
            bot.__history__[f"msg{i}"] = {
                "id": f"msg{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
            }

        mock = MockCompletionHandler([make_text_response("Summary.")])
        with handler(LiteLLMProvider()), handler(mock):
            bot._compact_history()

        # Recent messages should still be present
        remaining_ids = list(bot.__history__.keys())
        # The last `keep_recent` original messages should be there
        for i in range(10 - keep_recent, 10):
            assert f"msg{i}" in remaining_ids

    def test_compaction_triggered_by_template_call(self, tmp_path: Path):
        # Use a small max_history_len so compaction triggers
        bot = ChatBot(persist_dir=tmp_path, max_history_len=4)

        # Pre-populate with enough history to trigger compaction
        for i in range(6):
            bot.__history__[f"old{i}"] = {
                "id": f"old{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Old message {i}",
            }

        # The template call adds more messages; after it, compaction should trigger
        # We need 2 mock responses: one for the template call, one for compaction
        mock = MockCompletionHandler(
            [
                make_text_response("new reply"),
                make_text_response("Summary of old conversation."),
            ]
        )
        with handler(LiteLLMProvider()), handler(mock):
            bot.send("trigger compaction")

        # Compaction should have run, reducing history
        assert len(bot.__history__) <= bot.max_history_len + 4


# ---------------------------------------------------------------------------
# Tests: system prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """System prompt is augmented with handoff."""

    def test_base_docstring_used(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path)
        assert "persistent chat bot" in bot.__system_prompt__

    def test_handoff_appended(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path)
        bot.set_handoff("Processing query #42")
        prompt = bot.__system_prompt__
        assert "[HANDOFF FROM PRIOR SESSION]" in prompt
        assert "Processing query #42" in prompt

    def test_clean_prompt_no_extras(self, tmp_path: Path):
        bot = ChatBot(persist_dir=tmp_path)
        prompt = bot.__system_prompt__
        assert "[HANDOFF" not in prompt


# ---------------------------------------------------------------------------
# Tests: agent isolation
# ---------------------------------------------------------------------------


class TestAgentIsolation:
    """Multiple PersistentAgent instances with different persist_dirs
    are fully independent."""

    def test_two_agents_independent(self, tmp_path: Path):
        dir1 = tmp_path / "bot1"
        dir2 = tmp_path / "bot2"
        bot1 = ChatBot(persist_dir=dir1)
        ChatBot(persist_dir=dir2)  # create bot2's state dir

        bot1.set_handoff("bot1 work")
        bot1.save_checkpoint()

        bot2_restored = ChatBot(persist_dir=dir2)
        assert bot2_restored.handoff == ""

    def test_same_dir_different_agent_id(self, tmp_path: Path):
        bot1 = ChatBot(persist_dir=tmp_path, agent_id="alpha")
        bot2 = ChatBot(persist_dir=tmp_path, agent_id="beta")

        bot1.set_handoff("alpha work")
        bot1.save_checkpoint()

        bot2.set_handoff("beta work")
        bot2.save_checkpoint()

        # Reload
        r1 = ChatBot(persist_dir=tmp_path, agent_id="alpha")
        r2 = ChatBot(persist_dir=tmp_path, agent_id="beta")
        assert r1.handoff == "alpha work"
        assert r2.handoff == "beta work"


# ---------------------------------------------------------------------------
# Tests: compatibility with RetryLLMHandler
# ---------------------------------------------------------------------------


class TestRetryCompatibility:
    """PersistentAgent works with RetryLLMHandler."""

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

        bot = NumberBot(persist_dir=tmp_path)
        with (
            handler(LiteLLMProvider()),
            handler(RetryLLMHandler()),
            handler(mock),
        ):
            result = bot.pick()

        assert result == 42
        assert bot.handoff == ""
        assert bot._checkpoint_path.exists()
