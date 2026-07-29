"""Tests for `effectful.handlers.llm.completions`'s persistence support:
persisted `Agent`s and `SQLitePersister`.

These are deliberately independent of a real LLM wherever possible:
`_FakeAgentCalls` stands in for a full provider for anything that only cares
about *when/what* gets checkpointed. A couple of composition tests still pull
in `LiteLLMProvider` with a `MockCompletionHandler`, matching the pattern used
throughout the rest of this test suite.
"""

import dataclasses
import json
import pickle
import sqlite3
import threading
from pathlib import Path

import pytest
from litellm import ModelResponse

from effectful.handlers.llm import Agent, Template
from effectful.handlers.llm.harness.hooks import (
    completion,
)
from effectful.handlers.llm.harness.persisting import SQLitePersister
from effectful.handlers.llm.harness.providing import LiteLLMProvider
from effectful.handlers.llm.harness.retrying import TenacityRetryer
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Shared test doubles and fixtures
# ---------------------------------------------------------------------------


class _Bot(Agent):
    """A minimal test bot. Pass `agent_id` to make it persistent."""

    @Template.define
    def ask(self, q: str) -> str:
        """Answer: {q}"""
        raise NotHandled


@dataclasses.dataclass(kw_only=True)
class _StatefulBot(Agent):
    """A test bot with dataclass state to checkpoint alongside history.

    `agent_id` must be redeclared here as its own field -- `Agent` is
    deliberately not a dataclass itself, so it can't be inherited. Nothing
    else is needed: `__agent_id__`/`__persistent__` are derived lazily from
    `self.agent_id`, whichever way it ends up set.
    """

    agent_id: str | None = None
    counter: int = 0
    label: str = "x"

    @Template.define
    def ask(self, q: str) -> str:
        """Answer: {q}"""
        raise NotHandled


class _NestingBot(Agent):
    """A test bot with two templates, for nested-call tests."""

    @Template.define
    def outer(self, task: str) -> str:
        """Do: {task}"""
        raise NotHandled

    @Template.define
    def inner(self, q: str) -> str:
        """Answer: {q}"""
        raise NotHandled


class _PlainHelper(Agent):
    """A helper agent, always constructed without `agent_id` -- never persisted."""

    @Template.define
    def answer(self, q: str) -> str:
        """Answer: {q}"""
        raise NotHandled


@dataclasses.dataclass(kw_only=True)
class _DelegatingBot(Agent):
    """A bot that delegates to another agent.

    `agent_id` is redeclared for the same reason as `_StatefulBot` above.
    `helper` is excluded from checkpointing via `persist: False` metadata --
    otherwise it would get pickled as a duplicate, divergent copy of an agent
    that (if persistent) already checkpoints itself independently.
    """

    agent_id: str | None = None
    helper: Agent = dataclasses.field(metadata={"persist": False})

    @Template.define
    def run(self, task: str) -> str:
        """Do: {task}"""
        raise NotHandled


class _FakeAgentCalls(ObjectInterpretation):
    """Stands in for a full LLM provider.

    Each call pops the next configured turn and either:

    - appends a synthesised user/assistant message pair to the bound agent's
      history and returns the configured result, if the turn is a
      ``(user_content, assistant_content, result)`` tuple, or
    - delegates entirely to it, if the turn is a plain callable
      ``(template, args, kwargs) -> result`` -- letting a test simulate "the
      model called a tool that invoked another template" by just calling
      that template directly from Python.

    This lets persistence tests control the exact shape and timing of an
    agent's history without exercising the real completion loop.
    """

    def __init__(self, turns):
        self._turns = list(turns)
        self.call_count = 0
        self.calls: list[tuple[str, tuple, dict]] = []

    @implements(Template.__apply__)
    def _call(self, template, *args, **kwargs):
        self.calls.append((template.__name__, args, kwargs))
        n = self.call_count
        turn = self._turns[min(n, len(self._turns) - 1)]
        self.call_count += 1
        if callable(turn):
            return turn(template, args, kwargs)
        user_content, assistant_content, result = turn
        agent = getattr(template, "__agent__", None)
        if agent is not None:
            agent.__history__[f"u{n}"] = {
                "id": f"u{n}",
                "role": "user",
                "content": user_content,
            }
            agent.__history__[f"a{n}"] = {
                "id": f"a{n}",
                "role": "assistant",
                "content": assistant_content,
            }
        return result


class MockCompletionHandler(ObjectInterpretation):
    """Mock handler that returns pre-configured completion responses."""

    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.call_count = 0
        self.received_messages: list[list[dict]] = []

    @implements(completion)
    def _completion(self, messages=None, **kwargs):
        self.received_messages.append(list(messages) if messages else [])
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


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


def _load_row(db_path: Path, agent_id: str) -> tuple[bytes, str] | None:
    """Read a checkpoint row `(state_blob, history_json)` directly from the DB."""
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT state, history FROM checkpoints WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentPersistenceOptIn:
    """`Agent` itself carries persistence: passing an explicit `agent_id` is
    what opts an instance in, both to loading a checkpoint (if a handler is
    active when `__history__` is first touched) and to being checkpointed by
    `SQLitePersister`. Omitting it is a normal, transient agent."""

    def test_omitting_agent_id_yields_transient_agent(self):
        assert _Bot().__persistent__ is False

    def test_explicit_agent_id_marks_agent_persistent(self):
        assert _Bot(agent_id="p1").__persistent__ is True

    def test_construction_never_requires_a_handler(self):
        # Persistence is opt-in and best-effort: constructing (even with an
        # explicit agent_id) never touches the database and never raises,
        # regardless of whether a handler is active.
        bot = _Bot(agent_id="no-handler")
        assert dict(bot.__history__) == {}

    def test_agent_id_is_stable_across_instances(self):
        a, b = _Bot(agent_id="shared"), _Bot(agent_id="shared")
        assert a.__agent_id__ == b.__agent_id__ == "shared"

        # Contrast with a transient Agent: two instances always get distinct,
        # randomly-generated ids -- this is exactly the property that makes
        # cross-restart resumption possible for a persistent one.
        p1, p2 = _PlainHelper(), _PlainHelper()
        assert p1.__agent_id__ != p2.__agent_id__

    def test_bound_template_exposes_agent_instance(self):
        bot = _Bot(agent_id="x")
        assert bot.ask.__agent__ is bot


class TestCheckpointStateDefaults:
    """`SQLitePersister._checkpoint_state()` is a fixed static method, not an
    overridable hook -- it always walks declared dataclass fields, minus any
    marked `persist: False`."""

    def test_dataclass_fields_are_captured(self):
        bot = _StatefulBot(agent_id="s1", counter=5, label="hi")
        assert SQLitePersister._checkpoint_state(bot) == {"counter": 5, "label": "hi"}

    def test_non_dataclass_agent_has_empty_state(self):
        bot = _Bot(agent_id="p1")
        assert SQLitePersister._checkpoint_state(bot) == {}

    def test_persist_false_field_is_excluded(self):
        """`_DelegatingBot.helper` is itself an independently checkpointed
        agent -- excluding it prevents it from being embedded as a
        duplicate, divergent copy inside the delegator's own checkpoint."""
        orch = _DelegatingBot(agent_id="orch1", helper=_PlainHelper())
        assert SQLitePersister._checkpoint_state(orch) == {}

    def test_unpicklable_field_raises_instead_of_silently_dropping(self, tmp_path):
        """Unlike the old JSON-based design (which silently dropped any field
        that wasn't JSON-serialisable), pickling fails loudly."""

        @dataclasses.dataclass(kw_only=True)
        class _LockBot(Agent):
            agent_id: str | None = None
            handle: threading.Lock = dataclasses.field(default_factory=threading.Lock)

            @Template.define
            def ask(self, q: str) -> str:
                """Answer: {q}"""
                raise NotHandled

        db_path = tmp_path / "checkpoints.db"
        with (
            handler(_FakeAgentCalls([("hi", "hello!", "hello!")])),
            handler(SQLitePersister(db_path)),
        ):
            bot = _LockBot(agent_id="lock1")
            with pytest.raises(TypeError):
                bot.ask("hi")


class TestCheckpointPersistence:
    """DB-level checkpoint behavior. There's no manual `save()` entry point --
    checkpointing only happens as a side effect of a successful outermost
    `Template` call -- so these go through `_FakeAgentCalls` rather than
    poking the database directly."""

    def test_checkpoint_creates_db_with_wal_mode(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        with (
            handler(_FakeAgentCalls([("hi", "hello!", "hello!")])),
            handler(SQLitePersister(db_path)),
        ):
            bot = _Bot(agent_id="wal1")
            bot.ask("hi")

        assert db_path.exists()
        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"

    def test_checkpoint_round_trip_via_call(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        with (
            handler(_FakeAgentCalls([("hi", "hello!", "hello!")])),
            handler(SQLitePersister(db_path)),
        ):
            bot = _Bot(agent_id="f1")
            bot.ask("hi")

        row = _load_row(db_path, "f1")
        assert row is not None
        assert json.loads(row[1]) == list(bot.__history__.values())

    def test_checkpoint_upsert_is_idempotent_across_calls(self, tmp_path):
        """Two successful calls on the same agent still leave exactly one row."""
        db_path = tmp_path / "checkpoints.db"
        with (
            handler(
                _FakeAgentCalls([("first", "ok1", "ok1"), ("second", "ok2", "ok2")])
            ),
            handler(SQLitePersister(db_path)),
        ):
            bot = _Bot(agent_id="idem1")
            bot.ask("first")
            bot.ask("second")

        conn = sqlite3.connect(str(db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE agent_id = ?", ("idem1",)
        ).fetchone()[0]
        conn.close()
        assert count == 1
        row = _load_row(db_path, "idem1")
        assert row is not None
        assert len(json.loads(row[1])) == 4  # both calls' messages accumulated

    def test_multiple_agents_share_one_db_independently(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        with (
            handler(_FakeAgentCalls([("hi a", "hi a reply", "hi a reply")])),
            handler(SQLitePersister(db_path)),
        ):
            _Bot(agent_id="ma").ask("hi a")

        with (
            handler(_FakeAgentCalls([("hi b", "hi b reply", "hi b reply")])),
            handler(SQLitePersister(db_path)),
        ):
            _Bot(agent_id="mb").ask("hi b")

        row_a, row_b = _load_row(db_path, "ma"), _load_row(db_path, "mb")
        assert row_a is not None and row_b is not None
        assert json.loads(row_a[1])[0]["content"] == "hi a"
        assert json.loads(row_b[1])[0]["content"] == "hi b"

    def test_fresh_instance_same_agent_id_loads_prior_history_from_db(self, tmp_path):
        """The core "process restart" scenario: a brand-new instance,
        constructed under a brand-new `SQLitePersister` pointed at the same
        db, comes back populated -- but only once `__history__` is actually
        accessed while that handler is active (loading is lazy, not forced
        at construction)."""
        db_path = tmp_path / "checkpoints.db"
        with (
            handler(_FakeAgentCalls([("before restart", "ack", "ack")])),
            handler(SQLitePersister(db_path)),
        ):
            _Bot(agent_id="restart1").ask("before restart")

        with handler(SQLitePersister(db_path)):
            fresh = _Bot(agent_id="restart1")
            assert fresh.__history__["u0"]["content"] == "before restart"

    def test_history_access_outside_handler_scope_does_not_load(self, tmp_path):
        """Checkpoint loading is gated on whether a handler is active *when*
        `__history__` is first accessed -- not on construction. Accessing it
        with no handler installed just yields a fresh, empty history, same
        as a transient agent."""
        db_path = tmp_path / "checkpoints.db"
        with (
            handler(_FakeAgentCalls([("hi", "hello!", "hello!")])),
            handler(SQLitePersister(db_path)),
        ):
            _Bot(agent_id="scoped1").ask("hi")

        fresh = _Bot(agent_id="scoped1")
        assert dict(fresh.__history__) == {}


class TestAutomaticCheckpointing:
    def test_checkpoint_written_after_successful_call(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        fake = _FakeAgentCalls([("hi", "hello!", "hello!")])

        with handler(fake), handler(SQLitePersister(db_path)):
            bot = _Bot(agent_id="auto1")
            result = bot.ask("hi")

        assert result == "hello!"
        row = _load_row(db_path, "auto1")
        assert row is not None
        assert len(json.loads(row[1])) == 2

    def test_nothing_saved_on_exception(self, tmp_path):
        """An unhandled error mid-call leaves no checkpoint at all -- there is
        nothing meaningful to save: the failed call's own history never
        touched `agent.__history__` (see `LiteLLMProvider._call`), so a
        checkpoint taken at that point would just be an empty/stale row."""
        db_path = tmp_path / "checkpoints.db"

        def _boom(template, args, kwargs):
            raise RuntimeError("simulated failure")

        with pytest.raises(RuntimeError):
            with (
                handler(_FakeAgentCalls([_boom])),
                handler(SQLitePersister(db_path)),
            ):
                bot = _Bot(agent_id="fail1")
                bot.ask("hi")

        assert _load_row(db_path, "fail1") is None

    def test_failed_call_does_not_prevent_subsequent_checkpoint(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"

        def _boom(template, args, kwargs):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            with (
                handler(_FakeAgentCalls([_boom])),
                handler(SQLitePersister(db_path)),
            ):
                bot = _Bot(agent_id="clear1")
                bot.ask("first")

        assert _load_row(db_path, "clear1") is None

        with (
            handler(_FakeAgentCalls([("second", "ok", "ok")])),
            handler(SQLitePersister(db_path)),
        ):
            result = bot.ask("second")

        assert result == "ok"
        row = _load_row(db_path, "clear1")
        assert row is not None
        assert len(json.loads(row[1])) == 2

    def test_dataclass_state_checkpointed_alongside_history(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        fake = _FakeAgentCalls([("hi", "hello", "hello")])

        with handler(fake), handler(SQLitePersister(db_path)):
            bot = _StatefulBot(agent_id="state1", counter=7)
            bot.ask("hi")

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT state, history FROM checkpoints WHERE agent_id = ?", ("state1",)
        ).fetchone()
        conn.close()
        assert row is not None
        assert pickle.loads(row[0]) == {"counter": 7, "label": "x"}
        assert len(json.loads(row[1])) == 2


class TestNestingAndPersistence:
    """Builds on the cross-agent/same-agent nesting fix in `completions.py`:
    `SQLitePersister` must checkpoint exactly once per outermost call per
    agent, regardless of same-agent or cross-agent nesting."""

    def test_same_agent_nested_call_checkpoints_exactly_once(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"

        def _outer(template, args, kwargs):
            bot.__history__["u-outer"] = {
                "id": "u-outer",
                "role": "user",
                "content": "outer",
            }
            inner_result = bot.inner("nested")
            bot.__history__["a-outer"] = {
                "id": "a-outer",
                "role": "assistant",
                "content": f"outer done, inner said {inner_result}",
            }
            return "outer result"

        fake = _FakeAgentCalls([_outer, ("nested q", "inner reply", "inner reply")])
        persist = SQLitePersister(db_path)
        bot = _NestingBot(agent_id="nest1")

        # There's no separate `save()` to spy on (checkpoint-writing is
        # inlined into `_call`), so count checkpoint-connection opens
        # instead: one to lazily load history on the very first access (via
        # `Template.__get__`, when `bot.outer` is first bound), and one to
        # write the checkpoint after the outermost call returns -- the
        # nested call must not trigger a third, independent open.
        connection_opens = 0

        def _count_opens():
            nonlocal connection_opens
            connection_opens += 1
            return fwd()

        with (
            handler(fake),
            handler(persist),
            handler({SQLitePersister._checkpoint_connection: _count_opens}),
        ):
            result = bot.outer("go")

        assert result == "outer result"
        assert connection_opens == 2

        row = _load_row(db_path, "nest1")
        assert row is not None
        # Final persisted history includes both the outer call's own
        # messages and the nested call's messages (same shared history
        # object), written back exactly once by the outer call.
        assert len(json.loads(row[1])) == 4

    def test_nested_failure_saves_nothing(self, tmp_path):
        """A failure anywhere in the call -- nested or not -- means the
        outermost call never returns successfully, so nothing is checkpointed
        at all, for either the inner or outer template."""
        db_path = tmp_path / "checkpoints.db"

        def _outer(template, args, kwargs):
            bot.inner("boom")
            return "unreachable"

        def _inner_boom(template, args, kwargs):
            raise RuntimeError("inner tool failed")

        fake = _FakeAgentCalls([_outer, _inner_boom])
        persist = SQLitePersister(db_path)
        bot = _NestingBot(agent_id="nestfail1")

        with pytest.raises(RuntimeError):
            with handler(fake), handler(persist):
                bot.outer("go")

        assert _load_row(db_path, "nestfail1") is None

    def test_persistent_agent_delegates_to_plain_agent_via_tool(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        helper = _PlainHelper()

        def _run(template, args, kwargs):
            result = orch.helper.answer("sub-question")
            return f"orchestrated: {result}"

        fake = _FakeAgentCalls([_run, ("sub-question", "sub-answer", "sub-answer")])
        orch = _DelegatingBot(agent_id="orch1", helper=helper)

        with handler(fake), handler(SQLitePersister(db_path)):
            result = orch.run("go")

        assert result == "orchestrated: sub-answer"

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT agent_id FROM checkpoints").fetchall()
        conn.close()
        # Only the persistent bot gets a DB row; the transient helper doesn't.
        assert [r[0] for r in rows] == ["orch1"]
        # But the helper's own in-memory history was still correctly
        # populated -- SQLitePersister passing straight through for
        # non-persistent-agent templates must not interfere with that.
        assert len(helper.__history__) == 2

    def test_two_persistent_agents_cooperate_each_checkpoints_independently(
        self, tmp_path
    ):
        db_path = tmp_path / "checkpoints.db"
        helper = _Bot(agent_id="helper1")
        orch = _DelegatingBot(agent_id="orch3", helper=helper)

        def _run(template, args, kwargs):
            return f"orchestrated: {helper.ask('sub-question')}"

        fake = _FakeAgentCalls([_run, ("sub-question", "sub-answer", "sub-answer")])

        with handler(fake), handler(SQLitePersister(db_path)):
            result = orch.run("go")

        assert result == "orchestrated: sub-answer"

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT agent_id, history FROM checkpoints ORDER BY agent_id"
        ).fetchall()
        conn.close()
        histories = {r[0]: json.loads(r[1]) for r in rows}
        assert set(histories) == {"helper1", "orch3"}
        assert len(histories["helper1"]) == 2
        assert len(histories["orch3"]) == 0  # orch's own turn never appended messages


class TestHandlerComposition:
    def test_persistence_and_retry_compose_end_to_end(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        mock = MockCompletionHandler(
            [
                make_tool_call_response("nonexistent_tool", "{}"),
                make_text_response("done"),
            ]
        )

        with (
            handler(LiteLLMProvider(model="test")),
            handler(mock),
            handler(TenacityRetryer()),
            handler(SQLitePersister(db_path)),
        ):
            bot = _Bot(agent_id="compose1")
            result = bot.ask("go")

        assert result == "done"
        row = _load_row(db_path, "compose1")
        assert row is not None
        assert len(json.loads(row[1])) >= 2

    def test_call_without_persistence_handler_behaves_like_plain_agent(self):
        """A persistent Agent (constructed with an explicit `agent_id`) can be
        called normally even when no `SQLitePersister` is installed for that
        particular call -- it just behaves like a transient Agent: no
        checkpoint load, no save, no crash. Persistence is best-effort,
        gated entirely on whether a handler happens to be active whenever
        `__history__` is touched."""
        bot = _Bot(agent_id="nopersist1")

        mock = MockCompletionHandler([make_text_response("fine")])
        with handler(LiteLLMProvider(model="test")), handler(mock):
            result = bot.ask("go")

        assert result == "fine"
        assert len(bot.__history__) >= 2


class TestThreadSafety:
    def test_concurrent_calls_from_different_threads_preserve_db_integrity(
        self, tmp_path
    ):
        db_path = tmp_path / "checkpoints.db"
        persist = SQLitePersister(db_path)
        errors: list[Exception] = []

        def _call_one(i: int) -> None:
            try:
                with (
                    handler(
                        _FakeAgentCalls([(f"msg {i}", f"reply {i}", f"reply {i}")])
                    ),
                    handler(persist),
                ):
                    bot = _Bot(agent_id=f"thread-{i}")
                    bot.ask(f"msg {i}")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_call_one, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        conn = sqlite3.connect(str(db_path))
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        conn.close()
        assert integrity == "ok"
        assert count == 16
