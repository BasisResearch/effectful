import dataclasses
import json
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from effectful.handlers.llm.completions import get_agent_history
from effectful.handlers.llm.template import Agent, Template, get_bound_agent
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled


@Template.define
def summarize_context(transcript: str) -> str:
    """Summarise the following conversation transcript into a concise
    context summary. Preserve key facts, decisions, and any
    information the agent would need to continue working.

    Transcript:
    {transcript}"""
    raise NotHandled


class PersistentAgent(Agent):
    """An :class:`Agent` whose history can be persisted by :class:`PersistenceHandler`.

    This is a lightweight marker class. All persistence *behaviour*
    (checkpointing, handoff, DB I/O) lives in :class:`PersistenceHandler`,
    a composable handler following the same pattern as
    :class:`~effectful.handlers.llm.completions.RetryLLMHandler`.

    Unlike plain :class:`Agent` (which uses ``id(self)`` by default),
    ``PersistentAgent`` **requires** a stable ``agent_id`` so that
    checkpoints can be matched across process restarts.

    Override :meth:`checkpoint_state` and :meth:`restore_state` to persist
    custom subclass state alongside the message history.

    **Usage**::

        from pathlib import Path
        from effectful.handlers.llm.persistence import PersistentAgent, PersistenceHandler
        from effectful.handlers.llm import Template
        from effectful.handlers.llm.completions import LiteLLMProvider
        from effectful.ops.semantics import handler
        from effectful.ops.types import NotHandled

        class ResearchBot(PersistentAgent):
            \"""You are a research assistant that remembers prior sessions.\"""

            @Template.define
            def ask(self, question: str) -> str:
                \"""Answer: {question}\"""
                raise NotHandled

        bot = ResearchBot(agent_id="research-bot")

        with handler(LiteLLMProvider()), handler(PersistenceHandler(Path("./state/checkpoints.db"))):
            bot.ask("What is the capital of France?")
            # Kill process here, restart, and the bot resumes with context.
    """

    def __init__(self, *, agent_id: str):
        self.__agent_id__ = agent_id

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of subclass state to persist.

        The default implementation serialises all
        :func:`dataclasses.dataclass` fields.  Override this (and
        :meth:`restore_state`) for custom serialisation.
        """
        if not dataclasses.is_dataclass(self):
            return {}
        state: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            try:
                json.dumps(val)
                state[f.name] = val
            except (TypeError, ValueError):
                pass  # skip non-serialisable fields
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore subclass state from *state* dict.

        The default implementation sets each key as an attribute.
        Override this (and :meth:`checkpoint_state`) for custom
        deserialisation.
        """
        for key, value in state.items():
            setattr(self, key, value)


def _init_db(conn: sqlite3.Connection) -> None:
    """Create the checkpoints table and configure WAL mode for crash tolerance."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS checkpoints (
            agent_id TEXT PRIMARY KEY,
            handoff  TEXT NOT NULL DEFAULT '',
            state    TEXT NOT NULL DEFAULT '{}',
            history  TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    conn.commit()


class PersistenceHandler(ObjectInterpretation):
    """Handler that persists :class:`PersistentAgent` history to a SQLite database.

    Install alongside
    :class:`~effectful.handlers.llm.completions.LiteLLMProvider`::

        with handler(LiteLLMProvider()), handler(PersistenceHandler(Path("./state/checkpoints.db"))):
            bot.ask("question")

    Uses SQLite WAL mode for crash tolerance.  If the process is killed
    mid-write, SQLite's journal-based recovery ensures the database
    remains consistent.

    All state is read from and written to the database directly — no
    in-memory caching.  This makes the handler stateless (aside from
    nesting depth tracking) and easy to reason about.

    **Automatic checkpointing**:

    - **Before** each top-level template call: saves a checkpoint with a
      handoff note describing the in-progress work.
    - **After** each successful call: clears the handoff and saves again.
    - **On failure**: saves the checkpoint (with handoff) so the next
      session can resume.

    **Crash recovery**: on the next run, the handoff note from the prior
    crash is injected into the system prompt so the LLM can resume.

    **Nested calls** (e.g. a tool calling another template on the same
    agent) are passed through without additional checkpointing.

    Composes with :class:`~effectful.handlers.llm.completions.RetryLLMHandler`
    and :class:`CompactionHandler`::

        with (
            handler(LiteLLMProvider()),
            handler(RetryLLMHandler()),
            handler(CompactionHandler()),
            handler(PersistenceHandler(Path("./state/checkpoints.db"))),
        ):
            bot.ask("question")

    **Crash recovery example**::

        from pathlib import Path
        from effectful.handlers.llm.persistence import PersistentAgent, PersistenceHandler
        from effectful.handlers.llm import Template
        from effectful.handlers.llm.completions import LiteLLMProvider
        from effectful.ops.semantics import handler
        from effectful.ops.types import NotHandled

        class Bot(PersistentAgent):
            \"""You are a helpful assistant.\"""

            @Template.define
            def work(self, task: str) -> str:
                \"""Do: {task}\"""
                raise NotHandled

        bot = Bot(agent_id="worker")
        persist = PersistenceHandler(Path("./state/checkpoints.db"))

        # Session 1 — process crashes mid-call
        with handler(LiteLLMProvider(model="gpt-4o-mini")), handler(persist):
            bot.work("step 1")  # completes, checkpointed
            bot.work("step 2")  # process killed here

        # Session 2 — restart with the same db_path
        bot2 = Bot(agent_id="worker")
        persist2 = PersistenceHandler(Path("./state/checkpoints.db"))
        with handler(LiteLLMProvider(model="gpt-4o-mini")), handler(persist2):
            # History from session 1 is restored automatically.
            # The handoff note "Executing work ..." tells the LLM what
            # was in progress when the crash occurred.
            bot2.work("step 2")  # resumes with full context

    Use :meth:`save` for manual checkpointing outside the automatic flow
    (e.g. after initialising agent state in a choreography).

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._tls = threading.local()
        self._db_lock = threading.Lock()
        self._db_initialized = False

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection to the checkpoint database.

        Each call returns a fresh connection, making it safe to use from
        any thread.  WAL mode and table creation are applied once on the
        first call (guarded by ``_db_lock``).
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        if not self._db_initialized:
            with self._db_lock:
                if not self._db_initialized:
                    _init_db(conn)
                    self._db_initialized = True
        return conn

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""
        return self._db_path

    def _get_depths(self) -> dict[str, int]:
        if not hasattr(self._tls, "depths"):
            self._tls.depths = {}
        return self._tls.depths

    def _load_row(self, agent_id: str) -> tuple[str, str, str] | None:
        """Read a checkpoint row from the database.

        Returns ``(handoff, state_json, history_json)`` or ``None``.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT handoff, state, history FROM checkpoints WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        finally:
            conn.close()
        return row

    def _ensure_loaded(self, agent: PersistentAgent) -> bool:
        """Load an agent's checkpoint from the database into the in-process history.

        Safe to call multiple times — only loads once per agent (tracked
        via thread-local ``_loaded`` set to avoid re-seeding history that
        is already live in memory).

        Returns ``True`` if a checkpoint was found and loaded.
        """
        agent_id = agent.__agent_id__
        loaded = self._get_loaded()
        if agent_id in loaded:
            return False
        loaded.add(agent_id)

        row = self._load_row(agent_id)
        if row is None:
            return False

        _handoff, state_json, history_json = row
        agent.restore_state(json.loads(state_json))
        stored = get_agent_history(agent_id)
        stored.clear()
        stored.update({msg["id"]: msg for msg in json.loads(history_json)})
        return True

    def _get_loaded(self) -> set[str]:
        if not hasattr(self._tls, "loaded"):
            self._tls.loaded = set()
        return self._tls.loaded

    def save(self, agent: PersistentAgent, handoff: str = "") -> Path:
        """Write an agent's current state to the database and return the db path."""
        agent_id = agent.__agent_id__
        history = get_agent_history(agent_id)
        state_json = json.dumps(agent.checkpoint_state(), default=str)
        history_json = json.dumps(list(history.values()), default=str)

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO checkpoints (agent_id, handoff, state, history)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    handoff = excluded.handoff,
                    state   = excluded.state,
                    history = excluded.history
                """,
                (agent_id, handoff, state_json, history_json),
            )
            conn.commit()
        finally:
            conn.close()

        return self.db_path

    def _get_handoff(self, agent_id: str) -> str:
        """Return the current handoff note for *agent_id* (reads from DB)."""
        row = self._load_row(agent_id)
        if row is None:
            return ""
        return row[0]

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        agent = get_bound_agent(template)
        if not isinstance(agent, PersistentAgent):
            return fwd(template, *args, **kwargs)

        agent_id = agent.__agent_id__
        self._ensure_loaded(agent)

        # Nesting: only checkpoint for outermost call per agent
        depths = self._get_depths()
        depth = depths.get(agent_id, 0)
        depths[agent_id] = depth + 1
        is_outermost = depth == 0

        try:
            if is_outermost:
                # Inject prior-session handoff into system prompt
                prior_handoff = self._get_handoff(agent_id)
                if prior_handoff:
                    template.__system_prompt__ = (
                        f"{template.__system_prompt__}\n\n"
                        f"[HANDOFF FROM PRIOR SESSION] {prior_handoff}"
                    )

                # Record current call as handoff for crash recovery
                current_handoff = (
                    f"Executing {template.__name__} with args={repr(args)[:200]}"
                )
                self.save(agent, handoff=current_handoff)

            result = fwd(template, *args, **kwargs)

            if is_outermost:
                self.save(agent, handoff="")

            return result
        except BaseException:
            if is_outermost:
                # Preserve handoff so next session knows what was in progress
                self.save(agent, handoff=current_handoff)
            raise
        finally:
            depths[agent_id] = depth


class CompactionHandler(ObjectInterpretation):
    """Handler that compacts agent history when it exceeds a threshold.

    After each top-level template call on an :class:`Agent`, if the
    message history exceeds ``max_history_len``, older messages are
    summarised into a single context-summary message via an LLM call.::

        with handler(LiteLLMProvider()), handler(CompactionHandler(max_history_len=20)):
            agent.ask("question")  # history auto-compacted after call
    """

    def __init__(self, max_history_len: int = 50) -> None:
        self._max_history_len = max_history_len
        self._tls = threading.local()

    def _get_depths(self) -> dict[str, int]:
        if not hasattr(self._tls, "depths"):
            self._tls.depths = {}
        return self._tls.depths

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        agent = get_bound_agent(template)
        if not isinstance(agent, Agent):
            return fwd(template, *args, **kwargs)

        agent_id = agent.__agent_id__
        depths = self._get_depths()
        depth = depths.get(agent_id, 0)
        depths[agent_id] = depth + 1
        is_outermost = depth == 0

        try:
            result = fwd(template, *args, **kwargs)

            if is_outermost:
                history = get_agent_history(agent_id)
                if len(history) > self._max_history_len:
                    self._compact(agent_id, history)

            return result
        finally:
            depths[agent_id] = depth

    def _compact(self, agent_id: str, history: OrderedDict[str, Any]) -> None:
        keep_recent = max(self._max_history_len // 2, 4)
        items = list(history.items())
        if len(items) <= keep_recent:
            return

        split = len(items) - keep_recent
        # Never split between a tool_use and its tool_result(s).
        while split > 0 and items[split][1].get("role") == "tool":
            split -= 1
        if split <= 0:
            return

        old_items = items[:split]
        recent_items = items[split:]

        old_text_parts: list[str] = []
        for _, msg in old_items:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                content = " ".join(text_parts)
            if content:
                old_text_parts.append(f"[{role}]: {content}")
        old_transcript = "\n".join(old_text_parts)

        if not old_transcript.strip():
            return

        summary = summarize_context(old_transcript)

        summary_msg: dict[str, Any] = {
            "id": f"compaction-{agent_id}",
            "role": "user",
            "content": f"[CONTEXT SUMMARY FROM PRIOR CONVERSATION]\n{summary}",
        }
        history.clear()
        history[summary_msg["id"]] = summary_msg
        for key, msg in recent_items:
            history[key] = msg
