"""Checkpoint a Skill receiver's conversation and declared state to SQLite.

:class:`SQLitePersister` needs a stable ``__agent_id__`` on the receiver and
``HistoryBuilder`` in the handler stack. The standard :func:`~effectful.handlers.llm.harness.harness`
installs both when ``persist_db`` is set. Install it before the receiver's first
``__history__`` access (usually its first Skill call): restoration is lazy and
that first access is cached. Declare restart-safe dataclass fields for state that
must survive a process restart; dynamic attributes and live closures are not
saved. The database is trusted input because restoration uses pickle.

After a successful top-level bound turn, the checkpoint stores its committed messages
and declared dataclass fields. A nested turn can checkpoint before an enclosing
turn later fails; the history transaction does not undo that earlier checkpoint.
Fields marked ``metadata={"persist": False}`` and ``__agent_id__`` are excluded.
Serialization or database errors can surface after the turn's messages have
committed in memory; a checkpoint does not make that commit atomic with disk.

For example, give a dataclass receiver a stable identity and install the
database before its first Skill call::

    import dataclasses

    from effectful.handlers.llm import Skill
    from effectful.handlers.llm.harness import harness
    from effectful.ops.semantics import handler

    @dataclasses.dataclass
    class DurableAssistant:
        __agent_id__: str
        notes: list[str] = dataclasses.field(default_factory=list)

        @Skill.define
        def answer(self, question: str) -> str:
            \"\"\"Answer {question} using prior conversation and {self.notes}.\"\"\"

    with handler(harness(model="openai/gpt-5-mini", persist_db="agents.db")):
        assistant = DurableAssistant(__agent_id__="assistant:demo")
        print(assistant.answer("What did we decide last time?"))

Handler authors assembling a stack directly must include ``HistoryBuilder``
before ``SQLitePersister``. For a Skill that needs no Tools::

    from pathlib import Path
    from effectful.handlers.llm.harness.durability.retrying import TenacityRetryer
    from effectful.handlers.llm.harness.durability.persistence import SQLitePersister
    from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
    from effectful.handlers.llm.harness.hooks import AgentLoop
    from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
    from effectful.ops.semantics import handler

    with (
        handler(AgentLoop()),
        handler(LiteLLMConfigurer(model="openai/gpt-5-mini")),
        handler(HistoryBuilder()),
        handler(TenacityRetryer()),
        handler(SQLitePersister(Path("agents.db"))),
    ):
        assistant.answer("What did we decide last time?")
"""

import dataclasses
import json
import pathlib
import pickle
import sqlite3
import typing

from effectful.handlers.llm.harness.hooks import (
    PromptInjectingInterpretation,
    call_agent,
)
from effectful.handlers.llm.types import Agent, Skill
from effectful.ops.semantics import fwd
from effectful.ops.syntax import implements
from effectful.ops.types import Operation


class SQLitePersister(PromptInjectingInterpretation):
    """A successful turn on this receiver checkpoints its conversation and declared
    dataclass fields. Later processes can restore them using the same persistent
    identity. Read earlier messages as prior conversation, even if they were
    recorded in another process.

    Only declared fields are saved. A dynamic attribute or live function attached
    to ``self`` can survive in this process but will not be restored. A failed
    turn does not write a checkpoint; Python object mutations made before the
    failure are not rolled back in this process. A successful nested turn may
    already have checkpointed before an enclosing turn fails.
    """

    db_path: pathlib.Path

    @Operation.define
    @staticmethod
    def _checkpoint_connection() -> sqlite3.Connection | None:
        """Return a connection to the currently active `SQLitePersister`'s
        checkpoint database, or `None` if no persistence handler is installed.

        Purely a resource hook -- the handler's implementation just hands
        back a connection; `Agent.__history__` (see `types.py`) and
        `SQLitePersister` own all the query/serialisation logic around it.
        """
        return None

    def __init__(self, db_path: pathlib.Path) -> None:
        """Open (creating if absent) the checkpoint database.

        WAL mode buys crash tolerance: if the process is killed mid-write,
        SQLite's journal-based recovery keeps the database consistent.
        ``synchronous=NORMAL`` is the usual companion to WAL -- it trades an
        fsync per commit for one per checkpoint, which is the right trade when
        the alternative to a lost final commit is re-running the call.

        All state is read from and written to the database directly, with no
        in-memory cache to go stale, so several processes may share one file.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = pathlib.Path(db_path)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            # `history` is the message sequence as a JSON array, in order.
            # Kept in sync with the SELECT in `Agent.__history__` (types.py)
            # and the INSERT below.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    agent_id TEXT PRIMARY KEY,
                    state    BLOB NOT NULL DEFAULT x'',
                    history  TEXT NOT NULL DEFAULT '[]'
                )
                """
            )

    @implements(_checkpoint_connection.__func__)  # type: ignore[attr-defined]
    def _get_checkpoint_connection(self) -> sqlite3.Connection:
        """Open a new SQLite connection to the checkpoint database.

        Each call returns a fresh connection, making it safe to use from any
        thread. WAL mode and table creation are already applied by `__init__`.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @staticmethod
    def _checkpoint_state(agent: Agent) -> dict[str, typing.Any]:
        """The declared dataclass fields of `agent` that should be checkpointed.

        Only declared fields are ever considered -- not `agent.__dict__` at
        large -- so transient/cached attributes are excluded by default. A
        field can opt out explicitly with
        `dataclasses.field(metadata={"persist": False})`, which is required
        for any field that is itself an independently checkpointed `Agent`
        (otherwise it would be embedded as a duplicate, divergent copy inside
        this agent's own checkpoint). ``__agent_id__`` (if a `@dataclass` subclass
        redeclares it -- see `Agent`) is always excluded: it's already the
        row's primary key, fixed at construction time, with nothing to
        restore.

        Non-dataclass agents have no declared fields to walk, so they
        checkpoint history only.
        """
        if not dataclasses.is_dataclass(agent):
            return {}
        return {
            f.name: getattr(agent, f.name)
            for f in dataclasses.fields(agent)
            if f.name != "__agent_id__" and f.metadata.get("persist", True)
        }

    @implements(call_agent)
    def call_agent[**P, T](
        self, skill: Skill[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Checkpoint the agent after the call returns.

        The save happens *after* `fwd`, so nothing is written when the call
        raises: a `Skill` call's work happens against a private copy of the
        agent's history that is only written back on success (see
        `HistoryBuilder.call_agent`), so an interrupted call's partial exchange --
        and any other in-process state not captured by `__history__`, such as a
        REPL session -- is unrecoverable regardless of what this handler
        does.

        Two gates decide whether anything is written. The skill must be bound
        to an agent (``__history__``), and that agent must have been given an
        stable ``__agent_id__`` (see `Agent`): a transient agent, the default, is
        never written to the database, even when nested inside a persisted
        agent's call under this same handler.

        A successful nested call runs this rule too. On the same receiver, its
        messages are discarded by `HistoryBuilder`, so that checkpoint stores
        the previously committed history with any current declared field
        mutations. An outer success overwrites it with the outer turn's history.
        An outer failure leaves the nested checkpoint in place: message
        transactions do not make several checkpoints atomic.
        """
        result = fwd()
        if hasattr(skill, "__history__") and skill.__self__.__is_persistent__:  # type: ignore
            agent: Agent = skill.__self__  # type: ignore
            agent_id = agent.__agent_id__
            state_blob = pickle.dumps(self._checkpoint_state(agent))
            history_json = json.dumps(list(skill.__history__), default=str)
            with self._checkpoint_connection() as conn:  # type: ignore[union-attr]
                conn.execute(
                    """
                    INSERT INTO checkpoints (agent_id, state, history)
                    VALUES (?, ?, ?)
                    ON CONFLICT(agent_id) DO UPDATE SET
                        state   = excluded.state,
                        history = excluded.history
                    """,
                    (agent_id, state_blob, history_json),
                )

        return result
