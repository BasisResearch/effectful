"""Checkpointing of `Agent` history and state to a SQLite database.

`HistoryBuilder` is what opens the transaction a call's messages accumulate in,
so a stack without it has no history for this handler to checkpoint.
`~effectful.handlers.llm.harness.harness` assembles all of these; assemble them
by hand only to leave one out::

    with (
        handler(AgentLoop()),
        handler(LiteLLMConfigurer()),
        handler(HistoryBuilder()),
        handler(TenacityRetryer()),
        handler(SQLitePersister(Path("./state/checkpoints.db"))),
    ):
        bot.ask("question")

There is deliberately no crash-recovery "handoff" note written on restore: the
last successful checkpoint is already a complete, uncorrupted transcript, and
the caller is expected to simply retry the request that didn't finish.

.. rubric:: What a checkpoint holds

Durability needs both ``persist_db=...`` in the harness and a stable
``__agent_id__`` on the Skill-owning object. Install before the receiver's
``__history__`` is first read (binding a Skill method reads it): restoration is
lazy and cached, see `Agent.__history__`. A checkpoint holds the message history
and the declared dataclass fields `SQLitePersister._checkpoint_state` selects;
`SQLitePersister.call_agent` says when one is written and what a nested turn's
checkpoint means. Treat the SQLite file as trusted input, because restoration
unpickles it.

Declare a stable id and restart-safe fields, then install persistence before the
receiver's history is first accessed::

    import dataclasses

    from effectful.handlers.llm import Skill
    from effectful.handlers.llm.harness import harness
    from effectful.ops.semantics import handler


    @dataclasses.dataclass
    class DurableAssistant:
        \"\"\"Answer questions using state restored for one stable user identity.\"\"\"

        __agent_id__: str
        notes: list[str] = dataclasses.field(default_factory=list)

        @Skill.define
        def answer(self, question: str) -> str:
            \"\"\"Answer {question}, using prior conversation and {self.notes}.\"\"\"


    with handler(harness(model="openai/gpt-5-mini", persist_db="agents.db")):
        assistant = DurableAssistant(__agent_id__="assistant:eli")
        print(assistant.answer("What did we decide last time?"))

Use a namespaced identity whose meaning remains stable across restarts. Do not
put live closures or untrusted values in persisted fields. See
:mod:`effectful.handlers.llm.examples.acp.assistant` for session ids mapped to persistent Skill-owner
identities in a full application.
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
    """This conversation outlives the process. When a call you are answering
    returns, the whole exchange and the agent's declared fields are written to
    disk, and the next time this agent runs -- in a later process, days from now
    -- they are restored. The history you are reading may therefore begin long
    before this run started.

    So anything you set on the agent persists, and is worth setting
    deliberately: notes, accumulated findings, a running summary. Conversely,
    do not re-derive what an earlier turn already established and recorded; it
    is in front of you because it was saved, not because it was just computed.

    A call that raises saves nothing. If you are heading toward an error, an
    intermediate result you want kept should be recorded before the failure,
    not after it.
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

        Nested calls -- a tool invoking another skill on the same agent -- run
        this rule too, so each writes its own checkpoint on the way out. That is
        harmless rather than intended: the agent's history is one shared object,
        so the enclosing call's save overwrites the nested one with a superset,
        and the row left behind is the state as of the outermost return. A nested
        checkpoint can therefore outlive an enclosing turn that later fails:
        history transactions are not a distributed transaction over checkpoints.
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
