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
    """Handler that persists `Agent` history and state to a SQLite database.

    Install alongside `AgentLoop`, `LiteLLMConfigurer` and `HistoryBuilder`::

        with (
            handler(AgentLoop()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(SQLitePersister(Path("./state/checkpoints.db"))),
        ):
            bot.ask("question")

    `HistoryBuilder` is what opens the transaction a call's messages accumulate
    in, so a stack without it has no history for this handler to checkpoint.
    `harness` assembles all of these; assemble them by hand only to leave one out.

    Only agents constructed with an explicit `agent_id` (see `Agent`) are
    checkpointed -- a transient agent (the default) is never written to the
    database, even nested inside a persisted agent's call under the same
    handler.

    Uses SQLite WAL mode for crash tolerance: if the process is killed
    mid-write, SQLite's journal-based recovery keeps the database consistent.
    All state is read from and written to the database directly -- there is no
    in-memory cache to go stale.

    **Automatic checkpointing**: after each outermost call for a given agent
    *returns successfully*, its state and history are saved.

    **On an exception**, nothing is saved. A `Skill` call's work happens
    against a private copy of the agent's history that is only written back
    on success (see `AgentLoop._call`), so an interrupted call's partial
    exchange -- and any other in-process state not captured by `__history__`
    (e.g. a `PythonRepl` session) -- is unrecoverable regardless of what this
    handler does. There is deliberately no crash-recovery "handoff" note: the
    last successful checkpoint is already a complete, uncorrupted transcript,
    and the caller is expected to simply retry the request that didn't finish.

    **Nested calls** (e.g. a tool invoking another skill on the same
    agent) are passed through without additional checkpointing -- only the
    outermost call per agent saves.

    Composes with `TenacityRetryer`::

        with (
            handler(AgentLoop()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(SQLitePersister(Path("./state/checkpoints.db"))),
        ):
            bot.ask("question")

    Args:
        db_path: pathlib.Path to the SQLite database file.
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
        this agent's own checkpoint). `agent_id` (if a `@dataclass` subclass
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
    def _call[**P, T](self, skill: Skill[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        result = fwd()
        if hasattr(skill, "__history__") and skill.__self__.__is_persistent__:  # type: ignore
            agent: Agent = skill.__self__  # type: ignore
            agent_id = agent.__agent_id__
            state_blob = pickle.dumps(self._checkpoint_state(agent))
            history_json = json.dumps(list(skill.__history__), default=str)
            with self._checkpoint_connection() as conn:
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
