"""Choreographic programming for multi-agent LLM systems.

Write a single ``async`` function describing how agents interact from a global
perspective, then run it with automatic endpoint projection (EPP). Every agent
runs that same function as its own `asyncio.Task`; inter-agent communication
is handled automatically through a `TaskQueue`, and the whole process is
crash-tolerant and restartable.

## How it works

Each `step` in the choreography is assigned an incrementing step ID. Because
every agent runs the same program and every step's result is shared through
the queue, all agents allocate the same IDs in the same order. For a given
step, `EndpointProjection` either:

- **claims and executes** it, if the step's template belongs to this agent; or
- **awaits its result**, if it belongs to another agent.

Completed steps are recorded in the queue. On restart the program re-runs from
the top: completed steps return their cached result immediately, so execution
resumes at the first step that never finished.

## Why the program is async, and where the threads went

`effectful`'s handler stack is synchronous, top to bottom: `Template.__apply__`,
`fwd`, and everything in `effectful.handlers.llm.completions` down to
`litellm.completion` are blocking calls. Two consequences shape this module.

*A handler may never be a coroutine function.* `coproduct` wraps every handler
in a synchronous continuation (see `effectful.internals.runtime._set_prompt`),
so an ``async def`` handler would return an un-awaited coroutine, the wrapper's
``with handler(...)`` block would exit, and the body would later run outside the
interpretation that defines its `fwd`. `EndpointProjection` is therefore *not*
an `effectful.ops.syntax.ObjectInterpretation`; it is an ordinary object held
in a `contextvars.ContextVar` and driven explicitly by `step` and `scatter`.
That is also why the choreography spells its steps out with ``await`` rather
than calling ``architect.plan(spec)`` directly.

*A template call must still run on a thread.* `call` hands the blocking call to
a worker thread and awaits it, so an agent waiting on a peer costs a suspended
coroutine rather than a parked thread. Note that the naive alternative --
wrapping each agent's whole program in `asyncio.to_thread` -- deadlocks here:
agents block on each other, `asyncio.to_thread` draws from a default executor
of ``min(32, cpu_count + 4)`` workers, and waiting agents hold workers that the
agents they wait for can never get. `Choreography` sizes its own executor to
the number of agents for the same reason.

## Primitives

`step`
    A choreography step: bookkeeping in the queue plus execution on a worker
    thread. This is the unit of replay.
`call`
    Execution on a worker thread with no bookkeeping. Use it for work that is
    already covered by an enclosing step -- notably inside `scatter`.
`scatter`
    Distribute items across a pool of same-role agents by claim-based pull.

Several scatters run concurrently with `asyncio.gather`; agents belonging to
more than one group work on all of them at once::

    specs, tests, proofs = await asyncio.gather(
        scatter(blocks, spec_writer, lambda w, b: call(w.write_spec, b)),
        scatter(blocks, tester, lambda t, b: call(t.write_tests, b)),
        scatter(blocks, prover, lambda p, b: call(p.prove, b)),
    )

Step IDs are allocated when `step`/`scatter` is *called*, not when the returned
awaitable is *awaited*, so the IDs in a `asyncio.gather` are deterministic and
agree across agents.

## Example -- sequential choreography with a review loop

::

    import asyncio
    from pathlib import Path
    from typing import Literal, TypedDict

    from effectful.handlers.llm import Agent, Template
    from effectful.handlers.llm.completions import (
        LiteLLMProvider, RetryLLMHandler, SQLitePersister,
    )
    from effectful.handlers.llm.multi import (
        Choreography, PersistentTaskQueue, call, scatter, step,
    )
    from effectful.ops.types import NotHandled

    class ReviewResult(TypedDict):
        verdict: Literal["PASS", "NEEDS_FIXES"]
        feedback: str

    class Architect(Agent):
        \"\"\"You are a software architect.\"\"\"

        @Template.define
        def plan_modules(self, project_spec: str) -> str:
            \"\"\"Break this project into modules: {project_spec}\"\"\"
            raise NotHandled

    class Coder(Agent):
        \"\"\"You are a Python developer.\"\"\"

        @Template.define
        def implement_module(self, spec: str) -> str:
            \"\"\"Implement the module: {spec}\"\"\"
            raise NotHandled

    class Reviewer(Agent):
        \"\"\"You are a code reviewer.\"\"\"

        @Template.define
        def review_code(self, code: str) -> ReviewResult:
            \"\"\"Review this code: {code}\"\"\"
            raise NotHandled

    async def build_codebase(project_spec, architect, coder, reviewer):
        plan = await step(architect.plan_modules, project_spec)
        code = await step(coder.implement_module, plan)
        while True:
            result = await step(reviewer.review_code, code)
            if result["verdict"] == "PASS":
                return code
            code = await step(coder.implement_module, result["feedback"])

    architect = Architect(agent_id="architect")
    coder = Coder(agent_id="coder")
    reviewer = Reviewer(agent_id="reviewer")

    choreo = Choreography(
        build_codebase,
        agents=[architect, coder, reviewer],
        queue=PersistentTaskQueue(Path("./state/task_queue.db")),
        handlers=[
            LiteLLMProvider(model="gpt-4o-mini"),
            RetryLLMHandler(),
            SQLitePersister(Path("./state/checkpoints.db")),
        ],
    )
    # Kill at any point, restart, and it resumes where it left off.
    result = choreo.run(
        project_spec="Build a URL slugify library",
        architect=architect,
        coder=coder,
        reviewer=reviewer,
    )

## Example -- parallel scatter across multiple coders

::

    async def build_parallel(project_spec, architect, coder, reviewer):
        plan = await step(architect.plan_modules, project_spec)
        # Each module becomes a task; coders claim from the queue until none
        # remain -- natural load balancing.
        codes = await scatter(
            plan["modules"], coder,
            lambda c, mod: call(c.implement_module, str(mod)),
        )
        return [await step(reviewer.review_code, code) for code in codes]

    choreo = Choreography(
        build_parallel,
        agents=[architect, coder1, coder2, coder3, reviewer],
        queue=PersistentTaskQueue(Path("./state/task_queue.db")),
        handlers=[LiteLLMProvider(model="gpt-4o-mini"), RetryLLMHandler()],
    )
    # Pass a list for a role -- scatter distributes across all three coders.
    reviews = choreo.run(
        project_spec="Build textkit with slugify, wrap, and redact modules",
        architect=architect,
        coder=[coder1, coder2, coder3],
        reviewer=reviewer,
    )

"""

import abc
import asyncio
import concurrent.futures
import contextlib
import contextvars
import functools
import json
import sqlite3
import typing
import uuid
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from effectful.handlers.llm.template import Agent
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation
from effectful.ops.types import Interpretation

# ── Sentinel ───────────────────────────────────────────────────────


class _Missing:
    """The type of `MISSING`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


MISSING: typing.Final = _Missing()
"""Returned by `TaskQueue.get_result` for a task that is not done.

A distinct sentinel rather than ``None``, because ``None`` is a perfectly good
result for a step to have -- conflating the two makes a poll loop wait forever
for a step that already finished.
"""


# ── TaskQueue ──────────────────────────────────────────────────────


class TaskStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"


class TaskQueue(abc.ABC):
    """Abstract task queue with claim-based ownership.

    Subclasses implement persistent (file-based) or in-memory storage. All
    methods are coroutines: an implementation either completes on the event
    loop (`InMemoryTaskQueue`) or hands its I/O to a worker thread
    (`PersistentTaskQueue`).
    """

    @abc.abstractmethod
    async def submit(
        self,
        task_type: str,
        payload: dict,
        task_id: str | None = None,
    ) -> str:
        """Add a new task. Returns the task ID.

        Idempotent when *task_id* is given: if a task with that ID already
        exists, in any state, the call is a no-op. Choreography step IDs are
        deterministic, so every agent submits every task and exactly one
        submission wins.
        """

    @abc.abstractmethod
    async def claim_id(self, task_id: str, owner: str) -> dict | None:
        """Atomically claim the pending task with exactly this ID.

        Returns the task dict, or ``None`` if it is not pending (already
        claimed, done, failed, or absent).
        """

    @abc.abstractmethod
    async def claim_prefix(self, prefix: str, owner: str) -> dict | None:
        """Atomically claim the lowest-ordered pending task whose ID starts
        with *prefix*, or ``None`` if there is none."""

    @abc.abstractmethod
    async def complete(self, task_id: str, owner: str, result: Any = None) -> None:
        """Mark a claimed task as done with *result*."""

    @abc.abstractmethod
    async def fail(self, task_id: str, owner: str, error: str) -> None:
        """Mark a claimed task as failed."""

    @abc.abstractmethod
    async def get_result(self, task_id: str) -> Any:
        """Return the result of a completed task, or `MISSING`."""

    @abc.abstractmethod
    async def release_stale_claims(self, owner: str) -> int:
        """Return *owner*'s unfinished tasks to pending, and return how many.

        Called on startup for every agent. Both claimed *and failed* tasks are
        released: a claimed task belongs to a process that is no longer
        running, and a failed one has to be retried, since a step that stays
        failed is a step whose result never arrives.
        """

    @abc.abstractmethod
    async def pending_count(self) -> int:
        """Count tasks that are still pending."""

    @abc.abstractmethod
    async def all_done(self) -> bool:
        """``True`` if no pending or claimed tasks remain."""

    async def close(self) -> None:
        """Release any resources held by the queue. Idempotent."""


class InMemoryTaskQueue(TaskQueue):
    """In-memory task queue for testing or ephemeral workflows.

    Not crash-tolerant -- all state is lost when the process exits.

    No lock is needed, and none is taken: every method runs to completion on
    the event loop without awaiting, so no two can interleave. That does mean
    a single queue instance belongs to a single event loop.

    >>> import asyncio
    >>> async def demo():
    ...     q = InMemoryTaskQueue()
    ...     await q.submit("work", {}, task_id="t1")
    ...     task = await q.claim_id("t1", "worker-1")
    ...     await q.complete(task["id"], "worker-1", None)
    ...     return await q.get_result("t1"), await q.get_result("t2")
    >>> asyncio.run(demo())
    (None, MISSING)
    """

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}  # task_id -> task dict

    async def submit(
        self,
        task_type: str,
        payload: dict,
        task_id: str | None = None,
    ) -> str:
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]
        self._tasks.setdefault(
            task_id,
            {
                "id": task_id,
                "type": task_type,
                "payload": payload,
                "status": TaskStatus.PENDING,
                "owner": "",
                "result": MISSING,
            },
        )
        return task_id

    def _claim(self, task_id: str | None, owner: str) -> dict | None:
        if task_id is None:
            return None
        task = self._tasks[task_id]
        task["status"] = TaskStatus.CLAIMED
        task["owner"] = owner
        return dict(task)

    async def claim_id(self, task_id: str, owner: str) -> dict | None:
        task = self._tasks.get(task_id)
        pending = task is not None and task["status"] == TaskStatus.PENDING
        return self._claim(task_id if pending else None, owner)

    async def claim_prefix(self, prefix: str, owner: str) -> dict | None:
        return self._claim(
            next(
                (
                    tid
                    for tid in sorted(self._tasks)
                    if tid.startswith(prefix)
                    and self._tasks[tid]["status"] == TaskStatus.PENDING
                ),
                None,
            ),
            owner,
        )

    def _finish(self, task_id: str, status: TaskStatus, result: Any) -> None:
        task = self._tasks.get(task_id)
        if task is None or task["status"] != TaskStatus.CLAIMED:
            return
        task["status"] = status
        task["result"] = result

    async def complete(self, task_id: str, owner: str, result: Any = None) -> None:
        self._finish(task_id, TaskStatus.DONE, result)

    async def fail(self, task_id: str, owner: str, error: str) -> None:
        self._finish(task_id, TaskStatus.FAILED, {"error": error})

    async def get_result(self, task_id: str) -> Any:
        task = self._tasks.get(task_id)
        if task is None or task["status"] != TaskStatus.DONE:
            return MISSING
        return task["result"]

    async def release_stale_claims(self, owner: str) -> int:
        stale = [
            task
            for task in self._tasks.values()
            if task["owner"] == owner
            and task["status"] in (TaskStatus.CLAIMED, TaskStatus.FAILED)
        ]
        for task in stale:
            task["status"] = TaskStatus.PENDING
            task["owner"] = ""
            task["result"] = MISSING
        return len(stale)

    async def pending_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t["status"] == TaskStatus.PENDING)

    async def all_done(self) -> bool:
        return not any(
            t["status"] in (TaskStatus.PENDING, TaskStatus.CLAIMED)
            for t in self._tasks.values()
        )


def _init_queue_db(conn: sqlite3.Connection) -> None:
    """Create the tasks table and configure WAL mode for crash tolerance."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id      TEXT PRIMARY KEY,
            type    TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            status  TEXT NOT NULL DEFAULT 'pending',
            owner   TEXT NOT NULL DEFAULT '',
            result  TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
    conn.commit()


class PersistentTaskQueue(TaskQueue):
    """SQLite-backed task queue with claim-based ownership.

    All task state lives in a single SQLite database in WAL journal mode: if
    the process is killed mid-transaction, SQLite's journal-based recovery
    keeps the database consistent. Call `release_stale_claims` on restart to
    reclaim work from a crashed session -- `Choreography` does this for every
    agent before it starts.

    Claiming runs inside a ``BEGIN IMMEDIATE`` transaction, so the read that
    finds a pending task and the write that takes ownership of it cannot be
    split by another claimer. That holds *across processes*, which is the
    point: a lock would only order claims within one process, and this queue
    exists to be shared by processes that restart independently.

    Every operation runs on a private single-thread executor. One thread keeps
    the event loop free of blocking file I/O without letting short queue
    transactions queue up behind long LLM calls in a shared pool, and
    serialises this process's own access for free.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="taskqueue"
        )
        with contextlib.closing(self._connect()) as conn:
            _init_queue_db(conn)

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: transactions are opened explicitly (see _claim)
        # rather than inferred by the sqlite3 module from statement kind.
        conn = sqlite3.connect(str(self._db_path), timeout=10, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    async def _run[T](self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run *fn* against a fresh connection on the queue's own thread."""

        def _with_conn() -> T:
            with contextlib.closing(self._connect()) as conn:
                return fn(conn)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, _with_conn)

    async def submit(
        self,
        task_type: str,
        payload: dict,
        task_id: str | None = None,
    ) -> str:
        tid = str(uuid.uuid4())[:8] if task_id is None else task_id
        payload_json = json.dumps(payload, default=str)

        def _submit(conn: sqlite3.Connection) -> str:
            conn.execute(
                """
                INSERT OR IGNORE INTO tasks (id, type, payload, status, owner, result)
                VALUES (?, ?, ?, ?, '', NULL)
                """,
                (tid, task_type, payload_json, TaskStatus.PENDING),
            )
            return tid

        return await self._run(_submit)

    @staticmethod
    def _claim(
        conn: sqlite3.Connection, where: str, param: str, owner: str
    ) -> dict | None:
        """Claim the first pending task matching *where*, atomically.

        ``BEGIN IMMEDIATE`` takes SQLite's write lock up front, so the SELECT
        and the UPDATE below are one indivisible step for every reader,
        in this process or any other.
        """
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT id, type, payload FROM tasks "  # noqa: S608 - `where` is a literal
                f"WHERE status = ? AND {where} ORDER BY id LIMIT 1",
                (TaskStatus.PENDING, param),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            conn.execute(
                "UPDATE tasks SET status = ?, owner = ? WHERE id = ?",
                (TaskStatus.CLAIMED, owner, row[0]),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return {
            "id": row[0],
            "type": row[1],
            "payload": json.loads(row[2]),
            "status": TaskStatus.CLAIMED,
            "owner": owner,
        }

    async def claim_id(self, task_id: str, owner: str) -> dict | None:
        return await self._run(
            functools.partial(self._claim, where="id = ?", param=task_id, owner=owner)
        )

    async def claim_prefix(self, prefix: str, owner: str) -> dict | None:
        return await self._run(
            functools.partial(
                self._claim, where="id LIKE ?", param=prefix + "%", owner=owner
            )
        )

    async def _finish(self, task_id: str, status: TaskStatus, result: Any) -> None:
        result_json = json.dumps(result, default=str)

        def _update(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE tasks SET status = ?, result = ? WHERE id = ? AND status = ?",
                (status, result_json, task_id, TaskStatus.CLAIMED),
            )

        await self._run(_update)

    async def complete(self, task_id: str, owner: str, result: Any = None) -> None:
        await self._finish(task_id, TaskStatus.DONE, result)

    async def fail(self, task_id: str, owner: str, error: str) -> None:
        await self._finish(task_id, TaskStatus.FAILED, {"error": error})

    async def get_result(self, task_id: str) -> Any:
        def _get(conn: sqlite3.Connection) -> Any:
            row = conn.execute(
                "SELECT result FROM tasks WHERE id = ? AND status = ?",
                (task_id, TaskStatus.DONE),
            ).fetchone()
            # A done task always has a result column, even if it encodes None.
            return MISSING if row is None else json.loads(row[0])

        return await self._run(_get)

    async def release_stale_claims(self, owner: str) -> int:
        def _release(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                """
                UPDATE tasks SET status = ?, owner = '', result = NULL
                WHERE status IN (?, ?) AND owner = ?
                """,
                (TaskStatus.PENDING, TaskStatus.CLAIMED, TaskStatus.FAILED, owner),
            )
            return cursor.rowcount

        return await self._run(_release)

    async def pending_count(self) -> int:
        def _count(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = ?", (TaskStatus.PENDING,)
            ).fetchone()
            return row[0] if row else 0

        return await self._run(_count)

    async def all_done(self) -> bool:
        def _count(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status IN (?, ?)",
                (TaskStatus.PENDING, TaskStatus.CLAIMED),
            ).fetchone()
            return row[0] == 0 if row else True

        return await self._run(_count)

    async def close(self) -> None:
        self._pool.shutdown(wait=False)


# ── Endpoint projection ───────────────────────────────────────────


class ChoreographyError(Exception):
    """Raised when a choreography fails because one of its agents failed."""


_PROJECTION: contextvars.ContextVar["EndpointProjection | None"] = (
    contextvars.ContextVar("effectful_choreography_projection", default=None)
)

_IN_SCATTER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "effectful_choreography_in_scatter", default=False
)


class EndpointProjection:
    """Projects a choreographic program onto a single agent.

    Not an `effectful.ops.syntax.ObjectInterpretation`: it drives ``await``
    points, and `effectful` handlers must be synchronous (see the module
    docstring). Activate it for the current task with `activate`, after which
    `step` and `scatter` route through it.

    Args:
        agent: The agent this projection speaks for.
        queue: The shared task queue.
        poll_interval: Seconds between polls while awaiting a peer's result.
        executor: Thread pool for blocking template calls. ``None`` uses
            asyncio's default executor, which is only safe when agents do not
            wait on each other -- `Choreography` always passes its own.
    """

    def __init__(
        self,
        agent: Agent,
        queue: TaskQueue,
        poll_interval: float = 0.05,
        executor: concurrent.futures.Executor | None = None,
    ) -> None:
        self._agent = agent
        self._agent_id = agent.__agent_id__
        self._queue = queue
        self._poll = poll_interval
        self._executor = executor
        self._step = 0

    @property
    def agent(self) -> Agent:
        """The agent this projection speaks for."""
        return self._agent

    @contextlib.contextmanager
    def activate(self):
        """Make this the active projection for the current task."""
        token = _PROJECTION.set(self)
        try:
            yield self
        finally:
            _PROJECTION.reset(token)

    def _next_step(self) -> str:
        step_id = f"step-{self._step:04d}"
        self._step += 1
        return step_id

    async def _in_thread[T](self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Await *fn* on a worker thread, carrying the current context along.

        The context copy is what puts the agent's `effectful` handler stack --
        provider, retries, persistence -- in scope inside the worker.
        """
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            self._executor, functools.partial(ctx.run, fn, *args, **kwargs)
        )

    async def _await_result(self, task_id: str) -> Any:
        """Poll the queue until *task_id* has a result."""
        while (result := await self._queue.get_result(task_id)) is MISSING:
            await asyncio.sleep(self._poll)
        return result

    async def _run_step(
        self, step_id: str, template: Callable, args: tuple, kwargs: dict
    ) -> Any:
        agent = getattr(template, "__agent__", None)

        if agent is None:
            # Unbound template: not owned by anyone, so every agent runs it.
            return await self._in_thread(template, *args, **kwargs)

        if agent.__agent_id__ != self._agent_id:
            return await self._await_result(step_id)

        cached = await self._queue.get_result(step_id)
        if cached is not MISSING:
            return cached

        name = getattr(template, "__name__", "step")
        await self._queue.submit(name, {"agent": self._agent_id}, task_id=step_id)
        if await self._queue.claim_id(step_id, self._agent_id) is None:
            # Another process is running this step; wait for its result.
            return await self._await_result(step_id)

        try:
            result = await self._in_thread(template, *args, **kwargs)
        except Exception as e:
            # Cancellation deliberately falls through unmarked: the step stays
            # claimed and `release_stale_claims` frees it on the next run.
            await self._queue.fail(step_id, self._agent_id, str(e))
            raise
        await self._queue.complete(step_id, self._agent_id, result)
        return result

    async def _scatter[A: Agent, T, U](
        self,
        step_id: str,
        items: Sequence[T],
        agent: A | Sequence[A],
        fn: Callable[[A, T], Awaitable[U]],
    ) -> list[U]:
        agents = [agent] if isinstance(agent, Agent) else list(agent)
        agent_ids = {a.__agent_id__ for a in agents}
        me = typing.cast(A, self._agent)
        task_ids = [f"{step_id}:{i:04d}" for i in range(len(items))]

        # Every agent submits every task, but the IDs are deterministic and
        # submit is idempotent, so each is created exactly once.
        for i, task_id in enumerate(task_ids):
            await self._queue.submit(
                f"scatter-{step_id}", {"item_index": i}, task_id=task_id
            )

        if self._agent_id in agent_ids:
            while (
                task := await self._queue.claim_prefix(f"{step_id}:", self._agent_id)
            ) is not None:
                item = items[task["payload"]["item_index"]]
                token = _IN_SCATTER.set(True)
                try:
                    result = await fn(me, item)
                except Exception as e:
                    await self._queue.fail(task["id"], self._agent_id, str(e))
                    raise
                finally:
                    _IN_SCATTER.reset(token)
                await self._queue.complete(task["id"], self._agent_id, result)

        return [await self._await_result(task_id) for task_id in task_ids]


# ── Choreography primitives ───────────────────────────────────────


def call[**P, T](fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Awaitable[T]:
    """Await a blocking call -- usually a `Template` -- on a worker thread.

    No queue bookkeeping: the call is not a replayable choreography step. Use
    it for work already covered by an enclosing step, most importantly for the
    per-item function passed to `scatter`. Outside a choreography it is
    `asyncio.to_thread`.
    """
    projection = _PROJECTION.get()
    if projection is None:
        return asyncio.to_thread(fn, *args, **kwargs)
    return projection._in_thread(fn, *args, **kwargs)


def step[**P, T](
    template: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Awaitable[T]:
    """Take one step of a choreography, and return an awaitable for its result.

    The step ID is allocated here, when `step` is called, rather than when the
    result is awaited, so that concurrent steps still get the same IDs in the
    same order on every agent.

    Under `EndpointProjection`, the agent that owns *template* claims the step
    and executes it while the others await its result; a step that already
    completed on an earlier run returns its recorded result without calling the
    model. A template bound to no agent is executed by every agent.

    Outside a choreography this is just `call`.
    """
    projection = _PROJECTION.get()
    if projection is None:
        return call(template, *args, **kwargs)
    if _IN_SCATTER.get():
        raise RuntimeError(
            "step() cannot be used inside scatter(): its per-item work is not "
            "a choreography step, and allocating step IDs there would put the "
            "agents out of sync. Use call() instead."
        )
    return projection._run_step(projection._next_step(), template, args, kwargs)


def scatter[A: Agent, T, U](
    items: Sequence[T],
    agent: A | Sequence[A],
    fn: Callable[[A, T], Awaitable[U]],
) -> Awaitable[list[U]]:
    """Distribute *items* over *agent* by calling ``await fn(agent, item)``.

    *agent* may be a single agent or a pool of same-role agents. Under
    `EndpointProjection` each item becomes a task and the pool claims tasks
    until none remain, which balances load by construction: a fast agent takes
    more items. On restart, items that completed are returned from the queue
    and only the rest re-run.

    Results come back in *items* order, whoever computed them.

    Outside a choreography, items are processed sequentially, round-robin over
    the pool.

    .. warning::

        *fn* should only touch the agent it is handed, with `call`. Steps
        (`step`) and other agents' templates inside *fn* are not supported.
    """
    projection = _PROJECTION.get()
    if projection is None:
        return _scatter_default(items, agent, fn)
    return projection._scatter(projection._next_step(), items, agent, fn)


async def _scatter_default[A: Agent, T, U](
    items: Sequence[T],
    agent: A | Sequence[A],
    fn: Callable[[A, T], Awaitable[U]],
) -> list[U]:
    agents = [agent] if isinstance(agent, Agent) else list(agent)
    return [await fn(agents[i % len(agents)], item) for i, item in enumerate(items)]


# ── Choreography runner ───────────────────────────────────────────


def _first_exception(group: BaseExceptionGroup) -> BaseException:
    """The first leaf exception of a possibly nested exception group."""
    exc = group.exceptions[0]
    return _first_exception(exc) if isinstance(exc, BaseExceptionGroup) else exc


class Choreography:
    """Run a choreographic program with endpoint projection.

    Every agent runs *program* as its own `asyncio.Task`; `EndpointProjection`
    is what makes each of those tasks behave differently. Blocking template
    calls go to a thread pool sized to the number of agents, so no agent can be
    starved by another's model call.

    The tasks run in an `asyncio.TaskGroup`, which supplies the parts the
    threaded version had to build by hand: the first failure cancels the other
    agents, and the failure propagates to the caller. Cancellation cannot
    interrupt an LLM call that is already in flight on a worker thread, so a
    failing run waits for those to return before it raises.

    Args:
        program: The choreographic ``async`` function. All agents run it.
        agents: The agents participating in the choreography.
        queue: The task queue. Defaults to `InMemoryTaskQueue`; pass a
            `PersistentTaskQueue` for crash tolerance.
        handlers: Handlers installed per agent beneath the projection (LLM
            provider, retries, persistence).
        poll_interval: Seconds between polls while awaiting a peer's result.

    Example::

        choreo = Choreography(
            build_codebase,
            agents=[architect, coder, reviewer],
            queue=PersistentTaskQueue(Path("./state/task_queue.db")),
            handlers=[
                LiteLLMProvider(model="gpt-4o-mini"),
                RetryLLMHandler(),
                SQLitePersister(Path("./state/checkpoints.db")),
            ],
        )
        result = choreo.run(
            project_spec="Build a library...",
            architect=architect,
            coder=coder,
            reviewer=reviewer,
        )
    """

    def __init__(
        self,
        program: Callable[..., Awaitable[Any]],
        agents: Sequence[Agent],
        queue: TaskQueue | None = None,
        handlers: Sequence[Interpretation | ObjectInterpretation] | None = None,
        poll_interval: float = 0.05,
    ) -> None:
        self.program = program
        self.agents = list(agents)
        self.handlers = list(handlers or [])
        self.poll_interval = poll_interval
        self._queue = queue if queue is not None else InMemoryTaskQueue()

    @property
    def queue(self) -> TaskQueue:
        """The underlying task queue, for inspection or manual operations."""
        return self._queue

    def projection(
        self,
        agent: Agent,
        executor: concurrent.futures.Executor | None = None,
    ) -> EndpointProjection:
        """The `EndpointProjection` for one agent.

        Useful for driving a single agent yourself::

            proj = choreo.projection(agent)
            with handler(provider), proj.activate():
                result = await choreo.program(**kwargs)
        """
        return EndpointProjection(
            agent, self._queue, poll_interval=self.poll_interval, executor=executor
        )

    async def _agent_main(
        self,
        agent: Agent,
        executor: concurrent.futures.Executor,
        kwargs: dict[str, Any],
    ) -> Any:
        projection = self.projection(agent, executor=executor)
        with contextlib.ExitStack() as stack:
            for h in self.handlers:
                stack.enter_context(handler(h))
            stack.enter_context(projection.activate())
            try:
                return await self.program(**kwargs)
            except (asyncio.CancelledError, ChoreographyError):
                raise
            except Exception as e:
                raise ChoreographyError(
                    f"Agent {agent.__agent_id__!r} failed: {e}"
                ) from e

    async def run_async(self, **kwargs: Any) -> Any:
        """Run the choreography to completion.

        Keyword arguments are forwarded to the choreographic function. All
        agents compute the same result; that result is returned.

        On restart after a crash, completed steps return their recorded
        results and unfinished ones are released and re-run.

        Raises:
            ChoreographyError: If any agent fails.
        """
        for agent in self.agents:
            await self._queue.release_stale_claims(agent.__agent_id__)

        tasks: list[asyncio.Task] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(self.agents)), thread_name_prefix="choreo"
        ) as executor:
            try:
                async with asyncio.TaskGroup() as group:
                    tasks = [
                        group.create_task(
                            self._agent_main(agent, executor, kwargs),
                            name=f"choreo-{agent.__agent_id__}",
                        )
                        for agent in self.agents
                    ]
            except BaseExceptionGroup as group_error:
                raise _first_exception(group_error)

        return tasks[0].result()

    def run(self, **kwargs: Any) -> Any:
        """Run the choreography from synchronous code.

        Equivalent to ``asyncio.run(choreo.run_async(**kwargs))``; call
        `run_async` directly from inside a running event loop.
        """
        return asyncio.run(self.run_async(**kwargs))
