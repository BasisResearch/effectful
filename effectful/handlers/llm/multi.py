"""Choreographic programming for multi-agent LLM systems.

Write a single function describing how agents interact from a global
perspective, then run it with automatic endpoint projection (EPP).
Each agent gets its own thread, inter-agent communication is
handled automatically via a persistent :class:`TaskQueue`, and the
entire process is crash-tolerant and restartable.

**How it works.** The choreographic program is a plain Python function
whose arguments are agent instances. All agent threads run this same
function. The :class:`EndpointProjection` handler intercepts
:attr:`~effectful.handlers.llm.template.Template.__apply__`:

- When it is the current agent's template: claim a task in the
  queue, execute via ``fwd``, and store the result.
- When it is another agent's template: poll the queue until
  the result appears.

Each statement in the choreography is assigned an incrementing step ID.
Completed steps are persisted to disk. On restart, the program re-runs
from the start; completed steps return their cached results instantly,
and execution resumes from the first incomplete step.

Example — sequential choreography with a review loop::

    from pathlib import Path
    from typing import Literal, TypedDict

    from effectful.handlers.llm import Template
    from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
    from effectful.handlers.llm.multi import Choreography
    from effectful.handlers.llm.persistence import PersistenceHandler, PersistentAgent
    from effectful.ops.types import NotHandled

    class ModuleSpec(TypedDict):
        module_path: str
        description: str

    class PlanResult(TypedDict):
        modules: list[ModuleSpec]

    class ReviewResult(TypedDict):
        verdict: Literal["PASS", "NEEDS_FIXES"]
        feedback: str

    class Architect(PersistentAgent):
        \"\"\"You are a software architect.\"\"\"

        @Template.define
        def plan_modules(self, project_spec: str) -> PlanResult:
            \"\"\"Break this project into modules: {project_spec}\"\"\"
            raise NotHandled

    class Coder(PersistentAgent):
        \"\"\"You are a Python developer.\"\"\"

        @Template.define
        def implement_module(self, spec: str) -> str:
            \"\"\"Implement the module: {spec}\"\"\"
            raise NotHandled

    class Reviewer(PersistentAgent):
        \"\"\"You are a code reviewer.\"\"\"

        @Template.define
        def review_code(self, code: str) -> ReviewResult:
            \"\"\"Review this code: {code}\"\"\"
            raise NotHandled

    def build_codebase(
        project_spec: str,
        architect: Architect,
        coder: Coder,
        reviewer: Reviewer,
    ) -> str:
        plan = architect.plan_modules(project_spec)
        code = coder.implement_module(str(plan))
        while True:
            result = reviewer.review_code(code)
            if result["verdict"] == "PASS":
                return code
            code = coder.implement_module(result["feedback"])

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
            PersistenceHandler(Path("./state/checkpoints.db")),
        ],
    )
    # Kill at any point, restart, and it resumes where it left off.
    result = choreo.run(
        project_spec="Build a URL slugify library",
        architect=architect,
        coder=coder,
        reviewer=reviewer,
    )

Example — parallel scatter across multiple coders::

    from effectful.handlers.llm.multi import Choreography, PersistentTaskQueue, scatter

    def build_parallel(
        project_spec: str,
        architect: Architect,
        coder: Coder,
        reviewer: Reviewer,
    ) -> list[ReviewResult]:
        plan = architect.plan_modules(project_spec)
        # Each module becomes a task; coders claim from the queue
        # until none remain — natural load balancing.
        codes = scatter(
            plan["modules"], coder,
            lambda coder, mod: coder.implement_module(str(mod)),
        )
        return [reviewer.review_code(code) for code in codes]

    coder1 = Coder(agent_id="coder-1")
    coder2 = Coder(agent_id="coder-2")
    coder3 = Coder(agent_id="coder-3")

    choreo = Choreography(
        build_parallel,
        agents=[architect, coder1, coder2, coder3, reviewer],
        queue=PersistentTaskQueue(Path("./state/task_queue.db")),
        handlers=[LiteLLMProvider(model="gpt-4o-mini"), RetryLLMHandler()],
    )
    # Pass coder as a list — scatter distributes across all three
    reviews = choreo.run(
        project_spec="Build textkit with slugify, wrap, and redact modules",
        architect=architect,
        coder=[coder1, coder2, coder3],
        reviewer=reviewer,
    )

"""

import abc
import contextlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from effectful.handlers.llm.template import Agent, Template, get_bound_agent
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Interpretation, Operation

# ── TaskQueue ──────────────────────────────────────────────────────


class TaskStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"


class TaskQueue(abc.ABC):
    """Abstract task queue with claim-based ownership.

    Subclasses implement persistent (file-based) or in-memory storage.
    All methods are thread-safe.
    """

    @abc.abstractmethod
    def submit(
        self,
        task_type: str,
        payload: dict,
        task_id: str | None = None,
    ) -> str:
        """Add a new task.  Returns the task ID.

        Idempotent when *task_id* is specified: if a task with that ID
        already exists (in any state), the call is a no-op.
        """

    @abc.abstractmethod
    def claim(self, task_type: str, owner: str) -> dict | None:
        """Atomically claim the next pending task of the given type.

        Returns the task dict if one was claimed, or ``None``.
        """

    @abc.abstractmethod
    def claim_by_prefix(self, prefix: str, owner: str) -> dict | None:
        """Claim any pending task whose ID starts with *prefix*."""

    @abc.abstractmethod
    def complete(self, task_id: str, owner: str, result: Any = None) -> None:
        """Mark a claimed task as done with *result*."""

    @abc.abstractmethod
    def fail(self, task_id: str, owner: str, error: str) -> None:
        """Mark a claimed task as failed."""

    @abc.abstractmethod
    def get_result(self, task_id: str) -> Any | None:
        """Return the result of a completed task, or ``None``."""

    @abc.abstractmethod
    def release_stale_claims(self, owner: str) -> int:
        """Release tasks claimed by *owner* back to pending.

        Call on startup to reclaim work from a prior crashed session.
        """

    @abc.abstractmethod
    def pending_count(self, task_type: str | None = None) -> int:
        """Count pending tasks, optionally filtered by type."""

    @abc.abstractmethod
    def all_done(self) -> bool:
        """``True`` if no pending or claimed tasks remain."""


class InMemoryTaskQueue(TaskQueue):
    """In-memory task queue for testing or ephemeral workflows.

    Not crash-tolerant — all state is lost when the process exits.
    Thread-safe via a single lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, dict] = {}  # task_id -> task dict

    def submit(
        self,
        task_type: str,
        payload: dict,
        task_id: str | None = None,
    ) -> str:
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]
        with self._lock:
            if task_id in self._tasks:
                return task_id
            self._tasks[task_id] = {
                "id": task_id,
                "type": task_type,
                "payload": payload,
                "status": TaskStatus.PENDING,
                "owner": "",
                "result": None,
            }
            return task_id

    def claim(self, task_type: str, owner: str) -> dict | None:
        with self._lock:
            for task_id in sorted(self._tasks):
                task = self._tasks[task_id]
                if task["status"] == TaskStatus.PENDING and task["type"] == task_type:
                    task["status"] = TaskStatus.CLAIMED
                    task["owner"] = owner
                    return dict(task)
            return None

    def claim_by_prefix(self, prefix: str, owner: str) -> dict | None:
        with self._lock:
            for task_id in sorted(self._tasks):
                task = self._tasks[task_id]
                if task["status"] == TaskStatus.PENDING and task_id.startswith(prefix):
                    task["status"] = TaskStatus.CLAIMED
                    task["owner"] = owner
                    return dict(task)
            return None

    def complete(self, task_id: str, owner: str, result: Any = None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task["status"] != TaskStatus.CLAIMED:
                return
            task["status"] = TaskStatus.DONE
            task["result"] = result

    def fail(self, task_id: str, owner: str, error: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task["status"] != TaskStatus.CLAIMED:
                return
            task["status"] = TaskStatus.FAILED
            task["result"] = {"error": error}

    def get_result(self, task_id: str) -> Any | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None and task["status"] == TaskStatus.DONE:
                return task["result"]
            return None

    def release_stale_claims(self, owner: str) -> int:
        count = 0
        with self._lock:
            for task in self._tasks.values():
                if task["status"] == TaskStatus.CLAIMED and task["owner"] == owner:
                    task["status"] = TaskStatus.PENDING
                    task["owner"] = ""
                    count += 1
        return count

    def pending_count(self, task_type: str | None = None) -> int:
        with self._lock:
            return sum(
                1
                for t in self._tasks.values()
                if t["status"] == TaskStatus.PENDING
                and (task_type is None or t["type"] == task_type)
            )

    def all_done(self) -> bool:
        with self._lock:
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_status_type ON tasks(status, type)"
    )
    conn.commit()


class PersistentTaskQueue(TaskQueue):
    """SQLite-backed task queue with claim-based ownership.

    All task state is stored in a single SQLite database using WAL
    journal mode for crash tolerance.  If the process is killed
    mid-transaction, SQLite's journal-based recovery ensures the
    database remains consistent.

    Claiming a task atomically updates its status from ``pending`` to
    ``claimed`` inside a transaction, preventing double-claiming even
    across process restarts.

    The queue is fully crash-tolerant: call
    :meth:`release_stale_claims` on restart to reclaim work from a
    prior crashed session.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._db_initialized = False
        self._init_lock = threading.Lock()

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA busy_timeout=5000")
        if not self._db_initialized:
            with self._init_lock:
                if not self._db_initialized:
                    _init_queue_db(conn)
                    self._db_initialized = True
        return conn

    def submit(
        self,
        task_type: str,
        payload: dict,
        task_id: str | None = None,
    ) -> str:
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]
        payload_json = json.dumps(payload, default=str)
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO tasks (id, type, payload, status, owner, result)
                VALUES (?, ?, ?, ?, '', NULL)
                """,
                (task_id, task_type, payload_json, TaskStatus.PENDING),
            )
            conn.commit()
        finally:
            conn.close()
        return task_id

    def claim(self, task_type: str, owner: str) -> dict | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, type, payload, status, owner, result
                    FROM tasks
                    WHERE status = ? AND type = ?
                    ORDER BY id LIMIT 1
                    """,
                    (TaskStatus.PENDING, task_type),
                ).fetchone()
                if row is None:
                    return None
                task_id = row[0]
                conn.execute(
                    "UPDATE tasks SET status = ?, owner = ? WHERE id = ?",
                    (TaskStatus.CLAIMED, owner, task_id),
                )
                conn.commit()
                return {
                    "id": task_id,
                    "type": row[1],
                    "payload": json.loads(row[2]),
                    "status": TaskStatus.CLAIMED,
                    "owner": owner,
                    "result": json.loads(row[5]) if row[5] is not None else None,
                }
            finally:
                conn.close()

    def claim_by_prefix(self, prefix: str, owner: str) -> dict | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, type, payload, status, owner, result
                    FROM tasks
                    WHERE status = ? AND id LIKE ?
                    ORDER BY id LIMIT 1
                    """,
                    (TaskStatus.PENDING, prefix + "%"),
                ).fetchone()
                if row is None:
                    return None
                task_id = row[0]
                conn.execute(
                    "UPDATE tasks SET status = ?, owner = ? WHERE id = ?",
                    (TaskStatus.CLAIMED, owner, task_id),
                )
                conn.commit()
                return {
                    "id": task_id,
                    "type": row[1],
                    "payload": json.loads(row[2]),
                    "status": TaskStatus.CLAIMED,
                    "owner": owner,
                    "result": json.loads(row[5]) if row[5] is not None else None,
                }
            finally:
                conn.close()

    def complete(self, task_id: str, owner: str, result: Any = None) -> None:
        result_json = json.dumps(result, default=str)
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE tasks SET status = ?, result = ?
                WHERE id = ? AND status = ?
                """,
                (TaskStatus.DONE, result_json, task_id, TaskStatus.CLAIMED),
            )
            conn.commit()
        finally:
            conn.close()

    def fail(self, task_id: str, owner: str, error: str) -> None:
        error_json = json.dumps({"error": error}, default=str)
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE tasks SET status = ?, result = ?
                WHERE id = ? AND status = ?
                """,
                (TaskStatus.FAILED, error_json, task_id, TaskStatus.CLAIMED),
            )
            conn.commit()
        finally:
            conn.close()

    def get_result(self, task_id: str) -> Any | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT result FROM tasks WHERE id = ? AND status = ?",
                (task_id, TaskStatus.DONE),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0]) if row[0] is not None else None
        finally:
            conn.close()

    def release_stale_claims(self, owner: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    UPDATE tasks SET status = ?, owner = ''
                    WHERE status = ? AND owner = ?
                    """,
                    (TaskStatus.PENDING, TaskStatus.CLAIMED, owner),
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

    def pending_count(self, task_type: str | None = None) -> int:
        conn = self._connect()
        try:
            if task_type is None:
                row = conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE status = ?",
                    (TaskStatus.PENDING,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE status = ? AND type = ?",
                    (TaskStatus.PENDING, task_type),
                ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def all_done(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status IN (?, ?)",
                (TaskStatus.PENDING, TaskStatus.CLAIMED),
            ).fetchone()
            return row[0] == 0 if row else True
        finally:
            conn.close()


# ── scatter ────────────────────────────────────────────────────────


@Operation.define
def scatter(items: list, agent: Agent, fn: Callable) -> list:
    """Distribute *items* by calling ``fn(agent, item)`` for each item.

    **Default** (no EPP handler): sequential
    ``[fn(agent, item) for item in items]``.

    **Under** :class:`EndpointProjection`: each item becomes a task in
    the queue.  When a list of agents is passed for the same role
    (e.g. ``coder=[coder1, coder2]``), agents claim tasks until none
    remain — providing natural load balancing with crash recovery.
    On restart, completed items are returned from cache; only
    remaining items are re-executed.

    .. warning::

        ``fn`` should only call templates on the assigned agent.
        Cross-agent template calls inside scatter are not supported.
    """
    return [fn(agent, item) for item in items]


@Operation.define
def fan_out(groups: list[tuple[list, Agent, Callable]]) -> list[list]:
    """Run multiple scatter-like operations concurrently.

    Each element of *groups* is a ``(items, agent, fn)`` triple — the
    same arguments you would pass to :func:`scatter`.  Returns a list
    of result lists, one per group, in the same order as *groups*.

    **Default** (no EPP handler): sequential execution of each group::

        [
            [fn(agent, item) for item in items]
            for items, agent, fn in groups
        ]

    **Under** :class:`EndpointProjection`: all groups' items are
    submitted as tasks under a single step ID.  Agents from *every*
    group claim and execute work concurrently, so a spec-writer,
    tester, and prover can all be working at the same time rather
    than waiting for the previous scatter to finish.

    Example::

        spec_results, test_results, proof_results = fan_out([
            (spec_tasks, spec_writer,
             lambda w, b: w.write_spec(json.dumps(b, indent=2))),
            (test_tasks, tester,
             lambda t, b: t.write_tests_and_validate(json.dumps(b, indent=2))),
            (proof_tasks, prover,
             lambda p, b: p.prove_theorem(json.dumps(b, indent=2))),
        ])

    .. warning::

        ``fn`` should only call templates on the assigned agent.
        Cross-agent template calls inside fan_out are not supported.
    """
    return [[fn(agent, item) for item in items] for items, agent, fn in groups]


# ── Endpoint Projection ───────────────────────────────────────────


class ChoreographyError(Exception):
    """Raised when a choreography fails due to an agent error."""


class CancelledError(Exception):
    """Raised inside an agent thread when the choreography is cancelled."""


class EndpointProjection(ObjectInterpretation):
    """Handler that projects a choreographic program onto a single agent.

    Each template call in the choreography is assigned a step ID
    (incrementing counter).  Steps become tasks in the
    :class:`TaskQueue`.

    - **Own agent's templates**: check if the step is already done
      (cached); if not, claim the task, execute, and store the result.
    - **Other agent's templates**: poll the queue until the result
      appears.
    - **Unbound templates**: execute directly on all threads.

    Also handles :func:`scatter` for data-parallel distribution via
    claim-based pull.
    """

    def __init__(
        self,
        agent: Agent,
        queue: TaskQueue,
        agent_ids: frozenset[str],
        poll_interval: float = 0.1,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._agent = agent
        self._agent_id = agent.__agent_id__
        self._queue = queue
        self._agent_ids = agent_ids
        self._poll = poll_interval
        self._step = 0
        self._in_scatter = False
        self._cancel = cancel_event

    def _next_step(self) -> str:
        step_id = f"step-{self._step:04d}"
        self._step += 1
        return step_id

    def _check_cancelled(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise CancelledError("Choreography cancelled")

    def _wait_result(self, step_id: str) -> Any:
        """Poll queue until task result is available."""
        while True:
            self._check_cancelled()
            r = self._queue.get_result(step_id)
            if r is not None:
                return r
            time.sleep(self._poll)

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound = get_bound_agent(template)

        # Inside scatter: execute directly, no task management
        if self._in_scatter:
            if bound and bound.__agent_id__ == self._agent_id:
                return fwd(template, *args, **kwargs)
            raise RuntimeError(
                f"Cross-agent call in scatter: {self._agent_id} -> "
                f"{bound.__agent_id__ if bound else '?'}"
            )

        step_id = self._next_step()

        if bound is not None and bound.__agent_id__ == self._agent_id:
            # My template — check done cache, or claim and execute
            cached = self._queue.get_result(step_id)
            if cached is not None:
                return cached

            self._queue.submit(
                task_type=template.__name__,
                payload={"agent": self._agent_id},
                task_id=step_id,
            )
            task = self._queue.claim(template.__name__, self._agent_id)
            if task is None:
                # Already claimed (e.g. restarted while another thread
                # is executing) — poll for result
                return self._wait_result(step_id)

            try:
                result = fwd(template, *args, **kwargs)
                self._queue.complete(step_id, self._agent_id, result)
                return result
            except Exception as e:
                self._queue.fail(step_id, self._agent_id, str(e))
                raise

        elif bound is not None:
            # Another agent's template — poll for result
            return self._wait_result(step_id)

        else:
            # Unbound template — execute directly
            return fwd(template, *args, **kwargs)

    @implements(scatter)
    def _scatter(self, items: list, agent: Agent, fn: Callable) -> list:
        step_id = self._next_step()

        # agent may be a single Agent or a list of Agents (passed
        # transparently from choreo.run kwargs).  Normalize to a list.
        agents = agent if isinstance(agent, list) else [agent]
        scatter_ids = {a.__agent_id__ for a in agents}

        # Submit one task per item.  All agent threads execute this
        # loop, but submit() is idempotent on task_id — the
        # deterministic ID (step_id:index) ensures each task is
        # created exactly once regardless of how many threads call it.
        for i in range(len(items)):
            self._queue.submit(
                task_type=f"scatter-{step_id}",
                payload={"item_index": i},
                task_id=f"{step_id}:{i:04d}",
            )

        # If I'm a scatter agent, claim and execute until none left
        if self._agent_id in scatter_ids:
            while True:
                task = self._queue.claim_by_prefix(f"{step_id}:", self._agent_id)
                if task is None:
                    break
                idx = task["payload"]["item_index"]
                self._in_scatter = True
                try:
                    result = fn(self._agent, items[idx])
                    self._queue.complete(task["id"], self._agent_id, result)
                except Exception as e:
                    self._queue.fail(task["id"], self._agent_id, str(e))
                    raise
                finally:
                    self._in_scatter = False

        # Gather all results (blocking until done)
        return [self._wait_result(f"{step_id}:{i:04d}") for i in range(len(items))]

    @implements(fan_out)
    def _fan_out(self, groups: list[tuple[list, Agent, Callable]]) -> list[list]:
        step_id = self._next_step()

        # For each group, normalize agents and build a mapping from
        # agent_id → list of (group_index, items, fn) so each agent
        # knows which groups it participates in.
        group_agents: list[set[str]] = []
        group_fns: list[Callable] = []
        group_items: list[list] = []

        for g, (items, agent, fn) in enumerate(groups):
            agents = agent if isinstance(agent, list) else [agent]
            group_agents.append({a.__agent_id__ for a in agents})
            group_fns.append(fn)
            group_items.append(items)

        # Submit all tasks across all groups.  Deterministic IDs:
        # {step_id}:g{group}:{item_index}
        for g in range(len(groups)):
            for i in range(len(group_items[g])):
                self._queue.submit(
                    task_type=f"fan-{step_id}:g{g}",
                    payload={"group": g, "item_index": i},
                    task_id=f"{step_id}:g{g}:{i:04d}",
                )

        # Claim and execute tasks from my groups
        my_groups = [g for g in range(len(groups)) if self._agent_id in group_agents[g]]
        for g in my_groups:
            prefix = f"{step_id}:g{g}:"
            while True:
                task = self._queue.claim_by_prefix(prefix, self._agent_id)
                if task is None:
                    break
                idx = task["payload"]["item_index"]
                self._in_scatter = True
                try:
                    result = group_fns[g](self._agent, group_items[g][idx])
                    self._queue.complete(task["id"], self._agent_id, result)
                except Exception as e:
                    self._queue.fail(task["id"], self._agent_id, str(e))
                    raise
                finally:
                    self._in_scatter = False

        # Gather results per group (blocking)
        return [
            [
                self._wait_result(f"{step_id}:g{g}:{i:04d}")
                for i in range(len(group_items[g]))
            ]
            for g in range(len(groups))
        ]


# ── Choreography runner ───────────────────────────────────────────


class Choreography:
    """Run a choreographic program with endpoint projection.

    Each agent gets its own thread.  Template calls are routed via
    the :class:`TaskQueue`: the owning agent claims and executes,
    others poll for results.  On restart, completed steps are
    returned from cache.

    Args:
        program: The choreographic function.  All agent threads run
            this same function; EPP makes each thread behave
            differently.
        agents: The agents participating in the choreography.
        queue: The task queue to use.  Defaults to
            :class:`InMemoryTaskQueue` if not provided.  Pass a
            :class:`PersistentTaskQueue` for crash tolerance.
        handlers: Handler instances to install per-thread beneath
            the EPP handler (e.g. LLM provider, retry handler,
            persistence handler).
        poll_interval: Seconds between polling for task results
            (default 0.1).

    Example::

        choreo = Choreography(
            build_codebase,
            agents=[architect, coder, reviewer],
            queue=PersistentTaskQueue(Path("./state/task_queue.db")),
            handlers=[
                LiteLLMProvider(model="gpt-4o-mini"),
                RetryLLMHandler(),
                PersistenceHandler(Path("./state/checkpoints.db")),
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
        program: Callable[..., Any],
        agents: Sequence[Agent],
        queue: TaskQueue | None = None,
        handlers: Sequence[Interpretation | ObjectInterpretation] | None = None,
        poll_interval: float = 0.1,
    ) -> None:
        self.program = program
        self.agents = list(agents)
        self.handlers = list(handlers or [])
        self.poll_interval = poll_interval
        self._queue = queue if queue is not None else InMemoryTaskQueue()

    @property
    def queue(self) -> TaskQueue:
        """The underlying task queue (for inspection or manual ops)."""
        return self._queue

    def project(
        self,
        agent: Agent,
        cancel_event: threading.Event | None = None,
    ) -> EndpointProjection:
        """Return the EPP handler for a specific agent.

        Useful for manual thread management::

            proj = choreo.project(agent)
            with handler(provider), handler(proj):
                result = choreo.program(**kwargs)
        """
        return EndpointProjection(
            agent,
            self._queue,
            frozenset(a.__agent_id__ for a in self.agents),
            self.poll_interval,
            cancel_event=cancel_event,
        )

    def run(self, **kwargs: Any) -> Any:
        """Run the choreography to completion.

        Keyword arguments are forwarded to the choreographic function.
        Returns the result (identical across all agent threads).

        On restart after a crash, completed steps return cached
        results; stale claims are released and re-executed.

        Raises:
            ChoreographyError: If any agent thread fails.
        """
        # Release stale claims from prior crashed run
        for agent in self.agents:
            self._queue.release_stale_claims(agent.__agent_id__)

        cancel = threading.Event()
        results: dict[str, Any] = {}
        errors: list[tuple[str, BaseException]] = []
        lock = threading.Lock()

        def agent_main(agent: Agent) -> None:
            try:
                proj = self.project(agent, cancel_event=cancel)
                result = self._run_with_handlers(proj, **kwargs)
                with lock:
                    results[agent.__agent_id__] = result
            except CancelledError:
                pass  # another agent failed; this thread was cancelled
            except BaseException as e:
                cancel.set()  # signal other threads to stop
                with lock:
                    errors.append((agent.__agent_id__, e))

        threads = []
        for agent in self.agents:
            t = threading.Thread(
                target=agent_main,
                args=(agent,),
                name=f"choreo-{agent.__agent_id__}",
                daemon=True,
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        if errors:
            agent_id, exc = errors[0]
            raise ChoreographyError(f"Agent '{agent_id}' failed: {exc}") from exc

        # All agents compute the same result; return any.
        return next(iter(results.values()))

    def _run_with_handlers(self, proj: EndpointProjection, **kwargs: Any) -> Any:
        """Install handlers and EPP, then run the program."""
        with contextlib.ExitStack() as stack:
            for h in self.handlers:
                stack.enter_context(handler(h))
            # EPP outermost — intercepts before providers
            stack.enter_context(handler(proj))
            return self.program(**kwargs)
