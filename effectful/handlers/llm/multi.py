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
        state_dir=Path("./state"),
        handlers=[
            LiteLLMProvider(model="gpt-4o-mini"),
            RetryLLMHandler(),
            PersistenceHandler(Path("./state")),
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

    from effectful.handlers.llm.multi import Choreography, scatter

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
        state_dir=Path("./state"),
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

import contextlib
import json
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


class TaskQueue:
    """File-based task queue with claim-based ownership.

    Each task is a JSON file in *queue_dir*.  Claiming a task
    atomically renames it from ``<id>.pending.json`` to
    ``<id>.claimed.<owner>.json``, preventing double-claiming even
    across process restarts.

    The queue is fully crash-tolerant: call
    :meth:`release_stale_claims` on restart to reclaim work from a
    prior crashed session.
    """

    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _task_path(self, task_id: str, status: str, owner: str = "") -> Path:
        if owner:
            return self.queue_dir / f"{task_id}.{status}.{owner}.json"
        return self.queue_dir / f"{task_id}.{status}.json"

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
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]
        with self._lock:
            if list(self.queue_dir.glob(f"{task_id}.*")):
                return task_id
            task = {
                "id": task_id,
                "type": task_type,
                "payload": payload,
                "status": TaskStatus.PENDING,
                "owner": "",
                "result": None,
            }
            path = self._task_path(task_id, TaskStatus.PENDING)
            path.write_text(json.dumps(task, indent=2, default=str))
            return task_id

    def claim(self, task_type: str, owner: str) -> dict | None:
        """Atomically claim the next pending task of the given type.

        Returns the task dict if one was claimed, or ``None``.
        """
        with self._lock:
            for path in sorted(self.queue_dir.glob(f"*.{TaskStatus.PENDING}.json")):
                task = json.loads(path.read_text())
                if task["type"] != task_type:
                    continue
                task["status"] = TaskStatus.CLAIMED
                task["owner"] = owner
                claimed = self._task_path(task["id"], TaskStatus.CLAIMED, owner)
                try:
                    path.rename(claimed)
                except FileNotFoundError:
                    continue  # another thread claimed it
                claimed.write_text(json.dumps(task, indent=2, default=str))
                return task
        return None

    def claim_by_prefix(self, prefix: str, owner: str) -> dict | None:
        """Claim any pending task whose ID starts with *prefix*."""
        with self._lock:
            for path in sorted(self.queue_dir.glob(f"*.{TaskStatus.PENDING}.json")):
                fname = path.name.split(".")[0]
                if not fname.startswith(prefix):
                    continue
                task = json.loads(path.read_text())
                task["status"] = TaskStatus.CLAIMED
                task["owner"] = owner
                claimed = self._task_path(task["id"], TaskStatus.CLAIMED, owner)
                try:
                    path.rename(claimed)
                except FileNotFoundError:
                    continue
                claimed.write_text(json.dumps(task, indent=2, default=str))
                return task
        return None

    def complete(self, task_id: str, owner: str, result: Any = None) -> None:
        """Mark a claimed task as done with *result*."""
        claimed = self._task_path(task_id, TaskStatus.CLAIMED, owner)
        if not claimed.exists():
            return
        task = json.loads(claimed.read_text())
        task["status"] = TaskStatus.DONE
        task["result"] = result
        done = self._task_path(task_id, TaskStatus.DONE)
        tmp = done.with_suffix(".tmp")
        tmp.write_text(json.dumps(task, indent=2, default=str))
        tmp.replace(done)
        try:
            claimed.unlink()
        except FileNotFoundError:
            pass

    def fail(self, task_id: str, owner: str, error: str) -> None:
        """Mark a claimed task as failed."""
        claimed = self._task_path(task_id, TaskStatus.CLAIMED, owner)
        if not claimed.exists():
            return
        task = json.loads(claimed.read_text())
        task["status"] = TaskStatus.FAILED
        task["result"] = {"error": error}
        failed = self._task_path(task_id, TaskStatus.FAILED)
        claimed.rename(failed)
        failed.write_text(json.dumps(task, indent=2, default=str))

    def get_result(self, task_id: str) -> Any | None:
        """Return the result of a completed task, or ``None``."""
        done = self._task_path(task_id, TaskStatus.DONE)
        if done.exists():
            task = json.loads(done.read_text())
            return task.get("result")
        return None

    def release_stale_claims(self, owner: str) -> int:
        """Release tasks claimed by *owner* back to pending.

        Call on startup to reclaim work from a prior crashed session.
        """
        count = 0
        with self._lock:
            for path in self.queue_dir.glob(f"*.{TaskStatus.CLAIMED}.{owner}.json"):
                task = json.loads(path.read_text())
                task["status"] = TaskStatus.PENDING
                task["owner"] = ""
                pending = self._task_path(task["id"], TaskStatus.PENDING)
                path.rename(pending)
                pending.write_text(json.dumps(task, indent=2, default=str))
                count += 1
        return count

    def pending_count(self, task_type: str | None = None) -> int:
        """Count pending tasks, optionally filtered by type."""
        count = 0
        for path in self.queue_dir.glob(f"*.{TaskStatus.PENDING}.json"):
            if task_type is None:
                count += 1
            else:
                task = json.loads(path.read_text())
                if task["type"] == task_type:
                    count += 1
        return count

    def all_done(self) -> bool:
        """``True`` if no pending or claimed tasks remain."""
        for status in (TaskStatus.PENDING, TaskStatus.CLAIMED):
            if list(self.queue_dir.glob(f"*.{status}*")):
                return False
        return True


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

        # Submit one task per item
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


# ── Choreography runner ───────────────────────────────────────────


class Choreography:
    """Run a choreographic program with crash-tolerant endpoint projection.

    Each agent gets its own thread.  Template calls are routed via
    the :class:`TaskQueue`: the owning agent claims and executes,
    others poll for results.  On restart, completed steps are
    returned from cache.

    Args:
        program: The choreographic function.  All agent threads run
            this same function; EPP makes each thread behave
            differently.
        agents: The agents participating in the choreography.
        state_dir: Directory for the persistent task queue.
        handlers: Handler instances to install per-thread beneath
            the EPP handler (e.g. LLM provider, retry handler,
            persistence handler).
        poll_interval: Seconds between polling for task results
            (default 0.1).

    Example::

        choreo = Choreography(
            build_codebase,
            agents=[architect, coder, reviewer],
            state_dir=Path("./state"),
            handlers=[
                LiteLLMProvider(model="gpt-4o-mini"),
                RetryLLMHandler(),
                PersistenceHandler(Path("./state")),
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
        state_dir: Path,
        handlers: Sequence[Interpretation | ObjectInterpretation] | None = None,
        poll_interval: float = 0.1,
    ) -> None:
        self.program = program
        self.agents = list(agents)
        self.state_dir = Path(state_dir)
        self.handlers = list(handlers or [])
        self.poll_interval = poll_interval
        self._queue = TaskQueue(self.state_dir / "choreo_queue")

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
