"""Multi-agent system with crash-tolerant task queue.

Demonstrates:
- File-based task queue with claim-based ownership and pull-based agents
- Concurrent agent workers using threads
- Crash tolerance: Ctrl-C and restart, agents resume where they left off
- PersistentAgent for automatic checkpointing and context compaction

The scenario: a team of agents collaboratively builds a small Python library.
An architect agent breaks the project into module specs, a coder agent
writes each module, and a reviewer agent reviews and requests fixes.
All output is written to a workspace directory on disk.

Usage::

    # First run — agents start working
    python docs/source/multi_agent_example.py

    # Ctrl-C mid-run, then restart — agents pick up where they left off
    python docs/source/multi_agent_example.py

Requirements:
    pip install effectful[llm]
    export OPENAI_API_KEY=...   # or any LiteLLM-supported provider

"""

import json
import logging
import signal
import threading
import time
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.handlers.llm.persistence import PersistentAgent
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE = Path("./multi_agent_workspace")
STATE_DIR = WORKSPACE / ".state"
OUTPUT_DIR = WORKSPACE / "output"
MODEL = "gpt-4o-mini"

# The project to build
PROJECT_SPEC = """\
Build a small Python utility library called 'textkit' with these modules:
1. textkit/slugify.py — convert strings to URL-safe slugs
2. textkit/wrap.py — word-wrap text to a given width
3. textkit/redact.py — redact email addresses and phone numbers from text
Each module should have a clear public API, docstrings, and at least 3
test cases written as a separate test_<module>.py file.
"""


# ---------------------------------------------------------------------------
# Structured types
# ---------------------------------------------------------------------------


# Task payloads (typed for queue submit/claim consistency)
class PlanTaskPayload(TypedDict):
    project_spec: str


class CodeTaskPayload(TypedDict, total=False):
    """Module spec from architect, or fix request from reviewer."""

    module_path: str
    description: str
    public_api: str
    test_path: str


class ReviewTaskPayload(TypedDict, total=False):
    module_path: str
    test_path: str


# Template return types (constrained decoding for LLM output)
class ModuleSpec(TypedDict):
    """Schema for architect planning output — constrained decoding ensures valid shape."""

    module_path: str
    description: str
    public_api: str
    test_path: str


class PlanResult(TypedDict):
    """Wrapper for list output — LiteLLM requires a root object, not bare array."""

    modules: list[ModuleSpec]


class ReviewResult(TypedDict):
    """Schema for reviewer output — verdict constrained to PASS or NEEDS_FIXES."""

    verdict: Literal["PASS", "NEEDS_FIXES"]
    feedback: str


# ---------------------------------------------------------------------------
# Task Queue — file-based, crash-tolerant, claim-based ownership
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"


class TaskQueue:
    """File-based task queue with claim-based ownership.

    Each task is a JSON file in ``queue_dir``. Claiming a task atomically
    renames it from ``<id>.pending.json`` to ``<id>.claimed.<owner>.json``.
    This prevents double-claiming even across process restarts.
    """

    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _task_path(self, task_id: str, status: str, owner: str = "") -> Path:
        if owner:
            return self.queue_dir / f"{task_id}.{status}.{owner}.json"
        return self.queue_dir / f"{task_id}.{status}.json"

    def submit(self, task_type: str, payload: dict) -> str:
        """Add a new task to the queue. Returns the task ID."""
        task_id = str(uuid.uuid4())[:8]
        task = {
            "id": task_id,
            "type": task_type,
            "payload": payload,
            "status": TaskStatus.PENDING,
            "owner": "",
            "result": None,
        }
        path = self._task_path(task_id, TaskStatus.PENDING)
        path.write_text(json.dumps(task, indent=2))
        log.info("Submitted task %s: %s", task_id, task_type)
        return task_id

    def claim(self, task_type: str, owner: str) -> dict | None:
        """Atomically claim the next pending task of the given type.

        Returns the task dict if one was claimed, or None.
        """
        with self._lock:
            for path in sorted(self.queue_dir.glob(f"*.{TaskStatus.PENDING}.json")):
                task = json.loads(path.read_text())
                if task["type"] != task_type:
                    continue
                # Atomic claim via rename
                task["status"] = TaskStatus.CLAIMED
                task["owner"] = owner
                claimed_path = self._task_path(task["id"], TaskStatus.CLAIMED, owner)
                try:
                    path.rename(claimed_path)
                except FileNotFoundError:
                    continue  # Another thread claimed it
                claimed_path.write_text(json.dumps(task, indent=2))
                log.info("Agent %s claimed task %s", owner, task["id"])
                return task
        return None

    def complete(self, task_id: str, owner: str, result: dict | None = None) -> None:
        """Mark a claimed task as done."""
        claimed_path = self._task_path(task_id, TaskStatus.CLAIMED, owner)
        if not claimed_path.exists():
            return
        task = json.loads(claimed_path.read_text())
        task["status"] = TaskStatus.DONE
        task["result"] = result
        done_path = self._task_path(task_id, TaskStatus.DONE)
        claimed_path.rename(done_path)
        done_path.write_text(json.dumps(task, indent=2))
        log.info("Task %s completed by %s", task_id, owner)

    def fail(self, task_id: str, owner: str, error: str) -> None:
        """Mark a claimed task as failed."""
        claimed_path = self._task_path(task_id, TaskStatus.CLAIMED, owner)
        if not claimed_path.exists():
            return
        task = json.loads(claimed_path.read_text())
        task["status"] = TaskStatus.FAILED
        task["result"] = {"error": error}
        failed_path = self._task_path(task_id, TaskStatus.FAILED)
        claimed_path.rename(failed_path)
        failed_path.write_text(json.dumps(task, indent=2))
        log.info("Task %s failed: %s", task_id, error[:100])

    def release_stale_claims(self, owner: str) -> int:
        """Release tasks claimed by a specific owner back to pending.

        Used on startup to reclaim work from a prior crashed session.
        """
        count = 0
        with self._lock:
            for path in self.queue_dir.glob(f"*.{TaskStatus.CLAIMED}.{owner}.json"):
                task = json.loads(path.read_text())
                task["status"] = TaskStatus.PENDING
                task["owner"] = ""
                pending_path = self._task_path(task["id"], TaskStatus.PENDING)
                path.rename(pending_path)
                pending_path.write_text(json.dumps(task, indent=2))
                count += 1
        if count:
            log.info("Released %d stale claims from %s", count, owner)
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
        """True if no pending or claimed tasks remain."""
        for status in (TaskStatus.PENDING, TaskStatus.CLAIMED):
            if list(self.queue_dir.glob(f"*.{status}*")):
                return False
        return True


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class ArchitectAgent(PersistentAgent):
    """You are a software architect. Given a project specification, you break
    it into individual module implementation tasks. Each task should specify
    the module filename, its public API, and what tests to write.
    Be concrete and specific — the coder will follow your spec exactly.
    """

    def __init__(self, persist_dir: Path):
        super().__init__(persist_dir=persist_dir, agent_id="architect")
        self._output_dir = OUTPUT_DIR

    @Tool.define
    def read_existing_files(self) -> str:
        """List files already written to the output directory."""
        if not self._output_dir.exists():
            return "No files yet."
        files = sorted(self._output_dir.rglob("*.py"))
        if not files:
            return "No Python files yet."
        return "\n".join(str(f.relative_to(self._output_dir)) for f in files)

    @Template.define
    def plan_modules(self, project_spec: str) -> PlanResult:
        """Given this project specification, output a plan with a "modules" list.
        Each module spec has: module_path, description, public_api, test_path.

        Use `read_existing_files` to check what's already been written
        and skip those.

        Project spec:
        {project_spec}"""
        raise NotHandled


class CoderAgent(PersistentAgent):
    """You are an expert Python developer. Given a module specification,
    you write clean, well-documented Python code. You also write thorough
    test files. Output ONLY the Python source code, no markdown fences.
    """

    def __init__(self, persist_dir: Path):
        super().__init__(persist_dir=persist_dir, agent_id="coder")
        self._output_dir = OUTPUT_DIR

    @Tool.define
    def read_file(self, path: str) -> str:
        """Read a file from the output directory."""
        full = self._output_dir / path
        if full.exists():
            return full.read_text()
        return f"File not found: {path}"

    @Tool.define
    def write_file(self, path: str, content: str) -> str:
        """Write a file to the output directory."""
        full = self._output_dir / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    @Template.define
    def implement_module(self, module_spec: str) -> str:
        """Implement the following module specification. Use `write_file`
        to write both the module and its test file. Use `read_file` to
        check existing code if needed.

        Specification:
        {module_spec}"""
        raise NotHandled


class ReviewerAgent(PersistentAgent):
    """You are a senior code reviewer. You review Python modules for
    correctness, style, edge cases, and test coverage. Be specific
    about issues and provide actionable feedback.
    """

    def __init__(self, persist_dir: Path):
        super().__init__(persist_dir=persist_dir, agent_id="reviewer")
        self._output_dir = OUTPUT_DIR

    @Tool.define
    def read_file(self, path: str) -> str:
        """Read a file from the output directory."""
        full = self._output_dir / path
        if full.exists():
            return full.read_text()
        return f"File not found: {path}"

    @Template.define
    def review_module(self, module_path: str, test_path: str) -> ReviewResult:
        """Review the module at {module_path} and its tests at {test_path}.
        Use `read_file` to read them. Return verdict "PASS" or "NEEDS_FIXES"
        and feedback. If NEEDS_FIXES, explain exactly what to change."""
        raise NotHandled


# ---------------------------------------------------------------------------
# Worker loops
# ---------------------------------------------------------------------------

_shutdown = threading.Event()


def architect_worker(queue: TaskQueue) -> None:
    """Pull 'plan' tasks from the queue, generate module specs, submit
    'code' tasks for each module."""
    agent = ArchitectAgent(persist_dir=STATE_DIR)
    queue.release_stale_claims("architect")

    provider = LiteLLMProvider(model=MODEL)

    while not _shutdown.is_set():
        task = queue.claim("plan", "architect")
        if task is None:
            time.sleep(1)
            continue

        try:
            payload: PlanTaskPayload = task["payload"]
            with handler(provider), handler(RetryLLMHandler()):
                plan = agent.plan_modules(payload["project_spec"])

            for mod in plan["modules"]:
                queue.submit("code", mod)

            queue.complete(task["id"], "architect", {"modules": len(plan["modules"])})
        except Exception as e:
            log.exception("Architect failed on task %s", task["id"])
            queue.fail(task["id"], "architect", str(e))


def coder_worker(queue: TaskQueue) -> None:
    """Pull 'code' tasks, write modules and tests, submit 'review' tasks."""
    agent = CoderAgent(persist_dir=STATE_DIR)
    queue.release_stale_claims("coder")

    provider = LiteLLMProvider(model=MODEL)

    while not _shutdown.is_set():
        task = queue.claim("code", "coder")
        if task is None:
            time.sleep(1)
            continue

        try:
            payload: CodeTaskPayload = task["payload"]
            spec = json.dumps(payload, indent=2)
            with handler(provider), handler(RetryLLMHandler()):
                agent.implement_module(spec)

            queue.submit(
                "review",
                {
                    "module_path": payload["module_path"],
                    "test_path": payload.get("test_path", ""),
                },
            )
            queue.complete(task["id"], "coder")
        except Exception as e:
            log.exception("Coder failed on task %s", task["id"])
            queue.fail(task["id"], "coder", str(e))


def reviewer_worker(queue: TaskQueue) -> None:
    """Pull 'review' tasks, review code, submit 'fix' tasks if needed."""
    agent = ReviewerAgent(persist_dir=STATE_DIR)
    queue.release_stale_claims("reviewer")

    provider = LiteLLMProvider(model=MODEL)

    while not _shutdown.is_set():
        task = queue.claim("review", "reviewer")
        if task is None:
            time.sleep(1)
            continue

        try:
            payload: ReviewTaskPayload = task["payload"]
            with handler(provider), handler(RetryLLMHandler()):
                review = agent.review_module(
                    payload["module_path"],
                    payload.get("test_path", ""),
                )

            if review["verdict"] == "NEEDS_FIXES":
                fix_payload = {
                    **payload,
                    "description": f"Fix issues: {review['feedback']}",
                    "public_api": payload.get("public_api", ""),
                }
                queue.submit("code", fix_payload)

            queue.complete(
                task["id"],
                "reviewer",
                {"verdict": review["verdict"], "feedback": review["feedback"][:200]},
            )
        except Exception as e:
            log.exception("Reviewer failed on task %s", task["id"])
            queue.fail(task["id"], "reviewer", str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    queue = TaskQueue(STATE_DIR / "queue")

    # On first run (or if restarted after all tasks completed), seed the queue
    if queue.all_done() and not list(OUTPUT_DIR.rglob("*.py")):
        queue.submit("plan", {"project_spec": PROJECT_SPEC})
        log.info("Seeded queue with initial planning task")
    else:
        log.info("Resuming — found existing tasks in queue")

    # Handle Ctrl-C gracefully
    def on_signal(sig, frame):
        log.info("Shutdown signal received — finishing current tasks...")
        _shutdown.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Launch worker threads
    workers = [
        threading.Thread(
            target=architect_worker,
            args=(queue,),
            name="architect",
            daemon=True,
        ),
        threading.Thread(
            target=coder_worker,
            args=(queue,),
            name="coder",
            daemon=True,
        ),
        threading.Thread(
            target=reviewer_worker,
            args=(queue,),
            name="reviewer",
            daemon=True,
        ),
    ]

    for w in workers:
        w.start()

    # Wait until all work is done or shutdown
    while not _shutdown.is_set():
        if queue.all_done():
            log.info("All tasks completed!")
            _shutdown.set()
            break
        time.sleep(2)

    for w in workers:
        w.join(timeout=10)

    # Summary
    done = list((STATE_DIR / "queue").glob(f"*.{TaskStatus.DONE}.json"))
    failed = list((STATE_DIR / "queue").glob(f"*.{TaskStatus.FAILED}.json"))
    output_files = list(OUTPUT_DIR.rglob("*.py"))
    log.info(
        "Summary: %d tasks done, %d failed, %d output files",
        len(done),
        len(failed),
        len(output_files),
    )
    for f in output_files:
        log.info("  %s", f.relative_to(WORKSPACE))


if __name__ == "__main__":
    main()
