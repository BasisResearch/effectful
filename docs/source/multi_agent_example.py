"""Multi-agent system using choreographic endpoint projection.

Demonstrates:

- Choreographic programming: one ``async`` function describes the entire workflow
- Automatic endpoint projection: each agent runs it as its own `asyncio.Task`
- Crash tolerance: Ctrl-C and restart, agents resume where they left off
- Scatter: two coder agents share the implementation work via claim-based pull
- A persisted `Agent` (`agent_id` + `SQLitePersister`) for automatic checkpointing

The scenario: a team of agents collaboratively builds a small Python library.
An architect agent breaks the project into module specs, two coder agents
implement the modules in parallel (via `scatter`), and two reviewer agents
review modules in parallel and request fixes until everything passes.

Only the LLM calls occupy threads, and only while they are in flight: an agent
waiting on a peer's step is a suspended coroutine. See
`effectful.handlers.llm.multi` for why the program is written with explicit
``await step(...)`` rather than plain method calls.

Usage::

    # First run -- agents start working
    python docs/source/multi_agent_example.py

    # Ctrl-C mid-run, then restart -- agents pick up where they left off
    python docs/source/multi_agent_example.py

Requirements:
    pip install effectful[llm]
    export OPENAI_API_KEY=...   # or any LiteLLM-supported provider

"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Literal, TypedDict

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    RetryLLMHandler,
    SQLitePersister,
)
from effectful.handlers.llm.multi import (
    Choreography,
    ChoreographyError,
    PersistentTaskQueue,
    call,
    scatter,
    step,
)
from effectful.ops.types import NotHandled

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
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
# Structured types — constrained decoding for LLM output
# ---------------------------------------------------------------------------


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
# Agents
# ---------------------------------------------------------------------------


class ArchitectAgent(Agent):
    """You are a software architect. Given a project specification, you break
    it into individual module implementation tasks. Each task should specify
    the module filename, its public API, and what tests to write.
    Be concrete and specific — the coder will follow your spec exactly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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


class CoderAgent(Agent):
    """You are an expert Python developer. Given a module specification,
    you write clean, well-documented Python code. You also write thorough
    test files. Output ONLY the Python source code, no markdown fences.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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


class ReviewerAgent(Agent):
    """You are a senior code reviewer. You review Python modules for
    correctness, style, edge cases, and test coverage. Be specific
    about issues and provide actionable feedback.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
# Choreographic program — the entire multi-agent workflow in one function
# ---------------------------------------------------------------------------


async def build_project(
    project_spec: str,
    architect: ArchitectAgent,
    coder: CoderAgent,
    reviewer: ReviewerAgent,
) -> list[ReviewResult]:
    """Choreographic program describing the full build workflow.

    1. Architect breaks the project into module specs.
    2. Coders implement modules in parallel (scatter distributes via claim-based pull).
    3. Reviewers review modules in parallel; coders fix in parallel until all pass.
    """
    # Step 1: Architect plans modules.  Every agent awaits this same step; only
    # the architect actually calls the model.
    plan = await step(architect.plan_modules, project_spec)

    # Step 2: Scatter implementation across coders.
    # Each module becomes a task in the queue; coders claim until none remain.
    await scatter(
        plan["modules"],
        coder,
        lambda c, mod: call(c.implement_module, json.dumps(mod, indent=2)),
    )

    # Step 3: Review loop — keep fixing until reviewers accept all modules
    while True:
        reviews: list[ReviewResult] = await scatter(
            plan["modules"],
            reviewer,
            lambda r, mod: call(r.review_module, mod["module_path"], mod["test_path"]),
        )

        needs_fixes = [
            (mod, review)
            for mod, review in zip(plan["modules"], reviews)
            if review["verdict"] == "NEEDS_FIXES"
        ]

        if not needs_fixes:
            return reviews

        # Scatter fixes across coders, then re-review
        await scatter(
            needs_fixes,
            coder,
            lambda c, pair: call(
                c.implement_module,
                json.dumps({**pair[0], "fix_feedback": pair[1]["feedback"]}, indent=2),
            ),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create agents.  An explicit agent_id is what makes an Agent persistent:
    # its history and state are checkpointed by SQLitePersister below.
    architect = ArchitectAgent(agent_id="architect")
    coder1 = CoderAgent(agent_id="coder-1")
    coder2 = CoderAgent(agent_id="coder-2")
    reviewer1 = ReviewerAgent(agent_id="reviewer-1")
    reviewer2 = ReviewerAgent(agent_id="reviewer-2")

    # Build the choreography — tasks, thread pool, cancellation on failure and
    # crash recovery are all handled for you.
    choreo = Choreography(
        build_project,
        agents=[architect, coder1, coder2, reviewer1, reviewer2],
        queue=PersistentTaskQueue(STATE_DIR / "task_queue.db"),
        handlers=[
            LiteLLMProvider(model=MODEL),
            RetryLLMHandler(),
            SQLitePersister(STATE_DIR / "checkpoints.db"),
        ],
    )

    log.info("Starting multi-agent build (Ctrl-C to pause, re-run to resume)")

    try:
        reviews = choreo.run(
            project_spec=PROJECT_SPEC,
            architect=architect,
            coder=[coder1, coder2],
            reviewer=[reviewer1, reviewer2],
        )
    except ChoreographyError as e:
        log.error("Choreography failed: %s", e)
        return
    except KeyboardInterrupt:
        log.warning("Interrupted — re-run to resume from the last completed step")
        return
    finally:
        asyncio.run(choreo.queue.close())

    # Summary
    output_files = list(OUTPUT_DIR.rglob("*.py"))
    passed = sum(1 for r in reviews if r["verdict"] == "PASS")
    log.info(
        "Done: %d modules reviewed (%d passed), %d output files",
        len(reviews),
        passed,
        len(output_files),
    )
    for f in output_files:
        log.info("  %s", f.relative_to(WORKSPACE))


if __name__ == "__main__":
    main()
