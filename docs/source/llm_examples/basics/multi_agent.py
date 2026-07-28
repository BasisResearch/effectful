"""Multi-agent library build via choreographic endpoint projection.

Demonstrates:
- Choreographic programming: one ``async`` function describes the whole workflow
- Endpoint projection: every agent runs that function as its own `asyncio.Task`,
  executing the steps it owns and awaiting the ones it doesn't
- ``scatter``: two coders share the implementation work and two reviewers share
  the reviews, each item going to whichever agent is free
- A step log: interrupt the run and start it again, and the agents resume from
  the last step that finished
- Tools as ground truth: the reviewers run each module's tests rather than
  judging the code by reading it, so the fix loop turns on a fact

The scenario: a team of agents collaboratively builds a small Python library.
An architect breaks the project into module specs, coders implement the modules
in parallel, and reviewers run their tests and review them in parallel, sending
work back to the coders until everything passes.

The reviewers run generated test files as subprocesses, so this example
executes code the model wrote. Everything under
`effectful.handlers.llm.harness` already can -- it installs a Python REPL --
but it is worth knowing before pointing this at an untrusted project spec.

Only in-flight LLM calls occupy threads: an agent waiting on a peer's step is a
suspended coroutine. See `effectful.handlers.llm.choreographies` for why steps
are spelled ``await step(...)`` rather than as plain method calls.

Run it, interrupt it with Ctrl-C, and run it again to watch it pick up where it
left off::

    python -m effectful.handlers.llm.harness \\
        docs/source/llm_examples/basics/multi_agent.py --model gpt-4o-mini

Use ``--restart`` to forget the recorded steps and build from scratch, and pass
``--persist-db PATH`` to the harness to checkpoint each agent's own
conversation history alongside them.
"""

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Literal, TypedDict

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.choreographies import (
    Choreography,
    ChoreographyError,
    recorded_steps,
    scatter,
    step,
)

DEFAULT_TEST_TIMEOUT = 60
"""Seconds a generated test file gets before the reviewer gives up on it."""

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
# Structured output — constrained decoding for LLM output
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

    def __init__(self, output_dir: pathlib.Path, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir

    @Tool.define
    def read_existing_files(self) -> str:
        """List files already written to the output directory."""
        files = sorted(self.output_dir.rglob("*.py"))
        if not files:
            return "No Python files yet."
        return "\n".join(str(f.relative_to(self.output_dir)) for f in files)

    @Template.define
    def plan_modules(self, project_spec: str) -> PlanResult:
        """Given this project specification, output a plan with a "modules" list.
        Each module spec has: module_path, description, public_api, test_path.

        Use `read_existing_files` to check what's already been written
        and skip those.

        Project spec:
        {project_spec}"""


class CoderAgent(Agent):
    """You are an expert Python developer. Given a module specification,
    you write clean, well-documented Python code. You also write thorough
    test files. Output ONLY the Python source code, no markdown fences.
    """

    def __init__(self, output_dir: pathlib.Path, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir

    @Tool.define
    def read_file(self, path: str) -> str:
        """Read a file from the output directory."""
        full = self.output_dir / path
        return full.read_text() if full.exists() else f"File not found: {path}"

    @Tool.define
    def write_file(self, path: str, content: str) -> str:
        """Write a file to the output directory."""
        full = self.output_dir / path
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


class ReviewerAgent(Agent):
    """You are a senior code reviewer. You review Python modules for
    correctness, style, edge cases, and test coverage. You judge a module by
    running its tests, not only by reading it. Be specific about issues and
    provide actionable feedback.
    """

    def __init__(
        self,
        output_dir: pathlib.Path,
        test_timeout: float = DEFAULT_TEST_TIMEOUT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.test_timeout = test_timeout

    @Tool.define
    def read_file(self, path: str) -> str:
        """Read a file from the output directory."""
        full = self.output_dir / path
        return full.read_text() if full.exists() else f"File not found: {path}"

    @Tool.define
    def run_tests(self, test_path: str) -> str:
        """Run the test file at `test_path` with pytest and return its output."""
        try:
            result = subprocess.run(
                # `-o addopts=` because pytest would otherwise inherit the
                # addopts of whatever project the workspace happens to sit in.
                [sys.executable, "-m", "pytest", test_path, "-q", "--no-header"]
                + ["-o", "addopts=", "-p", "no:cacheprovider"],
                cwd=self.output_dir,
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Timed out after {self.test_timeout}s — the tests do not terminate."
        return f"exit code {result.returncode}\n\n{result.stdout[-4000:]}"

    @Template.define
    def review_module(self, module_path: str, test_path: str) -> ReviewResult:
        """Review the module at {module_path} and its tests at {test_path}.
        Use `read_file` to read them and `run_tests` to run the test file.

        Return verdict "PASS" or "NEEDS_FIXES" and feedback. A module whose
        tests do not all pass is "NEEDS_FIXES", whatever the code looks like;
        say which test failed and why. If the test itself is wrong, say that
        instead — either way the coder has something to fix."""


# ---------------------------------------------------------------------------
# Choreographic program — the entire multi-agent workflow in one function
# ---------------------------------------------------------------------------


async def build_project(
    project_spec: str,
    architect: ArchitectAgent,
    coder: CoderAgent,
    reviewer: ReviewerAgent,
    max_rounds: int,
) -> list[ReviewResult]:
    """Choreographic program describing the full build workflow.

    1. Architect breaks the project into module specs.
    2. Coders implement modules in parallel (scatter hands each to whoever is free).
    3. Reviewers run each module's tests and review it; coders fix what failed,
       for up to *max_rounds* rounds.
    """
    # Step 1: the architect plans the modules.  Every agent awaits this same
    # step; only the architect calls the model for it.
    plan = await step(architect.plan_modules, project_spec)

    # Step 2: scatter implementation across the coders.  Each coder takes the
    # next module as it becomes free, until none are left.
    await scatter(
        plan["modules"],
        coder,
        lambda c, mod: step(c.implement_module, json.dumps(mod, indent=2)),
    )

    # Step 3: review loop — keep fixing until the reviewers accept every module.
    # Bounded, because a reviewer and a coder that disagree would otherwise
    # trade rounds forever.  Every agent sees the same reviews, so they all
    # leave the loop on the same iteration.
    for _ in range(max_rounds):
        reviews: list[ReviewResult] = await scatter(
            plan["modules"],
            reviewer,
            lambda r, mod: step(r.review_module, mod["module_path"], mod["test_path"]),
        )

        needs_fixes = [
            (mod, review)
            for mod, review in zip(plan["modules"], reviews)
            if review["verdict"] == "NEEDS_FIXES"
        ]
        if not needs_fixes:
            return reviews

        await scatter(
            needs_fixes,
            coder,
            lambda c, pair: step(
                c.implement_module,
                json.dumps({**pair[0], "fix_feedback": pair[1]["feedback"]}, indent=2),
            ),
        )

    return reviews  # out of rounds; hand back the last verdicts as they stand


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=pathlib.Path,
        default=pathlib.Path("./multi_agent_workspace"),
        help="Directory to write the generated library into",
    )
    parser.add_argument(
        "--project-spec",
        type=str,
        default=PROJECT_SPEC,
        help="The project for the team to build",
    )
    parser.add_argument("--coders", type=int, default=2, help="Number of coder agents")
    parser.add_argument(
        "--reviewers", type=int, default=2, help="Number of reviewer agents"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="How many review-and-fix rounds to allow before giving up",
    )
    parser.add_argument(
        "--test-timeout",
        type=float,
        default=DEFAULT_TEST_TIMEOUT,
        metavar="SECONDS",
        help="How long a reviewer waits for a generated test file to finish",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Forget the steps recorded by earlier runs and build from scratch",
    )
    args = parser.parse_args()

    output_dir = args.workspace / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # An explicit agent_id is what makes an Agent persistent, and it is also how
    # endpoint projection tells the agents apart.
    architect = ArchitectAgent(output_dir, agent_id="architect")
    coders = [CoderAgent(output_dir, agent_id=f"coder-{i}") for i in range(args.coders)]
    reviewers = [
        ReviewerAgent(output_dir, args.test_timeout, agent_id=f"reviewer-{i}")
        for i in range(args.reviewers)
    ]

    # Steps completed by an earlier run are replayed instead of re-asking the
    # model, so an interrupted build resumes rather than starting over.
    log = args.workspace / ".state" / "steps.db"
    if args.restart:
        log.unlink(missing_ok=True)

    # Tasks, the thread pool and cancellation on failure are all handled for
    # you; the model handlers come from the harness.
    choreo = Choreography(
        build_project, agents=[architect, *coders, *reviewers], log=log
    )

    done = len(recorded_steps(log))
    print(
        f"Starting multi-agent build{f' ({done} steps already done)' if done else ''}"
    )
    try:
        reviews = choreo.run(
            project_spec=args.project_spec,
            architect=architect,
            coder=coders,
            reviewer=reviewers,
            max_rounds=args.max_rounds,
        )
    except ChoreographyError as e:
        print(f"Choreography failed: {e} — re-run to retry from this step")
        return
    except KeyboardInterrupt:
        print("Interrupted — re-run to resume from the last completed step")
        return

    passed = sum(1 for r in reviews if r["verdict"] == "PASS")
    print(f"\nDone: {len(reviews)} modules reviewed, {passed} passed")
    for f in sorted(output_dir.rglob("*.py")):
        print(f"  {f.relative_to(args.workspace)}")


if __name__ == "__main__":
    main()
