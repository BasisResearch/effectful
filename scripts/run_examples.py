"""Run the ``docs/source/llm_examples`` scripts against a live model.

This is the nightly counterpart to ``tests/test_handlers_llm_examples_basics.py``.
That suite replays recorded models, so it answers "does the Python still work"
on every build for free; only a live run answers "do these prompts still get a
usable answer out of a real model", which is the thing that decays silently as
providers change models underneath us.

Every example is run the way its docs tell a reader to run it -- through the
module launcher, with the example's own default arguments -- so what passes here
is what a reader can reproduce. `OVERRIDES` holds the few exceptions, each with
a reason.

Usage::

    uv run python scripts/run_examples.py --model gpt-4o-mini
    uv run python scripts/run_examples.py reasoning basics --model gpt-4o-mini

Exits non-zero if any example failed, and prints a summary of every outcome.

What "passed" means here is "exited zero", which is weaker than it sounds: an
example that handles its own errors reports them and exits zero anyway.
``autoformalization/auditing`` run against an unreachable model prints ``no
verdict (retries exhausted): 5`` and succeeds, because a claim it could not get
a verdict on is a result it is designed to report. That makes a misconfigured
key the dangerous failure -- it would turn the whole nightly green while nothing
was tested -- so :func:`preflight` reaches the model once up front and refuses
to run anything if it cannot.
"""

import argparse
import ast
import dataclasses
import enum
import pathlib
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "docs" / "source" / "llm_examples"

DEFAULT_TIMEOUT = 30 * 60
"""Seconds any one example may take before it is killed and reported failed.

Generous, because these are agent loops against a live provider and a slow
model is not a broken example -- but bounded, because a hung tool call would
otherwise hang the whole nightly job.
"""


class Skip(enum.StrEnum):
    """Why an example is not run here."""

    LEAN = "needs a Lean 4 + Mathlib toolchain (see the module docstring)"


@dataclasses.dataclass(frozen=True)
class Override:
    """A deviation from "run it the way the docs say"."""

    args: tuple[str, ...] = ()
    timeout: int = DEFAULT_TIMEOUT
    skip: Skip | None = None
    why: str = ""


OVERRIDES: dict[str, Override] = {
    "autoformalization/formalization": Override(skip=Skip.LEAN),
    "reasoning/aime2024": Override(
        args=("least-beautiful-base",),
        why="its parser requires one of several problem subcommands",
    ),
    "reasoning/continual": Override(
        args=("--budget", "20"),
        why="the default 400-press budget is a research run, not a smoke test",
    ),
    "autoformalization/auditing": Override(
        args=("--limit", "1"),
        why="--limit is the example's own documented cheap smoke test",
    ),
}


def example_scripts(dirs: list[str]) -> list[pathlib.Path]:
    """Every runnable example under `dirs` (or all of them), sorted."""
    roots = [EXAMPLES_DIR / d for d in dirs] if dirs else [EXAMPLES_DIR]
    return sorted(
        p
        for root in roots
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
        and p.name != "__init__.py"
        # A module with no `main` is a library its siblings import.
        and any(
            isinstance(node, ast.FunctionDef) and node.name == "main"
            for node in ast.parse(p.read_text(), filename=str(p)).body
        )
    )


def name_of(path: pathlib.Path) -> str:
    return str(path.relative_to(EXAMPLES_DIR).with_suffix(""))


def preflight(model: str) -> str | None:
    """Reach `model` once. Returns the reason it is unusable, or None.

    Cheapest possible request -- the reply is discarded, only that one arrived
    matters. This is the difference between a nightly that says "every example
    still works" and one that says "every example still exits zero while
    failing to reach a model", which look identical in the summary.
    """
    import litellm

    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


@dataclasses.dataclass
class Result:
    name: str
    status: str
    seconds: float
    detail: str = ""


def run_one(path: pathlib.Path, model: str, extra: list[str], retries: int) -> Result:
    """Run one example, retrying a failure `retries` times before believing it.

    A live agent loop is not deterministic: one of ``map_reduce``'s four
    concurrent evaluations can exhaust its retries on a malformed reply that the
    same model produces correctly a minute later. Failing the nightly on that
    trains everyone to ignore it, so a run that fails and then passes is
    reported as FLAKY -- visible, but not a failure, because it did pass.
    """
    name = name_of(path)
    override = OVERRIDES.get(name, Override())
    if override.skip is not None:
        print(f"::notice::skipping {name}: {override.skip}", flush=True)
        return Result(name, "SKIP", 0.0, str(override.skip))

    first: Result | None = None
    for attempt in range(retries + 1):
        result = _attempt(path, name, model, extra, override)
        if result.status == "PASS":
            if first is not None:
                return dataclasses.replace(
                    result,
                    status="FLAKY",
                    detail=f"passed on attempt {attempt + 1} after {first.detail}",
                )
            return result
        first = first or result
        if attempt < retries:
            print(f"::notice::{name} failed ({result.detail}), retrying", flush=True)
    assert first is not None
    return first


def _attempt(
    path: pathlib.Path,
    name: str,
    model: str,
    extra: list[str],
    override: Override,
) -> Result:
    argv = [
        sys.executable,
        "-m",
        "effectful.handlers.llm.harness",
        str(path),
        "--model",
        model,
        *extra,
        *override.args,
    ]
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}", flush=True)
    if override.why:
        print(f"not run with its own defaults: {override.why}", flush=True)
    print(f"$ {' '.join(argv)}", flush=True)

    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            timeout=override.timeout,
            # An example that reads stdin outside --interactive should die on
            # EOF rather than wait forever for a tty that is not there.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return Result(name, "TIMEOUT", elapsed, f"exceeded {override.timeout}s")
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        return Result(name, "FAIL", elapsed, f"exit {proc.returncode}")
    return Result(name, "PASS", elapsed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dirs",
        nargs="*",
        metavar="DIR",
        help="Example subdirectories to run (default: all of them)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to run every example against",
    )
    parser.add_argument(
        "--harness-arg",
        action="append",
        default=[],
        metavar="FLAG",
        help="Extra launcher flag, repeatable (e.g. --harness-arg=--num-retries=2)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Times to retry a failing example before believing the failure",
    )
    parser.add_argument(
        "--no-preflight",
        dest="preflight",
        action="store_false",
        help="Do not check that the model is reachable before running anything",
    )
    args = parser.parse_args()

    # An override names an example by path; a renamed example would otherwise
    # leave a dead entry here and silently go back to its own defaults, which
    # for the entries below means a research-sized run or a missing subcommand.
    stale = {n for n in OVERRIDES if not (EXAMPLES_DIR / f"{n}.py").exists()}
    if stale:
        parser.error(f"OVERRIDES names examples that no longer exist: {sorted(stale)}")

    scripts = example_scripts(args.dirs)
    if not scripts:
        parser.error(f"no runnable examples under {args.dirs or ['all']}")

    if args.preflight and (reason := preflight(args.model)) is not None:
        print(f"::error title=preflight::{args.model} is unreachable: {reason}")
        return 2

    results = [run_one(p, args.model, args.harness_arg, args.retries) for p in scripts]

    width = max(len(r.name) for r in results)
    print(f"\n{'=' * 72}\nSummary\n{'=' * 72}", flush=True)
    for r in results:
        detail = f"  ({r.detail})" if r.detail else ""
        print(f"{r.status:<8}{r.name:<{width}}  {r.seconds:6.1f}s{detail}")

    failed = [r for r in results if r.status in ("FAIL", "TIMEOUT")]
    flaky = [r for r in results if r.status == "FLAKY"]
    passed = sum(r.status in ("PASS", "FLAKY") for r in results)
    skipped = sum(r.status == "SKIP" for r in results)
    print(
        f"\n{passed} passed ({len(flaky)} flaky), "
        f"{len(failed)} failed, {skipped} skipped"
    )
    for r in flaky:
        print(f"::warning title={r.name}::{r.detail}")
    for r in failed:
        print(f"::error title={r.name}::{r.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
