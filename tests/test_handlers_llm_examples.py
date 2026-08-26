"""Live runs of every script under ``docs/source/llm_examples``.

Each example is run the way its docs tell a reader to run it -- through the
module launcher, as a subprocess, with the example's own default arguments -- so
what passes here is what a reader can reproduce. `OVERRIDES` holds the few
exceptions, each with a reason.

Where this runs. Pull requests get `basics`, in the LLM Integration Tests
workflow; the nightly workflow runs everything, which is what the
``nightly_example`` mark selects. Without an API key the whole module skips, so
the Test workflow collects it and moves on -- what a keyless build can say about
an example is said in ``test_handlers_llm_harness_launcher.py`` instead, which
checks statically that every example still imports and parses its flags.

There was briefly a suite that replayed recorded model responses, so this
coverage came free on every build. It was removed because its fixtures were
keyed on the assembled request, which includes the system prompt -- so every
fixture went stale whenever a prompt string in ``effectful/handlers/llm/``
changed, and those are under active development. Live runs cost tokens and
flake; stale fixtures cost a re-recording against a paid API every time an
unrelated docstring moved.
"""

import dataclasses
import enum
import os
import pathlib
import subprocess
import sys

import pytest
from _pytest.mark.structures import ParameterSet

from tests.conftest import (
    EFFECTFUL_LLM_MODEL,
    example_id,
    example_scripts,
    requires_llm,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_TIMEOUT = 3 * 60
"""Seconds any one example may take before it is killed and reported failed.

Generous, because these are agent loops against a live provider and a slow model
is not a broken example -- but bounded, because a hung tool call would otherwise
hang the job.
"""

RETRIES = int(os.getenv("EFFECTFUL_EXAMPLE_RETRIES", "1"))
"""Times to retry a failing example before believing the failure.

A live agent loop is not deterministic: one of ``map_reduce``'s four concurrent
evaluations can exhaust its retries on a malformed reply that the same model
handles correctly a minute later. Failing on that trains everyone to ignore the
result, so a run that fails and then passes warns instead.
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


def example_params() -> list[ParameterSet]:
    """Every example script, with the non-``basics`` ones marked for the nightly.

    The split is by cost, not by importance. ``basics`` is a handful of
    round-trips per example and gates pull requests; the other directories are
    agent searches and multi-round pipelines, and run once a night.
    """
    params = []
    for path in example_scripts():
        name = example_id(path)
        marks = [] if name.startswith("basics/") else [pytest.mark.nightly_example]
        params.append(pytest.param(path, id=name, marks=marks))
    return params


@pytest.fixture(scope="session")
def reachable_model() -> str:
    """The model to run examples against, having checked that it answers.

    `requires_llm` only knows whether a key is *configured*. This is the
    difference between a suite that says "every example still works" and one
    that says "every example still exits zero while failing to reach a model" --
    which look identical in the summary, because an example that handles its own
    errors reports them and exits zero anyway. ``autoformalization/auditing``
    against an unreachable model prints ``no verdict (retries exhausted): 5`` and
    succeeds, since a claim it could not get a verdict on is a result it is
    designed to report.
    """
    import litellm

    try:
        litellm.completion(
            model=EFFECTFUL_LLM_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as e:
        pytest.fail(
            f"{EFFECTFUL_LLM_MODEL} is configured but unreachable, so every "
            f"example below would be testing nothing: {type(e).__name__}: {e}"
        )
    return EFFECTFUL_LLM_MODEL


def run_example(
    path: pathlib.Path, override: Override, model: str
) -> subprocess.CompletedProcess:
    argv = [
        sys.executable,
        "-m",
        "effectful.handlers.llm.harness",
        str(path),
        "--model",
        model,
        *override.args,
    ]
    print(f"$ {' '.join(argv)}", flush=True)
    return subprocess.run(
        argv,
        cwd=REPO_ROOT,
        timeout=override.timeout,
        # An example that reads stdin outside --interactive should die on EOF
        # rather than wait forever for a tty that is not there.
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )


@requires_llm
@pytest.mark.example
@pytest.mark.parametrize("path", example_params())
def test_example_runs(path, reachable_model):
    name = example_id(path)
    override = OVERRIDES.get(name, Override())
    if override.skip is not None:
        pytest.skip(f"{name}: {override.skip}")
    if override.why:
        print(f"not run with its own defaults: {override.why}", flush=True)

    first: subprocess.CompletedProcess | None = None
    for attempt in range(RETRIES + 1):
        try:
            proc = run_example(path, override, reachable_model)
        except subprocess.TimeoutExpired:
            pytest.fail(f"{name} exceeded {override.timeout}s")
        if proc.returncode == 0:
            if first is not None:
                # Visible in the report without failing the run, since it passed.
                print(f"::warning title={name}::flaky, passed on attempt {attempt + 1}")
            assert proc.stdout.strip(), (
                f"{name} exited zero but printed nothing; these examples are "
                f"scripts whose output is the point"
            )
            return
        first = first or proc

    assert first is not None
    pytest.fail(
        f"{name} exited {first.returncode} on every one of {RETRIES + 1} "
        f"attempts\n--- stdout ---\n{first.stdout[-4000:]}"
        f"\n--- stderr ---\n{first.stderr[-4000:]}"
    )


def test_examples_were_actually_found():
    """Guard the parametrization: an empty glob would vacuously pass."""
    params = example_params()
    assert len(params) > 20
    names = {p.id for p in params}
    assert set(OVERRIDES) < names, "OVERRIDES names an example that no longer exists"
    assert sum(n.startswith("basics/") for n in names) > 5, "basics/ went missing"
