"""End-to-end runs of ``docs/source/llm_examples/basics`` against recorded models.

Every example here is run the way a reader runs it -- through the module
launcher, which supplies the handler stack -- but with the transport swapped for
`ReplayCompletions`, which serves each request from a JSON fixture recorded
once against a real model. That makes the whole suite offline, deterministic and
free, so it can run on every build rather than only where an API key exists.

What this does and does not cover. It covers the Python: that an example still
imports, that its skills and tools still decode the shape of reply the model
gave last time, that its control flow still reaches the end. It does *not* cover
whether the prompts still make sense to a live model -- a fixture cannot notice
that a model would now answer differently. That is the nightly workflow's job.

Re-record after changing an example's prompts or control flow::

    REBUILD_FIXTURES=true EFFECTFUL_LLM_MODEL=openrouter/openai/gpt-4o-mini \\
      uv run pytest tests/test_handlers_llm_examples_basics.py -n auto

Recording needs an API key and *overwrites* the fixtures of the examples it
runs; replaying needs neither key nor network.
"""

import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import sys
import typing

import pytest
from litellm.files.main import ModelResponse

from effectful.handlers.llm.harness import __main__ as launcher
from effectful.handlers.llm.harness import harness
from effectful.handlers.llm.harness.hooks import completion
from effectful.ops.semantics import coproduct, fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from tests.conftest import EFFECTFUL_LLM_MODEL

EXAMPLES_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "docs"
    / "source"
    / "llm_examples"
    / "basics"
)

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / "fixtures" / "examples"

REBUILD_FIXTURES = os.getenv("REBUILD_FIXTURES") == "true"

DUMP_REQUESTS = os.getenv("DUMP_REQUESTS")
"""Directory to write each request to, for diagnosing a fixture miss."""


# ============================================================================
# Which examples run here, and how
# ============================================================================

# Arguments that shrink an example to smoke-test size. The defaults are what a
# reader runs and what the nightly workflow runs; here a second question or a
# third chat turn would cost another recorded round-trip to re-exercise a path
# the first one already covered.
EXAMPLE_ARGS: dict[str, list[str]] = {
    "conversation": ["--messages", "Hi! Can you tell me about the Statue of Liberty?"],
    "guardrails": ["--queries", "What are great places to check out in NYC?"],
    "hitl": ["--max-steps", "2"],
    "text2sql": ["--questions", "What is the average salary by department?"],
}

# Examples whose *other* I/O is not the `completion` op, so replaying the model
# alone would still leave them talking to the network. They are covered live by
# the nightly workflow instead.
NETWORK_BOUND = {
    "rag": "embeds documents through litellm.embedding, which is not an op",
    "research_agent": "fetches article text from the Wikipedia API",
}


def basics_examples() -> list[pathlib.Path]:
    return sorted(p for p in EXAMPLES_DIR.glob("*.py") if p.name != "__init__.py")


# ============================================================================
# Recorded transport
# ============================================================================

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
"""The address in a default Python ``repr``, e.g. ``<function f at 0x104e8b7e0>``.

These reach the wire. A `Skill` whose annotation carries a pydantic validator is
described to the model as ``Annotated[FlightDetails, AfterValidator(func=<function
matches_request at 0x117e577e0>)]``, and the address in it changes every process
-- so two runs of ``flight_booking`` send prompts that differ in that one place.
Keying a fixture on the raw request would make it unreplayable; keying it on the
request with addresses elided makes it replay, and loses nothing, since an
address cannot be part of what the model was answering.
"""


def _salient(kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """The parts of a request that determine the reply.

    The model is deliberately not among them: a fixture records what some model
    said to this prompt, and pinning it to the name of the model that happened
    to record it would make every recording unusable under the model CI runs.
    """
    return {
        k: kwargs.get(k)
        for k in ("messages", "tools", "tool_choice", "response_format")
    }


class ReplayCompletions(ObjectInterpretation):
    """Serve `completion` from disk, recording it first if asked to.

    Installed *below* `LiteLLMConfigurer` rather than above it, so a replayed
    response is configured and post-processed exactly as a live one is -- the
    request it is keyed on is the fully assembled one, and the reply still
    passes back through ``_enforce_tool_choice`` on its way out.

    Fixtures are keyed by a hash of the request rather than by call ordinal,
    for two reasons the examples force. ``map_reduce`` fans its skill calls out
    across threads, so there is no stable ordinal to key on; and an ordinal
    would make the fixtures say nothing about *which* request they answer, so a
    reworded prompt would silently replay the old exchange instead of failing
    and asking to be re-recorded. Identical requests do recur -- a retry after
    malformed output re-sends the same messages -- so each key carries its own
    occurrence counter, which is what lets ``error_recovery`` replay a first
    answer that fails and a second that does not.
    """

    def __init__(self, name: str):
        self.dir = FIXTURE_DIR / name
        self.name = name
        self.seen: dict[str, int] = {}

    @staticmethod
    def _key(kwargs: dict[str, typing.Any]) -> str:
        """A digest of the request, modulo the addresses in `_ADDRESS`."""
        blob = json.dumps(_salient(kwargs), sort_keys=True, default=repr)
        return hashlib.sha256(_ADDRESS.sub("", blob).encode()).hexdigest()[:16]

    def _path(self, kwargs: dict[str, typing.Any]) -> pathlib.Path:
        key = self._key(kwargs)
        n = self.seen[key] = self.seen.get(key, -1) + 1
        if DUMP_REQUESTS:
            # Fixtures hold replies, not requests, so a miss on replay says only
            # that *something* about the request changed. Dumping both sides of
            # the comparison is how you find out what.
            dump = pathlib.Path(DUMP_REQUESTS) / self.name
            dump.mkdir(parents=True, exist_ok=True)
            (dump / f"{len(self.seen):03d}_{key}_{n}.json").write_text(
                json.dumps(_salient(kwargs), sort_keys=True, indent=2, default=repr)
            )
        return self.dir / f"{key}_{n}.json"

    @implements(completion)
    def _completion(self, *args, **kwargs):
        path = self._path(kwargs)
        if REBUILD_FIXTURES:
            response = fwd(*args, **kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response.model_dump_json(indent=2))
            return response
        if not path.exists():
            raise AssertionError(
                f"No recorded reply for this request from {self.name!r} "
                f"({path.name}). The example's prompts or control flow have "
                f"changed since the fixtures were recorded; re-record with "
                f"REBUILD_FIXTURES=true (see this module's docstring)."
            )
        return ModelResponse.model_validate(json.loads(path.read_text()))


@pytest.fixture
def run_example(monkeypatch):
    """Run an example through the launcher with its model recorded or replayed."""

    def run(path: pathlib.Path, *args: str) -> str:
        replay = ReplayCompletions(path.stem)
        if REBUILD_FIXTURES and replay.dir.exists():
            # A re-recording is a replacement, not an addition: leaving the old
            # files behind would leave fixtures for requests nothing makes any
            # more, indistinguishable from the ones still in use.
            shutil.rmtree(replay.dir)

        # `coproduct` gives its second argument precedence, so the harness's own
        # `completion` runs first and `fwd`s down into the recording.
        monkeypatch.setattr(
            launcher,
            "harness",
            lambda **kwargs: coproduct(replay, harness(**kwargs)),
        )
        # An example that reads stdin outside `--interactive` should fail here
        # rather than block a CI run forever.
        monkeypatch.setattr(sys, "stdin", io.StringIO())
        monkeypatch.setattr(sys, "argv", list(sys.argv))
        monkeypatch.setattr(sys, "path", list(sys.path))

        launcher.main([str(path), "--model", EFFECTFUL_LLM_MODEL, *args])
        return replay.name

    return run


# ============================================================================
# The examples
# ============================================================================


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "path",
    basics_examples(),
    ids=[p.stem for p in basics_examples()],
)
def test_example_runs(path, run_example, capsys):
    if path.stem in NETWORK_BOUND:
        pytest.skip(f"{path.stem}: {NETWORK_BOUND[path.stem]}")

    run_example(path, *EXAMPLE_ARGS.get(path.stem, []))

    # Every one of these examples is a script whose output is the point; a run
    # that printed nothing finished without doing what it demonstrates.
    assert capsys.readouterr().out.strip()


def test_examples_were_actually_found():
    """Guard the parametrization: an empty glob would vacuously pass."""
    assert len(basics_examples()) > 5
    assert set(NETWORK_BOUND) < {p.stem for p in basics_examples()}
    assert set(EXAMPLE_ARGS) < {p.stem for p in basics_examples()}
