"""Tests for the module launcher and the contract it has with the examples.

``python -m effectful.handlers.llm.harness <script> <flags>`` splits one command
line between two consumers: the flags the launcher understands and the flags that
belong to the script. Both halves of that contract are tested here -- the split
itself, and the examples' side of it, which is that a script brings no handler
stack of its own because the launcher supplies one.

All of it is offline: argument parsing and an AST walk, no model and no network.
"""

import argparse
import pathlib

import pytest

from effectful.handlers.llm.harness.__main__ import (
    _parse_args,
    _provider_config,
    _reasoning_effort_choices,
)

EXAMPLES_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "docs" / "source" / "llm_examples"
)


def example_modules() -> list[pathlib.Path]:
    """Every example module, including the shared sibling libraries."""
    return sorted(
        p
        for p in EXAMPLES_DIR.rglob("*.py")
        if "__pycache__" not in p.parts and p.name != "__init__.py"
    )


# ============================================================================
# Splitting the command line
# ============================================================================


def test_parse_args_splits_harness_and_script_flags():
    ns, rest = _parse_args(["s.py", "--model", "m", "--render", "--depth", "3"])
    assert ns.script == "s.py"
    assert ns.model == "m"
    assert ns.render is True
    assert rest == ["--depth", "3"]


@pytest.mark.parametrize(
    "script_flag",
    [
        "--mode",  # a prefix of --model
        "--re",  # a prefix of --render and --reasoning-effort
        "--num",  # a prefix of --num-retries
        "--type",  # a prefix of --type-checker
        "--persist",  # a prefix of --persist-db
    ],
)
def test_parse_args_does_not_claim_abbreviated_script_flags(script_flag):
    """A script flag that merely *prefixes* a harness flag belongs to the script.

    argparse abbreviation matching would otherwise read ``--mode`` as ``--model``,
    silently overwriting the model with the script's argument and dropping the flag
    the script was passed.
    """
    ns, rest = _parse_args(["s.py", "--model", "gpt-4o-mini", script_flag, "v"])
    assert ns.model == "gpt-4o-mini"
    assert rest == [script_flag, "v"]


def test_parse_args_still_rejects_a_bad_harness_flag_value():
    """Turning abbreviation off does not turn validation off."""
    with pytest.raises(SystemExit):
        _parse_args(["s.py", "--type-checker", "not-a-checker"])


def test_parse_args_requires_a_script():
    with pytest.raises(SystemExit):
        _parse_args(["--model", "gpt-4o-mini"])


def test_reasoning_effort_is_unset_unless_asked_for():
    """No ``--reasoning-effort`` means the parameter is absent from the request.

    It used to default to the sentinel ``"default"``, which was then sent on
    every call -- and OpenAI rejects that value outright (``Unsupported value:
    'reasoning_effort' does not support 'default' with this model``), so every
    reasoning model was unusable through the launcher with no flag passed at
    all.
    """
    ns, _ = _parse_args(["s.py", "--model", "gpt-5-mini"])
    assert ns.reasoning_effort is None
    assert "reasoning_effort" not in _provider_config(ns)


def test_reasoning_effort_is_forwarded_when_asked_for():
    ns, _ = _parse_args(["s.py", "--model", "gpt-5-mini", "--reasoning-effort", "low"])
    assert _provider_config(ns)["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    "model", ["gpt-5-mini", "gpt-5", "gpt-4o-mini", "claude-sonnet-5", "not-a-model"]
)
def test_provider_config_leaves_other_models_alone(model):
    """Only the models that need the Responses API are rewritten. An unknown
    name is left alone rather than rewritten on a guess."""
    ns, _ = _parse_args(["s.py", "--model", model])
    assert _provider_config(ns)["model"] == model


def test_gpt_5_4_plus_is_routed_to_the_responses_api():
    """A GPT-5.4+ model with no ``--reasoning-effort`` is addressed through the
    Responses API.

    OpenAI rejects function tools alongside reasoning on
    ``/v1/chat/completions`` for these models, and the harness always sends
    tools -- but litellm bridges to ``/v1/responses`` only when
    ``reasoning_effort`` is not None, so omitting the parameter (correct in
    itself) also opts the model out of the endpoint its tool calls require.
    """
    ns, _ = _parse_args(["s.py", "--model", "gpt-5.6-terra"])
    config = _provider_config(ns)
    assert config["model"] == "openai/responses/gpt-5.6-terra"
    assert "reasoning_effort" not in config


def test_an_explicit_reasoning_effort_routes_itself():
    """`reasoning_effort` triggers litellm's bridge on its own, so the model
    name is left untouched and the requested effort is what is sent."""
    ns, _ = _parse_args(
        ["s.py", "--model", "gpt-5.6-terra", "--reasoning-effort", "low"]
    )
    config = _provider_config(ns)
    assert config["model"] == "gpt-5.6-terra"
    assert config["reasoning_effort"] == "low"


def test_reasoning_effort_rejects_the_sentinel_value():
    """``"default"`` appears in the ``Literal`` on ``litellm.completion``'s
    signature but not in litellm's canonical ``REASONING_EFFORT``, and no
    provider accepts it. Rejecting it at the CLI beats a 400 a request later."""
    assert "default" not in (_reasoning_effort_choices() or [])
    with pytest.raises(SystemExit):
        _parse_args(["s.py", "--reasoning-effort", "default"])


def test_parser_has_abbreviation_disabled():
    """Pin the mechanism, so a future flag added to a fresh parser cannot
    reintroduce the hijack without this failing."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    assert parser.allow_abbrev is False
    ns, _ = _parse_args(["s.py", "--mod", "x"])
    assert ns.model != "x"


def test_example_modules_were_actually_found():
    """Guard the parametrization above: an empty glob would vacuously pass."""
    assert len(example_modules()) > 20
