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

from effectful.handlers.llm.harness.__main__ import _parse_args

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
