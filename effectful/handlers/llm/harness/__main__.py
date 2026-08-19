"""
A reusable harness for running `effectful.handlers.llm` example scripts.

The example scripts under ``docs/source/llm_examples`` share a fixed stack of
handlers -- a LiteLLM provider, a Python REPL, retry/decoding logic, and so on --
that turns a bare `Skill`/`Agent` into something runnable. This module
factors that stack into a single object, `harness`, so the scripts themselves
carry none of the boilerplate.

Run as a module it becomes a command-line launcher that wraps an arbitrary
script in the same context::

    python -m effectful.handlers.llm.harness <path_to_script.py> <harness_flags> <script_flags>

Harness flags are consumed here; other flags pass through to the script unchanged.
"""

import argparse
import inspect
import os
import pdb
import runpy
import sys
import textwrap
import typing

import litellm

from effectful.handlers.llm.harness import harness
from effectful.ops.semantics import handler


def _reasoning_effort_choices() -> list[str] | None:
    """The ``reasoning_effort`` values ``litellm.completion`` declares.

    Extracted from the ``Optional[Literal[...]]`` annotation on the live
    signature so the CLI choices track litellm exactly across upgrades. Returns
    ``None`` (leave the flag unrestricted) if the annotation isn't a Literal we
    can read, so a shape change in litellm degrades to accepting any string
    rather than breaking the launcher.
    """
    try:
        annotation = (
            inspect.signature(litellm.completion)
            .parameters["reasoning_effort"]
            .annotation
        )
        # Optional[Literal[...]] -> unwrap the Union, then read the Literal args.
        literals = [
            v
            for arg in typing.get_args(annotation)
            for v in typing.get_args(arg)
            if isinstance(v, str)
        ]
        return literals or None
    except Exception:
        return None


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split ``argv`` into harness options and pass-through script flags."""
    parser = argparse.ArgumentParser(
        prog=f"python -m {__spec__.name}" if __spec__ else None,
        description=textwrap.dedent(__doc__),
    )
    parser.add_argument("script", help="Path to the script to run")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=5,
        help="Number of retries for malformed/failing LLM output",
    )
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Whether to log LLM calls and metadata to Langfuse",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Live-render the streaming message history in the terminal",
    )
    parser.add_argument(
        "--dump-system-prompt",
        type=str,
        default=None,
        metavar="PATH",
        help="Dump the assembled system prompt to this Markdown file",
    )
    parser.add_argument(
        "--tool-choice",
        type=str,
        default="auto",
        choices=["required", "auto", "none"],
        help="Whether to require, allow, or disable tool calls (none means disabled)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="default",
        choices=_reasoning_effort_choices(),
        help="Reasoning effort forwarded to litellm.completion",
    )
    parser.add_argument(
        "--eval-provider",
        type=str,
        default="builtin",
        choices=["builtin", "restricted", "none"],
        help="Provider that runs model-authored Python",
    )
    parser.add_argument(
        "--type-checker",
        type=str,
        default="ty",
        choices=["mypy", "ty", "none"],
        help="Handler that type-checks model-authored Python before it runs",
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Drop into pdb post-mortem on an unhandled error (like `python -m pdb`)",
    )
    parser.add_argument(
        "--persist-db",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Checkpoint persisted Agent state/history to this SQLite database "
            "(installs SQLitePersister)"
        ),
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    litellm.drop_params = True
    ns, script_args = _parse_args(sys.argv[1:] if argv is None else argv)
    # The script should see only its own flags, under its own name.
    sys.argv = [ns.script, *script_args]
    # Mirror `python <script>`: put the script's directory on sys.path so it can
    # import sibling modules (e.g. a shared environment definition) by absolute name.
    # `runpy.run_path` runs the file as `__main__` with no package, so relative
    # imports can't work and this dir would otherwise be off the path.
    sys.path.insert(0, os.path.dirname(os.path.abspath(ns.script)))
    h = harness(
        num_retries=ns.num_retries,
        langfuse=ns.langfuse,
        render=ns.render,
        dump_system_prompt=ns.dump_system_prompt,
        persist_db=ns.persist_db,
        eval_provider=ns.eval_provider,
        type_checker=ns.type_checker,
        model=ns.model,
        tool_choice=ns.tool_choice,
        reasoning_effort=ns.reasoning_effort,
    )
    with handler(h):
        if ns.pdb:
            try:
                runpy.run_path(ns.script, run_name="__main__")
            except BaseException:
                # Post-mortem while the handler stack is still installed, so live
                # handler/session state is inspectable at the debugger prompt.
                pdb.post_mortem()
        else:
            runpy.run_path(ns.script, run_name="__main__")


if __name__ == "__main__":
    main()
