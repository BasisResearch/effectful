"""A reusable harness for running `effectful.handlers.llm` example scripts.

The example scripts under ``docs/source/llm_examples`` share a fixed stack of
handlers -- a LiteLLM provider, a Python REPL, retry/decoding logic, and so on --
that turns a bare `Template`/`Agent` into something runnable. This module
factors that stack into a single object, `harness`, so the scripts themselves
carry none of the boilerplate.

`harness` is a `contextlib.ContextDecorator`, so it can be used programmatically
either as a context manager or as a decorator::

    with harness(model="gpt-4o", render=True):
        main()

    @harness(model="gpt-4o")
    def main() -> None:
        ...

Run as a module it becomes a command-line launcher that wraps an arbitrary
script in the same context::

    python -m effectful.handlers.llm.harness <path_to_script.py> <harness_flags> <script_flags>

Harness flags (``--model``, ``--num-retries``, ``--langfuse``, ``--render``,
``--dump-system-prompt``, ``--tool-choice``, ``--reasoning-effort``,
``--eval-provider``, ``--type-checker``, ``--pdb``, ``--persist-db``) are consumed
here; every other flag is passed through to the script unchanged.
"""

import argparse
import contextlib
import inspect
import os
import pathlib
import pdb
import runpy
import sys
import textwrap
import typing

import litellm
import tenacity

from effectful.handlers.llm.harness.context import FrameworkDocumenter
from effectful.handlers.llm.harness.durability import TenacityRetryer
from effectful.handlers.llm.harness.execution.builtin import BuiltinExecutor
from effectful.handlers.llm.harness.execution.mypy import MypyTypeChecker
from effectful.handlers.llm.harness.execution.restricted import (
    RestrictedPythonExecutor,
)
from effectful.handlers.llm.harness.observability.dumping import SystemPromptDumper
from effectful.handlers.llm.harness.observability.rendering import (
    RichTerminalRenderer,
)
from effectful.handlers.llm.harness.observability.tracing import LangfuseTracer
from effectful.handlers.llm.harness.persistence import SQLitePersister
from effectful.handlers.llm.harness.provision import (
    LiteLLMProvider,
)
from effectful.handlers.llm.harness.synthesis.body import (
    FinalBodySynthesizer,
)
from effectful.handlers.llm.harness.synthesis.snippet import StatefulReplSynthesizer
from effectful.handlers.llm.harness.transaction import HistoryBuilder
from effectful.ops.semantics import handler

# The providers that run model-authored Python, by the name the CLI knows them
# under: `unsafe` runs it with the plain interpreter, `restricted` under
# RestrictedPython's language subset and a guarded environment.
EVAL_PROVIDERS: dict[str, typing.Callable[[], typing.Any]] = {
    "unsafe": BuiltinExecutor,
    "restricted": RestrictedPythonExecutor,
}

# The handlers that type-check model-authored Python before it is compiled and
# run, by the name the CLI knows them under. Independent of `EVAL_PROVIDERS`:
# either checker composes with either provider. `none` installs nothing, leaving
# the `type_check` operation's default rule, which passes everything.
TYPE_CHECKERS: dict[str, typing.Callable[[], typing.Any] | None] = {
    "mypy": MypyTypeChecker,
    "none": None,
}


class harness(contextlib.ContextDecorator):
    """Install the standard `effectful.handlers.llm` handler stack.

    Constructing a `harness` records the configuration; entering it (as a
    context manager, decorator, or via the module CLI) installs the handlers and
    exiting removes them. The handlers, in installation order, are:

    1. `LiteLLMProvider` -- the model backend.
    2. `FrameworkDocumenter` -- describe the framework's concepts in the system
       prompt.
    3. `HistoryBuilder` -- accumulate the message history of a call.
    4. `RichTerminalRenderer` -- live-render the streaming history (if ``render``).
    5. `SystemPromptDumper` -- dump the system prompt (if ``dump_system_prompt``).
    6. The ``type_checker`` (`TYPE_CHECKERS`), the ``eval_provider``
       (`EVAL_PROVIDERS`) and `StatefulReplSynthesizer` -- check and run
       model-authored Python.
    7. `FinalBodySynthesizer` -- synthesize a function and call it.
    8. `TenacityRetryer` -- retry malformed/failing model output.
    9. `LexicalReaders` -- expose lexically-scoped tools to the model.
    10. `SQLitePersister` -- checkpoint a persisted `Agent`'s state/history to
        SQLite after each successful call (if ``persist_db``).
    11. `LangfuseTracer` -- log calls to Langfuse (if ``langfuse``).

    Args:
        model: LLM model to use.
        num_retries: Attempts for malformed/failing model output.
        langfuse: Log LLM calls and metadata to Langfuse.
        render: Live-render the streaming message history in the terminal.
        dump_system_prompt: If set, dump the assembled system prompt to this
            Markdown file.
        tool_choice: ``tool_choice`` forwarded to the provider.
        reasoning_effort: ``reasoning_effort`` forwarded to the provider (and on
            to ``litellm.completion``); omitted from requests when ``None``.
        api_base: API base URL forwarded to the provider.
        api_key: API key forwarded to the provider.
        persist_db: If set, path to a SQLite database used to checkpoint a
            persisted `~effectful.handlers.llm.template.Agent`'s (one
            constructed with an explicit `agent_id`) state and history via
            `~effectful.handlers.llm.completions.SQLitePersister`.
        eval_provider: Which provider runs model-authored Python -- a key of
            `EVAL_PROVIDERS` (``"unsafe"`` or ``"restricted"``).
        type_checker: Which handler type-checks model-authored Python before it
            runs -- a key of `TYPE_CHECKERS` (``"mypy"``, or ``"none"`` to run
            generated code unchecked).
    """

    def __init__(
        self,
        *,
        model: str = "",
        num_retries: int = 5,
        langfuse: bool = False,
        render: bool = False,
        dump_system_prompt: str | os.PathLike[str] | None = None,
        tool_choice: str = "auto",
        reasoning_effort: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        persist_db: str | os.PathLike[str] | None = None,
        eval_provider: str = "unsafe",
        type_checker: str = "mypy",
    ) -> None:
        self.model = model
        self.num_retries = num_retries
        self.langfuse = langfuse
        self.render = render
        self.dump_system_prompt = dump_system_prompt
        self.tool_choice = tool_choice
        self.reasoning_effort = reasoning_effort
        self.api_base = api_base
        self.api_key = api_key
        self.persist_db = persist_db
        self.eval_provider = eval_provider
        self.type_checker = type_checker

    def __enter__(self) -> "harness":
        stack = contextlib.ExitStack()
        # Only forward `reasoning_effort` when set, so we don't send
        # `reasoning_effort=None` on every request to providers that reject it.
        provider_config: dict[str, str] = {}
        if self.reasoning_effort is not None:
            provider_config["reasoning_effort"] = self.reasoning_effort
        stack.enter_context(
            handler(
                LiteLLMProvider(
                    model=self.model,
                    tool_choice=self.tool_choice,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    **provider_config,
                )
            )
        )
        stack.enter_context(handler(FrameworkDocumenter()))
        stack.enter_context(handler(HistoryBuilder()))
        if self.render:
            stack.enter_context(handler(RichTerminalRenderer()))
        if self.dump_system_prompt:
            stack.enter_context(
                handler(SystemPromptDumper(path=pathlib.Path(self.dump_system_prompt)))
            )
        type_checker = TYPE_CHECKERS[self.type_checker]
        if type_checker is not None:
            stack.enter_context(handler(type_checker()))
        stack.enter_context(handler(EVAL_PROVIDERS[self.eval_provider]()))
        stack.enter_context(handler(StatefulReplSynthesizer()))
        stack.enter_context(handler(FinalBodySynthesizer()))
        stack.enter_context(
            handler(TenacityRetryer(stop=tenacity.stop_after_attempt(self.num_retries)))
        )
        # stack.enter_context(handler(LexicalReaders()))
        if self.persist_db is not None:
            stack.enter_context(handler(SQLitePersister(pathlib.Path(self.persist_db))))
        if self.langfuse:
            stack.enter_context(handler(LangfuseTracer()))
        self._stack = stack
        return self

    def __exit__(self, *exc_info) -> bool | None:
        return self._stack.__exit__(*exc_info)


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
        default=None,
        choices=_reasoning_effort_choices(),
        help="Reasoning effort forwarded to litellm.completion",
    )
    parser.add_argument(
        "--eval-provider",
        type=str,
        default="unsafe",
        choices=sorted(EVAL_PROVIDERS),
        help="Provider that runs model-authored Python",
    )
    parser.add_argument(
        "--type-checker",
        type=str,
        default="mypy",
        choices=sorted(TYPE_CHECKERS),
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
    ns, script_args = _parse_args(sys.argv[1:] if argv is None else argv)
    # The script should see only its own flags, under its own name.
    sys.argv = [ns.script, *script_args]
    # Mirror `python <script>`: put the script's directory on sys.path so it can
    # import sibling modules (e.g. a shared environment definition) by absolute name.
    # `runpy.run_path` runs the file as `__main__` with no package, so relative
    # imports can't work and this dir would otherwise be off the path.
    sys.path.insert(0, os.path.dirname(os.path.abspath(ns.script)))
    with harness(
        model=ns.model,
        num_retries=ns.num_retries,
        langfuse=ns.langfuse,
        render=ns.render,
        dump_system_prompt=ns.dump_system_prompt,
        tool_choice=ns.tool_choice,
        reasoning_effort=ns.reasoning_effort,
        api_base=os.environ.get("DS4_OPENAI_API_BASE", None)
        if ns.model == "openai/deepseek-v4-flash"
        else None,
        api_key=os.environ.get("DS4_OPENAI_API_KEY", None)
        if ns.model == "openai/deepseek-v4-flash"
        else None,
        persist_db=ns.persist_db,
        eval_provider=ns.eval_provider,
        type_checker=ns.type_checker,
    ):
        if ns.pdb:
            try:
                runpy.run_path(ns.script, run_name="__main__")
            except BaseException:
                # Post-mortem while the handler stack is still installed, so live
                # handler/session state is inspectable at the debugger prompt.
                pdb.post_mortem()
        else:
            runpy.run_path(ns.script, run_name="__main__")
