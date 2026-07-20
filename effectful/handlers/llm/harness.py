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
``--dump-system-prompt``) are consumed here; every other flag is passed through
to the script unchanged.
"""

import argparse
import contextlib
import os
import pathlib
import runpy
import sys

import tenacity

from effectful.handlers.llm.completions import (
    LangfuseTracer,
    LiteLLMProvider,
    PythonRepl,
    RetryLLMHandler,
    SynthesizeAndCall,
    SystemPromptDumper,
    TerminalRenderer,
)
from effectful.handlers.llm.evaluation import UnsafeEvalProvider
from effectful.ops.semantics import handler


class harness(contextlib.ContextDecorator):
    """Install the standard `effectful.handlers.llm` handler stack.

    Constructing a `harness` records the configuration; entering it (as a
    context manager, decorator, or via the module CLI) installs the handlers and
    exiting removes them. The handlers, in installation order, are:

    1. `LiteLLMProvider` -- the model backend.
    2. `TerminalRenderer` -- live-render the streaming history (if ``render``).
    3. `SystemPromptDumper` -- dump the system prompt (if ``dump_system_prompt``).
    4. `UnsafeEvalProvider` and `PythonRepl` -- run model-authored Python.
    5. `SynthesizeAndCall` -- synthesize a function and call it.
    6. `RetryLLMHandler` -- retry malformed/failing model output.
    7. `LexicalReaders` -- expose lexically-scoped tools to the model.
    8. `LangfuseTracer` -- log calls to Langfuse (if ``langfuse``).

    Args:
        model: LLM model to use.
        num_retries: Attempts for malformed/failing model output.
        langfuse: Log LLM calls and metadata to Langfuse.
        render: Live-render the streaming message history in the terminal.
        dump_system_prompt: If set, dump the assembled system prompt to this
            Markdown file.
        tool_choice: ``tool_choice`` forwarded to the provider.
        api_base: API base URL forwarded to the provider.
        api_key: API key forwarded to the provider.
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
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.num_retries = num_retries
        self.langfuse = langfuse
        self.render = render
        self.dump_system_prompt = dump_system_prompt
        self.tool_choice = tool_choice
        self.api_base = api_base
        self.api_key = api_key

    def __enter__(self) -> "harness":
        stack = contextlib.ExitStack()
        stack.enter_context(
            handler(
                LiteLLMProvider(
                    model=self.model,
                    tool_choice=self.tool_choice,
                    api_base=self.api_base,
                    api_key=self.api_key,
                )
            )
        )
        if self.render:
            stack.enter_context(handler(TerminalRenderer()))
        if self.dump_system_prompt:
            stack.enter_context(
                handler(SystemPromptDumper(path=pathlib.Path(self.dump_system_prompt)))
            )
        stack.enter_context(handler(UnsafeEvalProvider()))
        stack.enter_context(handler(PythonRepl()))
        stack.enter_context(handler(SynthesizeAndCall()))
        stack.enter_context(
            handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(self.num_retries)))
        )
        # stack.enter_context(handler(LexicalReaders()))
        if self.langfuse:
            stack.enter_context(handler(LangfuseTracer()))
        self._stack = stack
        return self

    def __exit__(self, *exc_info) -> bool | None:
        return self._stack.__exit__(*exc_info)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split ``argv`` into harness options and pass-through script flags."""
    parser = argparse.ArgumentParser(
        prog=f"python -m {__spec__.name}" if __spec__ else None,
        description=(
            "Run an effectful.handlers.llm script under the standard handler "
            "stack. Flags other than the harness flags below are passed through "
            "to the script unchanged."
        ),
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
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns, script_args = _parse_args(sys.argv[1:] if argv is None else argv)
    # The script should see only its own flags, under its own name.
    sys.argv = [ns.script, *script_args]
    with harness(
        model=ns.model,
        num_retries=ns.num_retries,
        langfuse=ns.langfuse,
        render=ns.render,
        dump_system_prompt=ns.dump_system_prompt,
        tool_choice=ns.tool_choice,
        api_base=os.environ.get("DS4_OPENAI_API_BASE", None)
        if ns.model == "openai/deepseek-v4-flash"
        else None,
        api_key=os.environ.get("DS4_OPENAI_API_KEY", None)
        if ns.model == "openai/deepseek-v4-flash"
        else None,
    ):
        runpy.run_path(ns.script, run_name="__main__")


if __name__ == "__main__":
    main()
