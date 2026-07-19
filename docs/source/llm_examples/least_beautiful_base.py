"""Solving hard problems by writing and running Python.

You are a careful problem solver and an expert Python programmer. You answer by
writing code, not by reasoning in prose alone: problems that are error-prone to
work out by hand are often easy to brute-force or verify with a short program.
"""

import argparse
import contextlib
import os
import pathlib

import tenacity

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import (
    LangfuseTracer,
    LexicalReaders,
    LiteLLMProvider,
    PythonRepl,
    RetryLLMHandler,
    SynthesizeAndCall,
    SystemPromptDumper,
    TerminalRenderer,
)
from effectful.handlers.llm.evaluation import UnsafeEvalProvider
from effectful.ops.semantics import handler


@Template.define
def least_beautiful_base(threshold: int) -> int:
    r"""Find the least integer base b >= 2 for which there are more than
    {threshold} ``b``-eautiful integers.

    A positive integer n is ``b``-eautiful if it has exactly two digits when
    written in base b and those two digits sum to ``sqrt(n)``. For example, 81
    is 13-eautiful because 81 = 6_3 in base 13 and 6 + 3 = sqrt(81).

    >>> least_beautiful_base(0)
    3
    >>> least_beautiful_base(1)
    7
    >>> least_beautiful_base(5)
    31
    >>> least_beautiful_base(7)
    211
    """


def main(args: argparse.Namespace) -> None:
    threshold = 10
    print(f"Least b with > {threshold} b-eautiful integers")
    print(f"Answer: {least_beautiful_base(threshold)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    with (
        handler(LiteLLMProvider(model=args.model, tool_choice="required", api_base="http://localhost:8030/v1", api_key="")),
        handler(TerminalRenderer()) if args.render else contextlib.nullcontext(),
        handler(SystemPromptDumper(path=pathlib.Path(args.dump_system_prompt)))
        if args.dump_system_prompt
        else contextlib.nullcontext(),
        handler(UnsafeEvalProvider()),
        handler(PythonRepl()),
        handler(SynthesizeAndCall()),
        handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(args.num_retries))),
        handler(LexicalReaders()),
        handler(LangfuseTracer()) if args.langfuse else contextlib.nullcontext(),
    ):
        main(args)
