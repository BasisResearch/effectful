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
def fix_typos(text: str) -> str:
    """Output the following text exactly, with no changes at all except for fixing
    the misspellings. Leave every other stylistic decision -- commas, US vs British
    spellings, capitalization, line breaks -- exactly as in the original:

    <text>{text}</text>

    Only misspelled words may change; every correctly spelled word and all
    punctuation and whitespace must be preserved verbatim. Identify the typos, then
    apply the corrections with code so that nothing else can drift.

    >>> fix_typos("We inctroduce a probablistic method in the presense of noise.")
    'We introduce a probabilistic method in the presence of noise.'
    >>> fix_typos("Teh quick borwn fox jumpps over the lazy dog.")
    'The quick brown fox jumps over the lazy dog.'
    """


def main(args: argparse.Namespace) -> None:
    text = (
        "We inctroduce a probablistic algorithm that estimates the "
        "timne-varying location in the presense of measurment noise."
    )
    print(f"Fix only the typos in:\n{text}")
    print(f"Answer: {fix_typos(text)}")


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
