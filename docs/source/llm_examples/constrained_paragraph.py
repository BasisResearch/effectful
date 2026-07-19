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
def constrained_paragraph(endings: list[str]) -> str:
    r"""Write a short paragraph whose sentences end, in order, with the words in
    {endings}: one sentence per word, each ending with that exact word.

    The examples below split the returned paragraph into sentences and compare the
    last word of each (lowercased, punctuation stripped) against the requested
    endings -- so a synthesized function must build text with the right shape:

    >>> import re
    >>> def endings_of(paragraph):
    ...     sents = [s for s in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if s]
    ...     return [re.findall(r"[A-Za-z']+", s)[-1].lower() for s in sents]
    >>> endings_of(constrained_paragraph(["walk", "tumbling", "another", "lunatic"]))
    ['walk', 'tumbling', 'another', 'lunatic']
    >>> endings_of(constrained_paragraph(["dawn", "river"]))
    ['dawn', 'river']
    """


def main(args: argparse.Namespace) -> None:
    endings = ["mountain", "whisper", "thunder"]
    print(f"Paragraph with sentences ending in {endings}")
    print(f"Answer: {constrained_paragraph(endings)}")


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
