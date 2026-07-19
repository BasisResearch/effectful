"""
In-context learning to solve problems with code across a conversation.
"""

import argparse
import collections.abc
import contextlib
import os
import pathlib

import tenacity

from effectful.handlers.llm import Agent, Template
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


class CountdownSolver(Agent):
    """
    You are a careful problem solver and an expert Python programmer. You answer by
    writing code, not by reasoning in prose alone: problems that are error-prone to
    work out by hand are often easy to brute-force or verify with a short program.
    """

    @Template.define
    def solve(self, numbers: collections.abc.Sequence[int], target: int) -> bool:
        """In the Countdown numbers game, decide whether {target} can be made from
        {numbers}, using each number exactly once and combining them with + - * /
        (every intermediate division must come out exact).

        >>> agent = CountdownSolver()
        >>> agent.solve([2, 3, 5], 11)
        True
        >>> agent.solve([1, 1], 5)
        False
        >>> agent.solve([4, 7, 8, 9], 100)
        True
        >>> agent.solve([5, 5, 5], 3)
        False
        """


def main(args: argparse.Namespace) -> None:
    agent = CountdownSolver()
    # Fresh examples (none appear in the docstring doctests), each paired with its
    # known-correct answer so we can validate the agent's output.
    test_examples: list[tuple[list[int], int, bool]] = [
        ([3, 6, 25, 50], 147, True),  # (50 - 25) * 6 - 3
        ([1, 2, 3, 4], 24, True),  # 1 * 2 * 3 * 4
        ([2, 4, 8], 9, False),  # all-even operands can never reach an odd target
    ]
    for numbers, target, expected in test_examples:
        print(f"Testing solve({numbers}, {target})...")
        answer = agent.solve(numbers, target)
        status = "OK" if answer == expected else "WRONG"
        print(
            f"[{status}] solve({numbers}, {target}): {answer} (expected {expected})"
        )
        assert answer == expected, (
            f"solve({numbers}, {target}) = {answer}, expected {expected}"
        )


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
