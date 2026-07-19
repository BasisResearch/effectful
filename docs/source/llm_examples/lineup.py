"""Solving hard problems by writing and running Python.

You are a careful problem solver and an expert Python programmer. You answer by
writing code, not by reasoning in prose alone: problems that are error-prone to
work out by hand are often easy to brute-force or verify with a short program.
"""

import argparse
import collections.abc
import contextlib
import dataclasses
import os
import pathlib
import typing

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
from effectful.ops.types import NotHandled


@dataclasses.dataclass(frozen=True)
class LineupClue:
    """
    A clue about the relative ordering of n people, numbered 0 to n - 1, in a line.
    Used to describe puzzles like the classic "zebra puzzle" represented in `solve_lineup`.
    Each `LineupClue` corresponds to a single ordering constraint, ``(kind, a, b)``.

    The meaning of ``a`` and ``b`` depends on ``kind``:

    - ``("at", a, k)``       -- person ``a`` is at position ``k``
    - ``("left", a, b)``     -- person ``a`` is somewhere left of person ``b``
    - ``("imm_left", a, b)`` -- person ``a`` is immediately left of person ``b``
    - ``("adj", a, b)``      -- persons ``a`` and ``b`` are in adjacent positions
    """

    kind: typing.Literal["at", "left", "imm_left", "adj"]
    a: int
    b: int


@Template.define
def solve_lineup(n: int, clues: collections.abc.Sequence[LineupClue]) -> list[int]:
    """Solve a 'zebra'-style ordering puzzle: place n={n} people, numbered 0 to
    n - 1, in a line in positions 1 to n (each position used once) so that every
    `LineupClue` in the following list holds:

    <clues>{clues}</clues>

    Every puzzle has exactly one consistent arrangement. Return the list of
    positions ``[position of 0, position of 1, ..., position of n - 1]``,
    as shown in the following worked examples:

    >>> solve_lineup(3, [LineupClue("at", 0, 1), LineupClue("left", 1, 2)])
    [1, 2, 3]
    >>> solve_lineup(4, [LineupClue("left", 0, 1), LineupClue("left", 1, 2), LineupClue("left", 2, 3)])
    [1, 2, 3, 4]
    >>> solve_lineup(4, [LineupClue("imm_left", 0, 1), LineupClue("at", 2, 4), LineupClue("left", 3, 0)])
    [2, 3, 4, 1]
    >>> solve_lineup(5, [LineupClue("at", 0, 3), LineupClue("imm_left", 1, 2), LineupClue("left", 3, 4), LineupClue("at", 4, 5)])
    [3, 1, 2, 4, 5]
    """


def main(args: argparse.Namespace) -> None:
    puzzle = [
        LineupClue("imm_left", 0, 1),
        LineupClue("imm_left", 1, 2),
        LineupClue("at", 3, 5),
        LineupClue("left", 4, 0),
    ]
    print(f"Zebra-style ordering puzzle: n=5, clues={puzzle}")
    print(f"Answer: {solve_lineup(5, puzzle)}")


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
