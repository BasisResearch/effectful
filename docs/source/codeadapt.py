"""CodeAdapt: solving hard problems by writing and running Python.

You are a careful problem solver and an expert Python programmer. You answer by
writing code, not by reasoning in prose alone: problems that are error-prone to
work out by hand are often easy to brute-force or verify with a short program.

You MUST use the ``submit_solution`` tool to give your final answer,
the harness will not accept a final answer in direct text.

You can use whatever other tools are available to develop your solution,
and refine incorrect attempts given feedback from failures of ``submit_solution``.
"""

import argparse
import contextlib
import os
import pathlib
from typing import Literal, NamedTuple

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
    raise NotHandled


class LineupClue(NamedTuple):
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

    kind: Literal["at", "left", "imm_left", "adj"]
    a: int
    b: int


@Template.define
def solve_lineup(n: int, clues: list[LineupClue]) -> list[int]:
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
    raise NotHandled


@Template.define
def countdown_reachable(numbers: list[int], target: int) -> bool:
    """In the Countdown numbers game, decide whether {target} can be made from
    {numbers}, using each number exactly once and combining them with + - * /
    (every intermediate division must come out exact).

    >>> countdown_reachable([2, 3, 5], 11)
    True
    >>> countdown_reachable([1, 1], 5)
    False
    >>> countdown_reachable([4, 7, 8, 9], 100)
    True
    >>> countdown_reachable([5, 5, 5], 3)
    False
    """
    raise NotHandled


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
    raise NotHandled


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
    raise NotHandled


@Template.define
def musr_object_placement(
    story: str, person: str, item: str, locations: list[str]
) -> str:
    """A MuSR object-placement question: a theory-of-mind puzzle. Read the story
    and decide, from {locations}, where {person} would look for the {item}.

    The answer is the last place {person} *saw* the {item}: the last move they
    watched, or any later moment they directly saw it somewhere; or its original
    location if they never saw it after that. A person's belief does not change
    while they are not watching, so where the {item} actually ends up and where
    {person} believes it is can differ.

    <story>{story}</story>

    >>> musr_object_placement(
    ...     "Danny set the earphones in the recording booth, then stepped out for a "
    ...     "call. While he was gone, Emma quietly moved them to the producer's desk.",
    ...     "Danny",
    ...     "earphones",
    ...     ["recording booth", "producer's desk"],
    ... )
    'recording booth'
    """
    raise NotHandled


def main(
    task: Literal["beautiful", "lineup", "countdown", "paragraph", "typos", "musr"],
) -> None:
    if task == "beautiful":
        threshold = 10
        print(f"Least b with > {threshold} b-eautiful integers")
        print(f"Answer: {least_beautiful_base(threshold)}")
    elif task == "lineup":
        puzzle = [
            LineupClue("imm_left", 0, 1),
            LineupClue("imm_left", 1, 2),
            LineupClue("at", 3, 5),
            LineupClue("left", 4, 0),
        ]
        print(f"Zebra-style ordering puzzle: n=5, clues={puzzle}")
        print(f"Answer: {solve_lineup(5, puzzle)}")
    elif task == "countdown":
        numbers, target = [3, 6, 25, 50], 147
        print(f"Countdown: reach {target} from {numbers}")
        print(f"Answer: {countdown_reachable(numbers, target)}")
    elif task == "paragraph":
        endings = ["mountain", "whisper", "thunder"]
        print(f"Paragraph with sentences ending in {endings}")
        print(f"Answer: {constrained_paragraph(endings)}")
    elif task == "typos":
        text = (
            "We inctroduce a probablistic algorithm that estimates the "
            "timne-varying location in the presense of measurment noise."
        )
        print(f"Fix only the typos in:\n{text}")
        print(f"Answer: {fix_typos(text)}")
    elif task == "musr":
        STUDIO_STORY = """\
In the heart of the bustling studio, Ricky, Emma, and Danny readied themselves \
for a day of creating magic. Ricky, the gifted singer-songwriter, had his \
precious notebook of lyrics on the producer's desk. Emma, their producer, was \
cognizant of the notebook's place at her desk. Across the room, Danny, the studio \
assistant, kept the earphones in the recording booth. They were all aware of the \
arrangement -- the notebook on the producer's desk, the earphones in the \
recording booth.

Ricky gently places his notebook onto the piano, then becomes engrossed in \
perfecting his song. Emma, engrossed in her thoughts, deftly moves the earphones \
to the producer's desk. At that moment Danny was in a stirring conversation with a \
visiting sound engineer; the visitor stood blocking Danny's general overview of \
the studio space.

Later, delicately lifting Ricky's notebook, Danny orchestrates its move to the \
producer's desk. At the desk, he glimpses a pair of earphones indirectly drawing \
his attention amidst his routine of tidying up. Meanwhile Emma, from inside a \
sound-proofed booth, was lost in reviewing already-recorded tracks, out of \
Danny's view."""
        person = "Danny"
        item = "earphones"
        locations = ["piano", "producer's desk", "recording booth"]
        answer = musr_object_placement(STUDIO_STORY, person, item, locations)
        print(f"MuSR: where would {person} look for the {item}?")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CodeAdapt: solve hard problems by writing and running code"
    )
    parser.add_argument(
        "--task",
        choices=("beautiful", "lineup", "countdown", "paragraph", "typos", "musr"),
        default="beautiful",
        help="Which problem to solve",
    )
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
        handler(LiteLLMProvider(model=args.model, tool_choice="required")),
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
        main(args.task)
