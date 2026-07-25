"""Solving hard problems by writing and running Python.

You are a careful problem solver and an expert Python programmer. You answer by
writing code, not by reasoning in prose alone: problems that are error-prone to
work out by hand are often easy to brute-force or verify with a short program.
"""

import argparse

from effectful.handlers.llm import Template


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endings",
        nargs="+",
        default=["mountain", "whisper", "thunder"],
        metavar="WORD",
        help="Words each sentence must end with, in order",
    )
    args = parser.parse_args()
    print(f"Paragraph with sentences ending in {args.endings}")
    print(f"Answer: {constrained_paragraph(args.endings)}")


if __name__ == "__main__":
    main()
