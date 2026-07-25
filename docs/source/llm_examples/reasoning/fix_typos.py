"""Solving hard problems by writing and running Python.

You are a careful problem solver and an expert Python programmer. You answer by
writing code, not by reasoning in prose alone: problems that are error-prone to
work out by hand are often easy to brute-force or verify with a short program.
"""

import argparse

from effectful.handlers.llm import Template


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        type=str,
        default=(
            "We inctroduce a probablistic algorithm that estimates the "
            "timne-varying location in the presense of measurment noise."
        ),
        help="Text whose typos should be fixed",
    )
    args = parser.parse_args()
    print(f"Fix only the typos in:\n{args.text}")
    print(f"Answer: {fix_typos(args.text)}")


if __name__ == "__main__":
    main()
