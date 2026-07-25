"""Solving hard problems by writing and running Python.

You are a careful problem solver and an expert Python programmer. You answer by
writing code, not by reasoning in prose alone: problems that are error-prone to
work out by hand are often easy to brute-force or verify with a short program.
"""

import argparse

from effectful.handlers.llm import Template


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Find the least base with more than this many b-eautiful integers",
    )
    args = parser.parse_args()
    print(f"Least b with > {args.threshold} b-eautiful integers")
    print(f"Answer: {least_beautiful_base(args.threshold)}")


if __name__ == "__main__":
    main()
