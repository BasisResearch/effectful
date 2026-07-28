"""Solving hard problems by writing and running Python.

You are a careful problem solver and an expert Python programmer. You answer by
writing code, not by reasoning in prose alone: problems that are error-prone to
work out by hand are often easy to brute-force or verify with a short program.

Each template below generalizes a single problem from the 2024 AIME II over
one or more of its constants; passing the original contest constant recovers
the official answer (noted per template).
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


@Template.define
def root_of_unity_product(n: int) -> int:
    r"""Let omega != 1 be a primitive n-th root of unity, for n = {n}. Find the
    remainder when the product, over k = 0, ..., n - 1, of
    (2 - 2 * omega^k + omega^(2k)) is divided by 1000.

    >>> root_of_unity_product(3)
    13
    >>> root_of_unity_product(5)
    41
    >>> root_of_unity_product(7)
    113
    >>> root_of_unity_product(13)
    321
    """


@Template.define
def max_chip_placements(k: int) -> int:
    r"""There is a collection of k^2 indistinguishable black chips and k^2
    indistinguishable white chips, for k = {k}. Find the number of ways to
    place some of these chips in the k^2 unit cells of a k-by-k grid so that
    all chips in the same row and all chips in the same column have the same
    color, and any additional chip placed on the grid would violate one or
    more of the previous two conditions.

    >>> max_chip_placements(1)
    2
    >>> max_chip_placements(2)
    6
    >>> max_chip_placements(3)
    38
    >>> max_chip_placements(5)
    902
    """


@Template.define
def count_symmetric_triples(n: int, target: int) -> int:
    r"""Find the number of triples of nonnegative integers (a, b, c) satisfying
    a + b + c = {n} and
    a^2*b + a^2*c + b^2*a + b^2*c + c^2*a + c^2*b = {target}.

    >>> count_symmetric_triples(3, 6)
    7
    >>> count_symmetric_triples(6, 48)
    13
    >>> count_symmetric_triples(9, 162)
    19
    >>> count_symmetric_triples(300, 6000000)
    601
    """


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="problem", required=True)

    p14 = subparsers.add_parser("least-beautiful-base", help="2024 AIME II Problem 14")
    p14.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Find the least base with more than this many b-eautiful integers",
    )

    p13 = subparsers.add_parser("root-of-unity-product", help="2024 AIME II Problem 13")
    p13.add_argument(
        "--n",
        type=int,
        default=13,
        help="Order of the root of unity",
    )

    p9 = subparsers.add_parser("max-chip-placements", help="2024 AIME II Problem 9")
    p9.add_argument(
        "--k",
        type=int,
        default=5,
        help="Side length of the grid (and number of chips of each color, k^2)",
    )

    p11 = subparsers.add_parser(
        "count-symmetric-triples", help="2024 AIME II Problem 11"
    )
    p11.add_argument("--n", type=int, default=300, help="Required sum a + b + c")
    p11.add_argument(
        "--target",
        type=int,
        default=6_000_000,
        help="Required value of a^2 b + a^2 c + b^2 a + b^2 c + c^2 a + c^2 b",
    )

    args = parser.parse_args()

    if args.problem == "least-beautiful-base":
        print(f"Least b with > {args.threshold} b-eautiful integers")
        print(f"Answer: {least_beautiful_base(args.threshold)}")
    elif args.problem == "root-of-unity-product":
        print(f"Product over {args.n}-th roots of unity, mod 1000")
        print(f"Answer: {root_of_unity_product(args.n)}")
    elif args.problem == "max-chip-placements":
        print(f"Maximal chip placements on a {args.k}-by-{args.k} grid")
        print(f"Answer: {max_chip_placements(args.k)}")
    elif args.problem == "count-symmetric-triples":
        print(f"Triples with a + b + c = {args.n} and symmetric sum = {args.target}")
        print(f"Answer: {count_symmetric_triples(args.n, args.target)}")


if __name__ == "__main__":
    main()
