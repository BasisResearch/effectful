"""Solving hard problems by writing and running Python.

You are a careful problem solver and an expert Python programmer. You answer by
writing code, not by reasoning in prose alone: problems that are error-prone to
work out by hand are often easy to brute-force or verify with a short program.
"""

import argparse
import collections.abc
import dataclasses
import typing

from effectful.handlers.llm import Template


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


def main() -> None:
    kinds = typing.get_args(LineupClue.__annotations__["kind"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of people in the line (used with --clue)",
    )
    parser.add_argument(
        "--clue",
        dest="clues",
        action="append",
        nargs=3,
        metavar=("KIND", "A", "B"),
        default=None,
        help=(
            f"An ordering constraint 'KIND A B' where KIND is one of "
            f"{'/'.join(kinds)} (e.g. --clue imm_left 0 1); repeatable"
        ),
    )
    args = parser.parse_args()

    if args.clues is not None:
        n = args.n
        clues = []
        for kind, a, b in args.clues:
            if kind not in kinds:
                parser.error(
                    f"invalid clue kind {kind!r}; choose from {'/'.join(kinds)}"
                )
            try:
                clues.append(LineupClue(kind, int(a), int(b)))
            except ValueError:
                parser.error(f"clue positions must be integers, got {a!r} {b!r}")
    else:
        n = 5
        clues = [
            LineupClue("imm_left", 0, 1),
            LineupClue("imm_left", 1, 2),
            LineupClue("at", 3, 5),
            LineupClue("left", 4, 0),
        ]

    print(f"Zebra-style ordering puzzle: n={n}, clues={clues}")
    print(f"Answer: {solve_lineup(n, clues)}")


if __name__ == "__main__":
    main()
