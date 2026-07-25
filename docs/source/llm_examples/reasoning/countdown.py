"""
In-context learning to solve problems with code across a conversation.
"""

import argparse
import collections.abc

from effectful.handlers.llm import Agent, Template


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numbers",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Numbers to combine (used with --target for a single problem)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target value to make from --numbers",
    )
    args = parser.parse_args()
    if (args.numbers is None) != (args.target is None):
        parser.error("--numbers and --target must be given together")

    agent = CountdownSolver()

    # A custom problem has no known answer to validate against, so just solve it.
    if args.numbers is not None:
        print(f"Testing solve({args.numbers}, {args.target})...")
        answer = agent.solve(args.numbers, args.target)
        print(f"solve({args.numbers}, {args.target}): {answer}")
        return

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
        print(f"[{status}] solve({numbers}, {target}): {answer} (expected {expected})")
        assert answer == expected, (
            f"solve({numbers}, {target}) = {answer}, expected {expected}"
        )


if __name__ == "__main__":
    main()
