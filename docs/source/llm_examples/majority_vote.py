"""Majority voting ensemble.

Demonstrates:
- Running the same template multiple times and taking a majority vote
- ``collections.Counter`` for tallying responses
"""

import argparse
import collections
import collections.abc
import enum

from effectful.handlers.llm import Template

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


class Answer(enum.StrEnum):
    yes = "yes"
    no = "no"
    maybe = "maybe"


@Template.define
def yes_or_no(question: str) -> Answer:
    """
    Answer the following yes/no/maybe question: {question}
    """


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------


def majority_vote[Q](
    oracle: collections.abc.Callable[[Q], Answer], query: Q, voters: int = 3
) -> tuple[Answer, int]:
    """Call ``oracle(query)`` multiple times and return the most common answer."""
    counter = collections.Counter(oracle(query) for _ in range(voters))
    return counter.most_common(1)[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-voters", type=int, default=3, help="Number of voters for majority vote"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Is Paris the capital of France?",
        help="Yes/no question to ask",
    )
    args = parser.parse_args()

    answer, count = majority_vote(yes_or_no, args.question, voters=args.num_voters)
    print(
        f"Question: {args.question}\nAnswer: {answer} (voted {count}/{args.num_voters})"
    )


if __name__ == "__main__":
    main()
