"""Majority voting ensemble.

Demonstrates:
- Running the same template multiple times and taking a majority vote
- ``collections.Counter`` for tallying responses
"""

import argparse
import collections
import collections.abc
import enum
import os

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

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
    raise NotHandled


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Majority voting ensemble for yes/no questions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
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

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    provider = LiteLLMProvider(model=args.model)
    with handler(provider):
        answer, count = majority_vote(yes_or_no, args.question, voters=args.num_voters)
        print(
            f"Question: {args.question}\nAnswer: {answer} (voted {count}/{args.num_voters})"
        )
