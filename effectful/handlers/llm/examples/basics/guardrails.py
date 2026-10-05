"""Travel advisor with input guardrails.

Demonstrates:
- A pre-condition on a skill's parameter: one skill's judgement, attached to
  the annotation as ``annotated_types.Predicate`` metadata, guarding another
  skill's input wherever that input comes from
- A post-condition on the return: an ordinary Python predicate, attached the
  same way, enforced by the decoder that reads the model's answer -- a
  rejection is fed back to the model, which then answers again
"""

import argparse
import typing

import annotated_types
import pydantic

from effectful.handlers.llm import Skill


@Skill.define
def is_safe_query(user_query: str) -> bool:
    """
    Determine whether the user's query is purely related to travel advice: {user_query}
    """


def is_concise_answer(answer: str) -> bool:
    """Determine whether the answer is concise (<100 words)."""
    return len(answer.split()) < 100


@Skill.define
def travel_query(
    user_query: typing.Annotated[str, annotated_types.Predicate(is_safe_query)],
) -> typing.Annotated[str, annotated_types.Predicate(is_concise_answer)]:
    """
    Produce a concise (<100 word) answer to: {user_query}
    """


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "What are great places to check out in NYC?",
            "Should I buy apple stocks?",
        ],
        metavar="QUERY",
        help="User queries to run through the travel-advice guardrail",
    )
    args = parser.parse_args()

    for query in args.queries:
        print(f"Query: {query}")
        try:
            print("Answer:", travel_query(query))
        except pydantic.ValidationError:
            # The guard reports only that its predicate failed, so the message
            # a person sees is the caller's to write.
            print(f"Rejected: '{query}' is not related to travel advice.")


if __name__ == "__main__":
    main()
