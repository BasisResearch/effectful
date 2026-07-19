"""Travel advisor with input guardrails.

Demonstrates:
- Using one template to validate/guard input before passing it to another
- Simple control-flow gating based on LLM classification
"""

import argparse

from effectful.handlers.llm import Template

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@Template.define
def travel_query(user_query: str) -> str:
    """
    Produce a concise (<100 word) answer to: {user_query}
    """


# ---------------------------------------------------------------------------
# Guarded agent
# ---------------------------------------------------------------------------


def answer_travel_query(user_query: str) -> str:
    """Only answer travel-related queries; reject everything else."""

    @Template.define
    def is_safe_query(user_query: str) -> bool:
        """
        Determine whether the user's query is purely related to travel advice: {user_query}
        """

    if is_safe_query(user_query):
        return travel_query(user_query)
    else:
        return f"Rejected: '{user_query}' is not related to travel advice."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        print(answer_travel_query(query))


if __name__ == "__main__":
    main()
