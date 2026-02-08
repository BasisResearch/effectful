"""Travel advisor with input guardrails.

Demonstrates:
- Using one template to validate/guard input before passing it to another
- Simple control-flow gating based on LLM classification
"""

import argparse
import os

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@Template.define
def travel_query(user_query: str) -> str:
    """
    Produce a concise (<100 word) answer to: {user_query}
    """
    raise NotHandled


@Template.define
def is_safe_query(user_query: str) -> bool:
    """
    Determine whether the user's query is purely related to travel advice: {user_query}
    """
    raise NotHandled


# ---------------------------------------------------------------------------
# Guarded agent
# ---------------------------------------------------------------------------


def answer_travel_query(user_query: str) -> str:
    """Only answer travel-related queries; reject everything else."""
    if is_safe_query(user_query):
        return travel_query(user_query)
    else:
        return f"Rejected: '{user_query}' is not related to travel advice."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze average ages concurrently")
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
    args = parser.parse_args()

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    provider = LiteLLMProvider(model=args.model)
    with handler(provider), handler(RetryLLMHandler(num_retries=5)):
        print(answer_travel_query("What are great places to check out in NYC?"))
        print(answer_travel_query("Should I buy apple stocks?"))
