"""Retrying tool execution failures.

Demonstrates:
- ``RetryLLMHandler`` surfacing tool exceptions back to the LLM as tool messages
- A flaky tool (``unstable_service``) that succeeds only after multiple attempts
- The contrast between an unhandled failure and a retry-handled success
"""

import argparse
import os

from tenacity import stop_after_attempt

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Flaky tool
# ---------------------------------------------------------------------------

call_count = 0
REQUIRED_RETRIES = 3


@Tool.define
def unstable_service() -> str:
    """Fetch data from an unstable external service. May require retries."""
    global call_count
    call_count += 1
    if call_count < REQUIRED_RETRIES:
        raise ConnectionError(
            f"Service unavailable! Attempt {call_count}/{REQUIRED_RETRIES}. Please retry."
        )
    return "{ 'status': 'ok', 'data': [1, 2, 3] }"


# ---------------------------------------------------------------------------
# Template (unstable_service auto-captured from lexical scope)
# ---------------------------------------------------------------------------


@Template.define
def fetch_data() -> str:
    """Use the unstable_service tool to fetch data."""
    raise NotHandled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retry LLM template calls when tools raise exceptions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=4,
        help="Number of retries for tool/decode failures",
    )
    args = parser.parse_args()

    provider = LiteLLMProvider(model=args.model)

    print("=== Without RetryLLMHandler ===")
    with handler(provider):
        try:
            result = fetch_data()
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

    # Reset for the retry-enabled run.
    call_count = 0

    print("\n=== With RetryLLMHandler ===")
    with (
        handler(provider),
        handler(RetryLLMHandler(stop=stop_after_attempt(args.num_retries))),
    ):
        result = fetch_data()
        print(f"Result: {result} (after {call_count} tool attempts)")
