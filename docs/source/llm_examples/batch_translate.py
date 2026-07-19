"""Batch translation with instruction injection.

Demonstrates:
- ``@Template.define`` for a translation template with injected instructions
"""

import argparse
import os

from tenacity import stop_after_attempt

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.handlers.llm.evaluation import RestrictedEvalProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Translation template
# ---------------------------------------------------------------------------


@Template.define
def translate(target_language: str, instructions: str = "") -> Template[[str], str]:
    """
    Write a `Template` that translates a string of English text into {target_language}
    If any instructions are provided, include them in the prompt: {instructions}
    """
    raise NotHandled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch translation with instruction injection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of steps before giving up",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=5,
        help="Number of retries for malformed LLM output",
    )
    args = parser.parse_args()

    provider = LiteLLMProvider(model=args.model)

    with (
        handler(provider),
        handler(RetryLLMHandler(stop=stop_after_attempt(args.num_retries))),
        handler(RestrictedEvalProvider()),
    ):
        translator = translate(
            target_language="french", instructions="Use formal language."
        )
        print(translator("hello, how are you? how is your day going?"))
