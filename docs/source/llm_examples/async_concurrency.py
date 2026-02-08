"""Fork/join async concurrency with templates.

Demonstrates:
- Running multiple LLM template calls concurrently with ``asyncio.gather``
- Using ``asyncio.to_thread`` to run synchronous template calls in parallel
"""

import argparse
import asyncio
import functools
import os

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Async template
# ---------------------------------------------------------------------------


@Template.define
def analyze_average_age(ages: list[int]) -> int:
    """Analyze the dataset of ages {ages} and return the average age of
    participants. Do not use any tools."""
    raise NotHandled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(provider: LiteLLMProvider):
    analysis = functools.partial(
        asyncio.to_thread, handler(provider)(analyze_average_age)
    )
    results = await asyncio.gather(
        analysis([25, 30, 35, 40]),
        analysis([20, 28, 17, 30]),
        analysis([22, 27, 31, 29]),
        analysis([24, 26, 32, 38]),
        analysis([21, 29, 33, 37]),
    )
    for i, result in enumerate(results):
        print(f"Group {i}: average age = {result}")


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
    asyncio.run(main(provider))
