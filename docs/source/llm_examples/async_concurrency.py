"""Fork/join async concurrency with templates.

Demonstrates:
- Running multiple LLM template calls concurrently with ``asyncio.gather``
- Using ``asyncio.to_thread`` to run synchronous template calls in parallel
"""

import asyncio
import functools

from effectful.handlers.llm import Template

# ---------------------------------------------------------------------------
# Async template
# ---------------------------------------------------------------------------


@Template.define
def analyze_average_age(ages: list[int]) -> int:
    """Analyze the dataset of ages {ages} and return the average age of
    participants. Do not use any tools."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    async def run() -> None:
        analysis = functools.partial(asyncio.to_thread, analyze_average_age)
        results = await asyncio.gather(
            analysis([25, 30, 35, 40]),
            analysis([20, 28, 17, 30]),
            analysis([22, 27, 31, 29]),
            analysis([24, 26, 32, 38]),
            analysis([21, 29, 33, 37]),
        )
        for i, result in enumerate(results):
            print(f"Group {i}: average age = {result}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
