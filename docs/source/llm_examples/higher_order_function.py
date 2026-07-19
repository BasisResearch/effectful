"""Generating higher-order functions that call other templates.

Demonstrates:
- A template returning a ``Callable``, evaluated via ``UnsafeEvalProvider``
- The synthesized function calling sub-templates (``write_chapter``,
  ``judge_chapter``) at runtime
- ``RetryLLMHandler`` to recover from transient validation/runtime errors
- ``inspect.getsource`` on the generated function
"""

import argparse
import inspect
from collections.abc import Callable
from typing import Literal

from effectful.handlers.llm import Template

# ---------------------------------------------------------------------------
# Sub-templates the generated function may call
# ---------------------------------------------------------------------------


@Template.define
def write_chapter(chapter_number: int, chapter_name: str) -> str:
    """Write a short story about {chapter_number}. Do not use any tools."""


@Template.define
def judge_chapter(story_so_far: str, chapter_number: int) -> bool:
    """Decide if the new chapter is coherent with the story so far. Do not use any tools."""


# ---------------------------------------------------------------------------
# Orchestrator template returning a callable
# ---------------------------------------------------------------------------


@Template.define
def write_multi_chapter_story(style: Literal["moral", "funny"]) -> Callable[[str], str]:
    """
    Generate a function that writes a story in style: {style} about the given topic.
    """


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic", type=str, default="a curious cat", help="Story topic"
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["moral", "funny"],
        default="moral",
        help="Story style",
    )
    args = parser.parse_args()

    print(f"\n=== Generating story function (style={args.style}) ===")
    story_fn = write_multi_chapter_story(args.style)
    print(inspect.getsource(story_fn))
    print(f"\n=== Running generated function on {args.topic!r} ===")
    print(story_fn(args.topic))


if __name__ == "__main__":
    main()
