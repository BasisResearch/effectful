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
import os
from collections.abc import Callable
from typing import Literal

from tenacity import stop_after_attempt

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.handlers.llm.evaluation import UnsafeEvalProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Sub-templates the generated function may call
# ---------------------------------------------------------------------------


@Template.define
def write_chapter(chapter_number: int, chapter_name: str) -> str:
    """Write a short story about {chapter_number}. Do not use any tools."""
    raise NotHandled


@Template.define
def judge_chapter(story_so_far: str, chapter_number: int) -> bool:
    """Decide if the new chapter is coherent with the story so far. Do not use any tools."""
    raise NotHandled


# ---------------------------------------------------------------------------
# Orchestrator template returning a callable
# ---------------------------------------------------------------------------


@Template.define
def write_multi_chapter_story(style: Literal["moral", "funny"]) -> Callable[[str], str]:
    """Generate a function that writes a story in style: {style} about the given topic.

    If you raise an exception, handle it yourself.
    The program can use helper functions defined elsewhere (DO NOT REDEFINE THEM):
    - write_chapter(chapter_number: int, chapter_name: str) -> str
    - judge_chapter(story_so_far: str, chapter_number: int) -> bool
    """
    raise NotHandled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a higher-order function that calls sub-templates"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
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
    parser.add_argument(
        "--num-retries",
        type=int,
        default=4,
        help="Number of retries for malformed LLM output",
    )
    args = parser.parse_args()

    provider = LiteLLMProvider(model=args.model)

    print("Sub-templates available to write_multi_chapter_story:")
    print(list(write_multi_chapter_story.tools.keys()))

    with (
        handler(RetryLLMHandler(stop=stop_after_attempt(args.num_retries))),
        handler(provider),
        handler(UnsafeEvalProvider()),
    ):
        print(f"\n=== Generating story function (style={args.style}) ===")
        story_fn = write_multi_chapter_story(args.style)
        print(inspect.getsource(story_fn))
        print(f"\n=== Running generated function on {args.topic!r} ===")
        print(story_fn(args.topic))
