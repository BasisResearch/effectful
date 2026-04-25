"""Template composition: templates can call other templates.

Demonstrates:
- Sub-templates auto-captured into an orchestrator template's lexical scope
- Inspecting ``write_story.tools`` to confirm sub-templates are exposed to the LLM
- The orchestrator dispatches to the right sub-template based on a style argument
"""

import argparse
import os

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Sub-templates
# ---------------------------------------------------------------------------


@Template.define
def story_with_moral(topic: str) -> str:
    """Write a short story about {topic} and end with a moral lesson. Do not use any tools."""
    raise NotHandled


@Template.define
def story_funny(topic: str) -> str:
    """Write a funny, humorous story about {topic}. Do not use any tools."""
    raise NotHandled


# ---------------------------------------------------------------------------
# Orchestrator template
# ---------------------------------------------------------------------------


@Template.define
def write_story(topic: str, style: str) -> str:
    """Write a story about {topic} in the style: {style}.
    Available styles: 'moral' for a story with a lesson, 'funny' for humor.
    Use story_funny for humor, story_with_moral for a story with a lesson.
    """
    raise NotHandled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Template composition with auto-captured sub-templates"
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
    args = parser.parse_args()

    assert story_with_moral in write_story.tools.values()
    assert story_funny in write_story.tools.values()
    print("Sub-templates available to write_story:", list(write_story.tools.keys()))

    provider = LiteLLMProvider(model=args.model)
    with handler(provider):
        print("\n=== Story with moral ===")
        print(write_story(args.topic, "moral"))
        print("\n=== Funny story ===")
        print(write_story(args.topic, "funny"))
