"""Template composition: templates can call other templates.

Demonstrates:
- Sub-templates auto-captured into an orchestrator template's lexical scope
- Inspecting ``write_story.tools`` to confirm sub-templates are exposed to the LLM
- The orchestrator dispatches to the right sub-template based on a style argument
"""

import argparse

from effectful.handlers.llm import Template

# ---------------------------------------------------------------------------
# Sub-templates
# ---------------------------------------------------------------------------


@Template.define
def story_with_moral(topic: str) -> str:
    """Write a short story about {topic} and end with a moral lesson. Do not use any tools."""


@Template.define
def story_funny(topic: str) -> str:
    """Write a funny, humorous story about {topic}. Do not use any tools."""


# ---------------------------------------------------------------------------
# Orchestrator template
# ---------------------------------------------------------------------------


@Template.define
def write_story(topic: str, style: str) -> str:
    """Write a story about {topic} in the style: {style}.
    Available styles: 'moral' for a story with a lesson, 'funny' for humor.
    Use story_funny for humor, story_with_moral for a story with a lesson.
    """


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Template composition with auto-captured sub-templates"
    )
    parser.add_argument(
        "--topic", type=str, default="a curious cat", help="Story topic"
    )
    args = parser.parse_args()

    print("\n=== Story with moral ===")
    print(write_story(args.topic, "moral"))
    print("\n=== Funny story ===")
    print(write_story(args.topic, "funny"))


if __name__ == "__main__":
    main()
