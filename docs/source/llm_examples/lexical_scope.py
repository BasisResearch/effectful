"""Composition via lexical scope: auto-captured sub-templates, invoked two ways.

Demonstrates:
- Module-level @Template.define sub-templates auto-captured into other templates'
  lexical scope, with no explicit registration
- An Agent grouping @Tool.define tools with a @Template.define orchestrator that
  calls those tools and the sub-templates directly (model-driven composition)
- A template returning a Callable: the model synthesizes a function that calls the
  same sub-templates when run (code-driven composition), via the eval provider
- inspect.getsource on the synthesized function
"""

import argparse
import inspect
from collections.abc import Callable
from typing import Literal

from effectful.handlers.llm import Agent, Template, Tool

# ---------------------------------------------------------------------------
# Sub-templates (module-level; auto-captured into the scopes below)
# ---------------------------------------------------------------------------


@Template.define
def story_with_moral(topic: str) -> str:
    """Write a short story about {topic} and end with a moral lesson."""


@Template.define
def story_funny(topic: str) -> str:
    """Write a funny, humorous story about {topic}."""


# ---------------------------------------------------------------------------
# (1) Model-driven composition: an orchestrator template calls tools and
#     sub-templates directly during its own turn.
# ---------------------------------------------------------------------------


class TripPlanner(Agent):
    """Plans a trip to a city with good weather and tells a story about visiting it."""

    @Tool.define
    def cities(self) -> list[str]:
        """Return a list of candidate destination cities."""
        return ["Chicago", "New York", "Barcelona"]

    @Tool.define
    def weather(self, city: str) -> str:
        """Given a city name, return a short description of its weather."""
        status = {"Chicago": "cold", "New York": "wet", "Barcelona": "sunny"}
        return status.get(city, "unknown")

    @Template.define
    def plan_trip_story(self, style: str) -> str:
        """Use the relevant tools to identify a city that has good (sunny)
        weather. Then write a short story about visiting that city in the requested
        style: {style}"""


# ---------------------------------------------------------------------------
# (2) Code-driven composition: a template synthesizes a function that calls the
#     same sub-templates when executed.
# ---------------------------------------------------------------------------


@Template.define
def write_story_fn(style: Literal["moral", "funny"]) -> Callable[[str], str]:
    """Generate a Python function that takes a topic string and returns a story
    about it in the {style} style. The function should delegate the writing to the
    `story_funny` sub-template for humor, or `story_with_moral` for a lesson."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--style",
        type=str,
        choices=["moral", "funny"],
        default="funny",
        help="Style of the story to produce",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="a curious cat",
        help="Topic for the synthesized story function to run on",
    )
    args = parser.parse_args()

    # (1) Model-driven: the orchestrator template calls tools and sub-templates.
    print("=== Orchestrator template (model-driven composition) ===")
    planner = TripPlanner()
    print(planner.plan_trip_story(args.style))

    # (2) Code-driven: the model synthesizes a function that calls the sub-templates.
    print(f"\n=== Synthesized higher-order function (style={args.style}) ===")
    story_fn = write_story_fn(args.style)
    print(inspect.getsource(story_fn))
    print(f"\n=== Running it on {args.topic!r} ===")
    print(story_fn(args.topic))


if __name__ == "__main__":
    main()
