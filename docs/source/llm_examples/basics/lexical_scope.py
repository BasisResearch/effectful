"""Composition via lexical scope: auto-captured sub-skills, invoked two ways.

Demonstrates:
- Module-level @Skill.define sub-skills auto-captured into other skills'
  lexical scope, with no explicit registration
- An Agent grouping @Tool.define tools with a @Skill.define orchestrator that
  calls those tools and the sub-skills directly (model-driven composition)
- A skill returning a Callable: the model synthesizes a function that calls the
  same sub-skills when run (code-driven composition), via the eval provider
- inspect.getsource on the synthesized function
"""

import argparse
import inspect
from collections.abc import Callable
from typing import Literal

from effectful.handlers.llm import Skill, Tool


@Skill.define
def story_with_moral(topic: str) -> str:
    """Write a short story about {topic} and end with a moral lesson. Do not use any tools."""


@Skill.define
def story_funny(topic: str) -> str:
    """Write a funny, humorous story about {topic}. Do not use any tools."""


class TripPlanner:
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

    @Skill.define
    def plan_trip_story(self, style: str) -> str:
        """Use the relevant tools to identify a city that has good (sunny)
        weather. Then write a short story about visiting that city in the requested
        style: {style}"""


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
    parser.add_argument(
        "--method",
        type=str,
        choices=["model", "code"],
        default="model",
        help="Whether to run the model-driven or code-driven composition",
    )
    args = parser.parse_args()

    if args.method == "model":
        # (1) Model-driven: the orchestrator skill calls tools and sub-skills.
        print("=== Orchestrator skill (model-driven composition) ===")
        planner = TripPlanner()
        print(planner.plan_trip_story(args.style))

    elif args.method == "code":

        @Skill.define
        def write_story_fn(style: Literal["moral", "funny"]) -> Callable[[str], str]:
            """Generate a Python function that takes a topic string and returns a story
            about it in the {style} style. The function should delegate the writing to the
            `story_funny` sub-skill for humor, or `story_with_moral` for a lesson."""

        # (2) Code-driven: the model synthesizes a function that calls the sub-skills.
        print(f"\n=== Synthesized higher-order function (style={args.style}) ===")
        story_fn = write_story_fn(args.style)
        print(inspect.getsource(story_fn))
        print(f"\n=== Running it on {args.topic!r} ===")
        print(story_fn(args.topic))


if __name__ == "__main__":
    main()
