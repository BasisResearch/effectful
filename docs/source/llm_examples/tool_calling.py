"""Tool calling: templates invoke Python callables exposed via ``@Tool.define``.

Demonstrates:
- ``@Tool.define`` for exposing a Python function to the model
- Lexical-scope auto-capture: tools defined alongside a template are made
  available to the LLM without explicit registration
- The model chains multiple tool calls to answer a multi-step query
"""

import argparse
import os

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@Tool.define
def cities() -> list[str]:
    """Return a list of cities that can be passed to `weather`."""
    return ["Chicago", "New York", "Barcelona"]


@Tool.define
def weather(city: str) -> str:
    """Given a city name, return a description of the weather in that city."""
    status = {"Chicago": "cold", "New York": "wet", "Barcelona": "sunny"}
    return status.get(city, "unknown")


# ---------------------------------------------------------------------------
# Template (cities and weather are auto-captured from lexical scope)
# ---------------------------------------------------------------------------


@Template.define
def vacation() -> str:
    """Use the provided tools to suggest a city that has good weather. Use only the `cities` and `weather` tools provided."""
    raise NotHandled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool calling with auto-captured lexical scope"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    args = parser.parse_args()

    provider = LiteLLMProvider(model=args.model)
    with handler(provider):
        print(vacation())
