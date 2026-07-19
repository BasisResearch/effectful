"""Structured output via dataclasses.

Demonstrates:
- Dataclass return types decoded from constrained LLM generation
- Round-tripping a dataclass: one template produces it, another consumes it
"""

import argparse
import dataclasses
import os

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KnockKnockJoke:
    whos_there: str
    punchline: str


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@Template.define
def write_joke(theme: str) -> KnockKnockJoke:
    """Write a knock-knock joke on the theme of {theme}. Do not use any tools."""
    raise NotHandled


@Template.define
def rate_joke(joke: KnockKnockJoke) -> bool:
    """Decide if {joke} is funny or not. Do not use any tools."""
    raise NotHandled


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def do_comedy(theme: str) -> None:
    joke = write_joke(theme)
    print("> You are onstage at a comedy club. You tell the following joke:")
    print(
        f"Knock knock.\nWho's there?\n{joke.whos_there}.\n"
        f"{joke.whos_there} who?\n{joke.punchline}"
    )
    if rate_joke(joke):
        print("> The crowd laughs politely.")
    else:
        print("> The crowd stares in stony silence.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured output via dataclasses")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    parser.add_argument(
        "--theme", type=str, default="lizards", help="Theme for the joke"
    )
    args = parser.parse_args()

    provider = LiteLLMProvider(model=args.model)
    with handler(provider):
        do_comedy(args.theme)
