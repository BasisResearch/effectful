"""Structured output via dataclasses.

Demonstrates:
- Dataclass return types decoded from constrained LLM generation
- Round-tripping a dataclass: one template produces it, another consumes it
"""

import argparse
import dataclasses

from effectful.handlers.llm import Template

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


@Template.define
def rate_joke(joke: KnockKnockJoke) -> bool:
    """Decide if {joke} is funny or not. Do not use any tools."""


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

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theme", type=str, default="lizards", help="Theme for the joke"
    )
    args = parser.parse_args()

    do_comedy(args.theme)


if __name__ == "__main__":
    main()
