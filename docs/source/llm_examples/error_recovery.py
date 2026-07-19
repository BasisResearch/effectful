"""Recovering from failed LLM output: flaky tools and invalid structured output.

A single task -- rate a movie after looking it up -- exercises both retry paths:

Demonstrates:
- RetryLLMHandler surfacing tool exceptions back to the LLM as tool messages, so a
  flaky tool (lookup_movie) can succeed after multiple attempts
- RetryLLMHandler feeding pydantic validation errors back to the LLM so it can
  correct structured output (a Rating) that fails validation
"""

import argparse
import dataclasses
import typing

from effectful.handlers.llm import Template, Tool

# ---------------------------------------------------------------------------
# Flaky tool (auto-captured into rate_movie's lexical scope)
# ---------------------------------------------------------------------------

call_count = 0
REQUIRED_RETRIES = 3


@Tool.define
def lookup_movie(title: str) -> str:
    """Look up facts about a movie from an (unreliable) database."""
    global call_count
    call_count += 1
    if call_count < REQUIRED_RETRIES:
        raise ConnectionError(
            f"Movie database unavailable! Attempt {call_count}/{REQUIRED_RETRIES}. Please retry."
        )
    return f"{title}: an acclaimed action film, widely regarded as a genre classic."


# ---------------------------------------------------------------------------
# Validated structured output
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Rating:
    """
    A movie rating, with a score (an integer from 1 to 5) and an explanation.
    The explanation MUST mention the score, otherwise it will be rejected as invalid.
    """

    score: typing.Literal[1, 2, 3, 4, 5]
    explanation: str

    def __post_init__(self):
        if self.score < 1 or self.score > 5:
            raise ValueError(f"score must be 1-5, got {self.score}")
        if str(self.score) not in self.explanation:
            raise ValueError(
                f"explanation must mention the score {self.score}, got '{self.explanation}'"
            )


# ---------------------------------------------------------------------------
# Template: uses the flaky tool, returns validated structured output
# ---------------------------------------------------------------------------


@Template.define
def rate_movie(movie_name: str) -> Rating:
    """Look up the movie {movie_name}, then give it a rating."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", type=str, default="Die Hard", help="Movie to rate")
    args = parser.parse_args()

    rating = rate_movie(args.movie)
    print(f"Rated {args.movie!r} after {call_count} tool attempts:")
    print(f"Score: {rating.score}/5")
    print(f"Explanation: {rating.explanation}")


if __name__ == "__main__":
    main()
