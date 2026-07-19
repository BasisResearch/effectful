"""Retrying failed LLM output: validation errors and tool failures.

Demonstrates:
- ``RetryLLMHandler`` feeding ``PydanticCustomError`` messages back to the LLM
  so it can correct structured output that fails validation
- ``RetryLLMHandler`` surfacing tool exceptions back to the LLM as tool messages,
  so a flaky tool (``unstable_service``) can succeed after multiple attempts
- ``functools.cache`` to make a template call deterministic in-process
"""

import argparse
import functools

import pydantic
from pydantic import field_validator
from pydantic_core import PydanticCustomError

from effectful.handlers.llm import Template, Tool

# ---------------------------------------------------------------------------
# Validated structured output
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass
class Rating:
    score: int
    explanation: str

    @field_validator("score")
    @classmethod
    def check_score(cls, v):
        if v < 1 or v > 5:
            raise PydanticCustomError(
                "invalid_score",
                "score must be 1–5, got {v}",
                {"v": v},
            )
        return v

    @field_validator("explanation")
    @classmethod
    def check_explanation_contains_score(cls, v, info):
        score = info.data.get("score", None)
        if score is not None and str(score) not in v:
            raise PydanticCustomError(
                "invalid_explanation",
                "explanation must mention the score {score}, got '{explanation}'",
                {"score": score, "explanation": v},
            )
        return v


@functools.cache
@Template.define
def give_rating_for_movie(movie_name: str) -> Rating:
    """Give a rating for {movie_name}. The explanation MUST include the numeric score. Do not use any tools."""


# ---------------------------------------------------------------------------
# Flaky tool (unstable_service auto-captured from lexical scope)
# ---------------------------------------------------------------------------

call_count = 0
REQUIRED_RETRIES = 3


@Tool.define
def unstable_service() -> str:
    """Fetch data from an unstable external service. May require retries."""
    global call_count
    call_count += 1
    if call_count < REQUIRED_RETRIES:
        raise ConnectionError(
            f"Service unavailable! Attempt {call_count}/{REQUIRED_RETRIES}. Please retry."
        )
    return "{ 'status': 'ok', 'data': [1, 2, 3] }"


@Template.define
def fetch_data() -> str:
    """Use the unstable_service tool to fetch data."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", type=str, default="Die Hard", help="Movie to rate")
    args = parser.parse_args()

    print("=== Retrying structured-output validation ===")
    rating = give_rating_for_movie(args.movie)
    print(f"Score: {rating.score}/5")
    print(f"Explanation: {rating.explanation}")

    print("\n=== Retrying tool execution failures ===")
    result = fetch_data()
    print(f"Result: {result} (after {call_count} tool attempts)")


if __name__ == "__main__":
    main()
