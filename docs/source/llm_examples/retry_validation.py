"""Retrying when structured-output validation fails.

Demonstrates:
- A pydantic dataclass with ``field_validator`` constraints
- ``RetryLLMHandler`` feeding ``PydanticCustomError`` messages back to the LLM
  so it can correct its output on a subsequent attempt
"""

import argparse
import os

import pydantic
from pydantic import field_validator
from pydantic_core import PydanticCustomError
from tenacity import stop_after_attempt

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

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


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


@Template.define
def give_rating_for_movie(movie_name: str) -> Rating:
    """Give a rating for {movie_name}. The explanation MUST include the numeric score. Do not use any tools."""
    raise NotHandled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retry on pydantic validation errors in LLM responses"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    parser.add_argument("--movie", type=str, default="Die Hard", help="Movie to rate")
    parser.add_argument(
        "--num-retries",
        type=int,
        default=4,
        help="Number of retries for malformed LLM output",
    )
    args = parser.parse_args()

    provider = LiteLLMProvider(model=args.model)

    print("=== Without RetryLLMHandler ===")
    with handler(provider):
        try:
            rating = give_rating_for_movie(args.movie)
            print(f"Score: {rating.score}/5\nExplanation: {rating.explanation}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n=== With RetryLLMHandler ===")
    with (
        handler(provider),
        handler(RetryLLMHandler(stop=stop_after_attempt(args.num_retries))),
    ):
        rating = give_rating_for_movie(args.movie)
        print(f"Score: {rating.score}/5")
        print(f"Explanation: {rating.explanation}")
