"""Map-reduce resume evaluation.

Demonstrates:
- Fan-out: evaluating multiple items independently with the same template
- Reduce: aggregating individual results into a summary
- ``asyncio.gather`` with ``asyncio.to_thread`` for parallel LLM calls
- Structured output with dataclasses
"""

import argparse
import asyncio
import dataclasses
import functools
import os

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Evaluation:
    name: str
    qualified: bool
    strengths: str
    weaknesses: str
    score: int  # 1-10


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@Template.define
def evaluate_resume(resume: str, job_description: str) -> Evaluation:
    """You are a hiring manager. Evaluate this resume against the job
    description and produce a structured evaluation.

    Job description: {job_description}

    Resume:
    {resume}

    Score from 1 (poor fit) to 10 (perfect fit).
    """
    raise NotHandled


@Template.define
def summarize_evaluations(job_description: str, evaluations_text: str) -> str:
    """You are a hiring manager summarizing candidate evaluations.

    Job description: {job_description}

    Individual evaluations:
    {evaluations_text}

    Provide a brief summary: rank the candidates from best to worst,
    highlight the top candidate, and note any concerns.
    """
    raise NotHandled


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

JOB_DESCRIPTION = (
    "Senior Python Developer: 5+ years Python experience, "
    "familiarity with web frameworks (Django/Flask), "
    "database design, and cloud deployment (AWS/GCP)."
)

RESUMES = [
    "Alice Chen - 7 years Python, Django expert, AWS certified, "
    "led team of 5, built microservices architecture at FinTech startup.",
    "Bob Smith - 3 years Python, 2 years JavaScript, some Flask experience, "
    "junior developer at small agency, strong communication skills.",
    "Carol Davis - 10 years software engineering, 6 years Python, "
    "GCP specialist, PostgreSQL expert, open-source contributor, "
    "previously senior engineer at Google.",
    "Dave Wilson - 4 years Python, self-taught, built several side projects, "
    "no professional experience with web frameworks or cloud platforms.",
]

# ---------------------------------------------------------------------------
# Map-reduce pipeline
# ---------------------------------------------------------------------------


async def map_reduce_evaluate(
    provider: LiteLLMProvider,
    resumes: list[str],
    job_description: str,
) -> str:
    """Evaluate resumes in parallel (map), then summarize (reduce)."""
    # Map: evaluate each resume concurrently
    evaluate = functools.partial(
        asyncio.to_thread,
        handler(provider)(handler(RetryLLMHandler(num_retries=3))(evaluate_resume)),
    )
    evaluations: list[Evaluation] = list(
        await asyncio.gather(*(evaluate(resume, job_description) for resume in resumes))
    )

    # Print individual evaluations
    for ev in evaluations:
        print(f"  {ev.name}: score={ev.score}/10, qualified={ev.qualified}")
        print(f"    + {ev.strengths}")
        print(f"    - {ev.weaknesses}")

    # Reduce: summarize all evaluations
    evaluations_text = "\n\n".join(
        f"Candidate: {ev.name}\n"
        f"Score: {ev.score}/10\n"
        f"Qualified: {ev.qualified}\n"
        f"Strengths: {ev.strengths}\n"
        f"Weaknesses: {ev.weaknesses}"
        for ev in evaluations
    )
    with handler(provider), handler(RetryLLMHandler(num_retries=3)):
        return summarize_evaluations(job_description, evaluations_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map-reduce resume evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
    args = parser.parse_args()

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    provider = LiteLLMProvider(model=args.model)

    print(f"Evaluating {len(RESUMES)} resumes for: {JOB_DESCRIPTION}\n")
    summary = asyncio.run(map_reduce_evaluate(provider, RESUMES, JOB_DESCRIPTION))
    print(f"\n{summary}")
