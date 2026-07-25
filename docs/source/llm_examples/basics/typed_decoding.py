"""Type-driven decoding: turning model output into typed Python values.

Demonstrates:
- Primitive decoding (int, bool) from templates that return a number / a decision
- Dataclass return types decoded from constrained generation
- Round-tripping a dataclass: one template produces it, others consume it as prompt input
- Synthesizing an executable Callable from a template (run via the eval provider)
- inspect.getsource on the synthesized function

The thread tying these together: you declare a Python return type and the model's
output is decoded into a real value of that type -- an int, a bool, a dataclass, or
an executable function -- as an auto-grader that poses a problem, solves it, and
checks its own work.
"""

import argparse
import dataclasses
import inspect
from collections.abc import Callable

from effectful.handlers.llm import Template

# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Problem:
    title: str
    description: str
    example_input: str
    example_output: str


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@Template.define
def pose_problem(topic: str) -> Problem:
    """Invent a small self-contained string-processing coding problem about {topic}.

    The problem must be solvable by a single Python function taking one string and
    returning one string. Fill in a short title, a clear description, and one
    worked example (``example_input`` and its correct ``example_output``). Do not
    use any tools."""


@Template.define
def estimate_difficulty(problem: Problem) -> int:
    """Rate the difficulty of {problem} from 1 (trivial) to 5 (very hard),
    returning just the integer. Do not use any tools."""


@Template.define
def write_solution(problem: Problem) -> Callable[[str], str]:
    """Write a Python function that solves {problem}: it takes the input string and
    returns the required output string. It must reproduce the worked example."""


@Template.define
def judge(problem: Problem, output: str) -> bool:
    """For {problem}, the candidate solution produced {output} when run on the
    example input. Decide whether that matches the expected ``example_output``.
    Do not use any tools."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        type=str,
        default="text processing",
        help="Topic the generated coding problem should be about",
    )
    args = parser.parse_args()

    # Dataclass return type, decoded from constrained generation.
    problem = pose_problem(args.topic)
    assert isinstance(problem, Problem)
    print(f"# {problem.title}\n{problem.description}")
    print(f"example: {problem.example_input!r} -> {problem.example_output!r}")

    # Primitive int decode; the dataclass is round-tripped back in as prompt input.
    difficulty = estimate_difficulty(problem)
    assert isinstance(difficulty, int)
    print(f"\nDifficulty: {difficulty}/5")

    # Synthesize an executable Callable and inspect its source.
    solution = write_solution(problem)
    assert callable(solution)
    print("\nGenerated solution:")
    print(inspect.getsource(solution))

    # Run the synthesized function, then decode a bool verdict from the judge.
    output = solution(problem.example_input)
    print(f"solution({problem.example_input!r}) == {output!r}")
    verdict = judge(problem, output)
    assert isinstance(verdict, bool)
    print("PASS" if verdict else "FAIL")


if __name__ == "__main__":
    main()
