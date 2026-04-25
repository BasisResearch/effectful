"""Chain-of-thought reasoning with structured self-loop.

Demonstrates:
- Structured output with a ``ThoughtStep`` dataclass
- An ``Agent`` that loops until it decides it has a final answer
- The LLM sees its own prior reasoning via ``Agent.__history__``
"""

import argparse
import dataclasses
import os

from tenacity import stop_after_attempt

from effectful.handlers.llm import Agent, Template
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ThoughtStep:
    reasoning: str
    conclusion: str
    is_final: bool


# ---------------------------------------------------------------------------
# Chain-of-thought agent
# ---------------------------------------------------------------------------


class Thinker(Agent):
    """Agent that reasons step-by-step until it reaches a final answer."""

    @Template.define
    def think(self, problem: str) -> ThoughtStep:
        """You are solving a problem step by step.

        Problem: {problem}

        Review the conversation history for any prior reasoning steps.
        Continue from where you left off. Break the problem into small,
        logical steps. Set is_final=true only when you have a complete,
        well-supported answer.
        """
        raise NotHandled

    def solve(self, problem: str, max_steps: int = 10) -> str:
        """Solve a problem by iterative chain-of-thought reasoning."""
        for i in range(max_steps):
            step = self.think(problem)
            print(f"  [step {i + 1}] {step.reasoning}")
            if step.is_final:
                return step.conclusion

        return step.conclusion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain-of-thought reasoning agent")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum reasoning steps before stopping",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=(
            "A farmer has 17 sheep. All but 9 run away. "
            "Then he buys 5 more. How many sheep does he have now?"
        ),
        help="The problem to solve",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=3,
        help="Number of retries for malformed LLM output",
    )
    args = parser.parse_args()

    provider = LiteLLMProvider(model=args.model)

    problems = [
        args.problem,
        (
            "If you have a 3-gallon jug and a 5-gallon jug, "
            "how do you measure exactly 4 gallons of water?"
        ),
    ]

    with (
        handler(provider),
        handler(RetryLLMHandler(stop=stop_after_attempt(args.num_retries))),
    ):
        for problem in problems:
            thinker = Thinker()
            print(f"\nProblem: {problem}")
            answer = thinker.solve(problem, max_steps=args.max_steps)
            print(f"Answer: {answer}")
