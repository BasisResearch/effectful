"""LLM-guided Towers of Hanoi solver with tool-based validation.

Adapted from https://github.com/BasisResearch/effectful/pull/404

Demonstrates:
- A static Pydantic ``Step`` model for structured output
- ``@Tool.define`` inside a closure to expose game-state validation as a tool
- ``RetryLLMHandler`` to retry on malformed LLM output
- Templates defined inside a function that auto-capture closure-scoped tools
"""

import argparse
import itertools
import os
from dataclasses import dataclass, field

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Step model
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """A single move: take the top disk from tower ``start`` and place it on
    tower ``end``.  Tower indices are zero-based."""

    start: int
    end: int
    explanation: str = field(default="")  # optional reasoning from the LLM


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------


@dataclass
class GameState:
    """State of a Towers of Hanoi game.

    Higher numbers represent larger disks, so ``(2, 1, 0)`` is a valid
    tower (largest on bottom).  The goal is to move all disks from the
    leftmost tower (index 0) to the rightmost tower (index -1).

    This is a plain ``dataclass`` (not a Pydantic model) so the type checker
    can see its methods.
    """

    size: int
    towers: tuple[tuple[int, ...], ...] = field(default=())

    def __post_init__(self):
        if self.size > 0 and not self.towers:
            self.towers = tuple(
                tuple(reversed(range(self.size))) if i == 0 else ()
                for i in range(self.size)
            )

    def apply(self, step: Step) -> "GameState":
        """Apply a move, returning the new state.  Raises ``ValueError`` if
        the move is invalid."""
        start, end = step.start, step.end
        if not (0 <= start < len(self.towers) and 0 <= end < len(self.towers)):
            raise ValueError(f"tower index out of range: ({start}, {end})")
        if len(self.towers[start]) == 0:
            raise ValueError(f"tower {start} is empty")
        if len(self.towers[end]) > 0 and self.towers[start][-1] > self.towers[end][-1]:
            raise ValueError(
                f"cannot place disk {self.towers[start][-1]} on top of "
                f"disk {self.towers[end][-1]}"
            )
        new_towers = [list(t) for t in self.towers]
        disk = new_towers[start].pop()
        new_towers[end].append(disk)
        return GameState(self.size, tuple(tuple(t) for t in new_towers))

    def is_done(self) -> bool:
        return all(len(t) == 0 for t in self.towers[:-1]) and all(
            self.towers[-1][i] > self.towers[-1][i + 1]
            for i in range(len(self.towers[-1]) - 1)
        )

    def valid_steps(self) -> list[Step]:
        steps = []
        for i, ti in enumerate(self.towers):
            for j, tj in enumerate(self.towers):
                if i == j or len(ti) == 0:
                    continue
                if len(tj) == 0 or ti[-1] < tj[-1]:
                    steps.append(Step(i, j))
        return steps

    def __str__(self) -> str:
        return " | ".join(str(list(t)) for t in self.towers)


# ---------------------------------------------------------------------------
# LLM move predictor
# ---------------------------------------------------------------------------


def predict_next_step(state: GameState) -> Step:
    """Ask the LLM to predict the next move.

    A ``get_valid_moves`` tool is defined in the closure so the template
    can query which moves are legal for the current game state.  A
    ``validate_move`` tool checks whether a proposed move is legal and
    raises ``ValueError`` if not — when wrapped by ``RetryLLMHandler``,
    this error is fed back to the LLM so it can correct itself.
    """
    valid = state.valid_steps()

    @Tool.define
    def get_valid_moves() -> list[Step]:
        """Return the list of valid moves for the current game state."""
        return valid

    @Tool.define
    def validate_move(proposed: Step) -> bool:
        """Check whether moving from tower ``start`` to tower ``end`` is legal."""
        return proposed in state.valid_steps()

    @Template.define
    def predict(game_state: GameState) -> Step:
        """Given the state of the game of Towers of Hanoi:

        {game_state}

        Predict the next step to complete the game (move all disks to the
        rightmost tower).  You MUST call get_valid_moves first to see which
        moves are legal, then pick the best one.  Give a brief reasoning.
        """
        raise NotHandled

    return predict(state)


# ---------------------------------------------------------------------------
# Solver loop
# ---------------------------------------------------------------------------


def solve_hanoi(state: GameState, max_steps: int = 30):
    """Solve Towers of Hanoi by repeatedly asking the LLM for the next move."""
    for i in itertools.count():
        print(f"step {i}: {state}")
        if state.is_done():
            print("Solved!")
            return
        if i >= max_steps:
            print("Gave up after max steps.")
            return

        step: Step = predict_next_step(state)
        try:
            state = state.apply(step)
            print(f"  move: {step.start} -> {step.end}")
        except ValueError as e:
            print(f"  attempt {i}: invalid move {step}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-guided Towers of Hanoi solver")
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
    parser.add_argument(
        "--game-size",
        type=int,
        default=3,
        help="Number of disks in the Towers of Hanoi game",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum number of steps before giving up",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=5,
        help="Number of retries for malformed LLM output",
    )
    args = parser.parse_args()

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    provider = LiteLLMProvider(model=args.model)

    with handler(provider), handler(RetryLLMHandler(num_retries=args.num_retries)):
        solve_hanoi(GameState(size=args.game_size), max_steps=args.max_steps)
