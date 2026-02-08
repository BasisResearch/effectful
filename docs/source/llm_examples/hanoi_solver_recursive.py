"""Recursive LLM-based Towers of Hanoi solver.

Adapted from https://github.com/BasisResearch/effectful/pull/404

Demonstrates:
- ``IsRecursive`` annotation to let a template call itself as a tool
- Recursive problem decomposition via LLM tool calls
- Post-hoc validation of the LLM-generated move sequence

The classic recursive algorithm for Tower of Hanoi is:

    hanoi(n, source, target, auxiliary):
        if n == 1: move disk from source to target
        else:
            hanoi(n-1, source, auxiliary, target)   # move n-1 disks out of the way
            move largest disk from source to target  # move the bottom disk
            hanoi(n-1, auxiliary, target, source)    # move n-1 disks to target

This solver defines a recursive ``Template`` that can call itself as a tool.
The LLM decomposes the n-disk problem into three sub-steps, making recursive
tool calls for the (n-1)-disk sub-problems, and returns the concatenated
list of moves.

See: https://en.wikipedia.org/wiki/Tower_of_Hanoi
"""

import argparse
import os
import typing
from dataclasses import dataclass, field

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.handlers.llm.template import IsRecursive
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


# ---------------------------------------------------------------------------
# Game state (for validation only)
# ---------------------------------------------------------------------------


@dataclass
class GameState:
    """State of a Towers of Hanoi game.

    Higher numbers represent larger disks, so ``(2, 1, 0)`` is a valid
    tower (largest on bottom).  The goal is to move all disks from the
    leftmost tower (index 0) to the rightmost tower (index -1).
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

    def __str__(self) -> str:
        return " | ".join(str(list(t)) for t in self.towers)


# ---------------------------------------------------------------------------
# Recursive LLM solver
# ---------------------------------------------------------------------------


@Template.define
def solve(
    n_disks: int, source: int, target: int, auxiliary: int
) -> typing.Annotated[list[Step], IsRecursive]:
    """Solve Tower of Hanoi: move {n_disks} disks from tower {source} to
    tower {target}, using tower {auxiliary} as temporary storage.

    Recursive strategy:
    - Base case (n_disks == 1): return [Step(start=source, end=target)]
    - Recursive case (n_disks > 1):
        1. Call solve(n_disks - 1, source, auxiliary, target) to move the
           top n_disks-1 disks out of the way onto the auxiliary tower.
        2. Move the largest disk: Step(start=source, end=target).
        3. Call solve(n_disks - 1, auxiliary, target, source) to move the
           n_disks-1 disks from auxiliary to the target tower.
        4. Return the concatenated list of all steps from (1), (2), and (3).
    """
    raise NotHandled


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_solution(size: int, steps: list[Step]) -> bool:
    """Apply all steps to the initial state and check that the puzzle is solved."""
    state = GameState(size=size)
    print(f"  initial: {state}")
    for i, step in enumerate(steps):
        try:
            state = state.apply(step)
            print(f"  step {i}: move {step.start} -> {step.end}  =>  {state}")
        except ValueError as e:
            print(f"  step {i}: INVALID move {step.start} -> {step.end}: {e}")
            return False
    if state.is_done():
        print(f"  Solved in {len(steps)} moves!")
        return True
    else:
        print(f"  Not solved after {len(steps)} moves. Final state: {state}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursive LLM-based Towers of Hanoi solver"
    )
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
        n = args.game_size
        print(f"Solving Tower of Hanoi with {n} disks...")
        steps = solve(n_disks=n, source=0, target=n - 1, auxiliary=1)
        print(f"\nLLM returned {len(steps)} steps. Validating...\n")
        validate_solution(n, steps)
