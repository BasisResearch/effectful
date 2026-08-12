"""LLM-based Towers of Hanoi solver with two strategies.

Two solving strategies share a common ``Step`` / ``GameState`` model and are
selected with ``--mode``:

- ``recursive`` — ask the LLM to return the full move list in one shot, using
  the classic recursive decomposition.
- ``iterative`` — ask the LLM for one move at a time, with tool-based
  validation.  Adapted from https://github.com/BasisResearch/effectful/pull/404.
  Demonstrates:

  - A static ``Step`` model for structured output
  - ``@Tool.define`` inside a closure to expose game-state validation as a tool
  - Skills defined inside a function that auto-capture closure-scoped tools
"""

import argparse
import dataclasses
import itertools

from effectful.handlers.llm import Skill, Tool


@dataclasses.dataclass
class Step:
    """A single move: take the top disk from tower ``start`` and place it on
    tower ``end``.  Tower indices are zero-based."""

    start: int
    end: int
    explanation: str = dataclasses.field(default="")  # optional reasoning from the LLM


@dataclasses.dataclass
class GameState:
    """State of a Towers of Hanoi game.

    Higher numbers represent larger disks, so ``(2, 1, 0)`` is a valid
    tower (largest on bottom).  The goal is to move all disks from the
    leftmost tower (index 0) to the rightmost tower (index -1).

    This is a plain ``dataclass`` (not a Pydantic model) so the type checker
    can see its methods.
    """

    size: int
    towers: tuple[tuple[int, ...], ...] = dataclasses.field(default=())

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
# Recursive solver
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


def solve_recursive(state: GameState) -> None:

    @Skill.define
    def solve(n_disks: int, source: int, target: int, auxiliary: int) -> list[Step]:
        """Solve Tower of Hanoi using recursion: move {n_disks} disks from tower {source} to
        tower {target}, using tower {auxiliary} as temporary storage.
        """

    size = state.size
    print(f"Solving Tower of Hanoi with {size} disks...")
    steps = solve(n_disks=size, source=0, target=size - 1, auxiliary=1)
    print(f"\nLLM returned {len(steps)} steps. Validating...\n")
    validate_solution(size, steps)


# ---------------------------------------------------------------------------
# Iterative solver
# ---------------------------------------------------------------------------


def predict_next_step(state: GameState) -> Step:
    """Ask the LLM to predict the next move.

    A ``get_valid_moves`` tool is defined in the closure so the skill
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

    @Skill.define
    def predict(game_state: GameState) -> Step:
        """Given the state of the game of Towers of Hanoi:

        {game_state}

        Predict the next step to complete the game (move all disks to the
        rightmost tower).  You MUST call get_valid_moves first to see which
        moves are legal, then pick the best one.  Give a brief reasoning.
        """

    return predict(state)


def solve_iterative(state: GameState, *, max_steps: int = 30) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("recursive", "iterative"),
        default="iterative",
        help="Solving strategy: full recursive solution or iterative one move at a time",
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
        help="Maximum number of steps before giving up (iterative mode only)",
    )
    args = parser.parse_args()

    state = GameState(size=args.game_size)
    if args.mode == "recursive":
        solve_recursive(state)
    else:
        solve_iterative(state, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
