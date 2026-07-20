"""Recursive LLM-based Towers of Hanoi solver."""

import argparse
import dataclasses

from effectful.handlers.llm import Template

# ---------------------------------------------------------------------------
# Step model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Step:
    """A single move: take the top disk from tower ``start`` and place it on
    tower ``end``.  Tower indices are zero-based."""

    start: int
    end: int


# ---------------------------------------------------------------------------
# Game state (for validation only)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GameState:
    """State of a Towers of Hanoi game.

    Higher numbers represent larger disks, so ``(2, 1, 0)`` is a valid
    tower (largest on bottom).  The goal is to move all disks from the
    leftmost tower (index 0) to the rightmost tower (index -1).
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

    def __str__(self) -> str:
        return " | ".join(str(list(t)) for t in self.towers)


# ---------------------------------------------------------------------------
# Recursive LLM solver
# ---------------------------------------------------------------------------


@Template.define
def solve(n_disks: int, source: int, target: int, auxiliary: int) -> list[Step]:
    """Solve Tower of Hanoi using recursion: move {n_disks} disks from tower {source} to
    tower {target}, using tower {auxiliary} as temporary storage.
    """


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-size",
        type=int,
        default=3,
        help="Number of disks in the Towers of Hanoi game",
    )
    args = parser.parse_args()

    n = args.game_size
    print(f"Solving Tower of Hanoi with {n} disks...")
    steps = solve(n_disks=n, source=0, target=n - 1, auxiliary=1)
    print(f"\nLLM returned {len(steps)} steps. Validating...\n")
    validate_solution(n, steps)


if __name__ == "__main__":
    main()
