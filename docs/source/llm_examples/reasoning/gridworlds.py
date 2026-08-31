import abc
import dataclasses
import enum
import typing


@dataclasses.dataclass(frozen=True, eq=True, unsafe_hash=True)
class State:
    """A raw grid observation: rows of integer color codes.

    A ``dataclass`` wrapping the grid rather than a bare ``tuple`` subclass -- Pydantic
    / ``Encodable`` need a real type with a core schema to move a ``State`` across the
    model boundary (as a spliced prompt value and in the ``step`` signature).

    The grid itself is left unstructured -- recovering objects (player, box, walls)
    from it is exactly the agent's job. ``grid`` is a tuple of tuples, so ``State`` is
    ``frozen`` and hashable and doubles as a BFS key that compares by value.
    """

    grid: tuple[tuple[int, ...], ...]

    def __str__(self) -> str:
        return "\n".join("".join(str(cell) for cell in row) for row in self.grid)


class Color(enum.IntEnum):
    """The palette the agent observes. Only the *dynamics* are hidden; the goal --
    ``BOX_ON_TARGET`` appearing -- is visible, so we never synthesize an is_goal."""

    FLOOR = 0
    WALL = 1
    PLAYER = 2
    BOX = 3
    TARGET = 4
    BOX_ON_TARGET = 5


class Action(enum.IntEnum):
    """The four moves. ``IntEnum`` so a value stays a plain ``int`` at the model
    boundary: the synthesized ``step`` sees actions as 0-3, exactly as the prompt says."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @property
    def delta(self) -> tuple[int, int]:
        return {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }[self]


class Transition(typing.NamedTuple):
    """One recorded step of ground truth in the Timeline."""

    before: State
    action: Action
    after: State


# A ``(row, column)`` cell coordinate in the grid.
type Position = tuple[int, int]


class Game(abc.ABC):
    """A hidden game with a visible goal. The agent must reverse-engineer the rules."""

    rows: int
    cols: int

    @abc.abstractmethod
    def observe(self) -> State:
        """Return the current grid observation."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: Action) -> tuple[State, bool]:
        """Apply an action to reality and return the new state and whether the goal is
        reached."""
        raise NotImplementedError


class PushGame(Game):
    """A tiny Sokoban-lite game. The player pushes a box onto a target."""

    # The rest of the specification is a comment, not part of the docstring,
    # because `__doc__` is *agent-visible*: `world_model_agent.py` passes it to
    # the Physicist as the `<hint>` it is allowed to know, immediately before
    # telling it the dynamics are hidden and must be inferred from the timeline.
    # Anything written above this line is handed to the agent; anything below it
    # is for the reader.
    #
    # The true mechanics -- a move steps the player one cell; stepping into the
    # box pushes it one further; walls block both -- are *not* revealed to the
    # agent. The box can only travel rightward toward the target here, so the
    # level has no dead ends: the agent always recovers once its model is
    # correct.

    rows: int
    cols: int
    walls: set[Position]
    player: Position
    box: Position
    targets: set[Position]

    def __init__(self) -> None:
        self.rows, self.cols = 4, 7
        self.walls = {
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if r in (0, self.rows - 1) or c in (0, self.cols - 1)
        }
        self.player = (1, 1)
        self.box = (1, 2)
        self.targets = {(1, 5)}

    def observe(self) -> State:
        grid = [[Color.FLOOR] * self.cols for _ in range(self.rows)]
        for r, c in self.walls:
            grid[r][c] = Color.WALL
        for r, c in self.targets:
            grid[r][c] = Color.TARGET
        br, bc = self.box
        grid[br][bc] = Color.BOX_ON_TARGET if self.box in self.targets else Color.BOX
        pr, pc = self.player
        grid[pr][pc] = Color.PLAYER
        return State(tuple(tuple(int(cell) for cell in row) for row in grid))

    def step(self, action: Action) -> tuple[State, bool]:
        dr, dc = action.delta
        pr, pc = self.player
        ahead = (pr + dr, pc + dc)
        if ahead in self.walls:
            pass  # blocked by a wall
        elif ahead == self.box:
            beyond = (ahead[0] + dr, ahead[1] + dc)
            if beyond not in self.walls:  # push the box (never a wall here)
                self.box = beyond
                self.player = ahead
        else:
            self.player = ahead
        return self.observe(), self.box in self.targets


@dataclasses.dataclass
class Corridor:
    """A button corridor: rooms in a row, each behind a hidden button code.

    Not a `Game` -- it observes as prose and acts by button letter rather than
    by grid and move -- but the same species of environment: hidden rules, a
    visible goal. ``press`` is the whole interface. A correct press advances the
    current room's progress ("click"); completing a code opens the door and
    moves to the next room; a wrong press resets the room's progress ("buzz").

    Living in this module rather than the driving script also keeps the
    mechanics out of the model's view, as with `PushGame`: the system prompt
    embeds the *skill's* module source, not this one's. The codes themselves
    are the answer key -- bind them only in locals of the driving script's
    ``main()``, never at module scope, which the model can read.
    """

    BUTTONS: typing.ClassVar[str] = "ABC"

    codes: list[str]
    room: int = 0
    progress: int = 0
    presses: int = 0
    events: list[str] = dataclasses.field(default_factory=list)

    @property
    def solved(self) -> bool:
        return self.room >= len(self.codes)

    def press(self, button: str) -> str:
        assert not self.solved
        self.presses += 1
        if button == self.codes[self.room][self.progress]:
            self.progress += 1
            if self.progress == len(self.codes[self.room]):
                self.room += 1
                self.progress = 0
                event = (
                    f"press {self.presses}: {button} -> CLICK; door {self.room} opens"
                )
            else:
                event = f"press {self.presses}: {button} -> click (progress {self.progress})"
        else:
            self.progress = 0
            event = f"press {self.presses}: {button} -> BUZZ; room {self.room} progress reset"
        self.events.append(event)
        return event

    def observe(self) -> str:
        recent = "\n".join(self.events[-12:]) or "(no presses yet)"
        return (
            f"room {self.room} of {len(self.codes)} "
            f"(codes are {len(self.codes[0])} presses long); "
            f"confirmed progress in this room: {self.progress}; "
            f"total presses so far: {self.presses}\n"
            f"recent events:\n{recent}"
        )


# A hand-written reference strategy for `Corridor` -- an expert-harness baseline
# to compare a from-scratch self-improving agent against. It lives here, not in
# the driving script, because the script's own source is embedded in the system
# prompt: a module-level literal there would hand the from-scratch condition the
# expert strategy verbatim and contaminate the comparison.
CORRIDOR_EXPERT_STRATEGY = """\
Discover each code by prefix extension: with a confirmed prefix P, try P+"A";
a buzz resets the room, so re-enter P and try P+"B", then P+"C". Every click
extends the confirmed prefix by one. Record each room's confirmed prefix in a
note *immediately* -- after any clear, notes are all you have. Once a room's
full code is known, replay it exactly; never re-derive a recorded code."""
