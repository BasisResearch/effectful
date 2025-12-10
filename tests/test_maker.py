import itertools
import logging
import sys
from abc import ABC, abstractmethod
from typing import Optional

import pydantic
from litellm import ConfigDict
from PIL import Image, ImageDraw
from pydantic.dataclasses import dataclass

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import (
    LiteLLMProvider,
    RetryLLMHandler,
)
from effectful.handlers.llm.sampling import KAheadSampler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

type Move = tuple[int, int]


class Step(ABC):
    @property
    @abstractmethod
    def start(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def end(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class GameState:
    """State of a game of towers of Hanoi where the initial state is a
    set of towers. We use higher numbers to represesnt smaller
    disks. So [1,2,3] is a valid tower. The towers are all stacked at
    the left at the start (self.towers[0]), and the goal is to move
    them to the rightmost tower (self.towers[-1])."""

    size: int
    towers: tuple[tuple[int, ...], ...]

    @classmethod
    def new(cls, size: int) -> "GameState":
        towers: list[list[int]] = [[] for _ in range(size)]
        towers[0] = list(reversed(range(size)))
        state: tuple[tuple[int, ...], ...] = tuple(tuple(tower) for tower in towers)
        return cls(size, state)

    def visualise_image(self) -> Image.Image:
        "Uses python graphics libraries to visualise the state of the hanoi game."
        tower_width = 150
        disk_height = 30
        base_height = 20
        pole_width = 10
        img_width = tower_width * len(self.towers)
        img_height = disk_height * (self.size + 1) + base_height + 50

        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)

        for tower_idx, tower in enumerate(self.towers):
            # Draw pole
            pole_x = tower_idx * tower_width + tower_width // 2
            pole_top = 40
            pole_bottom = img_height - base_height - 10
            draw.rectangle(
                [
                    pole_x - pole_width // 2,
                    pole_top,
                    pole_x + pole_width // 2,
                    pole_bottom,
                ],
                fill="brown",
            )

            # Draw base
            base_y = img_height - base_height - 10
            draw.rectangle(
                [
                    tower_idx * tower_width + 20,
                    base_y,
                    (tower_idx + 1) * tower_width - 20,
                    base_y + base_height,
                ],
                fill="gray",
            )

            # Draw disks
            for disk_idx, disk in enumerate(tower):
                disk_width_px = 30 + disk * 15
                disk_y = pole_bottom - (disk_idx + 1) * disk_height
                disk_x1 = pole_x - disk_width_px // 2
                disk_x2 = pole_x + disk_width_px // 2

                # Color gradient based on disk size
                color_intensity = int(255 * (disk / self.size))
                color = (color_intensity, 100, 255 - color_intensity)
                draw.rectangle(
                    [disk_x1, disk_y, disk_x2, disk_y + disk_height - 5],
                    fill=color,
                    outline="black",
                    width=2,
                )
        return img

    def visualise(self):
        img = self.visualise_image()
        img.show()

    def apply(self, step: Move) -> Optional["GameState"]:
        """
        Given a tower `start` and a target tower `end` moves the topmost disk to the end tower.
        """
        start, end = step

        if not (0 <= start < len(self.towers) and 0 <= end < len(self.towers)):
            return None

        # start tower is non empty
        if len(self.towers[start]) == 0:
            return None

        # end tower is a valid target
        if len(self.towers[end]) > 0 and self.towers[start][-1] > self.towers[end][-1]:
            return None

        # create state with the move applied
        new_towers = [list(tower) for tower in self.towers]
        disk = new_towers[start].pop()
        new_towers[end].append(disk)

        #
        new_state = GameState(
            size=self.size, towers=tuple(tuple(tower) for tower in new_towers)
        )
        return new_state

    def is_done(self) -> bool:
        return all(len(tower) == 0 for tower in self.towers[:-1]) and all(
            self.towers[-1][i] > self.towers[-1][i + 1]
            for i in range(len(self.towers[-1]) - 1)
        )

    def valid_steps(self) -> list[Move]:
        steps = []
        for i, tower_i in enumerate(self.towers):
            for j, tower_j in enumerate(self.towers):
                if i == j:
                    continue
                if len(tower_i) == 0:
                    continue
                # if tower_i's disk is smaller than tower_j's topmost, then it is valid to move from tower i to j
                if len(tower_j) == 0 or tower_i[-1] < tower_j[-1]:
                    steps.append((i, j))
        return steps


def build_validated_model(game_state: GameState) -> type[Step]:
    valid_steps = game_state.valid_steps()

    @pydantic.dataclasses.dataclass(frozen=True)
    class StepModel:
        start: int
        end: int
        explanation: str = ""
        model_config = ConfigDict(extra="forbid")

        @pydantic.field_validator("start", "end", mode="before")
        def validate_indices(cls, v, info):
            if isinstance(v, int):
                if not (0 <= v < len(game_state.towers)):
                    raise ValueError(f"{info.field_name} {v} out of range")
            else:
                raise TypeError("start/end must both be int")
            return v

        @pydantic.model_validator(mode="after")
        def validate_step(self):
            if (self.start, self.end) not in valid_steps:
                raise ValueError("step is not in {self.valid_steps}")
            return self

        def __hash__(self):
            return hash((self.start, self.end))

    return StepModel  # type: ignore


def predict_next_step(game_state: GameState) -> Move:
    ValidStep = build_validated_model(game_state)

    @Template.define
    def predict_next_step_inner(game_state) -> ValidStep:  # type: ignore
        """
        Given the state of the game of towers of Hanoi as follows:

        {game_state}

        Predict the next step to complete the game (moving all disks to the rightmost tower).

        Give a reasoning for your prediction, and return the step following the format:

        <step>start,end</step>

        where start and end are zero-based indices for the towers to move. Be concise and avoid wordy answers.
        """
        raise NotHandled

    s = predict_next_step_inner(game_state)
    return (s.start, s.end)


def solve_hanoi(state: GameState):
    log = []

    for i in itertools.count():
        print(f"step {i} - {state}")
        with handler(KAheadSampler()), handler(RetryLLMHandler()):
            step = predict_next_step(state)
        # track the step at each point
        if new_state := state.apply(step):
            log.append((state, step))

        state = new_state or state
        state.visualise()
        if state.is_done():
            break


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

with (
    handler(LiteLLMProvider(model_name="gpt-4o-mini")),
):
    solve_hanoi(state=GameState.new(3))
