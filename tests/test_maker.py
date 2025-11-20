import itertools
import logging
import random
import re
import sys
from collections import Counter
from typing import Optional

from openai import OpenAI
from pydantic.dataclasses import dataclass

from effectful.handlers import futures
from effectful.handlers.futures import Executor, ThreadPoolFuturesInterpretation
from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import LLMLoggingHandler, OpenAIAPIProvider
from effectful.ops.semantics import handler


@dataclass(frozen=True)
class Step:
    start: int
    end: int


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
        towers = [[] for _ in range(size)]
        towers[0] = list(reversed(range(size)))
        towers = tuple(tuple(tower) for tower in towers)
        return cls(size, towers)

    def visualise_text(self):
        max_disk = self.size
        width = max_disk * 2 + 3
        for i, tower in enumerate(self.towers):
            print(f"\nTower {i}:")
            for disk in reversed(tower):
                disk_width = (disk + 1) * 2 - 1
                padding = (max_disk - disk_width) // 2
                print(" " * padding + "=" * disk_width + " " * padding)
            print("=" * width)
        print()

    def visualise_image(self):
        "Uses python graphics libraries to visualise the state of the hanoi game."
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None
        # Pillow-based visualization
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
        if img:
            img.show()
        else:
            self.visualise_text()

    def apply(self, step: Step) -> Optional["GameState"]:
        """
        Given a tower `start` and a target tower `end` moves the topmost disk to the end tower.
        """
        start, end = step.start, step.end

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

    def steps_to_complete(self) -> int:
        """Compute the number of steps to complete the towers of hanoi from a given configuration if using the optimal algorithm."""
        # Count disks on each tower
        total_moves = 0

        # For each tower that's not the destination, we need to move all its disks
        for tower_idx, tower in enumerate(self.towers):
            if tower_idx == self.size - 1:
                continue

            # Number of disks on this tower
            n_disks = len(tower)

            if n_disks > 0:
                # Moving n disks from one peg to another requires 2^n - 1 moves
                total_moves += (2**n_disks) - 1

        return total_moves

    def is_done(self) -> bool:
        return all(len(tower) == 0 for tower in self.towers[:-1]) and all(
            self.towers[-1][i] > self.towers[-1][i + 1]
            for i in range(len(self.towers[-1]) - 1)
        )

    def valid_steps(self) -> list[Step]:
        steps = []
        for i, tower_i in enumerate(self.towers):
            for j, tower_j in enumerate(self.towers):
                if i == j:
                    continue
                if len(tower_i) == 0:
                    continue
                # if tower_i's disk is smaller than tower_j's topmost, then it is valid to move from tower i to j
                if len(tower_j) == 0 or tower_i[-1] < tower_j[-1]:
                    steps.append(Step(i, j))
        return steps


class MicroAgent:
    """Micro agent (based on MAKERS paper) responsible for predicting a single next step."""

    game_state: GameState

    def __init__(self, state: GameState):
        self.game_state = state

    @Template.define
    def predict_next_step(self) -> str:
        """
        Given the state of the game of towers of Hanoi as follows:

        {self.game_state}

        Predict the next step to complete the game (moving all disks to the rightmost tower).

        Give a reasoning for your prediction, and return the step following the format:

        <step>start,end</step>

        where start and end are zero-based indices for the towers to move. Be concise and avoid wordy answers.
        """
        pass

    def parse_response(self, response: str) -> Step | None:
        "Parse the predicted step from an LLM response."
        pattern = r"<step>\s*(\d+)\s*,\s*(\d+)\s*</step>"
        m = re.search(pattern, response)
        if not m:
            return None
        return Step(int(m.group(1)), int(m.group(2)))

    def has_no_red_flags(self, response: str) -> Step | None:
        """Returns the underlying step if the provided step has no red flags."""
        if len(response) > 450.0:  # based on a sample
            return None

        step = self.parse_response(response)
        if not step:
            return None
        if not (
            0 <= step.start < len(self.game_state.towers)
            and 0 <= step.end < len(self.game_state.towers)
        ):
            return None
        if step not in self.game_state.valid_steps():
            return None
        return step

    def get_vote(self):  # algorithm 3
        while True:
            resp = self.predict_next_step()
            if step := self.has_no_red_flags(resp):
                return step


class FirstToAheadMoveSelector:
    k: int
    game_state: GameState
    agents: list[MicroAgent]
    votes: Counter[Step]

    def __init__(self, state: GameState, no_agents=6, k=3):
        self.k = k
        self.game_state = state
        self.agents = [MicroAgent(self.game_state) for _ in range(no_agents)]
        self.votes = Counter()

    def do_voting(self) -> Step:  # algorithm 2
        # run n in parallel repeatedly until k come out in top
        while True:
            # submit a batch of votes
            for vote in futures.as_completed(
                Executor.submit(agent.get_vote) for agent in self.agents
            ):
                self.votes[vote] += 1
                max_other_votes = max(
                    (self.votes[o_vote] for o_vote in self.votes if o_vote != vote),
                    default=0,
                )
                if self.votes[vote] >= max_other_votes + self.k:
                    return vote


def calculate_average_sample_size():
    """Function I used to calculate the number 450. in the above code."""
    sizes = []
    samples = []

    with handler(OpenAIAPIProvider(OpenAI())):
        for _ in range(10):
            s = GameState.new(random.randint(3, 6))
            for i in range(100):
                step = random.choice(s.valid_steps())
                s = s.apply(step) or s
            resp = MicroAgent(s).predict_next_step()
            samples.append(resp)
            sizes.append(len(resp))
    return sum(sizes) / len(sizes)


def solve_hanoi(state: GameState):
    log = []

    for i in itertools.count():
        print(f"step {i} - {state}")
        step = FirstToAheadMoveSelector(state).do_voting()
        # track the step at each point
        log.append((state, step))

        state = state.apply(step)
        state.visualise()


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

with (
    handler(ThreadPoolFuturesInterpretation()),
    handler(OpenAIAPIProvider(OpenAI())),
    handler(LLMLoggingHandler()),
):
    solve_hanoi(state=GameState.new(3))
