"""A self-improving agent: the harness lives on `self`, the transcript is disposable.

Implements the inference-time loop shared by "Harness RL is Meta-Learning:
Training to Self-Improve at Test Time" (COLM 2026 submission) and "Continual
Harness: Online Adaptation for Self-Improving Foundation Agents" (arXiv
2605.09998) over the unmodified harness stack. Both papers have an agent revise
its own harness -- system prompt, sub-agents, skills, memory -- from experience,
mid-episode, with model weights frozen. Here that harness is the `Player`
agent's instance attributes:

- ``instructions`` is the mutable half of the system prompt (the papers' `p`),
  spliced into every request via the ``{self.instructions}`` hole below;
- ``notes`` is memory (`M`), catalogued by id and title in every request via
  ``{self.memory_catalog}`` and read in full through the REPL on demand;
- skills (`K`) and sub-agents (`G`) are whatever `Tool`s and `Skill`s the model
  binds onto ``self`` in the REPL -- bound is offered.

Every revision the papers route through meta-tools is an assignment to ``self``
in ``exec_code``, and their Refiner is not a component but a *moment*: the
compaction call where the agent promotes what matters onto ``self`` and discards
the transcript it no longer needs, atomically, with the REPL tool's
``clear="conversation"`` mode. That scope, rather than ``clear="turn"``, is the
one this task needs: a `Player`'s history spans every ``plan_presses`` call, so
dropping only the current call's own rounds would free almost nothing.
The schedule for that moment is itself harness content -- a sentence in the
instructions, revisable like everything else.

The task is a button corridor: each room's door opens after a hidden button
sequence, discoverable by trial (a wrong press resets the room). Codes are
stable within a run, so what the agent writes onto ``self`` -- discovered
codes, a discovery strategy, a solver skill -- genuinely transfers across
compactions: the harness carries the continuity, the transcript is
disposable.

Conditions (`--condition`): ``scratch`` starts with empty instructions and the
compaction protocol only; ``expert`` starts with a hand-written strategy,
the papers' hand-engineered-harness baseline. The minimalist baseline needs no
code at all: run with ``--tool-choice none`` and the model must answer directly.
"""

import argparse
import copy
import dataclasses
import random
import typing

from docs.source.llm_examples.reasoning.gridworlds import Corridor
from effectful.handlers.llm import Agent, Skill


@dataclasses.dataclass(frozen=True)
class Note:
    """One entry of the agent's memory: catalogued by id and title in every
    request, body read on demand through the REPL."""

    id: str
    title: str
    body: str
    importance: float = 1.0


@dataclasses.dataclass
class Player(Agent):
    """You are playing a button corridor. Each room's door opens only after its
    hidden button sequence is entered; a wrong press resets that room's
    progress; the sequences are stable for the whole game. Your score is the
    total number of presses, so rediscovering what you already learned is the
    main way to lose. Use the REPL to improve yourself as you go:

    - assign to `self.instructions` to rewrite your own standing strategy;
    - append `Note`s to `self.notes` for facts worth keeping (discovered codes,
      failed hypotheses, where you left off); read one back with
      `print(self.notes[i].body)`;
    - define reusable functions or `Skill`s in the REPL and assign them onto
      `self` (use the function's own name: `self.next_guess = next_guess`) --
      anything bound on `self` is offered to you as a tool on later calls,
      whereas one merely defined in the session is gone when you answer; if one
      stops earning its keep, `del`ete it.

    Do that writing *before* you answer, in the same call you learned it. The
    observation you are shown reports only recent events, so a press whose
    result you never wrote down is a press you will pay to make again. If this
    request tells you something the last one did not -- a button that clicked, a
    button that buzzed, a room completed -- record it, then answer.

    Compact as you go: once the transcript has served its purpose -- say, when a
    room's code is recorded in a note -- write what matters onto `self`, then
    call `exec_code` with `clear="conversation"`. That leaves the request, and
    the call you made it in: your message, the snippet and its output. Every
    earlier call goes, and `self` is untouched. So a code recorded in a note is
    a code you keep; a code you only ever read off the transcript is a code you
    will pay to rediscover.
    """

    instructions: str = ""
    notes: list[Note] = dataclasses.field(default_factory=list)

    @property
    def memory_catalog(self) -> str:
        entries = sorted(self.notes, key=lambda n: -n.importance)
        return (
            "\n".join(f"- [{n.id}] {n.title}" for n in entries)
            or "(no notes recorded yet)"
        )

    @Skill.define
    def plan_presses(self, observation: str) -> list[str]:
        """Decide the next button presses to attempt, given the current state
        of the corridor:

        <observation>
        {observation}
        </observation>

        Your standing strategy (rewrite it via `self.instructions` when you
        learn something structural):

        <instructions>
        {self.instructions}
        </instructions>

        Your memory catalog (bodies via `print(self.notes[i].body)` in the REPL):

        <memory>
        {self.memory_catalog}
        </memory>

        Return a short list of buttons (each one of "A", "B" or "C") to press
        next, in order. Presses are applied until one buzzes, so a plan past
        the first uncertain press is wasted only if that press is wrong.
        """


def harness_diff(agent: Agent, baseline: dict[str, typing.Any]) -> str:
    """A one-line-per-change report of how `agent`'s harness differs from
    `baseline` (a ``dict(vars(agent))`` snapshot taken before the run) --
    what the run authored, rebound, and retired, whatever the route."""
    current = {k: v for k, v in vars(agent).items() if not k.startswith("__")}
    previous = {k: v for k, v in baseline.items() if not k.startswith("__")}
    lines = []
    for name in sorted(current.keys() | previous.keys()):
        if name not in previous:
            lines.append(f"+ {name} = {current[name]!r}")
        elif name not in current:
            lines.append(f"- {name}")
        elif current[name] is not previous[name] and current[name] != previous[name]:
            lines.append(f"~ {name} = {current[name]!r}")
    return "\n".join(lines) or "(harness unchanged)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rooms", type=int, default=6, help="Number of rooms")
    parser.add_argument("--length", type=int, default=4, help="Code length per room")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the codes")
    parser.add_argument(
        "--budget", type=int, default=400, help="Press budget before giving up"
    )
    parser.add_argument(
        "--condition",
        choices=("scratch", "expert"),
        default="scratch",
        help="scratch: empty instructions; expert: a hand-written strategy",
    )
    args = parser.parse_args()

    # Imported here, not at module scope, for the same reason the expert
    # strategy's text lives in `gridworlds` rather than in this file: this
    # module's source is embedded in the system prompt, and a module-level
    # binding would additionally sit in every Skill's lexical scope -- either
    # way the scratch condition would be handed the expert strategy.
    from docs.source.llm_examples.reasoning.gridworlds import (
        CORRIDOR_EXPERT_STRATEGY,
    )

    # The answer key. Locals on purpose: module-level names are in every
    # Skill's lexical scope, printed into the system prompt, and readable
    # through the REPL.
    rng = random.Random(args.seed)
    codes = [
        "".join(rng.choice(Corridor.BUTTONS) for _ in range(args.length))
        for _ in range(args.rooms)
    ]
    optimal = sum(len(c) for c in codes)

    player = Player(
        instructions=CORRIDOR_EXPERT_STRATEGY if args.condition == "expert" else ""
    )
    # Deep, not shallow: `harness_diff` compares the run's attributes against
    # this snapshot, and most of what the agent does to its harness it does *in
    # place* -- `self.notes.append(...)`. A shallow copy shares that list, so
    # the before and after are the same object and every mutation reads as no
    # change at all.
    baseline = copy.deepcopy(vars(player))

    corridor = Corridor(codes=codes)
    while not corridor.solved and corridor.presses < args.budget:
        presses = player.plan_presses(corridor.observe())
        for button in presses:
            if corridor.solved or corridor.presses >= args.budget:
                break
            if button not in corridor.BUTTONS:
                corridor.events.append(f"ignored invalid button {button!r}")
                continue
            corridor.press(button)

    print(f"\nsolved: {corridor.solved}")
    print(f"presses: {corridor.presses} (optimal {optimal}, budget {args.budget})")
    print(f"\nharness changes this run:\n{harness_diff(player, baseline)}")

    assert corridor.solved, "ran out of press budget before the corridor was solved"


if __name__ == "__main__":
    main()
