"""Schema-style world-model agent: learn a hidden game by writing its rules as code.

Inspired by the "Schema" harness (https://schema-harness.github.io/), which has a
model play a game with hidden rules "like a physicist": write the game's mechanism
as an executable program, test it against the recorded history, and plan inside it.

Demonstrates:
- A ``Template`` returning a ``Callable`` -- the model's *world model* is executable
  Python, synthesized once and then run thousands of times by ordinary code
- An ``Agent`` whose persistent state is the memory: an append-only Timeline of real
  transitions that is fed back into every deliberation as ground truth
- Certification at synthesis: the model embeds recorded transitions as doctests in the
  ``step`` it writes, and the ``Callable`` decoder runs them -- a model that fails to
  reproduce recorded history is rejected and fed back before it is ever used
- Reality-outranks-model: during execution a single mispredict voids the rest of the
  plan and appends the surprising transition, forcing a re-theorize next round; a
  plain-Python BFS searches *inside* the synthesized model for free

The Timeline is external memory fed into every deliberation as ground truth -- and as
the doctests that certify each model -- so it is genuinely distinct from the Agent's
conversational history. We deliberately keep
the *notes* store out of this (a) variant: because the tiny game never overflows the
context window, a rewritable notes summary would only duplicate ``__history__``. At
ARC-AGI-3 scale, where context is auto-compacted, a curated notes file stops being
redundant and becomes the model's "weights" -- that is the (b) variant this omits.
"""

import argparse
import collections
import collections.abc
import dataclasses
import textwrap

from gridworlds import Action, Color, Game, State, Transition

from effectful.handlers.llm import Agent, Template


@dataclasses.dataclass
class Physicist(Agent):
    """Reverse-engineers the game by writing its ``step`` rule as Python code."""

    hint: str
    timeline: list[Transition] = dataclasses.field(default_factory=list)

    @Template.define
    def theorize(
        self, state: State, action: Action | None = None
    ) -> collections.abc.Callable[[State, Action], State]:
        """You are reverse-engineering a 2D grid game by writing its rules as code.
        You've been given a high-level hint about the game:

        <hint>
        {self.hint}
        </hint>

        Beyond that, the dynamics are hidden; infer them ONLY from these recorded transitions:

        <timeline>
        {self.timeline}
        </timeline>

        The current world state, which you will plan beyond using the model, is:

        <state>
        {state}
        </state>

        You may also have access to a prospective Action (although it may be None):

        <action>
        {action}
        </action>

        Think through the problem, using tools to explore and test hypotheses if necessary,
        and write a pure Python function ``step(state, action)``
        that reproduces every recorded transition exactly.
        The function's docstring **MUST** include all salient recorded transitions
        from the timeline as runnable doctests. If there are no recorded transitions,
        you do not need to include any doctests.
        """

    def plan(
        self,
        model: collections.abc.Callable[[State, Action], State],
        start: State,
        *,
        max_nodes: int = 5000,
    ) -> list[Action]:
        """Search *inside* the model for a plan reaching the goal (BOX_ON_TARGET). Free.

        Returns the action sequence to a goal state (``[]`` if ``start`` already wins),
        or ``[]`` if no plan is found within ``max_nodes``.
        """
        solved = lambda s: any(Color.BOX_ON_TARGET in row for row in s.grid)  # noqa: E731
        if solved(start):
            return []
        frontier: collections.deque[tuple[State, list[Action]]] = collections.deque(
            [(start, [])]
        )
        seen: set[State] = {start}
        while frontier and len(seen) < max_nodes:
            state, plan = frontier.popleft()
            for action in Action:
                try:
                    nxt = model(state, action)
                except Exception:
                    continue  # can't plan through a rule that crashes
                if nxt in seen:
                    continue
                if solved(nxt):
                    return plan + [action]
                seen.add(nxt)
                frontier.append((nxt, plan + [action]))
        return []

    def solve(self, env: Game, *, max_actions: int = 40) -> bool:
        """
        Outer loop: observe, deliberate, plan, execute.
        """
        while len(self.timeline) < max_actions:
            # Observe the current state of reality and print it.
            state = env.observe()
            print(f"\ncurrent grid ({len(self.timeline)} real actions spent):\n{state}")

            # Deliberate: synthesize a step() model; its embedded doctests certify it
            # against the recorded Timeline at decode time.
            model = self.theorize(state)

            # Plan inside the certified model for free; if none, take one probing step.
            plan = self.plan(model, state)
            if not plan:
                plan = [Action(len(self.timeline) % len(Action))]
                print(
                    f"[plan] no solution in model; probing with action {plan[0].name}"
                )
            else:
                print(f"[plan] found in model: {[a.name for a in plan]}")

            # Execute against reality, checking each prediction. A surprise voids the rest.
            for action in plan:
                predicted = model(state, action)
                actual, done = env.step(action)
                self.timeline.append(Transition(state, action, actual))
                state = actual
                # real_actions += 1  # removed
                if done:
                    print(
                        f"[execute] action {action.name} -> SOLVED in {len(self.timeline)} actions"
                    )
                    return True
                if actual != predicted:
                    print(f"[execute] action {action.name} -> surprise; plan voided")
                    break
                print(f"[execute] action {action.name} -> as predicted")

        print(f"\nGave up after {len(self.timeline)} actions.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        type=str,
        choices=[
            "push",
        ],
        default="push",
        help="Which hidden game to solve",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=40,
        help="Budget of real environment actions before giving up",
    )
    args = parser.parse_args()

    if args.env == "push":
        from gridworlds import PushGame

        game = PushGame()
    else:
        raise ValueError(f"Unknown environment {args.env}")

    assert game.__doc__, "Game must have a docstring hint for the agent"
    phys = Physicist(hint=textwrap.dedent(game.__doc__))
    solved = phys.solve(game, max_actions=args.max_actions)
    assert solved, "Failed to solve the game within the action budget."


if __name__ == "__main__":
    main()
