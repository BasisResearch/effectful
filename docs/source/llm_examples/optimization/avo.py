"""Agentic variation: the evaluator moves inside the model's reach (AVO).

Implements "AVO: Agentic Variation Operators for Autonomous Evolutionary Search"
(arXiv 2603.24517). Its argument is against the system the rest of this directory
implements. Evolutionary search with an LLM decomposes variation as
``Vary(P) = Generate(Sample(P))``, and the model is confined to ``Generate``: it emits
one candidate per invocation, having tested nothing. AVO replaces the entire operator
with a single autonomous agent run, ``Vary(P) = Agent(P, K, f)``, where the agent is
handed the lineage ``P``, a knowledge base ``K``, and -- the load-bearing part -- the
scoring function ``f`` itself, as a utility it may call whenever it likes. It edits,
evaluates, reads the diagnostics, repairs, and only then commits.

The whole of that change, in this library, is four lines of ordinary Python:

  1. **``f`` becomes a `Tool`** in the variation skill's lexical scope. The multi-turn
     agent loop the paper builds around it is what a `Skill` call already is -- the
     model calls `VariationAgent.evaluate` as many times as its budget allows before
     answering -- so the paper's central mechanism is a scope change, not a subsystem.
  2. **The lineage is in reach**: the incumbent arrives as a typed ``Kernel``
     argument, rendered into the prompt as its own source, and stays callable *by
     name* in the REPL, so the agent can time it against its candidate.
  3. **One persistent agent** spans the whole run, where `kernels.py` builds a fresh
     `Proposer` per iteration. Its history is the paper's memory, and is what the
     transfer episode below rides on.
  4. **The commit rule inverts**: the agent self-certifies, and the framework shrinks
     to `evolve_lineage` -- verify what came back, commit if it earns a place.

The domain is `kernels.py`'s, deliberately: both examples score through one
`evaluate_kernel`, so the difference between them is the operator and not the
yardstick. The artifact here is the kernel *itself*, which is the paper's artifact,
where `kernels.py` evolves the instruction that writes one.

Demonstrates:
- An evaluator handed to the model as a tool, and metered, so "how many times did the
  agent score something" is a number in the report rather than an unknown
- A synthesized callable passed *by reference* to a tool: the agent defines a kernel in
  the REPL and writes ``self.evaluate(candidate)``, which the harness type-checks as a
  real Python call expression -- no serialization of the artifact anywhere
- The stdlib as the knowledge base: ``help(math)`` in the REPL is the paper's
  documentation-retrieval channel, with no task-specific retrieval tool
- `TenacityRetryer` plus decode-time doctests as the paper's implicit repair stage
- An EVO control (``--evo``) that is the *same domain* through `optimize_anything`'s
  single-turn proposer, reported in both currencies -- though see below for why this
  domain cannot make that a test of anything
- Transfer (``--transfer``): the same agent, with its accumulated history, adapting its
  kernel to a sibling task -- the paper's 30-minute MHA -> GQA episode
"""

# Simplifications vs. the source:
# - The domain cannot discriminate the two operators, which is the load-bearing
#   simplification and the one to read first (see "What this example does not do"). The
#   task was chosen for having a measurable optimization ladder and a correctness cliff;
#   it should also have been chosen for having a ladder whose *order a model cannot
#   predict*, since that is the only condition under which holding the evaluator can
#   beat guessing. Every other entry in this list is a simplification of the paper's
#   setup; this one is a limitation of the experiment.
# - Pure-Python list transforms on a CPU, not CUDA kernels on a B200 against cuDNN and
#   FlashAttention-4. The baseline is a deliberately plain reference implementation
#   rather than months of expert tuning, so "3x the baseline" here and "3.5% over
#   cuDNN" there are not the same kind of quantity, and the gap between them is most
#   of what makes the paper's result hard -- and most of why an agent that can profile
#   is worth its cost there and is not here.
# - Minutes and a handful of steps against 7 days and 40 committed versions. The
#   paper's trajectory -- discrete jumps separated by plateaus -- needs a budget this
#   example does not spend; expect one large jump onto the best rung of the ladder and
#   then grinding, which is what the ladder above predicts.
# - No supervisor. The paper adds one that watches for stagnation and steers the search
#   when it plateaus, but never ablates it: there are no intervention counts and no
#   with/without arm, so its contribution is asserted rather than measured. At this
#   budget a stagnation trigger would likely never fire. Restoring it is a stagnation
#   counter plus one `redirect` skill on a *separate* agent -- separate so that it
#   reviews from outside the context that plateaued, and so that lexical scope keeps
#   `evaluate` out of its reach.
# - The knowledge base is the standard library rather than PTX ISA documentation, and
#   the agent reaches it the same way the paper's does: by reading docs in its own
#   session, not through a retrieval tool written for the task.
# - No git. The lineage is a list in memory; the paper commits each version, which is
#   how its run survives a restart. `Agent.__agent_id__` plus ``--persist-db`` would be
#   the equivalent here and is not wired up.
# - The two arms cannot be matched on both currencies, and this is not a defect of the
#   comparison but of what the two operators buy. One AVO step spends one proposal and
#   as many evaluator calls as its budget allows; one EVO iteration spends one of each.
#   ``--evo`` matches on evaluator calls, which is the currency the paper's claim is
#   about (the agent's advantage is supposed to be worth what it costs to test), and
#   the report prints both counts for both arms so a reader can see the other side of
#   the trade.
# - One run per arm and no variance estimate, against a timing ratio whose noise floor
#   is a few percent (visible in the seed's own score, which is 1.0 by construction and
#   does not measure as exactly 1.0). Differences smaller than that are noise, and the
#   report says so rather than ranking the arms.
# - Both arms see this module's source, because the harness puts it in the system
#   prompt. The test cases and the reference implementation are therefore *not* hidden
#   from either arm, whatever `kernels.py` says about its own: the enforceable
#   asymmetry between the arms is that the AVO agent is *offered* a metered evaluator
#   and the EVO proposer is not, and a control that chose to reconstruct scoring for
#   itself in the REPL would be neither prevented nor undetected -- the report prints
#   both arms' evaluator counts partly so that this stays visible.

import argparse
import collections.abc
import random
import time

from docs.source.llm_examples.optimization.kernels import (
    KERNEL_TASKS,
    Kernel,
    KernelTask,
    evaluate_kernel,
)
from docs.source.llm_examples.optimization.library import (
    Diagnostic,
    Evaluation,
    Lineage,
    Result,
    Rollout,
    evolve_lineage,
    optimize_anything,
    source_of,
)
from effectful.handlers.llm import Agent, Skill, Tool

# The timed configurations: the paper scores each kernel on a vector of benchmark
# shapes, and keeping several here is what stops a candidate from winning by being
# good at one size. Small enough that a whole evaluation costs a fraction of a second,
# because the agent runs many of them per step.
CONFIGS: tuple[int, ...] = (30_000, 100_000, 300_000)

TASKS: dict[str, KernelTask] = {task.name: task for task in KERNEL_TASKS}

# "MHA" and "GQA": the task evolved, and the sibling the result is transferred to.
TASK = TASKS["zscore"]
TRANSFER_TASK = TASKS["l2_normalize"]


class VariationAgent(Agent):
    """You are a performance engineer optimizing one small Python kernel, over a long
    run in which you will be asked for improvements repeatedly.

    You have a real evaluator: `evaluate` runs the kernel you hand it against the
    hidden correctness cases and then times it against the reference implementation on
    several input sizes. USE IT. Write a candidate in the REPL, evaluate it, read what
    came back, and fix what it tells you -- the whole point of the loop you are in is
    that you can measure instead of guessing. An idea you have not evaluated is not an
    improvement.

    Two things are worth remembering across the whole run, because you are the same
    agent each time: which optimizations actually paid, and which looked promising and
    did not. Say so when you answer, and if the transcript gets long, compact it.
    """

    def __init__(
        self,
        task: KernelTask,
        configs: collections.abc.Sequence[int] = CONFIGS,
        step_budget: int = 6,
    ):
        super().__init__()
        self.task = task
        self.configs = tuple(configs)
        self.step_budget = step_budget
        self.spent = 0  # evaluator calls in the current variation step
        self.total = 0  # ... and over the whole run, which is the reported currency

    @Tool.define
    def evaluate(self, kernel: Kernel) -> Evaluation:
        """Score a candidate kernel: correctness first, then speed.

        The kernel is run against hidden correctness cases -- including degenerate inputs
        the specification mentions -- and then timed on several large inputs against the
        reference implementation. The score is the geometric mean of the per-size
        speedups, and an incorrect kernel scores zero however fast it is.

        Pass the function itself, not its source: define it with the REPL tool and then
        call this one on the name you bound.
        """
        self.total += 1
        if self.spent >= self.step_budget:
            return Evaluation(
                score=0.0,
                diagnostics=[
                    Diagnostic(
                        "BUDGET EXHAUSTED",
                        f"This is NOT a score for your kernel -- it is a refusal to "
                        f"measure. You have used all {self.step_budget} evaluations "
                        f"for this step. Return the best kernel you have already "
                        f"measured.",
                    )
                ],
            )
        self.spent += 1
        evaluation = evaluate_kernel(kernel, self.task, self.configs)
        return Evaluation(
            score=evaluation.score,
            metrics=evaluation.metrics,
            diagnostics=[
                *evaluation.diagnostics,
                Diagnostic(
                    "evaluations remaining",
                    f"{self.step_budget - self.spent} of {self.step_budget} left in "
                    f"this step",
                ),
            ],
        )

    @Skill.define
    def vary(self, current: Kernel, evaluation: Evaluation) -> Kernel:
        """Improve this kernel.

        <specification>
        {self.task.spec}
        </specification>

        The current best kernel is below, and it is also bound in your REPL session as
        ``current``, so you can time your candidate against it directly.

        <current_kernel>
        {current}
        </current_kernel>

        Here is how it scored. Read it before writing anything: it reports the speedup
        at each input size, which failing cases there were, and the code that was
        measured.

        <evaluation>
        {evaluation}
        </evaluation>

        Work like an engineer, not like an oracle:

        - Write candidates in the REPL and hand each one to `evaluate`. You have a
          budget of evaluations per step; the diagnostics tell you how many are left.
        - When a candidate is slower or wrong, the diagnostics say which case failed
          and how fast it ran. Diagnose it before you try again.
        - Consult the standard library rather than guessing at it: ``help(math)``,
          ``import numpy`` and the like all work in the REPL, and whether a function
          exists and what it costs are both things you can check.
        - Watch the degenerate cases. The specification names them, and the fastest
          arithmetic is often exactly the arithmetic that gets them wrong.

        Return a kernel that you have MEASURED to be correct and at least as fast as
        the one you were given. It is re-scored after you return it, and it is rejected
        if it does not hold up so returning something you did not evaluate wastes the step.

        Your function's docstring MUST contain doctests certifying its contract, and
        they are run before your answer is accepted. Write at least one that checks the
        degenerate input the specification describes, prefixing each input line with
        the doctest prompt (three ``>`` characters and a space; it is spelled out
        rather than shown so that this instruction is not itself collected as a test).
        """


def run_avo(args: argparse.Namespace) -> tuple[Lineage[Kernel], VariationAgent]:
    """Agentic variation: one persistent agent, one lineage, the evaluator in reach."""
    agent = VariationAgent(TASK, CONFIGS, args.step_budget)

    def vary(current: Kernel, evaluation: Evaluation) -> Kernel:
        # The per-step budget resets here rather than inside the tool: the tool cannot
        # see where one variation step ends and the next begins, and the agent's
        # history spans all of them.
        agent.spent = 0
        return agent.vary(current, evaluation)

    lineage = evolve_lineage(
        vary=vary,
        evaluator=lambda kernel: evaluate_kernel(kernel, TASK, CONFIGS),
        seed=seed_kernel(TASK),
        budget=args.budget,
    )
    return lineage, agent


def run_evo(args: argparse.Namespace, rng: random.Random) -> Result:
    """The control: `optimize_anything` in single-task mode over the same artifact.

    Single-task mode, not a one-element dataset: with no dataset the Pareto objectives
    are the evaluation's own sub-scores, which here are the per-configuration speedups
    -- so the frontier can hold a candidate that wins only at one input size, which is
    the mode's whole purpose.
    """

    @Skill.define
    def propose_kernel(current: Kernel, feedback: list[Rollout]) -> Kernel:
        """You are a reflective optimizer. Rewrite this kernel to be faster.

        <current_kernel>
        {current}
        </current_kernel>

        Here is how it scored, including the speedup at each input size and any failing
        cases:

        <feedback>
        {feedback}
        </feedback>

        Diagnose what is costing the most, then write a faster kernel that computes exactly
        the same thing -- including on the degenerate inputs, since an incorrect kernel
        scores zero however fast it is.

        Your function's docstring MUST contain doctests certifying its contract, prefixing
        each input line with the doctest prompt (three ``>`` characters and a space).
        """

    return optimize_anything(
        evaluator=lambda kernel, _: evaluate_kernel(kernel, TASK, CONFIGS),
        proposer=propose_kernel,
        seed=seed_kernel(TASK),
        budget=args.evo_budget or args.budget * args.step_budget,
        selection=args.selection,
        rng=rng,
        task_name=TASK.name,
    )


def seed_kernel(task: KernelTask) -> Kernel:
    """Candidate zero: the reference implementation itself.

    Both arms start from it, and it is also the denominator of the score, so the seed
    measures 1.0 by construction -- give or take the timing noise, which is worth
    seeing. "The search improved on its seed" therefore means "it beat the
    straightforward implementation", with no baseline offset to argue about.
    """
    from docs.source.llm_examples.optimization.kernels import REFERENCE

    return REFERENCE[task.name]


def transfer(agent: VariationAgent, kernel: Kernel) -> tuple[Evaluation, Evaluation]:
    """The paper's MHA -> GQA episode: the same agent, a sibling task, one step.

    Nothing is reset. The agent keeps the history in which it discovered whatever it
    discovered about ``zscore``, and is asked for a kernel for a task that shares that
    task's shape but not its answer. The paper reports 30 minutes of autonomous
    adaptation for the same move.

    Returns the sibling task's seed evaluation and the adapted kernel's, so the report
    can state the transfer as a ratio against the reference rather than as a claim.
    """
    agent.task, agent.spent = TRANSFER_TASK, 0
    before = evaluate_kernel(seed_kernel(TRANSFER_TASK), TRANSFER_TASK, CONFIGS)
    adapted = agent.vary(kernel, before)
    return before, evaluate_kernel(adapted, TRANSFER_TASK, CONFIGS)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def speedups_of(evaluation: Evaluation) -> str:
    return ", ".join(
        f"{m.name.removeprefix('speedup@')}: {m.value:.2f}x"
        for m in evaluation.metrics
        if m.name.startswith("speedup@")
    )


def report_avo(lineage: Lineage[Kernel], agent: VariationAgent, seconds: float) -> None:
    """The trajectory, the two currencies, and the winning kernel."""
    print("\n" + "=" * 72)
    print(
        f"mode: agentic variation (AVO) | {lineage.attempts} variation steps in "
        f"{seconds:.0f}s"
    )
    print(
        f"seed {lineage.seed.score:.6g} -> best {lineage.best.score:.6g} "
        f"({len(lineage.versions) - 1} committed, {lineage.rejected} rejected)"
    )
    print(
        f"cost: {agent.total} evaluator calls by the agent + {lineage.evaluations} "
        f"verifications by the framework = {agent.total + lineage.evaluations} total, "
        f"over {lineage.attempts} proposals"
    )
    print("\nCommitted lineage:")
    for version in lineage.versions:
        print(
            f"  v{version.index} ({version.note}): {version.score:.4g}  "
            f"[{speedups_of(version.evaluation)}]"
        )
    print("\nBest kernel:")
    print((source_of(lineage.best.artifact) or repr(lineage.best.artifact)).rstrip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget", type=int, default=4, help="Variation steps (AVO arm)"
    )
    parser.add_argument(
        "--step-budget",
        type=int,
        default=6,
        help="Evaluator calls the agent may make within one variation step",
    )
    parser.add_argument(
        "--evo",
        action="store_true",
        help="Run the single-turn control instead: the same domain through "
        "optimize_anything, matched on evaluator calls",
    )
    parser.add_argument(
        "--evo-budget",
        type=int,
        default=0,
        help="Iterations for --evo; 0 matches the AVO arm's evaluator calls "
        "(--budget x --step-budget)",
    )
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="After evolving, adapt the winning kernel to a sibling task in one step "
        "-- the paper's MHA -> GQA episode",
    )
    parser.add_argument(
        "--selection",
        choices=["pareto", "best"],
        default="pareto",
        help="Candidate selection for the --evo control",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the control")
    args = parser.parse_args()

    if args.evo:
        evo = run_evo(args, random.Random(args.seed))
        print("\n" + "=" * 72)
        print(
            f"mode: EVO control | seed {evo.seed_score:.6g} -> "
            f"best {evo.best_score:.6g} ({evo.proposals} proposals, "
            f"{evo.evaluations} evaluator calls)"
        )
        print("\nBest kernel:")
        print((source_of(evo.best.artifact) or repr(evo.best.artifact)).rstrip())
        return

    started = time.monotonic()
    lineage, agent = run_avo(args)
    report_avo(lineage, agent, time.monotonic() - started)

    if args.transfer:
        before, after = transfer(agent, lineage.best.artifact)
        print("\n" + "=" * 72)
        print(
            f"Transfer to {TRANSFER_TASK.name}, one variation step with the agent's "
            f"history intact:"
        )
        print(f"  reference implementation: {before.score:.4g}")
        print(f"  adapted kernel:           {after.score:.4g}")
        print(f"  [{speedups_of(after)}]")


if __name__ == "__main__":
    main()
