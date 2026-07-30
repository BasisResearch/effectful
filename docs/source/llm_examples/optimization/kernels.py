"""Kernel instructions: multi-task search over a shared frontier (optimize_anything 5.2).

A dataset of related problems is supplied but no validation set, which selects the
paper's multi-task mode -- the one no prior LLM-evolution framework has. The frontier is
shared across tasks so a pattern discovered while working on one is available as a
parent when proposing for another, and at output time each task independently picks its
own best candidate off that frontier. Multi-task search therefore produces N specialized
artifacts that have all benefited from a common search, which is the distinction the
paper draws against generalization mode's single artifact.

The artifact is the *instruction that drives code generation*, exactly as the paper
evolves the prompt behind its CUDA kernels rather than the kernels themselves. Each
evaluation hands that instruction to a cheaper programmer model (``--worker-model``),
which writes the function, and scores what comes back.

Scoring borrows KernelBench's *shape* -- correctness against a reference
implementation, then wall-clock speedup against it -- and that shape is what makes the
domain optimizable at all: the worker writes a correct list transform from the bare
seed instruction on all five tasks, so correctness alone would saturate immediately.
Correctness is not, however, a floor the search stays above. An instruction that pushes
hard for speed makes the worker write kernels that fail, and those score zero; the gate
is a cliff the search repeatedly falls off, which the measured numbers below show.

Read the speedups with the baseline in mind, because it is not the paper's. KernelBench
compares against PyTorch, which is cuDNN and cuBLAS -- vendor-tuned code, and the reason
"87% match or beat the baseline" is a strong claim. The reference implementations here
are deliberately plain Python, explicit loops and ``append``, slow enough that ruff
objects to them in as many words. Beating them by 1.6x is beating unoptimized
interpreter code, not a tuned library, and the two numbers are not comparable.

The five tasks share their *failure modes* rather than their algorithm, which is what
gives cross-transfer something to transfer: three have a naive formulation that rescans
the whole prefix and is quadratic, and two turn on degenerate inputs a hurried
implementation skips. Both lessons are worth less than they sound. The references
already carry running state, so an instruction that teaches it recovers the baseline
rather than beating it -- 0.0 to about 1.0, a timeout fix wearing a speedup's clothes --
and the degenerate cases are stated in the specification text the worker is handed, so
the transferable insight there is "read the specification".

Demonstrates:
- Multi-task mode: per-task Pareto objectives on one shared frontier, per-task winners
  at output time, and a count of how many of those winners were last refined while the
  proposer was looking at a *different* task -- reported against the count chance alone
  would produce, because on five tasks with a two-task minibatch that null is 3.0 and
  the raw count says nothing on its own
- A single-task control (``--single-task``) that re-optimizes each task independently,
  which is the comparison the paper's 5.4 reports -- though see the simplifications on
  what "equivalent budget" does and does not mean here
- Side Information as compiler-style feedback: failing cases with expected and actual
  values, measured times and speedup, the traceback, and the code itself
- A correctness gate that a search for speed can and does fall foul of

Measured on 2026-07-29 with gpt-5.5 proposing and gpt-4.1-mini writing kernels, 10
iterations per arm. Mean speedup over the reference implementation:

    multi-task            1.336 -> 1.399   (31 evaluator calls)
      best single artifact       1.349     (the matched comparison; see below)
    single-task control   1.194 -> 1.407   (15 evaluator calls, 2 iterations per task)

The paper's 5.4 finding -- multi-task ahead of single-task at equivalent per-problem
budget -- does not reproduce here, and on this evidence nothing else does either. The
control ends slightly ahead on the headline number and well ahead on the gain (+17.9%
against +4.7%), while spending half the evaluator calls. Cross-task transfer came out
at 3 of 5 winners refined on another task against a chance expectation of exactly 3.0,
which is no signal at all.

None of that is a refutation of the paper, and it should not be read as one. The two
arms' seed measurements differ by 12% (1.336 against 1.194) for the *same* instruction
on the *same* five tasks, so the noise floor here is wider than either effect; the
multi-task headline is a maximum over a pool while the control's is a maximum over five
smaller pools; and one run of each, with 5 tasks against the paper's 31, cannot
separate a real difference from timing jitter. What this domain demonstrates is the
machinery -- a shared frontier, per-task selection off it, a control to compare against,
and a transfer statistic reported against its null -- not a result.

Read those numbers as provenance rather than as a current result: they were measured
before the prompts and the control reached their present form. The worker's prompt said
it would be tested "including degenerate inputs" and the proposer's said to prefer
"which degenerate inputs to handle explicitly, what to check before dividing, how to
treat boundaries" -- between them both of the two lessons this task family is built
around, named for the proposer in the prompt asking it to find them. And the control
passed a one-element dataset rather than running the engine's single-task mode, which
made it differ from the treatment arm in its selection rule as well as its task count
(see `run_kernel`). Re-running both arms under the present code is outstanding.

One thing any number here includes and cannot be separated from: the harness's
``TenacityRetryer`` sits above the worker model, so a kernel whose source does not
decode is fed its own error and asked again. Every speedup is therefore a speedup for
the instruction *plus that repair loop*, and an instruction that provokes
borderline-undecodable code is flattered by it.
"""

# Simplifications vs. the source:
# - Pure-Python list transforms on a CPU, not CUDA kernels on a V100 against
#   KernelBench's 31 PyTorch operations, and no NVCC in the loop. The baseline is
#   plain Python rather than a vendor-tuned library, so a speedup here is not the same
#   quantity the paper reports (see the header).
# - The score is mean speedup rather than the paper's fast_p(s) curve, and the side
#   information has no documentation-retrieval channel. With five tasks, fast_p could
#   be reported in 20% increments from the same per-task scores; it is not.
# - Budget is counted in optimizer iterations, not metric calls or dollars; the paper
#   spends ~3000 metric calls and $140 on this domain.
# - The two arms are NOT budget-matched in the paper's currency. Matching iterations,
#   as ``--single-task`` does by default, leaves multi-task with roughly three times the
#   evaluator calls -- it spends a full evaluation across all five tasks whenever a
#   proposal is accepted, while the control spends two calls on one task. ``report``
#   prints each arm's evaluation count, and ``--control-iterations`` exists to match on
#   that instead. Multi-task also takes its per-task maximum over a larger pool, which
#   favours it for reasons unrelated to transfer.
# - Nor can the two arms be made to differ in exactly one way. Single-task mode has no
#   per-example objectives to keep a frontier over, so choosing it necessarily changes
#   both the task count and what the Pareto objectives are. That is why the paper
#   introduces per-metric objectives for that mode, and it is a property of its own
#   comparison as much as of this one.
# - The headline gain is upward-biased on the multi-task side and cannot be negative
#   there: the result is a per-task maximum over the whole pool while the seed is a
#   single candidate's mean, so per-task max is >= the seed's score on every task by
#   construction. The report prints the best *single* artifact's mean alongside it,
#   which is the matched one-artifact-to-one-artifact comparison.
# - No repeats on either arm, and no variance estimate. The two seed measurements in
#   the numbers above differ by several percent for the same instruction on the same
#   tasks, which bounds the noise floor and is not far below the effects being compared.
# - Cross-task transfer is a lineage count over one run against one control, not the
#   paper's MT10/MT20 scaling study. It inspects only the last refinement rather than
#   full ancestry, so it is reported against its null expectation rather than alone.
# - The score is a wall-clock ratio, so it is only as reproducible as the machine is
#   quiet. The baseline is re-timed back to back with every candidate for exactly this
#   reason (see `measure_speedup`), which makes the ratio robust to a loaded machine but
#   not to one whose speed changes mid-measurement.

import argparse
import bisect
import collections.abc
import math
import random
import signal
import statistics
import threading
import time
import traceback
import zlib

import pydantic.dataclasses

from docs.source.llm_examples.optimization.library import (
    WORKER_MODEL,
    Candidate,
    Diagnostic,
    Evaluation,
    Metric,
    Result,
    Rollout,
    optimize_anything,
    report,
    source_of,
    worker,
)
from effectful.handlers.llm import Agent, Template

# A kernel is one list-to-list transform; every task in the family shares this
# signature so a single instruction can drive all of them.
type Kernel = collections.abc.Callable[[list[float]], list[float]]


class _Timeout(Exception):
    pass


@pydantic.dataclasses.dataclass(frozen=True)
class KernelTask:
    """One task in the family: what the transform must compute. The test cases are
    hidden -- they live in ``KERNEL_TESTS``, and reach the proposer only as the
    specific failures reported in Side Information."""

    name: str
    spec: str


# A family that shares its *failure modes* rather than its algorithm, which is what
# makes cross-transfer possible at all. Two lessons run through it. Three of the tasks
# have a naive formulation that recomputes over the whole prefix or window and is
# quadratic -- correct on the small cases, far too slow on the timed one -- so the
# transferable lesson is "carry running state instead of rescanning". The other two
# turn on degenerate inputs the specification states and a hurried implementation
# skips. An instruction that learns either lesson on one task collects points on the
# others, exactly as the paper's CUDA instruction learns coalescing once and spends it
# across 31 kernels.
KERNEL_TASKS: list[KernelTask] = [
    KernelTask(
        "count_smaller_before",
        "For each position i, output the number of earlier positions j < i whose value "
        "is strictly smaller than the value at i. The output has the same length as "
        "the input; an empty input gives an empty output.",
    ),
    KernelTask(
        "window_sum_101",
        "For each position i, output the sum of the values from index max(0, i - 100) "
        "through i inclusive -- a trailing window of up to 101 values, shorter near "
        "the start. The output has the same length as the input; an empty input gives "
        "an empty output.",
    ),
    KernelTask(
        "distinct_prefix_counts",
        "For each position i, output how many distinct values occur in the input up to "
        "and including position i. The output has the same length as the input; an "
        "empty input gives an empty output.",
    ),
    KernelTask(
        "l2_normalize",
        "Divide every value by the Euclidean (L2) norm of the whole input, so the "
        "result has unit norm. If that norm is exactly zero, every output value is "
        "0.0. The output has the same length as the input; an empty input gives an "
        "empty output.",
    ),
    KernelTask(
        "zscore",
        "Standardize the input: subtract the mean and divide by the population "
        "standard deviation. If that standard deviation is exactly zero, every output "
        "value is 0.0. The output has the same length as the input; an empty input "
        "gives an empty output.",
    ),
]

type Case = tuple[list[float], list[float]]

# The small cases are the contract, written out so the edge semantics are readable
# rather than implied by a reference implementation.
KERNEL_TESTS: dict[str, list[Case]] = {
    "count_smaller_before": [
        ([], []),
        ([5.0], [0.0]),
        ([3.0, 1.0, 2.0], [0.0, 0.0, 1.0]),
        ([2.0, 2.0, 1.0, 4.0], [0.0, 0.0, 0.0, 3.0]),
    ],
    "window_sum_101": [
        ([], []),
        ([5.0], [5.0]),
        ([1.0, 2.0, 3.0], [1.0, 3.0, 6.0]),
        ([-1.0, 1.0, -1.0, 1.0], [-1.0, 0.0, -1.0, 0.0]),
    ],
    "distinct_prefix_counts": [
        ([], []),
        ([5.0], [1.0]),
        ([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]),
        ([3.0, 1.0, 3.0, 2.0], [1.0, 2.0, 2.0, 3.0]),
    ],
    "l2_normalize": [
        ([], []),
        ([0.0, 0.0], [0.0, 0.0]),
        ([3.0, 4.0], [0.6, 0.8]),
        ([-3.0, 4.0], [-0.6, 0.8]),
    ],
    "zscore": [
        ([], []),
        ([5.0, 5.0, 5.0], [0.0, 0.0, 0.0]),
        ([2.0], [0.0]),
        ([1.0, 2.0, 3.0], [-1.224744871391589, 0.0, 1.224744871391589]),
    ],
}


# Written the plain way on purpose: explicit loops, ``append`` per element, arithmetic
# spelled out. This is the "straightforward implementation" a competent programmer
# reaches for first, and it is the baseline the score is a ratio against -- the role
# KernelBench's unoptimized PyTorch reference plays in the paper. Rewriting these with
# comprehensions, ``itertools.accumulate``, locally bound methods or a reciprocal
# multiply is exactly the headroom the search is asked to find. (The ``noqa``s below
# are load-bearing: ruff is right that a comprehension would be faster, and being
# slower than that is precisely this code's job.)


def _count_smaller_before(values: list[float]) -> list[float]:
    counts: list[float] = []
    seen: list[float] = []
    for x in values:
        counts.append(float(bisect.bisect_left(seen, x)))
        bisect.insort(seen, x)
    return counts


def _window_sum_101(values: list[float]) -> list[float]:
    out: list[float] = []
    running = 0.0
    for i in range(len(values)):
        running = running + values[i]
        if i >= 101:
            running = running - values[i - 101]
        out.append(running)
    return out


def _distinct_prefix_counts(values: list[float]) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for i in range(len(values)):
        seen.add(values[i])
        out.append(float(len(seen)))
    return out


def _l2_normalize(values: list[float]) -> list[float]:
    total = 0.0
    for i in range(len(values)):
        total = total + values[i] * values[i]
    norm = math.sqrt(total)
    out: list[float] = []
    for i in range(len(values)):
        out.append(0.0 if norm == 0.0 else values[i] / norm)  # noqa: PERF401
    return out


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    total = 0.0
    for i in range(len(values)):
        total = total + values[i]
    mean = total / len(values)
    variance = 0.0
    for i in range(len(values)):
        variance = variance + (values[i] - mean) * (values[i] - mean)
    deviation = math.sqrt(variance / len(values))
    out: list[float] = []
    for i in range(len(values)):
        out.append(  # noqa: PERF401
            0.0 if deviation == 0.0 else (values[i] - mean) / deviation
        )
    return out


# The baseline every candidate is measured against -- the straightforward linear
# implementation a competent programmer writes without thinking about speed. It is
# never shown to the model; it supplies the expected output of the timed case and the
# denominator of the speedup, exactly as KernelBench's PyTorch baseline does in the
# paper's 5.2.
REFERENCE: dict[str, Kernel] = {
    "count_smaller_before": _count_smaller_before,
    "window_sum_101": _window_sum_101,
    "distinct_prefix_counts": _distinct_prefix_counts,
    "l2_normalize": _l2_normalize,
    "zscore": _zscore,
}

# Size of the timed input per task, and the wall-clock ceiling that stops a quadratic
# implementation instead of letting it hang the run. The sizes are chosen so the
# reference finishes in tens of milliseconds and a rescanning implementation cannot
# finish at all: 30k elements of prefix rescanning is ~450M comparisons.
KERNEL_PERF: dict[str, tuple[int, float]] = {
    "count_smaller_before": (30_000, 5.0),
    "window_sum_101": (300_000, 5.0),
    "distinct_prefix_counts": (300_000, 5.0),
    "l2_normalize": (300_000, 5.0),
    "zscore": (300_000, 5.0),
}

TIMING_REPEATS = 3  # best of three: the standard robust estimator for a short run


def perf_input(task: str) -> list[float]:
    """The timed case's input: deterministic pseudo-random values, so every candidate
    is timed on exactly the same work. Seeded from a checksum of the task name rather
    than its length, which silently gave two tasks the same input the moment their
    names happened to match in length."""
    rng = random.Random(zlib.crc32(task.encode()))
    return [rng.uniform(-1.0, 1.0) for _ in range(KERNEL_PERF[task][0])]


def check_reference_agrees() -> bool:
    """The reference implementations must reproduce the written-out contract, or the
    timed case would be testing a different function than the small cases do.

    >>> check_reference_agrees()
    True
    """
    for name, cases in KERNEL_TESTS.items():
        for values, expected in cases:
            produced = REFERENCE[name](list(values))
            assert len(produced) == len(expected) and all(
                math.isclose(p, e, rel_tol=1e-9, abs_tol=1e-9)
                for p, e in zip(produced, expected)
            ), f"reference for {name} disagrees with the contract on {values}"
    return True


SEED_INSTRUCTION = "Write a Python function that implements the specification."


class Programmer(Agent):
    """You are an expert Python programmer. You implement exactly the specification
    you are given, following the engineering instruction you are handed, and you
    answer with code rather than prose."""

    @Template.define
    def write_kernel(self, instruction: str, task: KernelTask) -> Kernel:
        """Write ``kernel(values)``: a function taking a ``list[float]`` and returning
        a ``list[float]``, implementing this specification exactly.

        <specification>
        {task.spec}
        </specification>

        Follow this engineering instruction while you write it:

        <instruction>
        {instruction}
        </instruction>

        Standard library only. It is checked against hidden cases and then TIMED on a
        large input: a correct implementation scores the ratio of a straightforward
        reference implementation's time to yours, and an incorrect one scores zero
        however fast it is. Write it to be both right and fast.
        """


def _time_kernel(
    kernel: Kernel, values: list[float], ceiling: float
) -> tuple[list[float], float]:
    """Best-of-``TIMING_REPEATS`` wall-clock time for one kernel on one input, with an
    alarm so a quadratic implementation is stopped rather than left to hang."""

    def _alarm(signum: int, frame: object) -> None:
        raise _Timeout(f"exceeded the {ceiling}s ceiling on {len(values)} values")

    guarded = threading.current_thread() is threading.main_thread()
    best, produced = math.inf, []
    if guarded:
        previous = signal.signal(signal.SIGALRM, _alarm)
    try:
        for _ in range(TIMING_REPEATS):
            if guarded:
                signal.setitimer(signal.ITIMER_REAL, ceiling)
            start = time.perf_counter()
            produced = list(kernel(list(values)))
            best = min(best, time.perf_counter() - start)
            if guarded:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
    finally:
        if guarded:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)
    return produced, best


def measure_speedup(kernel: Kernel, task: str) -> tuple[float, Diagnostic]:
    """Correctness-gated speedup over the reference on the large input.

    This is the paper's KernelBench metric in miniature: a kernel that is wrong scores
    nothing, and a kernel that is right scores how many times faster than the baseline
    it runs. It is also what keeps this domain from saturating -- every model writes a
    correct list transform on the first try, so correctness alone would have nothing
    left to optimize. It is not thereby a *floor* the search stays above: an instruction
    that pushes hard for speed makes the worker write kernels that fail, and the run
    logs show the search falling off that cliff repeatedly.

    The baseline is re-timed next to every candidate rather than measured once and
    cached. That looks wasteful and is not: a wall-clock *ratio* is only meaningful if
    both sides saw the same machine, and timing the reference on an idle process while
    candidates are timed under load produces scores that swing by 5x with nothing about
    the code having changed. Interleaving the two costs milliseconds and makes the
    number reproducible.
    """
    values, (size, ceiling) = perf_input(task), KERNEL_PERF[task]
    expected, baseline = _time_kernel(REFERENCE[task], values, ceiling)
    try:
        produced, seconds = _time_kernel(kernel, values, ceiling)
    except _Timeout as exc:
        # Report the ceiling being hit and nothing else. Naming the likely cause here --
        # "rescanning earlier values for every position is quadratic; carry running
        # state" -- would hand the proposer the lesson the search is then credited with
        # discovering, and would be wrong besides on every other way of exceeding a
        # ceiling. Diagnosing is the proposer's job; the evaluator's is to say
        # accurately what happened.
        return 0.0, Diagnostic("speed", f"on {size} values: {exc}")
    except Exception as exc:
        return 0.0, Diagnostic(
            "speed", f"on {size} values this raised {type(exc).__name__}: {exc}"
        )
    if len(produced) != len(expected) or not all(
        math.isclose(p, e, rel_tol=1e-7, abs_tol=1e-7)
        for p, e in zip(produced, expected)
    ):
        return 0.0, Diagnostic(
            "speed", f"wrong output on the {size}-value input, so speed does not count"
        )
    return baseline / seconds, Diagnostic(
        "speed",
        f"{size} values in {seconds * 1e3:.1f}ms against the reference "
        f"implementation's {baseline * 1e3:.1f}ms measured back to back -- "
        f"{baseline / seconds:.2f}x",
    )


def evaluate_instruction(
    instruction: str, task: KernelTask | None, model: str
) -> Evaluation:
    """Synthesize a kernel under the candidate instruction, check it, and time it.

    The score is the measured speedup over the reference implementation, gated on
    correctness: any failing case scores zero, however fast the code is. That is the
    paper's KernelBench setup (correctness against the reference, then wall-clock
    against the PyTorch baseline), and it is what gives this domain something to climb.
    The Side Information is the failing cases with expected and actual values, the
    measured times and speedup, the traceback if it crashed, and the code itself.

    The metrics are only read in single-task mode, where the engine's Pareto objectives
    are an evaluation's sub-scores rather than a dataset's examples -- which is what the
    ``--single-task`` control runs. ``score`` has to be one of them because it is the
    number the report reads as the headline.
    """
    assert task is not None, "the kernel domain always has a dataset"
    try:
        with worker(model):
            kernel = Programmer().write_kernel(instruction, task)
    except Exception:
        return Evaluation(
            score=0.0,
            diagnostics=[
                Diagnostic("task", f"{task.name}: {task.spec}"),
                Diagnostic("synthesis failed", traceback.format_exc(limit=2).strip()),
            ],
        )

    cases = KERNEL_TESTS[task.name]
    passed = 0
    missed: list[Diagnostic] = []
    start = time.perf_counter()
    for values, expected in cases:
        try:
            produced = list(kernel(list(values)))
            ok = len(produced) == len(expected) and all(
                math.isclose(p, e, rel_tol=1e-9, abs_tol=1e-9)
                for p, e in zip(produced, expected)
            )
        except Exception as exc:
            ok, produced = False, f"raised {type(exc).__name__}: {exc}"  # type: ignore[assignment]
        if ok:
            passed += 1
        else:
            missed.append(
                Diagnostic(
                    "failing case",
                    f"kernel({values}) returned {produced}, expected {expected}",
                )
            )
    elapsed = time.perf_counter() - start

    speedup, timing = measure_speedup(kernel, task.name)
    correct = passed == len(cases) and speedup > 0.0

    # Only the first couple of failures go back: a wall of them buries the signal. Say
    # how many were withheld, so the proposer is not told a partial list is the whole one.
    failures = missed[:2]
    if len(missed) > len(failures):
        failures.append(
            Diagnostic(
                "further failures",
                f"{len(missed) - len(failures)} more case(s) also failed and are not "
                f"shown here",
            )
        )
    diagnostics = [Diagnostic("task", f"{task.name}: {task.spec}")]
    diagnostics += failures or [Diagnostic("correctness", "all small cases passed")]
    diagnostics.append(timing)
    diagnostics.append(
        Diagnostic("small-case timing", f"{len(cases)} cases in {elapsed * 1e3:.2f}ms")
    )
    diagnostics.append(
        Diagnostic("code under test", (source_of(kernel) or "(unavailable)").strip())
    )
    diagnostics.append(
        Diagnostic(
            "verdict",
            f"correct, and {speedup:.2f}x the reference implementation's speed -- the "
            f"score IS that ratio, so a correct but ordinary implementation scores "
            f"about 1.0 and only a faster one improves"
            if correct
            else f"{passed}/{len(cases)} small cases passed; an incorrect kernel "
            f"scores zero no matter how fast it is",
        )
    )
    return Evaluation(
        score=speedup if correct else 0.0,
        metrics=[
            Metric("score", speedup if correct else 0.0),
            Metric("cases_passed", float(passed)),
        ],
        diagnostics=diagnostics,
    )


class Proposer(Agent):
    """You are a reflective optimizer. You are shown the current instruction, the
    scores the code written under it achieved, and diagnostic side information
    explaining *why*, and you return a better instruction. You do not mutate blindly:
    you first read the diagnostics to decide which failure mode is costing the most,
    then you write the guidance that addresses it."""

    @Template.define
    def propose_instruction(self, current: str, feedback: list[Rollout]) -> str:
        """You are optimizing the INSTRUCTION handed to a programmer model that
        implements small list-transform functions. The instruction below is the
        artifact -- it is reused for every task in a family, so it must say things
        that are true of all of them.

        <current_instruction>
        {current}
        </current_instruction>

        Here is how the code written under it fared on a couple of tasks, including
        the specific test cases that failed:

        <feedback>
        {feedback}
        </feedback>

        Diagnose the failures, then rewrite the instruction so a programmer following
        it would not make them again. Prefer guidance that would still apply to a task
        you have not been shown over anything specific to one task -- an instruction
        that solves one task by naming its answer is worthless on the others.

        Return the improved instruction as plain text, nothing else.
        """


# ---------------------------------------------------------------------------
# Wiring and main
# ---------------------------------------------------------------------------


def run_kernel(args: argparse.Namespace, rng: random.Random) -> Result:
    """Multi-task by default. ``--single-task`` runs the paper's control instead: each
    task optimized independently, which is the comparison its 5.4 ablation reports.

    The control runs the engine's *single-task* mode, with the task bound in the
    evaluator's closure and no dataset at all, so its Pareto objectives are the
    evaluation's own sub-scores. Passing ``dataset=[task]`` would look equivalent and is
    not: that is multi-task mode with one example, a frontier over a single objective on
    which every tie is non-dominated and selection collapses to greedy, so the control
    would differ from the treatment arm in its selection rule as well as its task count.

    It is worth being clear that there is still no configuration in which the two arms
    differ by exactly one thing. Single-task mode necessarily changes both the number of
    tasks *and* what the objectives are, since with one task there are no per-example
    objectives to keep a frontier over -- which is precisely why the paper introduces
    per-metric objectives for that mode. The paper's comparison has the same property.

    The two arms are also matched on optimizer iterations, not on evaluator calls, and
    those are not the same thing -- multi-task pays for a full five-task evaluation
    whenever a proposal is accepted. ``--control-iterations`` sets the control's per-task
    budget directly, which is how to match on the evaluation counts ``report`` prints
    for both arms.
    """
    proposer = lambda instruction, feedback: Proposer().propose_instruction(  # noqa: E731
        instruction, feedback
    )
    if not args.single_task:
        return optimize_anything(
            evaluator=lambda i, t: evaluate_instruction(i, t, args.worker_model),
            proposer=proposer,
            seed=SEED_INSTRUCTION,
            dataset=KERNEL_TASKS,
            budget=args.budget,
            minibatch_size=args.minibatch,
            selection=args.selection,
            use_side_info=not args.no_side_info,
            rng=rng,
        )

    def evaluator_for(task: KernelTask) -> collections.abc.Callable[..., Evaluation]:
        """Bind the task, so the engine sees a single-task problem with no dataset."""

        def evaluate(instruction: str, _: object) -> Evaluation:
            return evaluate_instruction(instruction, task, args.worker_model)

        return evaluate

    # The control is five independent runs, and its report has to be an aggregate of all
    # five: reusing one run's ``Result`` and overwriting a few of its fields would print
    # that run's frontier, iteration count and "best artifact" as though they were the
    # whole control's.
    runs: list[Result] = []
    per_task: list[tuple[str, Candidate, float]] = []
    per_problem = args.control_iterations or max(1, args.budget // len(KERNEL_TASKS))
    for task in KERNEL_TASKS:
        print(f"\n[single-task control] {task.name} ({per_problem} iterations)")
        result: Result = optimize_anything(
            evaluator=evaluator_for(task),
            proposer=proposer,
            seed=SEED_INSTRUCTION,
            budget=per_problem,
            selection=args.selection,
            use_side_info=not args.no_side_info,
            rng=rng,
            task_name=task.name,
        )
        runs.append(result)
        per_task.append((task.name, result.best, result.best_score))

    best_run = max(runs, key=lambda r: r.best_score)
    return Result(
        mode=f"single-task control ({len(runs)} independent runs)",
        # No pool and no objectives: this arm has five separate frontiers over five
        # disjoint objective sets, and there is no honest way to merge them. Pooling
        # the candidates would ask the Pareto machinery to compare a count_smaller_before
        # score against a zscore one -- it raises, and it should. An empty pool tells
        # `report` there is no shared frontier here, which is exactly the difference
        # from the multi-task arm that this control exists to isolate.
        pool=[],
        history=[step for r in runs for step in r.history],
        objectives=[],
        seed_score=statistics.fmean(r.seed_score for r in runs),
        best=best_run.best,
        best_score=statistics.fmean(score for _, _, score in per_task),
        per_task=per_task,
        evaluations=sum(r.evaluations for r in runs),
        proposals=sum(r.proposals for r in runs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=10, help="Optimizer iterations")
    parser.add_argument(
        "--minibatch", type=int, default=2, help="Tasks per reflection step"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for selection and minibatches"
    )
    parser.add_argument(
        "--worker-model",
        default=WORKER_MODEL,
        help="Model that writes the kernels; the harness's --model is the proposer, "
        "as in the paper's proposer/worker split",
    )
    parser.add_argument(
        "--selection",
        choices=["pareto", "best"],
        default="pareto",
        help="Candidate selection; 'best' mutates the best average instead, which is "
        "the naive alternative the paper's 4.3 argues against rather than an ablation "
        "it runs",
    )
    parser.add_argument(
        "--no-side-info",
        action="store_true",
        help="Score-only feedback: the paper's SI ablation",
    )
    parser.add_argument(
        "--control-iterations",
        type=int,
        default=0,
        help="Per-task iterations for --single-task; 0 divides --budget across the "
        "tasks, which matches the arms on iterations rather than evaluator calls",
    )
    parser.add_argument(
        "--single-task",
        action="store_true",
        help="Run the single-task control instead of multi-task search: each task "
        "optimized independently at the same per-problem budget",
    )
    args = parser.parse_args()

    assert check_reference_agrees()
    result = run_kernel(args, random.Random(args.seed))
    report(result, selection=args.selection, side_info=not args.no_side_info)
    # No assertion that the score improved: in multi-task mode it cannot go down. The
    # headline is a per-task maximum over the pool and the seed is one candidate in it,
    # so ``best_score >= seed_score`` holds however badly the search does, and asserting
    # it would only look like a check. `report` prints the matched single-artifact
    # comparison next to it, which can go down and is the number to read.


if __name__ == "__main__":
    main()
