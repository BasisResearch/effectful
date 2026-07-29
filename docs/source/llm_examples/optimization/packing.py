"""Circle packing: single-task search over a code artifact (optimize_anything 5.3).

Pack ``n`` non-overlapping circles into the unit square so the sum of their radii is as
large as possible. The artifact *is* the solution here -- there is no dataset, and the
evaluator scores the candidate directly -- which is the paper's single-task mode, and
the mode AlphaEvolve and OpenEvolve operate in.

This script is set up to be a genuine attempt at the paper's result rather than an
illustration of the loop, so it follows 5.3 and Appendix G closely:

  * The artifact has the paper's signature, ``pack(n, time_budget, current_best)``
    (its evolved packer is ``main(timeout, current_best_solution)``, Appendix K.6), and
    is handed the best packing found so far to polish. That is what lets a search of a
    few dozen evaluations reach a competitive number instead of restarting each time --
    and it is why `optimize_anything` needs a ``state_key``, since an artifact's score
    now depends on when it ran.
  * It may use whatever numeric libraries are actually installed, because the paper's
    winner is an LP over radii whose dual variables give gradients for a local
    optimizer over centres -- unreachable in the standard library. The prompt reports
    what ``importlib`` finds rather than a fixed list, so the example still runs where
    scipy is absent.
  * The Pareto objectives are Mechanism 3's run-distribution metrics -- max score, mean
    score, stability, improvement rate over repeated runs -- rather than a single
    number, which is what keeps structurally different packers alive on the frontier.
  * The search evolves *two* modules on one shared frontier, the packer and a refiner
    instruction, which is Mechanism 2's leapfrogging. This needs nothing from the
    engine: a candidate is a `PackSystem`, and "which module to mutate" is a branch in
    this script's proposer.

The evaluator is deterministic Python throughout, so no model sits in the scoring path
and every number in the trace is measured.

Demonstrates:
- A ``Template`` returning a ``Callable`` whose *own* doctests are the decode-time
  contract: a packer that does not return ``n`` feasible circles is fed its error by
  ``TenacityRetryer`` and never reaches the evaluator -- the paper's refiner stage,
  for free
- Side Information as a typed value: geometric diagnostics, sub-scores, the spread
  across repeated runs, and with ``--visual-si`` a rendered ``PIL.Image`` of the
  current packing, all spliced into the proposer's prompt through the Encodable bridge
- Multi-module search with no engine support at all -- code and refiner instruction
  compete on the one frontier
- Search over state the evaluator depends on, declared through ``state_key``
- Scoring that cannot be gamed by a tolerance: the sum of radii is measured *after*
  shrinking the packing to exact feasibility

Measured on 2026-07-29 with gpt-5.5 proposing, on the paper's own instance
(``--num-circles 26 --time-budget 20``). Three runs from the 6x6 grid seed at 2.1667
reached 2.6083 (4 iterations), 2.6147 (6 iterations) and 2.6359831 (stopped after 10).
The paper reports 2.63598 for optimize_anything, 2.635 for AlphaEvolve and 2.6307 for
OpenEvolve at 200 evaluations, so the best of those runs matches the paper's figure at
the precision it is quoted, in about ten evaluations. Read that best-of-three
carefully: the spread across runs (2.608 to 2.636) is wider than the gap between the
published systems, so one run of this script is not a ranking of anything.

The winning artifact is an algorithm, not a remembered answer, which matters because
the optimum for n=26 is published and a model could simply recite it. It builds the 4n
wall constraints and n(n-1)/2 separation constraints programmatically over 3n
variables, hands them to SLSQP warm-started from the incumbent packing, repairs every
iterate to exact feasibility before scoring it, and keeps the best -- the same shape as
the paper's evolved artifact, reached in two accepted proposals. It contains no
coordinate table and no special case for 26.
"""

# Simplifications vs. the source:
# - Budget is counted in optimizer iterations rather than metric calls or dollars; the
#   paper spends 63 evaluations and $3.18 on this domain.
# - The four Mechanism-3 objectives are named but not defined in the paper; "stability"
#   and "improvement rate" here are this script's reading of those names.
# - The two modules alternate by a coin flip (``--code-share``); the paper does not say
#   how it splits attention between them.
# - ``--no-side-info`` and ``--selection best`` were last measured at n=10 on an earlier
#   version of this script, before the incumbent, the numeric libraries and the second
#   module -- and at an instance whose ceiling every arm hits within a few iterations.
#   Testing the paper's 93.96% score-only figure means re-running them at n=26 here.
# - No island model / MAP-Elites, which the paper also drops.

import argparse
import collections
import collections.abc
import dataclasses
import importlib.util
import io
import math
import random
import signal
import statistics
import threading
import time
import traceback
import typing

import pydantic.dataclasses
from PIL import Image

from docs.source.llm_examples.optimization.library import (
    Diagnostic,
    Evaluation,
    Metric,
    Result,
    Rollout,
    artifact_key,
    optimize_anything,
    report,
    source_of,
)
from effectful.handlers.llm import Agent, Template


@pydantic.dataclasses.dataclass(frozen=True)
class Circle:
    """A circle in the unit square: centre and radius."""

    x: float
    y: float
    r: float

    def __str__(self) -> str:
        return f"({self.x:.4f}, {self.y:.4f}) r={self.r:.4f}"


# ``pack(n, time_budget, current_best)`` is the paper's artifact signature -- its
# evolved circle packer is ``main(timeout, current_best_solution)`` (Appendix K.6). The
# artifact is handed its own time budget *and* the best packing found so far, so a
# candidate can polish the incumbent instead of starting over every time. That is what
# lets a search of a few dozen evaluations reach a competitive number, and it is why
# ``optimize_anything`` needs a ``state_key``: with the incumbent threaded through, an
# artifact's score depends on when it ran.
type Packer = collections.abc.Callable[[int, float, list[Circle] | None], list[Circle]]

# A kernel is one list-to-list transform; every task in the multi-task domain shares
# this signature so a single instruction can drive all of them.
type Kernel = collections.abc.Callable[[list[float]], list[float]]


def feasible(circles: collections.abc.Sequence[Circle], tol: float = 1e-9) -> bool:
    """True when every circle lies inside the unit square and no two overlap.

    In the synthesized packer's lexical scope, so the doctests the model must write
    can call it -- that is what makes "the artifact obeys its contract" checkable at
    decode time rather than at evaluation time.

    >>> feasible([Circle(0.25, 0.25, 0.25), Circle(0.75, 0.75, 0.25)])
    True
    >>> feasible([Circle(0.5, 0.5, 0.6)])
    False
    """
    return worst_violation(circles) <= tol


def worst_violation(circles: collections.abc.Sequence[Circle]) -> float:
    """How badly the packing breaks its constraints: the largest overlap depth or
    out-of-square excursion, and 0.0 for a feasible packing.

    >>> worst_violation([Circle(0.5, 0.5, 0.5)])
    0.0
    >>> round(worst_violation([Circle(0.5, 0.5, 0.5), Circle(0.5, 0.5, 0.5)]), 6)
    1.0
    """
    worst = 0.0
    for c in circles:
        if c.r <= 0.0:
            worst = max(worst, 1.0 - c.r)
        worst = max(worst, c.r - c.x, c.r - c.y, c.x + c.r - 1.0, c.y + c.r - 1.0)
    for i, a in enumerate(circles):
        for b in circles[i + 1 :]:
            worst = max(worst, a.r + b.r - math.hypot(a.x - b.x, a.y - b.y))
    return max(0.0, worst)


def total_radius(circles: collections.abc.Sequence[Circle]) -> float:
    """The reported sum of the radii -- what the packer claims.

    >>> total_radius([Circle(0.25, 0.25, 0.25), Circle(0.75, 0.75, 0.25)])
    0.5
    """
    return sum(c.r for c in circles)


# The paper's winning packer is a bilevel optimizer: an LP over radii whose duals give
# exact gradients for L-BFGS-B over centres, plus CMA-ES exploration (Appendix K.6).
# None of that is reachable in the standard library, so the proposer is told what is
# actually importable here rather than a fixed list -- scipy and numpy arrive in this
# repo transitively, and the example still runs where they do not.
NUMERIC_LIBRARIES = [
    name
    for name in ("numpy", "scipy.optimize", "scipy.spatial")
    if importlib.util.find_spec(name) is not None
]


def numeric_toolbox() -> str:
    """What the synthesized packer may import, as a sentence for the prompt."""
    if not NUMERIC_LIBRARIES:
        return (
            "Only the Python standard library is available -- no numpy, no scipy -- so "
            "write the numerics yourself."
        )
    return (
        f"These numeric libraries are installed and you should use them: "
        f"{', '.join(NUMERIC_LIBRARIES)}. A linear program over the radii for fixed "
        f"centres is exact (maximize the sum subject to r_i + r_j <= |c_i - c_j| and "
        f"r_i <= the distance to each wall), its dual variables give you the gradient "
        f"of the objective with respect to the centres, and that gradient drives a "
        f"local optimizer over the centres. Alternating those two is far stronger "
        f"than any hand-rolled heuristic."
    )


def feasible_scale(circles: collections.abc.Sequence[Circle]) -> float:
    """The largest factor ``s <= 1`` for which scaling every radius by ``s`` makes the
    packing exactly feasible -- 1.0 for a packing with room to spare, 0.0 for one that
    cannot be rescued.

    The score is ``s * total_radius``, and that is deliberate. Scoring the *reported*
    radii against a tolerance invites the artifact to overshoot by just under it: the
    first version of this example was gamed exactly that way, by a packer that added
    4e-10 to every radius and noted in a comment that this stayed inside the
    evaluator's 1e-9 feasibility check. Shrinking to exact feasibility instead of
    thresholding removes the incentive -- an inflated radius is scaled straight back
    out, and it drags every other circle down with it -- and it replaces the
    feasible/infeasible cliff with a gradient the proposer can actually climb.

    >>> feasible_scale([Circle(0.25, 0.25, 0.25), Circle(0.75, 0.75, 0.25)])
    1.0
    >>> feasible_scale([Circle(0.5, 0.5, 1.0)])
    0.5
    """
    scale = 1.0
    for c in circles:
        wall = min(c.x, c.y, 1.0 - c.x, 1.0 - c.y)
        if c.r <= 0.0 or wall <= 0.0:
            return 0.0  # a non-positive radius or a centre outside the square
        scale = min(scale, wall / c.r)
    for i, a in enumerate(circles):
        for b in circles[i + 1 :]:
            scale = min(scale, math.hypot(a.x - b.x, a.y - b.y) / (a.r + b.r))
    return max(0.0, min(1.0, scale))


@dataclasses.dataclass(frozen=True)
class PackSystem:
    """The artifact of the packing domain: *two* modules, not one.

    The paper optimizes the code artifact and a refiner prompt together on a single
    shared Pareto front (its Mechanism 2, "multi-module Pareto leapfrogging"), and
    credits that coordination for the circle-packing result: the refiner discovers an
    LP-based approach while the code module is still a weak heuristic, the code module
    absorbs it and catches up, the refiner pushes further with sequential LP, the code
    absorbs that too. Each module's advance is the foundation for the other's next one,
    and a code mutation that scores zero is recovered rather than lost, because the
    refiner can rewrite it.

    Modelling that needs nothing from the engine: a candidate is a ``PackSystem``, and
    the domain's proposer decides on each iteration which module to mutate -- rewrite
    the packer directly, or rewrite the refiner instruction and apply it to the packer.
    Both paths produce a new ``PackSystem`` that lands on the same frontier. ``origin``
    records which module produced it, so the leapfrogging is countable afterwards; it
    is deliberately left out of ``__str__`` so it does not perturb the cache key.
    """

    packer: Packer
    refiner: str
    origin: str = "seed"

    def __str__(self) -> str:
        return (
            f"<packer>\n{artifact_key(self.packer)}\n</packer>\n"
            f"<refiner_instruction>\n{self.refiner}\n</refiner_instruction>"
        )


# The refiner module's starting point: a generic repair instruction, which the search
# is free to turn into something specific about packing.
SEED_REFINER = (
    "Look at the diagnostics, find the single change that would raise the score the "
    "most, and make it."
)


class Proposer(Agent):
    """You are a reflective optimizer. You are shown the current artifact, the score
    it achieved, and diagnostic side information explaining *why* it scored that way,
    and you return a strictly better artifact. You do not mutate blindly: you first
    read the diagnostics to decide which failure mode is costing the most, then you
    make the change that addresses it -- and you are willing to replace the whole
    approach with a different one when the diagnostics say the current approach has
    saturated."""

    @Template.define
    def propose_packer(
        self, current: Packer, feedback: list[Rollout], n: int, toolbox: str
    ) -> Packer:
        """Write an improved ``pack(n, time_budget, current_best)`` that packs {n}
        non-overlapping circles into the unit square [0,1]x[0,1], maximizing the SUM OF
        THE RADII. The circles may have different radii. Return a list of
        ``Circle(x, y, r)``.

        The current artifact scored as follows. Read the diagnostics before you write
        anything: they tell you which circles are jammed, where the slack is, how the
        score varied across repeated runs, and whether the packing is even feasible.

        <feedback>
        {feedback}
        </feedback>

        Your arguments:
        - ``n``: how many circles. Read it; never hardcode a table for one size.
        - ``time_budget``: seconds you may spend. Poll ``time.monotonic()`` and return
          your best packing before it expires. Spend it -- returning early wastes
          search you were given.
        - ``current_best``: the best packing found so far (a list of ``Circle``), or
          ``None`` on the first call. POLISH IT. Starting from the incumbent and
          improving it is how a handful of evaluations reaches a strong number;
          restarting from scratch every time throws that away. Keep it as one of your
          starting configurations even when you also try fresh ones, and never return
          something worse than what you were handed.

        {toolbox}

        What actually wins here, in rough order of value: alternate between solving for
        the radii given the centres (exact, and cheap) and moving the centres to make
        room (local optimization from the radii problem's sensitivities); keep several
        structurally different starting configurations -- hexagonal, edge-biased,
        corner-anchored, farthest-point -- because the good packings for different
        ``n`` look nothing alike; give up on equal radii entirely, since a few large
        circles with small fillers in the gaps beats any uniform arrangement; and when
        progress stalls, relocate the smallest circles into corners and re-solve rather
        than nudging everything.

        Other constraints:
        - You are scored on the sum of radii AFTER the whole packing is shrunk to exact
          feasibility, so an overlap costs you proportionally and padding a radius to
          sit just inside a tolerance gains you nothing: it is scaled straight back
          out, and it shrinks every other circle with it.
        - The evaluator runs you several times and looks at the distribution, so
          randomness is fine and a stable, repeatable method is worth more than a lucky
          one.
        - It must work for ANY ``n``: the evaluator also reports how you do on a size
          you were not asked about.

        Your function's docstring MUST contain doctests certifying the contract, and
        they are run before your artifact is accepted -- this is the decode-time gate
        the paper builds a separate "refiner" stage for. ``Circle``, ``feasible``,
        ``total_radius`` and ``worst_violation`` are in scope. Write at least a
        doctest that binds ``cs = pack(5, 0.5, None)`` and checks
        ``len(cs) == 5 and feasible(cs)``, prefixing each input line with the doctest
        prompt (three ``>`` characters and a space; it is spelled out rather than
        shown so that this instruction is not itself collected as a test).

        The doctest below certifies the same contract on the other decode path, where
        the harness synthesizes this Template's body: it calls this Template
        recursively -- routed to your own submission, so it costs nothing -- and runs
        the packer that comes back.

        >>> _packer = Proposer().propose_packer(seed_packer, [], 4, numeric_toolbox())
        >>> _circles = _packer(4, 0.5, None)
        >>> len(_circles) == 4 and feasible(_circles)
        True
        """

    @Template.define
    def propose_packer_visual(
        self,
        current: Packer,
        feedback: list[Rollout],
        n: int,
        toolbox: str,
        render: Image.Image,
    ) -> Packer:
        """Write an improved ``pack(n, time_budget, current_best)`` that packs {n}
        non-overlapping circles into the unit square, maximizing the SUM OF THE RADII.

        Here is what the current packing actually looks like:

        {render}

        and here is what the evaluator measured:

        <feedback>
        {feedback}
        </feedback>

        Use the picture: wasted space, circles that could grow, and regions that want a
        different arrangement are visible in it in a way they are not in the numbers.

        {toolbox}

        Then apply the same rules as before -- read ``n``, spend ``time_budget``,
        polish the ``current_best`` you are handed rather than restarting, never return
        an infeasible packing -- and put doctests in your docstring certifying that
        ``cs = pack(5, 0.5, None)`` yields ``len(cs) == 5 and feasible(cs)``, each
        input line prefixed with the doctest prompt (three ``>`` characters and a
        space). ``Circle``, ``feasible``, ``total_radius`` and ``worst_violation`` are
        in scope.
        """

    @Template.define
    def refine_packer(
        self,
        current: Packer,
        instruction: str,
        feedback: list[Rollout],
        n: int,
        toolbox: str,
    ) -> Packer:
        """Apply a refinement instruction to a packing algorithm.

        This is the *refiner module* being spent: another module of the search evolved
        the instruction below, and your job is to carry it out on the current
        ``pack(n, time_budget, current_best)`` faithfully -- not to substitute your own
        plan for it.

        <refinement_instruction>
        {instruction}
        </refinement_instruction>

        Here is how the current packer scored, for context on what the instruction is
        reacting to:

        <feedback>
        {feedback}
        </feedback>

        {toolbox}

        Return the revised packer for n={n}. It keeps the same contract: read ``n``,
        spend ``time_budget``, polish ``current_best`` rather than discarding it, never
        return an infeasible packing, and carry doctests in the docstring certifying
        that ``cs = pack(5, 0.5, None)`` yields ``len(cs) == 5 and feasible(cs)`` (each
        input line prefixed with the doctest prompt -- three ``>`` characters and a
        space). If the packer you were given is broken, repair it: recovering a failed
        mutation is exactly what this module is for. ``Circle``, ``feasible``,
        ``total_radius`` and ``worst_violation`` are in scope.
        """

    @Template.define
    def propose_refiner(
        self, current: str, packer: Packer, feedback: list[Rollout]
    ) -> str:
        """You are optimizing the REFINER INSTRUCTION -- the second module of this
        search. It is a short natural-language directive that another model applies to
        the current packing algorithm to produce the next one, so it is where a
        *strategy* can be discovered and held even while the code lags behind it.

        <current_instruction>
        {current}
        </current_instruction>

        The algorithm it will be applied to is above, and here is how that algorithm
        scored:

        <feedback>
        {feedback}
        </feedback>

        Write a better instruction. It should name the specific structural change worth
        making next -- switch how the radii are solved for, change how the centres
        move, add a different seeding strategy, escape a saturated configuration, or
        repair a broken implementation -- in enough detail that a competent programmer
        could carry it out without guessing, while still being an instruction rather
        than the code itself. Aim past the current implementation: this module is
        valuable precisely when it is ahead of the code.

        Return the instruction as plain text, nothing else.
        """

    @Template.define
    def bootstrap_packer(self, objective: str, n: int, toolbox: str) -> Packer:
        """Seedless mode: there is no artifact yet, only a goal.

        <objective>
        {objective}
        </objective>

        {toolbox}

        Write the first version of ``pack(n, time_budget, current_best)`` for n={n}: it
        returns a list of ``Circle(x, y, r)`` filling the unit square without overlaps.
        Read ``n``, spend ``time_budget``, and start from ``current_best`` when it is
        not ``None``. Your docstring MUST contain doctests certifying that
        ``cs = pack(5, 0.5, None)`` yields ``len(cs) == 5 and feasible(cs)``, each
        input line prefixed with the doctest prompt (three ``>`` characters and a
        space). ``Circle``, ``feasible`` and ``total_radius`` are in scope.
        """


# ---------------------------------------------------------------------------
# The task and its evaluator -- deterministic Python, the ground truth every
# candidate is scored by.
# ---------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _run_packer(
    packer: Packer, n: int, time_budget: float, current_best: list[Circle] | None
) -> tuple[list[Circle], float]:
    """Run a synthesized packer under a hard wall-clock backstop, returning its
    circles and how long it took. The artifact is *given* its budget and the incumbent
    packing, and is expected to honour both; the alarm only catches one that does not.
    """

    def _alarm(signum: int, frame: object) -> None:
        raise _Timeout(f"pack() ignored its {time_budget}s budget")

    guarded = threading.current_thread() is threading.main_thread()
    if guarded:
        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, time_budget * 3.0 + 5.0)
    start = time.perf_counter()
    try:
        circles = list(packer(n, time_budget, current_best))
    finally:
        elapsed = time.perf_counter() - start
        if guarded:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)
    return circles, elapsed


def packing_score(circles: collections.abc.Sequence[Circle], n: int) -> float:
    """The one number: the sum of radii after shrinking the packing to exact
    feasibility, and zero for a packing of the wrong size.

    >>> packing_score([Circle(0.25, 0.25, 0.25), Circle(0.75, 0.75, 0.25)], 2)
    0.5
    >>> packing_score([Circle(0.5, 0.5, 1.0)], 1)
    0.5
    """
    if len(circles) != n:
        return 0.0
    return feasible_scale(circles) * total_radius(circles)


def generality_check(packer: Packer, n: int) -> Diagnostic:
    """Side information only, never scored: how the packer does on an instance size it
    was not asked about. A genuine algorithm keeps its edge here; a table of
    coordinates for one ``n`` falls back to whatever it does by default, and the
    proposer gets to see that it did."""
    other = n + 3
    baseline = total_radius(seed_packer(other, 0.1, None))
    try:
        circles, _ = _run_packer(packer, other, 0.5, None)
    except Exception as exc:
        return Diagnostic(
            "generality", f"pack({other}, ...) raised {type(exc).__name__}: {exc}"
        )
    if len(circles) != other:
        return Diagnostic(
            "generality",
            f"pack({other}, ...) returned {len(circles)} circles, not {other}",
        )
    achieved = packing_score(circles, other)
    return Diagnostic(
        "generality",
        f"on the unrequested size n={other} this packer scores {achieved:.6f} vs "
        f"{baseline:.6f} for the naive grid -- "
        + (
            "it generalizes"
            if achieved > baseline
            else "no better than the grid, so it is a table rather than an algorithm"
        ),
    )


def trajectory_metrics(scores: list[float]) -> list[Metric]:
    """The Pareto objectives of the paper's single-task search.

    Its Mechanism 3 says the front is kept across "max score, mean score, EMA
    stability, improvement rate", and that this is what keeps greedy, LP, SLP, bilevel
    L-BFGS and CMA-ES candidates alive at once. Those are properties of a *run
    distribution*, not of one packing, so the evaluator runs each packer several times
    and reports the shape of the result:

      * ``max_score``    -- the best packing it found, the headline number
      * ``mean_score``   -- what it achieves typically, not at its luckiest
      * ``stability``    -- negated mean absolute change between consecutive runs, so
                            a method that lands in the same place every time scores 0
                            and an erratic one scores negative
      * ``improvement``  -- how much the best improves from the first run to the last,
                            per extra run: the value of giving this artifact more time

    The paper names these four but does not define them, so the last two are this
    example's reading of the names. All four are higher-is-better, which the Pareto
    machinery requires.

    >>> [str(m) for m in trajectory_metrics([1.0, 1.0, 1.0])]
    ['max_score=1', 'mean_score=1', 'stability=0', 'improvement=0']
    """
    if not scores:
        return [
            Metric("max_score", 0.0),
            Metric("mean_score", 0.0),
            Metric("stability", 0.0),
            Metric("improvement", 0.0),
        ]
    deltas = [abs(b - a) for a, b in zip(scores, scores[1:])]
    drift = statistics.fmean(deltas) if deltas else 0.0
    return [
        Metric("max_score", max(scores)),
        Metric("mean_score", statistics.fmean(scores)),
        Metric("stability", -drift if drift else 0.0),  # not -0.0
        Metric(
            "improvement",
            (max(scores) - scores[0]) / len(scores) if len(scores) > 1 else 0.0,
        ),
    ]


PACKING_REPEATS = 3


def evaluate_packing(
    packer: Packer, n: int, time_budget: float, current_best: list[Circle] | None
) -> tuple[Evaluation, list[Circle]]:
    """Score a packer and explain the score.

    Ground truth, deterministic Python: the score is the sum of radii after shrinking
    the packing to exact feasibility (see ``feasible_scale``), so overlap is paid for
    proportionally rather than at a cliff and there is no tolerance to exploit.

    The packer is run ``PACKING_REPEATS`` times, each time handed the incumbent, and
    the run distribution becomes the Pareto objectives (``trajectory_metrics``). The
    repeats are not redundancy: these artifacts use randomised restarts, so "how good
    is it typically" and "does it stay there" are different questions from "how good
    was its best run", and the paper keeps candidates that win any of them.

    Everything the evaluator learns on the way -- which circles are jammed, where the
    slack is, whether the radii are suspiciously uniform, the spread across repeats,
    how it fares on a size it was not asked about, and the traceback if it crashed --
    goes back as Side Information. The best packing found is returned alongside the
    evaluation so the caller can make it the incumbent without paying for another run.
    """
    scores: list[float] = []
    best: list[Circle] = []
    elapsed = 0.0
    for _ in range(PACKING_REPEATS):
        try:
            circles, seconds = _run_packer(packer, n, time_budget, current_best)
        except Exception:
            return Evaluation(
                score=0.0,
                metrics=trajectory_metrics([0.0]),
                diagnostics=[
                    Diagnostic("crash", traceback.format_exc(limit=3).strip()),
                    Diagnostic(
                        "fix",
                        "pack(n, time_budget, current_best) must return a list of "
                        "Circle without raising",
                    ),
                ],
            ), []
        elapsed = max(elapsed, seconds)
        scores.append(packing_score(circles, n))
        if scores[-1] >= max(scores):
            best = circles

    metrics = trajectory_metrics(scores)
    score = max(scores)
    diagnostics: list[Diagnostic] = [
        Diagnostic(
            "repeats",
            f"{PACKING_REPEATS} runs scored "
            + ", ".join(f"{s:.6f}" for s in scores)
            + f"; the score is the best of them ({score:.6f})",
        ),
        Diagnostic("runtime", f"{elapsed:.2f}s of a {time_budget:.2f}s budget per run"),
    ]
    if current_best is not None:
        incumbent = packing_score(current_best, n)
        diagnostics.append(
            Diagnostic(
                "incumbent",
                f"the packing handed to you scored {incumbent:.6f}; this artifact "
                + (
                    f"improved it by {score - incumbent:.6f}"
                    if score > incumbent
                    else "did not improve on it, which means the time went nowhere -- "
                    "start from what you are given"
                ),
            )
        )

    if len(best) != n:
        return Evaluation(
            score=0.0,
            metrics=metrics,
            diagnostics=diagnostics
            + [Diagnostic("count", f"returned {len(best)} circles, expected {n}")],
        ), []

    violation = worst_violation(best)
    total = total_radius(best)
    scale = feasible_scale(best)
    smallest = min(c.r for c in best)
    diagnostics.append(generality_check(packer, n))

    if violation > 1e-9:
        offenders = sorted(best, key=lambda c: -c.r)[:3]
        diagnostics += [
            Diagnostic(
                "infeasible",
                f"worst constraint violation {violation:.6f} (overlap depth or "
                f"excursion outside the unit square). Radii sum to {total:.6f} as "
                f"returned, but every radius has to shrink by a factor of "
                f"{scale:.6f} before the packing is legal, so the score is "
                f"{score:.6f}. Place centres so the radii need no shrinking.",
            ),
            Diagnostic("largest circles", ", ".join(str(c) for c in offenders)),
        ]
        return Evaluation(score=score, metrics=metrics, diagnostics=diagnostics), best

    # Feasible: report where the slack is, so the proposer knows what to grow.
    slacks = []
    for i, a in enumerate(best):
        gap = min(
            [a.x - a.r, a.y - a.r, 1.0 - a.x - a.r, 1.0 - a.y - a.r]
            + [
                math.hypot(a.x - b.x, a.y - b.y) - a.r - b.r
                for j, b in enumerate(best)
                if j != i
            ]
        )
        slacks.append((gap, i, a))
    loosest = sorted(slacks, reverse=True)[:3]
    tightest = sorted(slacks)[:3]
    diagnostics += [
        Diagnostic(
            "room to grow",
            "; ".join(
                f"circle {i} {c} has {gap:.4f} of free space" for gap, i, c in loosest
            )
            or "every circle is jammed",
        ),
        Diagnostic(
            "jammed circles",
            "; ".join(f"circle {i} {c} slack {gap:.6f}" for gap, i, c in tightest),
        ),
        Diagnostic(
            "radius spread",
            f"largest {max(c.r for c in best):.4f}, smallest {smallest:.4f} -- "
            + (
                "nearly uniform, which is usually suboptimal"
                if max(c.r for c in best) - smallest < 0.01
                else "non-uniform"
            ),
        ),
    ]
    return Evaluation(score=score, metrics=metrics, diagnostics=diagnostics), best


def seed_packer(
    n: int, time_budget: float, current_best: list[Circle] | None
) -> list[Circle]:
    """The naive baseline every packing run starts from: equal circles on the
    tightest square grid that fits them, unless it is handed something better.

    >>> cs = seed_packer(4, 0.1, None)
    >>> len(cs) == 4 and feasible(cs)
    True
    """
    side = math.ceil(math.sqrt(n))
    r = 1.0 / (2 * side)
    grid = [
        Circle(x=(i % side) * 2 * r + r, y=(i // side) * 2 * r + r, r=r)
        for i in range(n)
    ]
    if current_best is not None and packing_score(current_best, n) > packing_score(
        grid, n
    ):
        return list(current_best)
    return grid


def render_packing(circles: collections.abc.Sequence[Circle], n: int) -> Image.Image:
    """Render a packing as a PNG, so Side Information can be visual. Nothing about the
    loop changes: an image is simply another Encodable the proposer's prompt splices."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import patches, pyplot

    figure = pyplot.figure(figsize=(4, 4), dpi=110)
    axes = figure.add_subplot(111, aspect="equal")
    axes.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, linewidth=1.5))
    for i, c in enumerate(circles):
        axes.add_patch(patches.Circle((c.x, c.y), c.r, alpha=0.45))
        axes.annotate(str(i), (c.x, c.y), ha="center", va="center", fontsize=7)
    axes.set_xlim(-0.05, 1.05)
    axes.set_ylim(-0.05, 1.05)
    axes.set_title(f"n={n}  sum of radii = {total_radius(circles):.4f}")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    pyplot.close(figure)
    buffer.seek(0)
    return Image.open(buffer)


# ---------------------------------------------------------------------------
# Wiring the domain to the engine.
# ---------------------------------------------------------------------------


def run_pack(args: argparse.Namespace, rng: random.Random) -> Result:
    """Single-task search over a two-module system, with the incumbent threaded through.

    Three things here are the paper's setup rather than a simplification of it: the
    artifact is handed the best packing found so far, the Pareto objectives are the
    run-distribution metrics of its Mechanism 3, and the search alternates between two
    modules on one shared frontier (Mechanism 2). The incumbent is ordinary mutable
    state in this closure -- and because it makes an evaluation depend on more than the
    artifact, it is declared to the engine as a ``state_key``.
    """
    n, budget = args.num_circles, args.time_budget
    toolbox = numeric_toolbox()
    incumbent: list[list[Circle]] = [[]]  # a one-slot cell: the best packing so far

    def state_key() -> str:
        return f"{packing_score(incumbent[0], n):.12f}"

    def evaluator(system: PackSystem, _: None) -> Evaluation:
        evaluation, best = evaluate_packing(
            system.packer, n, budget, incumbent[0] or None
        )
        # Whatever the candidate managed becomes the incumbent for everything after it,
        # so later artifacts start where this one stopped -- the paper's
        # ``current_best_solution``, which is why this domain declares a ``state_key``.
        if packing_score(best, n) > packing_score(incumbent[0], n):
            incumbent[0] = best
        return evaluation

    def mutate_code(system: PackSystem, feedback: list[Rollout]) -> PackSystem:
        if not args.visual_si:
            packer = Proposer().propose_packer(system.packer, feedback, n, toolbox)
        else:
            try:
                circles, _ = _run_packer(system.packer, n, budget, incumbent[0] or None)
                packer = Proposer().propose_packer_visual(
                    system.packer, feedback, n, toolbox, render_packing(circles, n)
                )
            except Exception:  # a packer that crashes has nothing to show
                packer = Proposer().propose_packer(system.packer, feedback, n, toolbox)
        return PackSystem(packer=packer, refiner=system.refiner, origin="code")

    def mutate_refiner(system: PackSystem, feedback: list[Rollout]) -> PackSystem:
        # The refiner module advances in two steps: rewrite the instruction, then spend
        # it on the current code. The instruction can therefore describe a strategy the
        # code does not implement yet -- which is exactly the leapfrogging the paper
        # describes, and also how a broken packer gets repaired instead of abandoned.
        instruction = Proposer().propose_refiner(
            system.refiner, system.packer, feedback
        )
        packer = Proposer().refine_packer(
            system.packer, instruction, feedback, n, toolbox
        )
        return PackSystem(packer=packer, refiner=instruction, origin="refiner")

    def proposer(system: PackSystem, feedback: list[Rollout]) -> PackSystem:
        mutate = mutate_code if rng.random() < args.code_share else mutate_refiner
        return mutate(system, feedback)

    def bootstrap(objective: str) -> PackSystem:
        return PackSystem(
            packer=Proposer().bootstrap_packer(objective, n, toolbox),
            refiner=SEED_REFINER,
            origin="bootstrap",
        )

    return optimize_anything(
        evaluator=evaluator,
        proposer=proposer,
        seed=(
            None
            if args.seedless
            else PackSystem(packer=seed_packer, refiner=SEED_REFINER)
        ),
        bootstrap=bootstrap,
        objective=(
            f"Pack {n} non-overlapping circles into the unit square so that the sum "
            f"of their radii is as large as possible."
        ),
        budget=args.budget,
        selection=args.selection,
        use_side_info=not args.no_side_info,
        rng=rng,
        task_name=f"pack-{n}",
        state_key=state_key,
    )


# ---------------------------------------------------------------------------
# Reporting and main
# ---------------------------------------------------------------------------


def module_note(result: Result) -> str:
    """Which module each surviving candidate came from. Multi-module search is only
    visible if you count it: this is the paper's leapfrogging, made checkable."""
    origins = collections.Counter(
        c.artifact.origin for c in result.pool if isinstance(c.artifact, PackSystem)
    )
    winner = result.best.artifact
    return (
        "Modules on the shared frontier: "
        + ", ".join(f"{k} {v}" for k, v in sorted(origins.items()))
        + "; the best candidate came from the "
        + f"{winner.origin if isinstance(winner, PackSystem) else 'unknown'} module"
    )


def render_system(artifact: typing.Any) -> str:
    """Both modules of the winning system: the instruction, then the code."""
    if not isinstance(artifact, PackSystem):
        return str(artifact)
    packer = source_of(artifact.packer) or repr(artifact.packer)
    return (
        f"--- refiner instruction ---\n{artifact.refiner}\n\n"
        f"--- packer ---\n{packer.rstrip()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-circles",
        type=int,
        default=10,
        help="Circles to pack; the paper's instance is 26",
    )
    parser.add_argument("--budget", type=int, default=8, help="Optimizer iterations")
    parser.add_argument(
        "--time-budget",
        type=float,
        default=2.0,
        help="Seconds a synthesized packer is given per call",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for selection and module choice"
    )
    parser.add_argument(
        "--code-share",
        type=float,
        default=0.5,
        help="Fraction of iterations that mutate the code module rather than the "
        "refiner module (both live on the one shared frontier)",
    )
    parser.add_argument(
        "--selection",
        choices=["pareto", "best"],
        default="pareto",
        help="Candidate selection; 'best' is the paper's greedy ablation",
    )
    parser.add_argument(
        "--no-side-info",
        action="store_true",
        help="Score-only feedback: the paper's SI ablation",
    )
    parser.add_argument(
        "--visual-si",
        action="store_true",
        help="Send a rendered image of the packing as side information",
    )
    parser.add_argument(
        "--seedless",
        action="store_true",
        help="Bootstrap candidate zero from a natural-language objective",
    )
    args = parser.parse_args()

    result = run_pack(args, random.Random(args.seed))
    report(
        result,
        selection=args.selection,
        side_info=not args.no_side_info,
        notes=[module_note(result)],
        render_artifact=render_system,
    )
    assert result.best_score >= result.seed_score, (
        f"optimization went backwards: {result.seed_score} -> {result.best_score}"
    )


if __name__ == "__main__":
    main()
