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
    scipy is absent. It reports *only* that; see `numeric_toolbox`.
  * The Pareto objectives are Mechanism 3's run-distribution metrics rather than a
    single number, which is what keeps structurally different packers alive on the
    frontier -- three of them here rather than the paper's four, for the reason
    `trajectory_metrics` gives.
  * The search evolves *two* modules on one shared frontier, the packer and a refiner
    instruction, which is Mechanism 2's leapfrogging. This needs nothing from the
    engine: a candidate is a `PackSystem`, and "which module to mutate" is a branch in
    this script's proposer.

The evaluator is deterministic Python throughout, so no model sits in the scoring path
and every number in the trace is measured.

Demonstrates:
- A ``Skill`` returning a ``Callable`` whose *own* doctests are the decode-time
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
- A control that says what the search is worth: ``--baseline`` runs the same problem
  for the same wall-clock with no model anywhere in it

What the numbers are
--------------------

Measured on 2026-07-30 with gpt-5.5 proposing, on the paper's own instance
(``--num-circles 26 --time-budget 20 --budget 10``), one run per configuration, each
from the 6x6 grid seed at 2.1666667:

  * default -- Pareto selection, side information on: **2.6359831**, 3 of 10 proposals
    accepted, 15 evaluations
  * ``--no-side-info``: 2.6319369, 2 of 10 accepted, 19 evaluations
  * ``--selection best``: 2.6317302, 5 of 10 accepted, 17 evaluations

The paper reports 2.63598 on this instance, against 2.635 for AlphaEvolve and 2.6307
for OpenEvolve at 200 evaluations, so the default arm matches its value to every digit
it gives. A
repeat of that arm reached the same 2.6359831 after four iterations before the
wall-clock hazard below wedged it at iteration 5; two runs landing on exactly that value
from different proposals is what a real local optimum looks like, and it is the
strongest evidence here that the artifact solves the problem rather than reciting a
published answer for 26. Four things have to be said before any of this is read as a
reproduction.

*Most of the distance is scipy's, not the search's.* ``--baseline`` runs warm-started
random-restart SLSQP with no LLM in the loop for the same packer wall-clock the search
spends (10 iterations x 3 repeats x 20s = 600s), and it reaches **2.6342924** in 13116
restarts -- past OpenEvolve at 200 evaluations, within 0.0007 of AlphaEvolve, and above
both ablation arms. The default arm's first accepted proposal scores
2.6342924 exactly: the first competent artifact the search writes is doing what the
control does, digit for digit, and the whole remaining margin of the run is 0.0017. So
the honest statement of this domain's result is that a reflective search whose artifacts
call a constrained optimizer beat a plain call to the same optimizer by about 0.0017 at
matched wall-clock. The paper reports no such control, which is why its own margin over
specialized systems is not attributable either.

*The run's number and the artifact's are different numbers, and the gap varies by arm.*
Every candidate is handed the best packing found so far and told never to return worse,
so a score accumulates the work of everything before it -- a candidate returning its
input unchanged is recorded at the full incumbent value. This is the paper's setup, not
a deviation from it: its evolved packer takes ``current_best_solution`` too, so its
2.63598 is a trajectory number in the same way. `cold_start_note` re-runs the winner
with no incumbent and prints both. The default arm's winner scores 2.6359831 cold
against the run's 2.6359831 -- it inherited nothing and reaches the headline from the
grid on its own. ``--selection best``'s winner scores 2.6319369 cold against a run
number of 2.6317302, marginally *better* alone than in the run. ``--no-side-info``'s
winner scores 2.5416318 cold against 2.6319369, so 0.09 of its score is other
candidates' work rather than its own, and that arm's headline is the least attributable
of the three.

*Score-only feedback costs almost nothing here, and the paper's ablation figure does not
reproduce.* ``--no-side-info`` reaches 2.6319369, which is 99.85% of the side-information
arm (99.1% of the distance from the seed), against the 93.96% the paper's Table 4
reports. The direction is the paper's and the magnitude is not, and the trace says why:
the first accepted proposal in every arm jumps from the grid to a warm-started SLSQP
restart loop and lands within 0.005 of the best number any arm reaches, after which all
of them grind in the fourth decimal. Diagnostics naming which circles are jammed cannot
be worth much when the remaining headroom is 0.004 and the artifact's own optimizer is
already searching it. That is a fact about this domain rather than a refutation of the
paper's: on a task whose ceiling is one competent artifact away from the seed, the SI
ablation has almost nothing to measure.

*Greedy selection is not distinguishable from Pareto here, because the frontier never
holds more than one candidate.* ``--selection best`` reached 2.6317302 against the
default arm's 2.6359831. That looks like support for 4.3's argument against collapsing
the frontier to an average, and it is not: every one of these runs ends "Pareto frontier
(1 candidate(s) survive)", and every accepted proposal prunes exactly one candidate. The
three objectives (max, mean and worst over the repeats) move together for packers this
close to deterministic, so dominance is total, and Pareto selection spends the run
choosing from a pool of one. With a single run per arm and no variance estimate, a 0.004
difference between two configurations that both reduce to "mutate the only candidate
there is" measures nothing about the selection rule, and the mechanism the difference
would have to come from -- structurally different packers kept alive by complementary
strengths -- never appears in the trace. 4.3 is untested here, not confirmed.

`module_note` gives Mechanism 2 the same treatment. Across the three runs the refiner
module's accepted proposals carry mean minibatch gains of +0.4676, +0.2313 and +0.2320;
the code module's carry +0.000845 and +0.000353, and in the score-only arm it had
nothing accepted at all. The refiner wins the one move that matters, off the grid seed,
and the code module grinds out everything after it in the fourth decimal and beyond,
which makes the two modules' gain figures a statement about when each ran rather than
about how good either is. The modules
do alternate -- accepted gains arrive as ``refiner -> code -> code`` in the default arm
and ``refiner -> code -> code -> refiner -> code`` in ``--selection best``, three
handovers -- but a handover means only that the other module produced the next accepted
gain, not that it was ahead of its partner. What the counts do establish is that the
second module is not decoration: it produced the first accepted gain in all three runs
and the winning artifact in the score-only arm.

Iterations 7 and 9 of the ``--selection best`` run both read ``2.6317302 -> 2.6317302
ACCEPTED``, as does iteration 3 of the default arm at 2.6342924. The accept gate is a
bare ``after > before``, so a child that clamped to the incumbent and improved it in the
eighth decimal is admitted and its parent pruned. That is Algorithm 1's accept rule as
written, and it is the mechanism by which a candidate contributing nothing carries the
run's whole score forward.

The winning artifact is an algorithm rather than a remembered answer, with one
qualification worth stating. It builds the 4n wall constraints and n(n-1)/2 separation
constraints programmatically over 3n variables, hands them to SLSQP warm-started from
the incumbent packing, repairs every iterate to exact feasibility before scoring it, and
keeps the best; it is recognisably the same program as `baseline_packing`, which is the
point above, and it contains no coordinate table. It does contain
``random.Random(15 if n == 26 else 1000003 + 7919 * n)`` -- a restart seed picked for the
instance it was asked about. So its zero cold-start gap says it reliably reproduces its
own lucky restart sequence at n=26, which is a weaker claim than reliably finding
2.6359831. Run directly at a size nobody asked it about, and given the same 20s the
search gave it, it scores 2.7827752 for n=29 against 2.4166667 for the grid, so it is an
algorithm on the evidence rather than on its author's word -- but see the note on
`generality_check` below, which does not give it that budget and concludes the opposite.
"""

# Simplifications vs. the source:
# - Budget is counted in optimizer iterations rather than metric calls or dollars; the
#   paper spends 63 evaluations and $3.18 on this domain.
# - Three Mechanism-3 objectives, not the paper's four: it names them without defining
#   them, and two of the four readings attempted here measured noise or rewarded doing
#   nothing (see `trajectory_metrics`).
# - The two modules alternate by a coin flip (``--code-share``); the paper does not say
#   how it splits attention between them. There is no per-module score to plot, so
#   Mechanism 2's leapfrogging curve is not reproduced -- only which module produced
#   each accepted gain (see `PackSystem` and `module_note`).
# - `generality_check` gives the packer 0.5s, and an artifact that honours a short budget
#   by returning its safe fallback is reported as "a table rather than an algorithm" on
#   that basis. It says exactly that about the winning artifact above, which scores
#   2.7827752 at n=29 against the grid's 2.4166667 when given the run's own 20s. So this
#   diagnostic currently distinguishes "bails out when rushed" from "hardcodes an
#   answer" not at all, and the proposer is shown the wrong conclusion every iteration.
#   Raising its budget would cost a full extra packer run per evaluation, which is why
#   the cheap version is here; the trade is not free either way.
# - One run per configuration and no variance estimate, on a domain where the three
#   configurations measured span 2.6317 to 2.6360 -- a range as wide as the whole gap
#   between the published systems the headline is compared against.
# - The wall-clock backstop in `_run_packer` is best-effort, not a guarantee. It is a
#   Python-level ``SIGALRM``, and the handler only runs when the interpreter next gets
#   control, so a synthesized packer sitting in a long C call or one that has moved work
#   into a subprocess can hang a run indefinitely -- which does happen. A real bound
#   needs process isolation, which this example does not do.
# - The accept gate is ``after > before`` with no minimum improvement, so a proposal
#   worth a billionth of the score is accepted and prunes its parent. Adding a threshold
#   would depart from Algorithm 1 as written, so it is documented rather than changed.
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
from effectful.handlers.llm import Agent, Skill


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
    """What the synthesized packer may import, as a sentence for the prompt.

    Only that, and the restraint is the point. The paper's *result* on this domain is a
    particular algorithm -- an exact LP over the radii for fixed centres, its duals as
    gradients on the centres, the two alternated (Appendix K.6) -- and naming it in the
    prompt that asks the search to find it turns a search into a transcription. What
    belongs here is the part the proposer cannot discover for itself, because it cannot
    import anything to find out: which libraries exist.
    """
    if not NUMERIC_LIBRARIES:
        return (
            "Only the Python standard library is available -- no numpy, no scipy -- so "
            "write the numerics yourself."
        )
    return (
        f"These numeric libraries are installed and you may import them: "
        f"{', '.join(NUMERIC_LIBRARIES)}."
    )


def feasible_scale(circles: collections.abc.Sequence[Circle]) -> float:
    """The largest factor ``s <= 1`` for which scaling every radius by ``s`` makes the
    packing exactly feasible -- 1.0 for a packing with room to spare, 0.0 for one that
    cannot be rescued.

    The score is ``s * total_radius``, and that is deliberate. Scoring the *reported*
    radii against a tolerance invites the artifact to overshoot by just under it: a
    packer that adds 4e-10 to every radius sits inside a 1e-9 feasibility check and
    collects the difference, and a search will find that before it finds a better
    packing. Shrinking to exact feasibility instead of thresholding removes the
    incentive -- an inflated radius is scaled straight back out, and it drags every
    other circle down with it -- and it replaces the feasible/infeasible cliff with a
    gradient the proposer can actually climb.

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
    absorbs that too. Each module's advance is the foundation for the other's next one.

    Modelling that needs nothing from the engine: a candidate is a ``PackSystem``, and
    the domain's proposer decides on each iteration which module to mutate -- rewrite
    the packer directly, or rewrite the refiner instruction and apply it to the packer.
    Both paths produce a new ``PackSystem`` that lands on the same frontier. ``origin``
    records which module produced it, so the leapfrogging is countable afterwards
    (`module_note`); it is deliberately left out of ``__str__`` so it does not perturb
    the cache key.

    Two things the paper credits to this mechanism are *not* reachable here:

      * The paper's "a failed code mutation is recovered rather than lost, because the
        refiner can rewrite it" cannot happen in this implementation. The refiner is
        applied to a parent drawn from the pool, and the accept gate never admits a
        candidate that scored zero, so the packer handed to the refiner has always
        already passed. Only a broken *seed* could be repaired this way.
      * The leapfrogging the paper measures is a per-module score curve -- code at 0.98
        while the refiner is at 1.93, then the reverse. There is no per-module score to
        plot here: the refiner only ever affects the world through the packer it
        rewrites, so a ``PackSystem`` has one score and the two modules share it. What
        `module_note` can honestly report is which module produced each accepted gain
        and whether the two alternate, which is the observable shadow of leapfrogging
        rather than the measurement itself.
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

    @Skill.define
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
        the harness synthesizes this Skill's body: it calls this Skill
        recursively -- routed to your own submission, so it costs nothing -- and runs
        the packer that comes back.

        >>> _packer = Proposer().propose_packer(seed_packer, [], 4, numeric_toolbox())
        >>> _circles = _packer(4, 0.5, None)
        >>> len(_circles) == 4 and feasible(_circles)
        True
        """

    @Skill.define
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

    @Skill.define
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

    @Skill.define
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

    @Skill.define
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
    proposer gets to see that it did.

    Read the verdict with the 0.5s budget in mind. It is short because this is the one
    diagnostic that costs an extra run of the packer, and it is short enough that an
    artifact which returns a safe fallback rather than a half-finished optimization when
    rushed is indistinguishable here from one that memorised an answer -- so a "table"
    verdict is evidence about the packer's behaviour under a tight budget, not proof
    that it fails to generalize.
    """
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

      * ``max_score``   -- the best packing it found, the headline number
      * ``mean_score``  -- what it achieves typically, not at its luckiest
      * ``worst_score`` -- the floor it is guaranteed not to fall below

    Three, not the paper's four. It names its four without defining them, and the two
    obvious readings of the missing pair do not survive contact with this evaluator:

      * "Improvement rate" as best-minus-first over the repeats measures which draw came
        out best, because the repeats are *independent* runs of the same artifact on the
        same input rather than a sequence of refinements. It is a property of the random
        seed, and no arrangement of it can say what it wants to say -- how much the
        artifact would gain from more time -- without actually giving it more time.
      * "Stability" as any scale-free measure of run-to-run agreement is maximized by an
        artifact that reliably does nothing. A deterministic packer takes the best
        attainable value whatever it scores, so it is non-dominated on that axis
        permanently and can never be pruned. An objective a do-nothing candidate wins
        outright does not keep algorithmic families alive on the frontier; it keeps junk
        alive on it.

    ``worst_score`` is the consistency objective that is not gameable that way: it
    rewards an artifact for being reliable *at a good level*, and a candidate that
    reliably scores nothing is last on it rather than first. All three are
    higher-is-better, which the Pareto machinery requires.

    >>> [str(m) for m in trajectory_metrics([1.0, 1.0, 1.0])]
    ['max_score=1', 'mean_score=1', 'worst_score=1']
    >>> [str(m) for m in trajectory_metrics([0.4, 1.0])]
    ['max_score=1', 'mean_score=0.7', 'worst_score=0.4']
    """
    if not scores:
        return [
            Metric("max_score", 0.0),
            Metric("mean_score", 0.0),
            Metric("worst_score", 0.0),
        ]
    return [
        Metric("max_score", max(scores)),
        Metric("mean_score", statistics.fmean(scores)),
        Metric("worst_score", min(scores)),
    ]


PACKING_REPEATS = 3


def evaluate_packing(
    packer: Packer,
    n: int,
    time_budget: float,
    current_best: list[Circle] | None,
    *,
    diagnose: bool = True,
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

    ``diagnose=False`` skips the generality check, which is the one diagnostic that
    costs a whole extra run of the packer. The SI ablation passes it: on a domain whose
    budget is wall-clock, an arm whose diagnostics are discarded unread must not be
    charged for producing them, or the ablation measures the bill as well as the effect.
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
    if diagnose:
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
# The control the paper does not run.
# ---------------------------------------------------------------------------


def baseline_packing(
    n: int, seconds: float, rng: random.Random
) -> tuple[list[Circle], int]:
    """Warm-started random-restart SLSQP with no model anywhere in the loop.

    This is the comparison 5.3 is missing, and it is the one that decides what the
    headline number means. The paper's only comparators on this instance are other
    LLM-driven program-search systems -- AlphaEvolve at 2.635, OpenEvolve at 2.6307 --
    so nothing in it separates "reflective search found a good algorithm" from
    "reflective search wrote a competent call to a constrained optimizer". Those are
    very different claims, and on an instance whose published values sit within a few
    thousandths of each other the difference is the whole result.

    So this function is a fair opponent rather than a straw man: it builds the same
    ``4n`` wall constraints and ``n(n-1)/2`` separation constraints over the same ``3n``
    variables that the search's winning artifacts converge on, supplies the analytic
    constraint Jacobian, and spends its budget on restarts -- half from random centres,
    half warm-started with jitter from its own incumbent, which is the same advantage
    `run_pack` gives the artifacts through ``current_best``. What it does not have is a
    model choosing what to try next. Whatever margin the search shows over this is the
    part attributable to reflection.

    Returns the best packing found and how many restarts fitted in the budget.
    """
    import numpy as np
    from scipy.optimize import minimize

    upper, lower = np.triu_indices(n, k=1)
    rows = np.arange(len(upper))
    eye, zero = np.eye(n), np.zeros((n, n))
    # d(wall constraints)/d(x, y, r): constant, so it is built once.
    walls_jac = np.vstack(
        [
            np.hstack([eye, zero, -eye]),  # x - r >= 0
            np.hstack([zero, eye, -eye]),  # y - r >= 0
            np.hstack([-eye, zero, -eye]),  # 1 - x - r >= 0
            np.hstack([zero, -eye, -eye]),  # 1 - y - r >= 0
        ]
    )

    def constraints(v: typing.Any) -> typing.Any:
        x, y, r = v[:n], v[n : 2 * n], v[2 * n :]
        walls = np.concatenate([x - r, y - r, 1.0 - x - r, 1.0 - y - r])
        gap = np.hypot(x[upper] - x[lower], y[upper] - y[lower]) - r[upper] - r[lower]
        return np.concatenate([walls, gap])

    def constraints_jac(v: typing.Any) -> typing.Any:
        x, y = v[:n], v[n : 2 * n]  # the separation gradient does not involve r
        dx, dy = x[upper] - x[lower], y[upper] - y[lower]
        distance = np.maximum(np.hypot(dx, dy), 1e-12)
        pairs = np.zeros((len(upper), 3 * n))
        pairs[rows, upper], pairs[rows, lower] = dx / distance, -dx / distance
        pairs[rows, n + upper], pairs[rows, n + lower] = dy / distance, -dy / distance
        pairs[rows, 2 * n + upper] = pairs[rows, 2 * n + lower] = -1.0
        return np.vstack([walls_jac, pairs])

    gradient = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    deadline = time.monotonic() + seconds
    best: list[Circle] = []
    restarts = 0
    while time.monotonic() < deadline:
        restarts += 1
        if best and restarts % 2 == 0:  # polish the incumbent
            start = np.array(
                [c.x for c in best] + [c.y for c in best] + [c.r for c in best]
            )
            start[: 2 * n] += np.array([rng.gauss(0.0, 0.02) for _ in range(2 * n)])
        else:  # a fresh configuration
            start = np.array(
                [rng.random() for _ in range(2 * n)]
                + [0.5 / math.ceil(math.sqrt(n))] * n
            )
        np.clip(start, 0.0, 1.0, out=start)
        try:
            solved = minimize(
                lambda v: -v[2 * n :].sum(),
                start,
                jac=lambda _: gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=[
                    {"type": "ineq", "fun": constraints, "jac": constraints_jac}
                ],
                options={"maxiter": 200, "ftol": 1e-10},
            )
        except Exception:  # a restart that fails to converge is simply skipped
            continue
        found = [
            Circle(
                x=float(solved.x[i]),
                y=float(solved.x[n + i]),
                r=float(solved.x[2 * n + i]),
            )
            for i in range(n)
        ]
        if packing_score(found, n) > packing_score(best, n):
            best = found
    scale = feasible_scale(best)
    return [Circle(x=c.x, y=c.y, r=c.r * scale) for c in best], restarts


def baseline_note(n: int, seconds: float, rng: random.Random) -> str:
    """Run the no-LLM control and say what it means, or say why it could not run."""
    if importlib.util.find_spec("scipy.optimize") is None:
        return (
            "No-LLM baseline: not available, because scipy is not installed here. The "
            "search's numbers are therefore unattributed -- there is nothing to say how "
            "much of the distance from the seed is the reflection and how much is the "
            "optimizer the artifacts call."
        )
    started = time.monotonic()
    circles, restarts = baseline_packing(n, seconds, rng)
    return (
        f"No-LLM baseline (warm-started random-restart SLSQP, no model in the loop): "
        f"{packing_score(circles, n):.7f} for n={n} in "
        f"{time.monotonic() - started:.0f}s over {restarts} restarts. This is the "
        f"comparison the paper's 5.3 does not report, and the number the search has to "
        f"beat for its margin to be about reflection rather than about scipy."
    )


# ---------------------------------------------------------------------------
# Wiring the domain to the engine.
# ---------------------------------------------------------------------------


def run_pack(args: argparse.Namespace, rng: random.Random) -> tuple[Result, list[str]]:
    """Single-task search over a two-module system, with the incumbent threaded through.

    Three things here are the paper's setup rather than a simplification of it: the
    artifact is handed the best packing found so far, the Pareto objectives are the
    run-distribution metrics of its Mechanism 3, and the search alternates between two
    modules on one shared frontier (Mechanism 2). The incumbent is ordinary mutable
    state in this closure -- and because it makes an evaluation depend on more than the
    artifact, it is declared to the engine as a ``state_key``.

    Returns the result and the per-iteration module choices, which the engine has no
    reason to know about and `module_note` needs in order to say which module each
    accepted gain came from.
    """
    n, budget = args.num_circles, args.time_budget
    toolbox = numeric_toolbox()
    incumbent: list[list[Circle]] = [[]]  # a one-slot cell: the best packing so far
    origins: list[str] = []

    def state_key() -> str:
        return f"{packing_score(incumbent[0], n):.12f}"

    def evaluator(system: PackSystem, _: None) -> Evaluation:
        evaluation, best = evaluate_packing(
            system.packer,
            n,
            budget,
            incumbent[0] or None,
            diagnose=not args.no_side_info,
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
        code = rng.random() < args.code_share
        # Recorded before the call, so the list stays aligned with the trace even when
        # the proposal dies at decode: the engine appends a Step either way.
        origins.append("code" if code else "refiner")
        return (mutate_code if code else mutate_refiner)(system, feedback)

    def bootstrap(objective: str) -> PackSystem:
        return PackSystem(
            packer=Proposer().bootstrap_packer(objective, n, toolbox),
            refiner=SEED_REFINER,
            origin="bootstrap",
        )

    # Annotated rather than inferred: this domain has no dataset, so the element type
    # would infer as ``None`` and fail the engine's ``E: Example`` bound.
    result: Result = optimize_anything(
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
    return result, origins


# ---------------------------------------------------------------------------
# Reporting and main
# ---------------------------------------------------------------------------


def module_note(result: Result, origins: list[str]) -> str:
    """Which module did the work, as far as it can honestly be attributed.

    Multi-module search is only visible if you count it, and counting the survivors on
    the frontier is not enough: it says which module's proposals lasted, not which one
    moved the score. So this also reports, per module, how many proposals it made, how
    many were accepted, and the mean minibatch gain when they were -- and the order the
    accepted gains arrived in, which is where the paper's leapfrogging would show up as
    the two modules handing off to each other.

    What it deliberately does not report is a per-module score, because there isn't one
    (see `PackSystem`). A handover in the sequence below means the other module produced
    the next accepted gain; it does not mean that module was ahead of its partner.
    """
    surviving = collections.Counter(
        c.artifact.origin for c in result.pool if isinstance(c.artifact, PackSystem)
    )
    proposed: collections.Counter[str] = collections.Counter()
    accepted: collections.Counter[str] = collections.Counter()
    gains: dict[str, list[float]] = {"code": [], "refiner": []}
    sequence: list[str] = []
    for step, origin in zip(result.history, origins):
        proposed[origin] += 1
        if step.accepted:
            accepted[origin] += 1
            gains[origin].append(step.after - step.before)
            sequence.append(origin)

    winner = result.best.artifact
    handovers = sum(a != b for a, b in zip(sequence, sequence[1:]))
    return "\n".join(
        [
            "Modules: "
            + "; ".join(
                f"{name} proposed {proposed[name]}, accepted {accepted[name]}"
                + (
                    f", mean minibatch gain {statistics.fmean(gains[name]):+.6f}"
                    if gains[name]
                    else ""
                )
                for name in ("code", "refiner")
                if proposed[name]
            ),
            "Surviving frontier by module: "
            + (", ".join(f"{k} {v}" for k, v in sorted(surviving.items())) or "none"),
            "The best candidate came from the "
            + f"{winner.origin if isinstance(winner, PackSystem) else 'unknown'} module",
            f"Accepted gains in order: {' -> '.join(sequence) or 'none'}"
            + (f" ({handovers} handover(s))" if len(sequence) > 1 else ""),
        ]
    )


def cold_start_note(result: Result, n: int, time_budget: float) -> str:
    """What the winning artifact scores on its own, with no incumbent to polish.

    The headline of a run with the incumbent threaded through belongs to the *run*, not
    to the artifact credited with it. Every candidate is handed the best packing found so
    far and told never to return something worse, so its score accumulates the work of
    everything that ran before it: a candidate that returned its input unchanged would be
    recorded at the full incumbent value. That is the paper's own setup, not a deviation
    from it -- its evolved packer takes ``current_best_solution`` too, so its 2.63598 is
    a trajectory number in exactly the same way -- but it means "the winning artifact
    reached X" is not a statement this search is entitled to make. Running the winner
    once from nothing is, and it costs one evaluation.
    """
    system = result.best.artifact
    if not isinstance(system, PackSystem):
        return ""
    evaluation, _ = evaluate_packing(
        system.packer, n, time_budget, None, diagnose=False
    )
    return (
        f"Cold start: the winning artifact alone, handed no incumbent, scores "
        f"{evaluation.score:.7f} against the run's {result.best_score:.7f}. The "
        f"difference is what it inherited from the candidates before it rather than "
        f"earned -- the run's number is the search's, the cold-start number is the "
        f"artifact's."
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
        "--visual-si",
        action="store_true",
        help="Send a rendered image of the packing as side information",
    )
    parser.add_argument(
        "--seedless",
        action="store_true",
        help="Bootstrap candidate zero from a natural-language objective",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the no-LLM control instead of the search: random-restart SLSQP for "
        "the wall-clock the search would have spent on packers",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=0.0,
        help="Seconds for --baseline; 0 matches the search's packer time, which is "
        "--budget x --time-budget x the evaluator's repeats",
    )
    args = parser.parse_args()

    if args.baseline:
        seconds = (
            args.baseline_seconds or args.budget * PACKING_REPEATS * args.time_budget
        )
        print(
            f"[baseline] no LLM, n={args.num_circles}, {seconds:.0f}s "
            + (
                "(as given)"
                if args.baseline_seconds
                else f"(= {args.budget} iterations x {PACKING_REPEATS} repeats x "
                f"{args.time_budget:.0f}s, the packer time the search would spend)"
            )
        )
        print(baseline_note(args.num_circles, seconds, random.Random(args.seed)))
        return

    result, origins = run_pack(args, random.Random(args.seed))
    report(
        result,
        selection=args.selection,
        side_info=not args.no_side_info,
        notes=[
            note
            for note in (
                module_note(result, origins),
                cold_start_note(result, args.num_circles, args.time_budget),
            )
            if note
        ],
        render_artifact=render_system,
    )
    # No assertion that the score improved: one would look reassuring and could not
    # fail. The headline is a maximum over the surviving pool, the seed is a candidate
    # in it, and a pruned seed was by definition dominated by a survivor. The numbers
    # worth checking are the two the notes above print -- the winner's cold-start score,
    # and what --baseline reaches with no model at all.


if __name__ == "__main__":
    main()
