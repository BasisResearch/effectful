"""optimize_anything: one reflective Pareto search over typed artifacts.

Implements the core of "optimize_anything: Unified Text Optimization can Outperform
Specialized Systems" (OpenReview 7M28lVzVUq). The paper's observation is that an
enormous range of problems -- a CUDA kernel, a packing algorithm, a scheduling
policy, an agent architecture, a system prompt -- are all *the same problem*:
improve an artifact that some evaluator scores. Its system is GEPA's reflective
Pareto search (Agrawal et al., ICLR 2026) lifted off prompts onto arbitrary
artifacts, plus three claims about what that lifting requires: Side Information as a
first-class evaluator contract, a *refiner* step that catches malformed generations
before evaluation, and three optimization modes selected by which data you supply.

Its Algorithm 1 is short enough to quote:

    P <- [seed];  evaluate the seed; record per-objective scores
    while budget remains:
        k <- ParetoSelect(P)          # sample in proportion to frontier frequency
        M <- a minibatch of 2-3 examples
        run candidate k on M, collecting scores *and* side information
        k' <- Reflect(k, scores, SI)  # the LLM proposes a revision
        if k' improves on M: evaluate it fully, admit it, prune dominated candidates
    return the best candidate

Everything except ``Reflect`` is ordinary Python, and that is the example's point.
Each of the paper's mechanisms falls out of an effectful idiom rather than a
subsystem:

  * **Side Information is just the evaluator's typed return value.** The paper needs
    a ``side_info`` dict and a serializer to carry stack traces, sub-scores, and
    rendered images to the proposer. Here ``evaluate`` returns an ``Evaluation``
    (score, sub-score ``Metric``s, ``Diagnostic``s) and the proposer's prompt splices
    it with ``{feedback}`` -- the Encodable bridge already knows how to put a typed
    value in the model's context. Because SI is *any* Encodable, the rendered
    ``PIL.Image`` of the current packing is SI too (``--visual-si``), through the same
    path ``image_input.py`` and ``illustration.py`` use. The paper's SI ablation is a
    flag here (``--no-side-info``), and it changes one line: which view of the
    ``Evaluation`` the proposer is shown.

  * **The "refiner" is ``TenacityRetryer`` plus decode-time certification.** The paper
    reports needing a dedicated step for "malformed code blocks, import errors, syntax
    issues ... essential for code and agent artifacts where minor formatting errors
    cause complete evaluation failure". A code artifact here is a ``Template``
    returning a ``Callable``: the model's source is parsed, type-checked against the
    requested signature, compiled, and its own doctests are run at decode time, so a
    malformed candidate is fed its own error and revised before it ever reaches the
    evaluator. The packer must carry doctests proving it returns ``n`` feasible
    circles (the doctest-certification idiom of ``world_model_agent.py``), so
    "the artifact at least runs and obeys its contract" is true by construction.

  * **"Serialize the artifact as a string" is the step you get to skip.** The search
    loop is generic in the artifact type ``A``; each domain keeps a *typed* artifact
    with its own typed proposer -- a ``Packer`` callable, a prompt ``str`` -- and the
    loop never sees a string. One algorithm, three artifact types, no adapter layer.

  * **The three modes are a function signature.** ``optimize_anything`` takes
    ``dataset`` and ``valset``; neither means single-task (the artifact *is* the
    solution, and the Pareto objectives are its sub-scores), ``dataset`` alone means
    multi-task (one shared frontier, one specialized artifact per task), both means
    generalization (search on train, select on held-out val). Pareto selection,
    minibatching, the accept-if-improves rule, dominated-candidate pruning, and the
    content-addressed evaluation cache are plain Python, and stay plain Python.

Three domains instantiate it, mirroring the paper's own:

  * ``pack`` (single-task, paper 5.3): pack ``n`` circles in the unit square to
    maximize the sum of radii, on the paper's own instance with ``--num-circles 26``.
    This domain is set up to be a genuine attempt at the paper's result rather than an
    illustration of the loop, so it follows 5.3 and Appendix G closely: the artifact
    has the paper's signature ``pack(n, time_budget, current_best)`` and is handed the
    incumbent packing to polish; it may use whatever numeric libraries are actually
    installed (scipy and numpy here), since the paper's winner is an LP over radii with
    dual-variable gradients driving a local optimizer over centres; the Pareto
    objectives are the run-distribution metrics of Mechanism 3 (max, mean, stability,
    improvement rate over repeated runs) rather than one score; and the search evolves
    *two* modules on one shared frontier -- the packer and a refiner instruction --
    which is the leapfrogging of Mechanism 2. The evaluator is deterministic Python
    throughout, so no model sits in the scoring path.
  * ``prompt`` (generalization, paper A.3): optimize a system prompt for constrained
    writing -- an exact word count, an initial-letter rule, a banned letter -- scored
    by deterministic Python on held-out instances. SI is the paper's design: the
    instance, the model's reasoning, what it produced, and a per-constraint account of
    what went wrong. As in the paper the artifact is optimized *for a cheaper model*
    (``--worker-model``) than the one proposing it, which in effectful is one scoped
    handler. (The paper uses AIME here; a 2026-class model saturates any arithmetic
    set small enough to embed in an example, which is why the task is constraint
    tracking instead -- see the domain's comment.)
  * ``kernel`` (multi-task, paper 5.2): four related list-transform tasks whose
    artifact is the *instruction that drives code generation*, exactly as the paper
    evolves the prompt behind its CUDA kernels. The frontier is shared, each task
    picks its own best instruction off it, and the run reports how much of the winning
    lineage was refined while looking at a *different* task -- cross-transfer, counted.

Demonstrates:
- One generic search loop over a *typed* artifact ``A``: ``Template``s returning a
  ``Callable`` (code) and returning ``str`` (a prompt) drive the same optimizer, so
  the paper's unification claim needs no serialization layer
- Side Information as a typed evaluator return value spliced into the proposer's
  prompt -- text diagnostics, sub-scores, and (with ``--visual-si``) a rendered
  ``PIL.Image``, all through the ordinary Encodable path
- Decode-time certification standing in for the paper's refiner: a synthesized
  ``Packer`` must carry doctests proving it returns ``n`` feasible circles, and a
  candidate that fails is corrected by ``TenacityRetryer`` before it is evaluated
- Pareto-frontier selection over per-example (multi-task/generalization) or per-metric
  (single-task) objectives, with frontier-frequency sampling and dominated pruning --
  all plain Python, ablatable with ``--selection best``
- Multi-module search with no engine support at all: the packing domain's artifact is a
  ``PackSystem`` of code *and* a refiner instruction, and "which module to mutate" is a
  branch in the domain's proposer, so both modules compete on the one frontier
- Search over state the evaluator depends on: threading the paper's
  ``current_best_solution`` makes scoring impure, and ``state_key`` is how a domain
  declares that so the evaluation cache stays honest
- The paper's two headline ablations as flags: ``--no-side-info`` (score-only
  feedback) and ``--selection best`` (greedy instead of Pareto)
- Three optimization modes selected purely by which arguments are supplied, including
  seedless mode (``--seedless``), where a natural-language objective bootstraps
  candidate zero
- The paper's proposer/target-model split as a scoped handler: the evaluator's model
  calls run under a cheap ``--worker-model`` nested inside the harness's stack, so
  "optimize this artifact for that model" costs one ``with`` block
"""

# What this actually reproduces, measured on 2026-07-29 (gpt-5.5 proposing,
# gpt-4.1-mini working, 8 to 10 iterations per run -- three or four orders of magnitude
# below the paper's budgets, so read these as "the mechanism runs", not as
# replications). Where a claim did not reproduce, that is recorded here rather than
# tuned away:
#
#   pack, n=26  the paper's own instance, ``--num-circles 26 --time-budget 20``. Three
#               runs from the 6x6 grid seed at 2.1667 reached 2.6083 (4 iterations),
#               2.6147 (6 iterations) and 2.6359831 (stopped after 10). The paper
#               reports 2.63598 for optimize_anything, 2.635 for AlphaEvolve and
#               2.6307 for OpenEvolve at 200 evaluations, so the best of those runs
#               matches the paper's figure at the precision it is quoted, in about ten
#               evaluations. Read that best-of-three carefully: the spread across runs
#               (2.608 to 2.636) is wider than the gap between the published systems,
#               so one run of this example is not a ranking of anything.
#
#               The winning artifact is an algorithm, not a remembered answer, which
#               matters because the optimum for n=26 is published and a model could
#               simply recite it. It builds the 4n wall constraints and n(n-1)/2
#               separation constraints programmatically over 3n variables, hands them
#               to SLSQP warm-started from the incumbent packing, repairs every iterate
#               to exact feasibility before scoring it, and keeps the best -- the same
#               shape as the paper's evolved artifact, reached in two accepted
#               proposals. It contains no coordinate table and no special case for 26.
#   pack, ablations
#               ``--no-side-info`` reached 1.585 against 1.591, and ``--selection best``
#               also reached 1.591: both measured at n=10 on the *earlier* version of
#               this domain, before the incumbent, the numeric libraries and the second
#               module were added, and both at an instance whose ceiling every arm hits
#               within a few iterations. They are recorded because they were run, not
#               because they test the paper's 93.96% -- doing that means re-running them
#               at n=26 on the current configuration.
#   prompt      train 0.733 -> 0.867, held-out val 0.800 -> 0.800. The search improved
#               the prompt on the instances it saw and none of that transferred: a
#               negative result on the paper's headline generalization claim at this
#               scale, on five training instances.
#   kernel      multi-task, mean speedup over the reference implementation
#               1.134 -> 1.603 across five tasks, with per-task winners drawn from
#               *different* candidates on the shared frontier, and 3 of the 5 winners
#               last refined while the proposer was looking at a different task.
#               The single-task control (``--single-task``, same two iterations per
#               problem) went 1.224 -> 1.285. Multi-task ahead of single-task at equal
#               per-problem budget is the direction of the paper's 5.4, though the two
#               arms start from differently-timed seeds, so compare the gains (+41% vs
#               +5%) rather than the endpoints. Several iterations scored zero because
#               an instruction pushing for speed made the worker write incorrect
#               kernels -- the correctness gate doing its job, and a failure mode the
#               paper's limitations section predicts.
#
# Simplifications vs. the source. The ``pack`` domain is deliberately held to the
# paper's setup -- incumbent threading, real numeric libraries, Mechanism 3's Pareto
# objectives, Mechanism 2's two modules -- so the gaps below are mostly about the other
# two domains and about budget:
# - Scale. The paper spends $1-$145 per domain over thousands of metric calls.
#   ``--num-circles 26`` is the paper's instance and the defaults are far too small for
#   it; see the measured note above for what a real attempt reached.
# - Budget is counted in optimizer iterations, not metric calls or dollars. The
#   proposer/worker split is a single ``--worker-model`` for both model-in-the-loop
#   domains, where the paper tunes proposer and target model per domain.
# - ``prompt`` optimizes for constrained writing rather than AIME 2022-2025, and it
#   selects the winner on the same held-out set it then reports, where the paper keeps
#   a third split. Read its val number as selection-biased.
# - ``kernel`` runs pure-Python list transforms on a CPU, not CUDA kernels on a V100
#   against KernelBench, and scores mean speedup rather than the paper's fast_p(s)
#   curve. Its side information has no documentation-retrieval channel.
# - No island model / MAP-Elites -- AlphaEvolve's structure, which the paper also drops.
# - The four Mechanism-3 objectives are named but not defined in the paper; "stability"
#   and "improvement rate" here are this example's reading of those names, and only the
#   ``pack`` domain uses them at all.
# - Acceptance is a mean-improvement test on the minibatch with no statistical
#   correction, and dominated candidates are pruned immediately rather than archived
#   with a novelty criterion.
# - The evaluation cache is an in-memory dict keyed by the artifact's source (or the
#   string itself) and the example name -- content-addressed in spirit, not persisted.
# - Cross-task transfer is reported as a post-hoc lineage count over one run, against
#   one single-task control (``--domain kernel --single-task``) -- not the paper's
#   MT10/MT20 scaling study, and with no repeats to put an error bar on either arm.
# - Multi-module search alternates the two modules by a coin flip (``--code-share``);
#   the paper does not say how it splits attention between them.
# - The kernel domain's score is a wall-clock ratio, so it is only as reproducible as
#   the machine is quiet. The baseline is re-timed back to back with every candidate
#   for exactly this reason (see ``measure_speedup``), which makes the ratio robust to
#   a loaded machine but not to a machine whose speed changes mid-measurement.

import argparse
import bisect
import collections
import collections.abc
import dataclasses
import importlib.util
import inspect
import io
import linecache
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

from effectful.handlers.llm import Agent, Template
from effectful.handlers.llm.harness.providing import LiteLLMProvider
from effectful.ops.semantics import handler

# The one piece of handler boilerplate in this file, and it is load-bearing. Every
# domain the paper reports optimizes an artifact *for a specific model*: a prompt for
# GPT-4.1-mini, an agent architecture for Gemini Flash, kernels behind GPT-5. The
# proposer is strong; the model the artifact runs on is cheap, and the whole point is
# to lift the cheap one. In effectful "run this call on a different model" is a scoped
# handler, so ``worker(...)`` is all the machinery that split needs -- no config
# system, no per-template model registry.
#
# One consequence worth knowing: this provider is installed *inside* the harness's
# stack, so it takes precedence over the ``TenacityRetryer`` for the calls it covers
# and their decoding failures are not retried. That is the behaviour we want here --
# a prompt whose answers do not decode has failed, and the evaluator scores it zero
# with the traceback as side information rather than quietly repairing it.
WORKER_MODEL = "openai/gpt-4.1-mini"


def worker(model: str) -> typing.Any:
    """Scope a call to the cheap model the artifact is being optimized *for*."""
    return handler(LiteLLMProvider(model=model))


# ---------------------------------------------------------------------------
# Side Information: the evaluator's contract.
#
# The paper's central API claim is that an evaluator returns a score *and* whatever
# diagnostics it can produce, and that the proposer reads them. Here that contract is
# a return type. Note the absence of a dict: values that cross the model boundary use
# ``list``s of small dataclasses, because strict tool schemas reject free-form dicts.
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class Metric:
    """One sub-score of an evaluation. Higher is always better, by construction -- the
    Pareto machinery compares metrics directly, so a "lower is better" quantity is
    negated by the evaluator that produces it."""

    name: str
    value: float

    def __str__(self) -> str:
        return f"{self.name}={self.value:.6g}"


@pydantic.dataclasses.dataclass(frozen=True)
class Diagnostic:
    """One piece of Side Information: a named, human-readable explanation of *why* the
    artifact scored what it scored -- a violated constraint, a failing test case, a
    traceback, a timing. This is the signal the paper argues is the text-optimization
    analogue of a gradient."""

    name: str
    detail: str

    def __str__(self) -> str:
        return f"{self.name}: {self.detail}"


@pydantic.dataclasses.dataclass(frozen=True)
class Evaluation:
    """What an evaluator returns: a score, optional sub-scores, and optional Side
    Information. ``score_only`` is the paper's SI ablation -- the same evaluation with
    its diagnostics withheld, which is all "score-only feedback" means."""

    score: float
    metrics: list[Metric] = dataclasses.field(default_factory=list)
    diagnostics: list[Diagnostic] = dataclasses.field(default_factory=list)

    def score_only(self) -> "Evaluation":
        return Evaluation(score=self.score)

    def __str__(self) -> str:
        lines = [f"score: {self.score:.6g}"]
        if self.metrics:
            lines.append("metrics: " + ", ".join(str(m) for m in self.metrics))
        lines.extend(f"- {d}" for d in self.diagnostics)
        return "\n".join(lines)


@pydantic.dataclasses.dataclass(frozen=True)
class Rollout:
    """One (example, evaluation) pair handed to the proposer. In single-task mode the
    example is the artifact itself, so ``example`` names the task."""

    example: str
    evaluation: Evaluation

    def __str__(self) -> str:
        return f"<{self.example}>\n{self.evaluation}\n</{self.example}>"


# ---------------------------------------------------------------------------
# The generic engine. No LLM appears below this line except through ``proposer``,
# which is the domain's Template call: everything else -- selection, minibatching,
# acceptance, pruning, caching -- is ordinary Python.
# ---------------------------------------------------------------------------


class Example(typing.Protocol):
    """What the loop needs of a dataset element: a name to key objectives and the
    evaluation cache by. Domains supply richer dataclasses; the member is a read-only
    property so a frozen dataclass satisfies it."""

    @property
    def name(self) -> str: ...


type Evaluator[A, E] = collections.abc.Callable[[A, E | None], Evaluation]
type Reflect[A] = collections.abc.Callable[[A, list[Rollout]], A]


def source_of(fn: object) -> str | None:
    """The source of a synthesized callable, or ``None`` if it cannot be recovered.

    ``inspect.getsource`` tokenizes the block it finds and raises for more reasons than
    its documented OSError/TypeError -- a synthesized function whose block ends inside
    a multi-line string raises ``tokenize.TokenError``, and a run once died on exactly
    that. When block extraction fails the whole synthesized module is still sitting in
    ``linecache``, and for both of this function's jobs -- keying the cache and showing
    the user what the search wrote -- the module is as good as the block.
    """
    try:
        return inspect.getsource(fn)  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        path = inspect.getsourcefile(fn) or fn.__code__.co_filename  # type: ignore[arg-type, attr-defined]
        lines = linecache.getlines(path)
        return "".join(lines) or None
    except Exception:
        return None


def artifact_key(artifact: object) -> str:
    """Content address of an artifact: its source if it is synthesized code, else its
    text. Two candidates with the same key have the same evaluation, which is what
    makes the cache sound."""
    if callable(artifact):
        return source_of(artifact) or repr(artifact)
    return str(artifact)


@dataclasses.dataclass
class Candidate[A]:
    """One artifact in the pool, with its score on every objective. ``refined_on``
    records which examples were in the minibatch that produced it -- the lineage the
    multi-task cross-transfer report reads."""

    index: int
    artifact: A
    scores: dict[str, float]
    parent: int | None
    generation: int
    refined_on: list[str] = dataclasses.field(default_factory=list)

    @property
    def average(self) -> float:
        return statistics.fmean(self.scores.values()) if self.scores else 0.0

    def dominates(self, other: "Candidate[A]", objectives: list[str]) -> bool:
        """Pareto dominance: at least as good everywhere, strictly better somewhere."""
        at_least = all(self.scores[j] >= other.scores[j] for j in objectives)
        strictly = any(self.scores[j] > other.scores[j] for j in objectives)
        return at_least and strictly


def pareto_frontier[A](
    pool: list[Candidate[A]], objectives: list[str]
) -> list[Candidate[A]]:
    """The non-dominated candidates: everything that is the best at *something*."""
    return [
        c
        for c in pool
        if not any(o.dominates(c, objectives) for o in pool if o is not c)
    ]


def pareto_select[A](
    pool: list[Candidate[A]], objectives: list[str], rng: random.Random
) -> Candidate[A]:
    """GEPA's selection rule, which the paper adopts verbatim: among non-dominated
    candidates, sample in proportion to how many objectives a candidate is *best* at.
    A specialist that wins one objective stays reachable; a generalist that wins many
    is reached often."""
    frontier = pareto_frontier(pool, objectives)
    weights = [
        sum(
            1
            for j in objectives
            if c.scores[j] >= max(f.scores[j] for f in frontier) - 1e-12
        )
        for c in frontier
    ]
    if not any(weights):  # degenerate scores -- fall back to uniform
        return rng.choice(frontier)
    return rng.choices(frontier, weights=weights, k=1)[0]


def best_select[A](
    pool: list[Candidate[A]], objectives: list[str], rng: random.Random
) -> Candidate[A]:
    """The ablation the paper argues against: always mutate the best average, which
    collapses the frontier's complementary strengths into one number."""
    return max(pool, key=lambda c: c.average)


@dataclasses.dataclass
class Step:
    """One iteration of the loop, kept for the trace and the convergence report."""

    iteration: int
    parent: int
    before: float
    after: float
    accepted: bool
    best: float


@dataclasses.dataclass
class Result[A, E]:
    """The outcome of a run: the pool (so the surviving frontier can be inspected),
    the trace, and the mode-appropriate answer."""

    mode: str
    pool: list[Candidate[A]]
    history: list[Step]
    objectives: list[str]
    seed_score: float
    best: Candidate[A]
    best_score: float
    per_task: list[tuple[str, Candidate[A], float]] = dataclasses.field(
        default_factory=list
    )
    evaluations: int = 0
    proposals: int = 0

    def frontier(self) -> list[Candidate[A]]:
        return pareto_frontier(self.pool, self.objectives)

    def iterations_to(self, threshold: float) -> int | None:
        """First iteration whose best-so-far reached ``threshold`` -- the convergence
        measure the paper's SI ablation reports as a speedup."""
        for step in self.history:
            if step.best >= threshold:
                return step.iteration
        return None


def mode_of(dataset: object, valset: object) -> str:
    if dataset is None:
        return "single-task"
    return "generalization" if valset is not None else "multi-task"


def optimize_anything[A, E: Example](
    *,
    evaluator: Evaluator[A, E],
    proposer: Reflect[A],
    seed: A | None = None,
    bootstrap: collections.abc.Callable[[str], A] | None = None,
    objective: str | None = None,
    dataset: list[E] | None = None,
    valset: list[E] | None = None,
    budget: int = 8,
    minibatch_size: int = 2,
    selection: str = "pareto",
    use_side_info: bool = True,
    rng: random.Random | None = None,
    task_name: str = "task",
    state_key: collections.abc.Callable[[], str] | None = None,
) -> Result[A, E]:
    """The paper's Algorithm 1, generic in the artifact type.

    The mode is a function of the arguments and nothing else: no ``dataset`` is
    single-task (objectives are the artifact's sub-score metrics), ``dataset`` alone is
    multi-task (objectives are the tasks; each task selects its own artifact off the
    shared frontier), and ``dataset`` + ``valset`` is generalization (search on train,
    select on held-out val). Seedless mode -- ``seed=None`` with an ``objective`` and a
    ``bootstrap`` -- lets the model write candidate zero.

    ``state_key`` names whatever *else* an evaluation depends on. It exists because the
    paper hands each circle-packing artifact the best solution found so far
    (``main(timeout, current_best_solution)``), which means the evaluator stops being a
    pure function of the artifact and a content-addressed cache silently starts serving
    stale scores. A domain that threads state like that says so here; every other
    domain leaves it ``None`` and keeps the plain content address.
    """
    rng = rng or random.Random(0)
    mode = mode_of(dataset, valset)
    cache: dict[tuple[str, str, str], Evaluation] = {}
    counters = {"evaluations": 0, "proposals": 0}

    def evaluate(artifact: A, example: E | None) -> Evaluation:
        """Content-addressed evaluation: the paper's caching, in three lines. It earns
        its keep because an evaluation can itself be an LLM call."""
        key = (
            artifact_key(artifact),
            example.name if example else task_name,
            state_key() if state_key else "",
        )
        if key not in cache:
            counters["evaluations"] += 1
            cache[key] = evaluator(artifact, example)
        return cache[key]

    def score_on(artifact: A, examples: list[E] | list[None]) -> dict[str, float]:
        """Per-objective scores. With a dataset the objectives are the examples; with
        none, they are the single evaluation's sub-score metrics (the paper: "single-
        task search admits only one data point, so per-example tracking reduces to
        per-metric tracking")."""
        if dataset is None:
            evaluation = evaluate(artifact, None)
            scores = {m.name: m.value for m in evaluation.metrics}
            # The headline is only its own objective when the evaluator offers nothing
            # finer; adding it alongside metrics that already contain it would just
            # double that dimension's weight in the frontier-frequency sampling.
            if not scores:
                scores["score"] = evaluation.score
            return scores
        return {
            e.name: evaluate(artifact, e).score for e in typing.cast(list[E], examples)
        }

    def headline(candidate: Candidate[A]) -> float:
        """The number a human reads. Averaging heterogeneous sub-scores is meaningless
        in single-task mode, where the artifact's own score is the answer; with a
        dataset the objectives *are* the per-example scores, so the average is right."""
        if dataset is not None:
            return candidate.average
        return candidate.scores.get("max_score") or candidate.scores.get("score", 0.0)

    pool_examples: list[E] | list[None] = (
        typing.cast(list[E] | list[None], dataset) if dataset is not None else [None]
    )

    # --- candidate zero -----------------------------------------------------
    if seed is None:
        if bootstrap is None or objective is None:
            raise ValueError(
                "seedless mode needs both an ``objective`` and a ``bootstrap``"
            )
        print("[seedless] bootstrapping candidate 0 from the objective ...")
        seed = bootstrap(objective)
    root = Candidate(
        index=0,
        artifact=seed,
        scores=score_on(seed, pool_examples),
        parent=None,
        generation=0,
    )
    pool: list[Candidate[A]] = [root]
    # Candidate indices come from a monotonic counter, not ``len(pool)``: pruning
    # removes dominated candidates, so a length-derived index would be reused and the
    # trace's parent links would silently point at the wrong artifact.
    minted = 1
    objectives = sorted(root.scores)
    history: list[Step] = []
    best_so_far = headline(root)
    print(
        f"[{mode}] {len(objectives)} objective(s): {', '.join(objectives)}\n"
        f"[seed] score {best_so_far:.8g}"
    )

    select = pareto_select if selection == "pareto" else best_select

    # --- the loop -----------------------------------------------------------
    for iteration in range(1, budget + 1):
        parent = select(pool, objectives, rng)

        # A minibatch of 2-3 examples, not the whole set: the paper's second Pareto
        # ingredient, so reflection is focused instead of trying to fix everything.
        minibatch: list[E] | list[None]
        if dataset is None:
            minibatch = [None]
        else:
            minibatch = rng.sample(dataset, k=min(minibatch_size, len(dataset)))

        rollouts = [
            Rollout(
                example=e.name if e is not None else task_name,
                evaluation=(
                    evaluate(parent.artifact, e)
                    if use_side_info
                    else evaluate(parent.artifact, e).score_only()
                ),
            )
            for e in minibatch
        ]
        before = statistics.fmean(r.evaluation.score for r in rollouts)

        # The only LLM call in the loop: reflect over the minibatch and its SI.
        counters["proposals"] += 1
        try:
            child_artifact = proposer(parent.artifact, rollouts)
        except Exception as exc:
            # Retries are exhausted, so the decode-time gate has rejected this proposal
            # for good and the iteration is lost. The reason is worth printing: it is
            # usually the artifact's own doctests failing, which is the paper's refiner
            # step doing its job where you can see it.
            reason = " ".join(str(exc).split())[:200]
            print(
                f"  iter {iteration}: proposal rejected at decode "
                f"({type(exc).__name__}: {reason})"
            )
            history.append(
                Step(iteration, parent.index, before, before, False, best_so_far)
            )
            continue

        after = statistics.fmean(evaluate(child_artifact, e).score for e in minibatch)

        accepted = after > before
        if accepted:
            # Only now pay for a full evaluation -- the paper's ordering.
            child = Candidate(
                index=minted,
                artifact=child_artifact,
                scores=score_on(child_artifact, pool_examples),
                parent=parent.index,
                generation=parent.generation + 1,
                refined_on=[r.example for r in rollouts],
            )
            minted += 1
            pool.append(child)
            kept = pareto_frontier(pool, objectives)
            dropped = len(pool) - len(kept)
            pool = kept
            best_so_far = max(best_so_far, max(headline(c) for c in pool))
        else:
            dropped = 0

        history.append(
            Step(iteration, parent.index, before, after, accepted, best_so_far)
        )
        print(
            f"  iter {iteration}: parent #{parent.index} (gen {parent.generation}) "
            f"minibatch {before:.8g} -> {after:.8g} "
            f"{'ACCEPTED' if accepted else 'rejected'}"
            + (f", pruned {dropped}" if dropped else "")
            + f", best {best_so_far:.8g}"
        )

    # --- what "best" means depends on the mode ------------------------------
    per_task: list[tuple[str, Candidate[A], float]] = []
    if mode == "generalization":
        # Search used the train set; the answer is whatever generalizes. Only the
        # frontier is re-evaluated on val, since evaluation costs model calls.
        frontier = pareto_frontier(pool, objectives)
        val_scores = {
            c.index: statistics.fmean(
                evaluate(c.artifact, e).score for e in typing.cast(list[E], valset)
            )
            for c in frontier
        }
        best = max(frontier, key=lambda c: val_scores[c.index])
        best_score = val_scores[best.index]
        seed_score = statistics.fmean(
            evaluate(root.artifact, e).score for e in typing.cast(list[E], valset)
        )
    elif mode == "multi-task":
        # N specialized artifacts: each task picks its own best off the shared
        # frontier, which is exactly where cross-transfer shows up. The run's score is
        # therefore the mean over those per-task winners, not any single candidate's
        # average -- the paper's "each task independently selects its own best
        # candidate from the frontier". Scoring one artifact across all tasks would
        # understate multi-task mode by construction, since specializing is the point.
        for e in typing.cast(list[E], dataset):
            winner = max(pool, key=lambda c: c.scores[e.name])
            per_task.append((e.name, winner, winner.scores[e.name]))
        best = max(pool, key=lambda c: c.average)
        best_score = statistics.fmean(score for _, _, score in per_task)
        seed_score = root.average
    else:
        best = max(pool, key=headline)
        best_score = headline(best)
        seed_score = headline(root)

    return Result(
        mode=mode,
        pool=pool,
        history=history,
        objectives=objectives,
        seed_score=seed_score,
        best=best,
        best_score=best_score,
        per_task=per_task,
        evaluations=counters["evaluations"],
        proposals=counters["proposals"],
    )


# ---------------------------------------------------------------------------
# The proposer: the one agent in the file. Its templates differ only in the artifact
# type they return -- a Callable for code, a str for a prompt -- which is what lets
# one loop optimize both without a serialization layer.
# ---------------------------------------------------------------------------


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

    @Template.define
    def propose_prompt(self, current: str, feedback: list[Rollout]) -> str:
        """You are optimizing the SYSTEM PROMPT given to a small model that writes
        sentences under hard constraints -- an exact word count, a required initial
        letter for every word, and a letter that must not appear. The prompt below is
        the artifact; return an improved one.

        <current_prompt>
        {current}
        </current_prompt>

        Here is how it did on a few instances, with the model's own reasoning, the
        sentence it produced, and a per-constraint account of what went wrong:

        <feedback>
        {feedback}
        </feedback>

        Diagnose before you rewrite. The constraints are always stated in the task
        itself, so the prompt's job is not to repeat them but to supply a *method*
        that makes them stick: how to construct the sentence so the count is right by
        construction, how to check each constraint separately rather than trusting a
        glance, what to do on finding a violation, and which failure the feedback
        shows is currently costing the most. Encode that as durable, general
        instructions -- the prompt is scored on instances you have not seen, with
        different topics, counts, letters and banned letters, so never mention a
        specific instance, and never write a sentence yourself.

        Return the improved prompt as plain text, nothing else.
        """

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
        it would not make them again. Prefer transferable engineering discipline
        (which degenerate inputs to handle explicitly, what to check before dividing,
        how to treat boundaries) over anything specific to one task -- an instruction
        that solves one task by naming its answer is worthless on the others.

        Return the improved instruction as plain text, nothing else.
        """


# ---------------------------------------------------------------------------
# Domain 1: circle packing -- single-task search over a code artifact (paper 5.3).
#
# The evaluator is deterministic Python, so this domain shows the loop with no model
# in the scoring path at all: every number in the trace is measured.
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
# Domain 2: prompt optimization -- generalization mode (paper A.3).
#
# The artifact is a string, the evaluator is itself a model call, and the objective is
# to do well on instances the search never saw.
#
# The task is constrained writing, scored by deterministic Python (the shape of
# ``constrained_paragraph.py``, with a checker attached). That choice is forced: the
# paper optimizes prompts for AIME, and a 2026-class model -- including the cheap ones
# -- answers any short arithmetic or number-theory question a self-contained example
# can carry, so the seed prompt saturates and the search has nothing to climb. Writing
# a sentence to an exact word count, an initial-letter rule and a banned letter is a
# constraint-tracking task these models genuinely fail, the checker is exact, and the
# lever a better prompt provides is real method -- count before answering, verify each
# constraint separately, revise once -- rather than a formatting convention.
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class Writing:
    """One constrained-writing instance. The constraints are in the question, so
    nothing is hidden from the answering model: what a better prompt supplies is the
    *method* for satisfying them, which is exactly what the paper's optimized prompts
    encode."""

    name: str
    topic: str
    words: int
    initial: str
    banned: str

    @property
    def question(self) -> str:
        return (
            f"Write a single sentence about {self.topic}. It must contain exactly "
            f"{self.words} words, every word must begin with the letter "
            f"'{self.initial}', and the letter '{self.banned}' must not appear "
            f"anywhere in the sentence."
        )


WRITINGS: list[Writing] = [
    Writing("sailing", "sailing", 6, "s", "e"),
    Writing("markets", "morning markets", 7, "m", "a"),
    Writing("cats", "curious cats", 5, "c", "i"),
    Writing("planets", "distant planets", 8, "p", "o"),
    Writing("bridges", "old bridges", 6, "b", "u"),
    Writing("trains", "night trains", 7, "t", "e"),
    Writing("gardens", "walled gardens", 5, "g", "s"),
    Writing("harbours", "quiet harbours", 6, "h", "i"),
    Writing("lanterns", "paper lanterns", 7, "l", "o"),
    Writing("rivers", "wide rivers", 5, "r", "a"),
]

TRAIN = [
    w
    for w in WRITINGS
    if w.name in {"sailing", "markets", "cats", "planets", "bridges"}
]
VAL = [w for w in WRITINGS if w not in TRAIN]

SEED_PROMPT = "Answer the question."


@pydantic.dataclasses.dataclass(frozen=True)
class Answer:
    """What the answering model returns: its reasoning and its final answer."""

    reasoning: str
    final: str


@Template.define
def answer_question(instructions: str, question: str) -> Answer:
    """{instructions}

    <question>
    {question}
    </question>
    """


def words_of(sentence: str) -> list[str]:
    """The sentence's words, stripped of punctuation and lowercased.

    >>> words_of("Silent ships sail; south, softly.")
    ['silent', 'ships', 'sail', 'south', 'softly']
    """
    cleaned = "".join(
        c if c.isalpha() or c.isspace() or c == "'" else " " for c in sentence
    )
    return [w for w in cleaned.lower().split() if w]


def score_writing(sentence: str, task: Writing) -> tuple[float, list[Diagnostic]]:
    """Check the three constraints and explain every miss.

    Partial credit on purpose: a 0/1 verdict would make most of the search invisible,
    while per-constraint credit is a gradient the proposer can climb -- and the
    per-constraint breakdown *is* the side information.

    A sentence that satisfies all three constraints of the first instance (six words,
    every word starting with 's', no letter 'e' anywhere) scores 1.0:

    >>> score, _ = score_writing("Ships sail south, ships spin swiftly.", WRITINGS[0])
    >>> score
    1.0

    while the near-miss "Silent ships sail south, softly singing." -- same six words,
    same initial, but 'silent' smuggles in an 'e' -- loses exactly one third:

    >>> score, _ = score_writing("Silent ships sail south, softly singing.", WRITINGS[0])
    >>> round(score, 4)
    0.6667
    """
    words = words_of(sentence)
    count_ok = len(words) == task.words
    starting = [w for w in words if w.startswith(task.initial)]
    initial_ratio = len(starting) / len(words) if words else 0.0
    banned_hits = sentence.lower().count(task.banned)

    diagnostics = [
        Diagnostic("sentence", repr(sentence)),
        Diagnostic(
            "word count",
            f"{len(words)} words {words}, needed exactly {task.words}"
            if not count_ok
            else f"exactly {task.words} words, as required",
        ),
        Diagnostic(
            "initial letter",
            f"{len(starting)}/{len(words)} words begin with '{task.initial}'"
            + (
                ""
                if initial_ratio == 1.0
                else f"; offending words: {[w for w in words if not w.startswith(task.initial)]}"
            ),
        ),
        Diagnostic(
            "banned letter",
            f"the letter '{task.banned}' appears {banned_hits} time(s) and must not appear"
            if banned_hits
            else f"the letter '{task.banned}' does not appear, as required",
        ),
    ]
    score = (
        (1.0 if count_ok else 0.0) + initial_ratio + (1.0 if banned_hits == 0 else 0.0)
    ) / 3.0
    return score, diagnostics


def evaluate_prompt(prompt: str, task: Writing | None, model: str) -> Evaluation:
    """Run one writing instance under the candidate prompt and score it.

    The Side Information follows the paper's design for this domain: the instance, the
    model's reasoning, what it produced, and a per-constraint account of what went
    wrong -- not merely that something did.
    """
    assert task is not None, "the prompt domain always has a dataset"
    try:
        with worker(model):
            produced = answer_question(prompt, task.question)
    except Exception:
        return Evaluation(
            score=0.0,
            diagnostics=[
                Diagnostic("task", task.question),
                Diagnostic("crash", traceback.format_exc(limit=2).strip()),
            ],
        )
    score, diagnostics = score_writing(produced.final, task)
    return Evaluation(
        score=score,
        diagnostics=[
            Diagnostic("task", task.question),
            Diagnostic("reasoning", produced.reasoning),
            *diagnostics,
            Diagnostic(
                "verdict",
                f"scored {score:.2f} of 1.00 -- one third for the exact word count, one "
                f"third for the fraction of words with the right initial, one third for "
                f"avoiding the banned letter",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Domain 3: kernel instructions -- multi-task search (paper 5.2).
#
# The artifact is the instruction that drives code generation, evaluated on several
# related tasks at once. Each task selects its own best instruction off one shared
# frontier, which is where the paper's cross-task transfer lives.
# ---------------------------------------------------------------------------


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
    is timed on exactly the same work."""
    rng = random.Random(len(task))
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

        Standard library only. It is checked against hidden cases, including
        degenerate inputs, and then TIMED on a large input: a correct implementation
        scores the ratio of a straightforward reference implementation's time to
        yours, and an incorrect one scores zero however fast it is. Write it to be
        both right and fast.
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
    correct list transform on the first try, but "correct" is the floor here, not the
    goal.

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
    except Exception as exc:
        return 0.0, Diagnostic(
            "speed",
            f"on {size} values this raised {type(exc).__name__}: {exc}. Rescanning "
            f"earlier values for every position is quadratic; carry running state.",
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
    against the PyTorch baseline), and it is what gives this domain something to climb
    -- correctness is the floor, not the target. The Side Information is the failing
    cases with expected and actual values, the measured times and speedup, the
    traceback if it crashed, and the code itself.
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
    failures: list[Diagnostic] = []
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
        elif len(failures) < 2:  # the first couple of failures are the useful ones
            failures.append(
                Diagnostic(
                    "failing case",
                    f"kernel({values}) returned {produced}, expected {expected}",
                )
            )
    elapsed = time.perf_counter() - start

    speedup, timing = measure_speedup(kernel, task.name)
    correct = passed == len(cases) and speedup > 0.0

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
        metrics=[Metric("cases_passed", float(passed))],
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Wiring the domains to the one loop. This is the paper's declarative API claim: the
# only thing that differs between "optimize a packing algorithm" and "optimize a
# prompt" is which seed, evaluator and proposer you hand over.
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


def run_prompt(args: argparse.Namespace, rng: random.Random) -> Result:
    return optimize_anything(
        evaluator=lambda prompt, q: evaluate_prompt(prompt, q, args.worker_model),
        proposer=lambda prompt, feedback: Proposer().propose_prompt(prompt, feedback),
        seed=SEED_PROMPT,
        dataset=TRAIN,
        valset=VAL,
        budget=args.budget,
        minibatch_size=args.minibatch,
        selection=args.selection,
        use_side_info=not args.no_side_info,
        rng=rng,
    )


def run_kernel(args: argparse.Namespace, rng: random.Random) -> Result:
    """Multi-task by default. ``--single-task`` runs the paper's control instead: each
    task optimized independently with the same per-task budget, which is the
    comparison its 5.4 ablation reports."""
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

    per_task: list[tuple[str, Candidate, float]] = []
    seeds: list[float] = []
    merged: Result | None = None
    per_problem = max(1, args.budget // len(KERNEL_TASKS))
    for task in KERNEL_TASKS:
        print(f"\n[single-task control] {task.name} ({per_problem} iterations)")
        result = optimize_anything(
            evaluator=lambda i, t: evaluate_instruction(i, t, args.worker_model),
            proposer=proposer,
            seed=SEED_INSTRUCTION,
            dataset=[task],
            budget=per_problem,
            minibatch_size=1,
            selection=args.selection,
            use_side_info=not args.no_side_info,
            rng=rng,
        )
        per_task.append((task.name, result.best, result.best_score))
        seeds.append(result.seed_score)
        merged = result
    assert merged is not None
    merged.per_task = per_task
    merged.mode = "multi-task (single-task control)"
    merged.best_score = statistics.fmean(s for _, _, s in per_task)
    merged.seed_score = statistics.fmean(seeds)
    return merged


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(result: Result, args: argparse.Namespace) -> None:
    print("\n" + "=" * 72)
    print(
        f"mode: {result.mode} | selection: {args.selection} | "
        f"side information: {'off' if args.no_side_info else 'on'}"
    )
    print(
        f"seed {result.seed_score:.6g} -> best {result.best_score:.6g} "
        f"after {len(result.history)} iterations "
        f"({result.proposals} proposals, {result.evaluations} evaluations)"
    )
    accepted = sum(step.accepted for step in result.history)
    print(f"accepted proposals: {accepted}/{len(result.history)}")

    frontier = result.frontier()
    print(f"\nPareto frontier ({len(frontier)} candidate(s) survive):")
    for c in sorted(frontier, key=lambda c: -c.average):
        scores = ", ".join(f"{k}={v:.4g}" for k, v in sorted(c.scores.items()))
        print(f"  #{c.index} (gen {c.generation}, parent {c.parent}): {scores}")

    if result.per_task:
        print("\nPer-task winners (multi-task selects off the shared frontier):")
        for name, candidate, score in result.per_task:
            print(f"  {name}: candidate #{candidate.index} scored {score:.4g}")
        transferred = [
            name
            for name, candidate, _ in result.per_task
            if candidate.refined_on and name not in candidate.refined_on
        ]
        print(
            f"\nCross-task transfer: {len(transferred)}/{len(result.per_task)} winning "
            f"artifacts were last refined while looking at a *different* task"
            + (f" ({', '.join(transferred)})" if transferred else "")
        )

    if any(isinstance(c.artifact, PackSystem) for c in result.pool):
        # Multi-module search: which module each surviving candidate came from is the
        # paper's leapfrogging, and the only way to see it is to count.
        origins = collections.Counter(
            c.artifact.origin for c in result.pool if isinstance(c.artifact, PackSystem)
        )
        winner = result.best.artifact
        print(
            "\nModules on the shared frontier: "
            + ", ".join(f"{k} {v}" for k, v in sorted(origins.items()))
            + f"; the best candidate came from the "
            f"{winner.origin if isinstance(winner, PackSystem) else 'unknown'} module"
        )

    print("\nBest artifact:")
    artifact = result.best.artifact
    if isinstance(artifact, PackSystem):
        print(f"--- refiner instruction ---\n{artifact.refiner}\n")
        print("--- packer ---")
        artifact = artifact.packer
    if callable(artifact):
        print((source_of(artifact) or repr(artifact)).rstrip())
    else:
        print(artifact)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=["pack", "prompt", "kernel"],
        default="pack",
        help="pack: single-task code artifact; prompt: generalization; kernel: multi-task",
    )
    parser.add_argument("--budget", type=int, default=8, help="Optimizer iterations")
    parser.add_argument(
        "--minibatch", type=int, default=2, help="Examples per reflection step"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for selection/minibatches"
    )
    parser.add_argument(
        "--num-circles", type=int, default=10, help="Circles to pack (paper uses 26)"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=2.0,
        help="Seconds a synthesized packer is given per call",
    )
    parser.add_argument(
        "--code-share",
        type=float,
        default=0.5,
        help="Pack domain: fraction of iterations that mutate the code module rather "
        "than the refiner module (both live on the one shared frontier)",
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
        help="Send a rendered image of the packing as side information (pack domain)",
    )
    parser.add_argument(
        "--seedless",
        action="store_true",
        help="Bootstrap candidate zero from a natural-language objective (pack domain)",
    )
    parser.add_argument(
        "--worker-model",
        default=WORKER_MODEL,
        help="Model the artifact is optimized FOR in the prompt/kernel domains; the "
        "harness's --model is the proposer, as in the paper's proposer/worker split",
    )
    parser.add_argument(
        "--single-task",
        action="store_true",
        help="Kernel domain: run the single-task control instead of multi-task search",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    runners = {"pack": run_pack, "prompt": run_prompt, "kernel": run_kernel}
    result = runners[args.domain](args, rng)
    report(result, args)

    assert result.best_score >= result.seed_score, (
        f"optimization went backwards: {result.seed_score} -> {result.best_score}"
    )


if __name__ == "__main__":
    main()
