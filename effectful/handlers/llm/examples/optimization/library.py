"""The optimize_anything engine: reflective Pareto search over a typed artifact.

Shared machinery for the three runnable examples in this directory, which implement
"optimize_anything: Unified Text Optimization can Outperform Specialized Systems"
(OpenReview 7M28lVzVUq). The paper's observation is that an enormous range of problems
-- a CUDA kernel, a packing algorithm, a scheduling policy, an agent architecture, a
system prompt -- are all *the same problem*: improve an artifact that some evaluator
scores. Its system is GEPA's reflective Pareto search (Agrawal et al., ICLR 2026)
lifted off prompts onto arbitrary artifacts.

Its Algorithm 1 is short enough to quote, and `optimize_anything` below is it:

    P <- [seed];  evaluate the seed; record per-objective scores
    while budget remains:
        k <- ParetoSelect(P)          # sample in proportion to frontier frequency
        M <- a minibatch of 2-3 examples
        run candidate k on M, collecting scores *and* side information
        k' <- Reflect(k, scores, SI)  # the LLM proposes a revision
        if k' improves on M: evaluate it fully, admit it, prune dominated candidates
    return the best candidate

Everything except ``Reflect`` is ordinary Python, and that is the point of the split
between this module and its three siblings: what lives here is the whole algorithm, and
what lives in `packing.py`, `prompting.py` and `kernels.py` is only ever an artifact
type, an evaluator, and a prompt. Each of the paper's mechanisms falls out of an
effectful idiom rather than a subsystem:

  * **Side Information is just the evaluator's typed return value.** The paper needs a
    ``side_info`` dict and a serializer to carry stack traces, sub-scores and rendered
    images to the proposer. Here an evaluator returns an `Evaluation` (score, sub-score
    `Metric`s, `Diagnostic`s) and the proposer's prompt splices it with ``{feedback}``
    -- the Encodable bridge already knows how to put a typed value in the model's
    context. Because SI is *any* Encodable, a rendered ``PIL.Image`` of the current
    packing is SI too, through the same path ``image_input.py`` uses. The paper's SI
    ablation is a flag in each script, and it changes one line: which view of the
    `Evaluation` the proposer is shown.

  * **The paper's "refiner" is `TenacityRetryer` plus decode-time certification.** It
    reports needing a dedicated step for "malformed code blocks, import errors, syntax
    issues ... essential for code and agent artifacts where minor formatting errors
    cause complete evaluation failure". A code artifact here is a ``Skill``
    returning a ``Callable``: the model's source is parsed, type-checked against the
    requested signature, compiled, and its own doctests are run at decode time, so a
    malformed candidate is fed its own error and revised before it ever reaches the
    evaluator.

  * **"Serialize the artifact as a string" is the step you get to skip.** This loop is
    generic in the artifact type ``A``. `packing.py` optimizes a callable (in fact a
    pair of modules), `prompting.py` and `kernels.py` optimize strings, and nothing in
    between ever sees a serialized artifact or needs an adapter.

  * **The three modes are a function signature.** `optimize_anything` takes ``dataset``
    and ``valset``; neither means single-task (the artifact *is* the solution, and the
    Pareto objectives are its sub-scores), ``dataset`` alone means multi-task (one
    shared frontier, one specialized artifact per task), both means generalization
    (search on train, select on held-out val). Pareto selection, minibatching, the
    accept-if-improves rule, dominated-candidate pruning and the content-addressed
    evaluation cache are plain Python, and stay plain Python.

The three scripts, each runnable on its own and each documenting what it does and does
not reproduce of its section of the paper:

  * `packing.py`   -- circle packing, single-task search over a code artifact (5.3)
  * `prompting.py` -- prompt optimization, generalization mode (A.3)
  * `kernels.py`   -- kernel instructions, multi-task search (5.2)

A fourth script implements the *successor* claim. "AVO: Agentic Variation Operators"
(arXiv 2603.24517) argues that the single-turn proposer above is itself the bottleneck:
confined to `Reflect`, the model cannot test a candidate, read its traceback, or revise
before committing. AVO replaces the whole variation operator with an autonomous agent
that holds the evaluator as a tool. `evolve_lineage` below is the small amount of
framework that survives that change, and:

  * `avo.py`       -- agentic variation on a kernel, against this engine as the control
"""

import collections.abc
import contextlib
import dataclasses
import inspect
import linecache
import random
import statistics
import typing

import pydantic.dataclasses

from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
from effectful.ops.semantics import handler

WORKER_MODEL = "openai/gpt-4.1-mini"


@contextlib.contextmanager
def worker(model: str):
    """Scope a call to the cheap model the artifact is being optimized *for*."""
    with handler(LiteLLMConfigurer(model=model)):
        yield


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
# which is the domain's Skill call: everything else -- selection, minibatching,
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
    its documented OSError/TypeError: a synthesized function whose block ends inside a
    multi-line string raises ``tokenize.TokenError``, which is reachable often enough to
    end a run. When block extraction fails the whole synthesized module is still sitting
    in ``linecache``, and for both of this function's jobs -- keying the cache and
    showing the user what the search wrote -- the module is as good as the block.
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
    text.

    Two candidates with the same key have the same evaluation *provided the evaluator
    is a function of the artifact and the example and nothing else*. That proviso is
    the whole of the cache's soundness, and one domain breaks it: `packing.py` hands
    each artifact the best packing found so far, so its score depends on when it ran.
    Such a domain must declare a ``state_key`` (see `optimize_anything`), which joins
    the key; a content address alone would serve a stale score from before the
    incumbent moved.
    """
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
    """Always mutate the best average, collapsing the frontier's complementary
    strengths into one number.

    This is the naive alternative the paper's 4.3 argues against -- "averaging hides
    which aspects are strong and which are weak" -- not an ablation it runs. The
    scripts expose it as ``--selection best`` so the argument can be checked here, and
    none of them has yet spent the budget to check it.
    """
    return max(pool, key=lambda c: c.average)


@dataclasses.dataclass
class Step:
    """One iteration of the loop, kept for the printed trace."""

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
# The agentic variation operator: single-lineage evolution.
#
# "AVO: Agentic Variation Operators for Autonomous Evolutionary Search" (arXiv
# 2603.24517) makes one change to everything above, and it is a change of *scope*
# rather than of machinery. Where `optimize_anything` decomposes variation as
# ``Vary(P) = Generate(Sample(P))`` and confines the model to a single-turn
# ``Generate``, AVO replaces the whole operator with one autonomous agent run,
# ``Vary(P) = Agent(P, K, f)``: the scoring function ``f`` is a tool the agent may
# call, so it can edit, evaluate, read the diagnostics, and revise before committing
# anything.
#
# Almost none of that needs code here, which is the point. The multi-turn loop the
# paper builds is what a `Skill` call already is, and handing over ``f`` is a `Tool`
# in the skill's lexical scope. What *is* left for a framework is small enough to fit
# below: hold the lineage, verify what the agent returns, and commit it if it earns a
# place. `avo.py` supplies the agent and the domain.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Version[A]:
    """One committed artifact in a lineage: the paper's git commit, in memory.

    Only *committed* versions are recorded. Everything the agent tried and discarded
    on the way stays inside its own variation step -- the paper's "500 optimization
    directions" behind 40 versions -- which is why a lineage's length is a count of
    successes rather than of work done.
    """

    index: int
    artifact: A
    evaluation: Evaluation
    note: str = ""

    @property
    def score(self) -> float:
        return self.evaluation.score


def commits(candidate: Evaluation, incumbent: Evaluation) -> bool:
    """The paper's commit rule: "passes correctness checks and matches or improves
    the benchmark score relative to the best committed version so far".

    Two clauses, and the first is doing real work. ``score > 0`` is the correctness
    gate -- an evaluator in this directory scores a wrong artifact zero however fast
    it is -- and without it the "matches" half of the second clause would happily
    commit a broken candidate on top of a broken incumbent, forever.

    "Matches" is kept rather than tightened to a strict improvement, because it is
    what the paper says and because a lateral move is how a search leaves a plateau.
    `evolve_lineage` separately refuses a candidate that *is* the incumbent, which is
    the only way matching could be a pure no-op.

    >>> commits(Evaluation(score=1.2), Evaluation(score=1.0))
    True
    >>> commits(Evaluation(score=1.0), Evaluation(score=1.0))
    True
    >>> commits(Evaluation(score=0.9), Evaluation(score=1.0))
    False
    >>> commits(Evaluation(score=0.0), Evaluation(score=0.0))
    False
    """
    return candidate.score > 0.0 and candidate.score >= incumbent.score


@dataclasses.dataclass
class Lineage[A]:
    """The outcome of a single-lineage run: what was committed, and what it cost.

    ``evaluations`` counts only this loop's own verification calls. The agent's calls
    to ``f`` during its variation steps are the domain's to count (see `avo.py`), and
    they are the larger number by far -- which is precisely the currency on which an
    agentic operator is expensive and a single-turn one is cheap, so a comparison that
    quotes one arm's proposals against the other's evaluations is not a comparison.
    """

    versions: list[Version[A]]
    attempts: int = 0
    rejected: int = 0
    evaluations: int = 0

    @property
    def seed(self) -> Version[A]:
        return self.versions[0]

    @property
    def best(self) -> Version[A]:
        # A commit never lowers the score, so this is the last version; taken as a
        # maximum anyway, so the invariant is enforced rather than assumed.
        return max(self.versions, key=lambda v: v.score)


def evolve_lineage[A](
    *,
    vary: collections.abc.Callable[[A, Evaluation], A],
    evaluator: collections.abc.Callable[[A], Evaluation],
    seed: A,
    budget: int,
) -> Lineage[A]:
    """Continuous single-lineage evolution: the paper's outer loop, which is all of
    the framework that survives once variation becomes an agent.

    Each step hands the incumbent and its evaluation to ``vary`` -- one autonomous
    agent run -- and re-scores whatever comes back. The re-scoring is not
    redundant. The agent has already called ``f`` itself, probably several times, but
    the number that goes in the lineage has to be the framework's own measurement of
    the artifact it was actually given: an agent that reports a score it did not
    achieve, or that returns a different kernel than the one it tested, is caught here
    rather than being taken at its word. It costs one evaluation per step.

    ``vary`` raising is an ordinary outcome, not a crash: the decode-time gate rejects
    malformed artifacts after its retries are exhausted, and the step is simply lost.
    """
    counters = {"evaluations": 0}

    def evaluate(artifact: A) -> Evaluation:
        counters["evaluations"] += 1
        return evaluator(artifact)

    lineage = Lineage(versions=[Version(0, seed, evaluate(seed), note="seed")])
    print(f"[avo] seed scores {lineage.seed.score:.6g}")

    for step in range(1, budget + 1):
        incumbent = lineage.best
        lineage.attempts += 1
        try:
            child = vary(incumbent.artifact, incumbent.evaluation)
        except Exception as exc:
            lineage.rejected += 1
            reason = " ".join(str(exc).split())[:200]
            print(
                f"  step {step}: no candidate -- the variation step failed at decode "
                f"({type(exc).__name__}: {reason})"
            )
            continue

        if artifact_key(child) == artifact_key(incumbent.artifact):
            lineage.rejected += 1
            print(f"  step {step}: rejected -- returned the incumbent unchanged")
            continue

        evaluation = evaluate(child)
        if commits(evaluation, incumbent.evaluation):
            lineage.versions.append(
                Version(len(lineage.versions), child, evaluation, note=f"step {step}")
            )
            print(
                f"  step {step}: COMMITTED v{lineage.versions[-1].index} "
                f"{incumbent.score:.6g} -> {evaluation.score:.6g}"
            )
        else:
            lineage.rejected += 1
            print(
                f"  step {step}: rejected {evaluation.score:.6g} "
                f"against the incumbent's {incumbent.score:.6g}"
            )

    lineage.evaluations = counters["evaluations"]
    return lineage


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def transfer_note(result: "Result") -> str:
    """How many per-task winners were last refined while the proposer was looking at a
    different task -- stated against the count chance alone would produce.

    Two things make the raw count meaningless on its own. A winner counts as
    "transferred" unless its own task was in the minibatch that produced it, so under a
    null where any candidate is equally likely to win any task the expected count is
    already ``tasks * (1 - minibatch/tasks)`` -- 3.0 on five tasks with a two-task
    minibatch, which is most of the range the statistic can take. And a task whose
    winner is the *seed* was never refined at all: it has no minibatch, so it cannot
    have transferred, and counting it as one turns a run with no transfer whatsoever
    into "4/5, above chance". Both the count and the null are therefore taken over the
    refined winners only, each contributing its own minibatch size rather than a pooled
    mean, and seed winners are reported separately.

    It remains one run, and it looks only at the last refinement rather than at full
    ancestry.
    """
    winners = result.per_task
    seeds = [name for name, candidate, _ in winners if not candidate.refined_on]
    refined = [(name, c) for name, c, _ in winners if c.refined_on]
    transferred = [name for name, c in refined if name not in c.refined_on]
    # Per candidate, since minibatches can differ in size: the chance of *not* drawing
    # this task into the minibatch that produced its winner.
    expected = sum(1.0 - len(c.refined_on) / len(winners) for _, c in refined)
    if not refined:
        return (
            f"\nCross-task transfer: not measurable -- all {len(winners)} winners are "
            f"the seed, which was never refined on any task"
        )
    verdict = (
        "above chance"
        if len(transferred) > expected
        else "at or below chance, so this run is no evidence of transfer"
    )
    return (
        f"\nCross-task transfer: {len(transferred)}/{len(refined)} *refined* winners "
        f"were last refined while looking at a different task"
        + (f" ({', '.join(transferred)})" if transferred else "")
        + f"; chance alone would give {expected:.1f} -- {verdict}"
        + (
            f". The other {len(seeds)} winner(s) ({', '.join(seeds)}) are the "
            f"unrefined seed and are excluded from both figures"
            if seeds
            else ""
        )
    )


def report(
    result: "Result",
    *,
    selection: str,
    side_info: bool,
    notes: collections.abc.Sequence[str] = (),
    render_artifact: collections.abc.Callable[[typing.Any], str] | None = None,
) -> None:
    """Print what a run did: the trace's summary, the surviving frontier, per-task
    winners where the mode has them, and the winning artifact.

    ``notes`` and ``render_artifact`` are how a domain adds its own reading without the
    library having to know about it -- `packing.py` uses them to count which of its two
    modules the frontier came from, and to print a two-part artifact.
    """
    print("\n" + "=" * 72)
    print(
        f"mode: {result.mode} | selection: {selection} | "
        f"side information: {'on' if side_info else 'off'}"
    )
    print(
        f"seed {result.seed_score:.6g} -> best {result.best_score:.6g} "
        f"after {len(result.history)} iterations "
        f"({result.proposals} proposals, {result.evaluations} evaluations)"
    )
    accepted = sum(step.accepted for step in result.history)
    print(f"accepted proposals: {accepted}/{len(result.history)}")
    # The multi-task headline is a per-task maximum over the pool while the seed is a
    # single candidate, so it cannot be negative and is biased upward by the size of the
    # pool. The best single artifact's mean is the matched comparison: one artifact
    # against one artifact, on the same tasks. It is only meaningful when every
    # candidate was scored on the same objectives, which is false for an arm that
    # aggregates independent runs.
    comparable = result.pool and all(
        set(c.scores) == set(result.objectives) for c in result.pool
    )
    if result.per_task and comparable:
        best_single = max(c.average for c in result.pool)
        print(
            f"best single artifact (matched comparison, since the headline above is a "
            f"per-task maximum over {len(result.pool)} candidates): {best_single:.6g}"
        )

    if result.pool:
        frontier = result.frontier()
        print(f"\nPareto frontier ({len(frontier)} candidate(s) survive):")
        for c in sorted(frontier, key=lambda c: -c.average):
            scores = ", ".join(f"{k}={v:.4g}" for k, v in sorted(c.scores.items()))
            print(f"  #{c.index} (gen {c.generation}, parent {c.parent}): {scores}")
    else:
        print("\n(no shared frontier: this run is several independent searches)")

    if result.per_task:
        print("\nPer-task winners:")
        for name, candidate, score in result.per_task:
            print(f"  {name}: candidate #{candidate.index} scored {score:.4g}")
        # Cross-task transfer is only a question where there was one frontier to
        # transfer across; for independent runs it is zero by construction, and saying
        # so as though it were a measurement would be worse than not saying it.
        if result.pool:
            print(transfer_note(result))

    for note in notes:
        print(f"\n{note}")

    print("\nBest artifact:")
    artifact = result.best.artifact
    if render_artifact is not None:
        print(render_artifact(artifact))
    elif callable(artifact):
        print((source_of(artifact) or repr(artifact)).rstrip())
    else:
        print(artifact)
