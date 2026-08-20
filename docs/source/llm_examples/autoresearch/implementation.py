"""MARS: budget-aware modular ML engineering as a cost-constrained tree search.

Implements the core of "MARS: Modular Agent with Reflective Search for Automated
AI Research" (arXiv:2602.02660). The paper's diagnosis is that LLM coding agents
for machine-learning engineering "generate monolithic scripts that ignore
execution costs and causal factors": they write one big program in a vacuum, never
budgeting the expensive model-evaluation step and never learning *why* one attempt
beat another. Its fix rests on three pillars: (1) *budget-aware planning* via a
cost-constrained MCTS that balances solution quality against compute expense; (2)
*modular construction* via a Design -> Decompose -> Implement pipeline that breaks a
repository into modules instead of one script; and (3) *comparative reflective
memory* that compares solution paths to solve credit assignment and transfer
lessons across branches. Each pillar falls out of ordinary effectful idioms:

  * The evaluator is the ground truth, and we do not fake it. Like ``formalization``
    with the Lean compiler, MARS's load-bearing part is the *expensive model
    evaluation* the paper says agents ignore, so we make it real: a synthesized
    pipeline is scored by a deterministic, re-runnable Python ``evaluate`` that
    returns both a quality metric (tour length) and the *measured execution cost*
    of running it. The MCTS reward and the budget are therefore real numbers, not
    simulated ones -- the same "ground truth a claim certifies against" role that
    ``investigation``'s evaluator plays for its Solution.

  * Decompose *is* the search's action space. The Designer emits a typed ``Design``
    -- an ordered list of ``ModuleSpec``s, each carrying a few candidate
    implementation *strategies* -- and those strategies are exactly the branching
    actions of the MCTS tree. Coordination lives in a structured value threaded
    between agents (the Outline idiom of ``writing`` / the StyledPlot idiom of
    ``illustration``), and the tree is a handful of ordinary calls playing from it.

  * Implement is code synthesis with a decode-time contract. Each module is a real
    ``Stage`` callable the harness compiles from the Implementer's code (the
    ``Callable``-return idiom of ``investigation``'s Solver). Its docstring carries a
    doctest that certifies "the module returns a valid tour (a permutation)" at
    decode time -- run on a *different, smaller* instance than the evaluation one, so
    a stage that hardcodes the problem size fails the doctest and is fed back by
    ``TenacityRetryer`` (the render-doctest grounding of ``illustration`` /
    ``countdown``). "The module must produce a valid tour" is grounding by
    construction; the Test Executor / Error Analyzer of the paper are this evaluator
    plus the harness's retry feedback.

  * Budget-aware MCTS is a plain-Python loop. Selection uses the paper's
    cost-aware UCT -- exploitation + exploration - alpha * (cost / budget) -- so the
    search favors high-reward, low-cost branches; a rollout budget caps the search,
    which is the point (enumerating every design x implementation would be far too
    expensive, so the agent must *plan* which to try). Reward and cost backpropagate
    up the path exactly as in textbook MCTS.

  * Comparative reflective memory is an LLM comparison feeding a refine loop. Every
    few rollouts the Reflector contrasts a high-reward branch with a low-reward one
    and emits a transferable ``Reflection`` (the paper's credit assignment); lessons
    accumulate in a ``Memory`` spliced into later Implement prompts, and adding a
    lesson invalidates the synthesis cache so subsequent branches re-implement with
    it -- cross-branch transfer made observable, an LLM-judge comparison (as in
    ``writing``'s reviewer) driving a re-implementation loop.

Demonstrates:
- A real, re-runnable evaluator as ground truth: synthesized module ``Callable``s
  are scored by deterministic Python (tour length) at a *measured* execution cost,
  so the MCTS reward and the search budget are real, not simulated
- Budget-aware MCTS in plain Python: the paper's cost-aware UCT (exploit + explore
  - alpha * cost/budget) plus a rollout budget, over a tree whose actions are the
  ``Design``'s per-module strategies
- A typed ``Design`` emitted by one agent that *is* the search's action space --
  Decompose-as-data threaded through Implement (the Outline idiom)
- Code synthesis with a decode-time contract: each module is a ``Callable`` whose
  doctest certifies it returns a valid permutation, fed back by ``TenacityRetryer``,
  and run on a different instance so it cannot hardcode the problem size
- Comparative reflective memory: an LLM comparison of a strong vs. weak branch
  emits a lesson spliced into later syntheses (invalidating the cache), so
  cross-branch transfer is observable
- Decode-time certification of the ``Design``'s shape (>= 2 strategies per module,
  unique names), and per-field guidance via ``field(metadata={"description": ...})``
"""

# Simplifications vs. the source:
# - One planted MLE-style task, run end to end, not MLE-Bench's Kaggle repositories.
#   The task is budget-constrained Euclidean TSP; the "repository" is a short
#   pipeline of composed ``Stage`` functions rather than a multi-file project, and
#   quality is tour length -- a stand-in for a real benchmark metric. This shows the
#   Design-Decompose-Implement *shape* and the cost/quality tradeoff, not ML at scale.
# - Cost is measured wall-clock execution time of the synthesized pipeline (min over
#   repeats), a real but machine- and noise-dependent proxy for MARS's "expensive
#   model evaluation"; the budget is a rollout cap plus this cost feeding UCT, not a
#   token-accounted API budget. Because the cost signal is real (hence noisy), which
#   design wins can vary run to run -- apt for a genuine cost, but not a fixed golden
#   output (contrast the deterministic-corpus examples).
# - The MCTS is small (a shallow tree of a few modules x a few strategies) and
#   rollouts complete by random strategy choice rather than a learned default policy;
#   there is no progressive widening. It demonstrates the cost-aware search shape.
# - Reflective memory compares the current best vs. worst successful branch every few
#   rollouts and splices lessons textually; there is no embedding store and no reward
#   re-weighting of tree nodes from lessons (a lesson acts only by re-implementation).
#   The paper's "63% of lessons come from cross-branch transfer" is reported here only
#   as a simple post-hoc count on one task, not reproduced as a statistic.
# - The Implementer writes pure Python over the given cities; there is no separate
#   refactor/debug sub-loop beyond the harness's synth + doctest + retry.
# - No ContextVar: unlike ``investigation``/``formalization``, nothing certifies
#   against per-run mutable state -- the module doctest checks a structural invariant
#   and the evaluator is handed its stages explicitly, so ground truth stays local.

import argparse
import collections.abc
import dataclasses
import inspect
import math
import random
import time

import pydantic

from effectful.handlers.llm import Agent, Skill

# A field's ``metadata={"description": ...}`` is inlined by pydantic into that
# field's JSON schema, which the harness renders into the system prompt as part of a
# skill's argument (and structured-output) spec. So per-field guidance reaches the
# model *through the type* -- used below only where the field name and type don't
# already say it, so no prompt has to repeat it.


# ---------------------------------------------------------------------------
# The task and its evaluator -- the ground truth every candidate is scored by. This
# is the load-bearing part we do not fake: a real, re-runnable Python evaluator that
# returns both a quality metric and the *measured* execution cost of running the
# synthesized pipeline (MARS's "expensive model evaluation", made real).
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class City:
    """A point in the plane the tour must visit."""

    x: float
    y: float


@pydantic.dataclasses.dataclass(frozen=True)
class Task:
    """One MLE-style engineering task: visit every city once and return, minimizing
    total Euclidean distance. A stand-in for a benchmark whose metric is expensive to
    evaluate and whose best solution trades quality against compute."""

    cities: tuple[City, ...]


# A tour is a permutation of city indices; the pipeline's job is to reorder it to
# shorten the round trip. A Stage is one module of the pipeline: it takes the cities
# and the current tour and returns an improved tour. Uniform typing makes the modules
# compose by a plain fold and keeps synthesis robust.
type Tour = list[int]
type Stage = collections.abc.Callable[[list[City], Tour], Tour]


def distance(a: City, b: City) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def tour_length(cities: collections.abc.Sequence[City], tour: Tour) -> float:
    """Total length of the closed tour that visits ``cities`` in ``tour`` order."""
    n = len(tour)
    return sum(distance(cities[tour[i]], cities[tour[(i + 1) % n]]) for i in range(n))


def _validate_tour(tour: Tour, n: int) -> None:
    """A stage's output must be a permutation of all ``n`` city indices, or it is not
    a valid tour -- the invariant the module doctest also enforces at decode time."""
    if sorted(tour) != list(range(n)):
        raise ValueError(
            f"a stage returned {tour}, which is not a valid tour: it must be a "
            f"permutation of every city index 0..{n - 1} exactly once"
        )


def run_pipeline(cities: collections.abc.Sequence[City], stages: list[Stage]) -> Tour:
    """Fold the identity tour through every stage, certifying each stage's output is a
    valid permutation. A stage that returns garbage raises -- the same
    certification-by-construction the doctest makes at decode time."""
    tour: Tour = list(range(len(cities)))
    for stage in stages:
        tour = list(stage(list(cities), tour))
        _validate_tour(tour, len(cities))
    return tour


# Repeat the pipeline a few times and take the minimum runtime: the standard robust
# estimator for a small computation's cost, damping OS/scheduler noise.
COST_REPEATS = 5


def evaluate(task: Task, stages: list[Stage]) -> tuple[float, float]:
    """Run the assembled pipeline and return ``(tour_length, cost_seconds)`` -- the
    real quality metric and the measured execution cost. Both feed the MCTS: length
    becomes the reward, cost enters cost-aware UCT and the budget. Raises (via
    ``run_pipeline``) if any stage produces an invalid tour."""
    cities = list(task.cities)
    best_cost = math.inf
    tour: Tour = list(range(len(cities)))
    for _ in range(COST_REPEATS):
        start = time.perf_counter()
        tour = run_pipeline(cities, stages)
        best_cost = min(best_cost, time.perf_counter() - start)
    return tour_length(cities, tour), best_cost


# ---------------------------------------------------------------------------
# Structured artifacts crossing between agents.
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class ModuleSpec:
    """One stage of the pipeline the Designer decomposes the task into: what it should
    accomplish, plus the candidate implementation *strategies* that become the MCTS
    branching actions for this stage."""

    name: str
    intent: str = dataclasses.field(
        metadata={
            "description": "What this stage does to the tour it receives (e.g. build "
            "an initial ordering, or locally improve the incoming tour). Every stage "
            "takes the cities and the current tour and returns a valid tour."
        }
    )
    strategies: list[str] = dataclasses.field(
        metadata={
            "description": "Two or three distinct, concrete implementation approaches "
            "for this stage (e.g. 'nearest-neighbour construction', '2-opt local "
            "search', 'or-opt segment moves'). Each becomes one search action."
        }
    )


@pydantic.dataclasses.dataclass(frozen=True)
class Design:
    """The Design/Decompose artifact: the pipeline's architecture as an ordered list
    of modules. This one structured value *is* the MCTS action space -- every
    downstream Implement call and every tree action reads it."""

    analysis: str = dataclasses.field(
        metadata={
            "description": "A short reading of the task: what makes a good tour and "
            "how the modules cooperate to produce one."
        }
    )
    modules: list[ModuleSpec]

    def __post_init__(self) -> None:
        if not self.modules:
            raise ValueError("a design must have at least one module (pipeline stage)")
        names = [m.name for m in self.modules]
        if len(set(names)) != len(names):
            raise ValueError(f"module names must be unique, got {names}")
        for m in self.modules:
            if len(set(m.strategies)) < 2:
                raise ValueError(
                    f"module {m.name!r} must offer at least two distinct strategies "
                    f"(the search needs branching actions), got {m.strategies}"
                )

    def __str__(self) -> str:
        lines = [f"analysis: {self.analysis}"]
        for i, m in enumerate(self.modules):
            lines.append(f"  module {i} [{m.name}]: {m.intent}")
            lines += [f"    - {s}" for s in m.strategies]
        return "\n".join(lines)


@pydantic.dataclasses.dataclass(frozen=True)
class PipelineChoice:
    """One decided stage of a full pipeline: which module, and the strategy chosen for
    it. A list of these is the path the MCTS committed to."""

    module: str
    strategy: str


@pydantic.dataclasses.dataclass(frozen=True)
class RolloutSummary:
    """A finished branch handed to the Reflector: which strategies it chose, and the
    real outcome the evaluator measured. The Reflector compares two of these to assign
    credit."""

    choices: list[PipelineChoice]
    tour_length: float
    cost_seconds: float
    reward: float = dataclasses.field(
        metadata={
            "description": "The search reward: fractional improvement of the tour over "
            "the trivial identity ordering, in [0, 1] (higher is better)."
        }
    )

    def __str__(self) -> str:
        picks = " -> ".join(f"{c.module}:{c.strategy}" for c in self.choices)
        return (
            f"[{picks}] length={self.tour_length:.1f} "
            f"cost={self.cost_seconds * 1e3:.2f}ms reward={self.reward:.3f}"
        )


@pydantic.dataclasses.dataclass(frozen=True)
class Reflection:
    """The Reflector's credit-assignment output: one transferable lesson drawn from
    comparing a strong branch against a weak one."""

    lesson: str = dataclasses.field(
        metadata={
            "description": "A concrete, transferable engineering lesson about how to "
            "implement a stage better -- grounded in the difference between the two "
            "branches, not generic advice. It will be shown to future implementers."
        }
    )
    applies_to: str = dataclasses.field(
        metadata={
            "description": "Which module or strategy this lesson informs, so a future "
            "implementer knows when it is relevant."
        }
    )

    def __str__(self) -> str:
        return f"({self.applies_to}) {self.lesson}"


@dataclasses.dataclass
class Memory:
    """The comparative reflective memory: the accumulated lessons, spliced into later
    Implement prompts. Not frozen -- reflections are appended as the search learns."""

    reflections: list[Reflection] = dataclasses.field(default_factory=list)

    def digest(self) -> str:
        """The lessons rendered for an Implement prompt; empty guidance when none."""
        if not self.reflections:
            return "(no lessons learned yet)"
        return "\n".join(f"- {r}" for r in self.reflections)


# ---------------------------------------------------------------------------
# The three agents. All are closed-book: no search tools -- the only "tool" is code
# synthesis (the harness's `submit_solution` tool) and the deterministic evaluator. This is the
# distinctive shape of a coding agent, versus the literature examples' search tools.
# ---------------------------------------------------------------------------


class Designer(Agent):
    """You are the Design & Decompose agent that opens the pipeline. Instead of writing
    one monolithic script, you break the task into a short, ordered pipeline of modules
    (stages), and for each module you propose a few concrete implementation strategies
    for a downstream search to choose among."""

    @Skill.define
    def design(self, task: Task) -> Design:
        """Analyze the task and decompose a solution into an ordered pipeline of two
        or three modules. Each module is a stage that takes the cities and the current
        tour and returns an improved, valid tour; a natural decomposition is
        construction (build an initial tour) followed by one or more local-improvement
        stages. For each module, propose two or three *distinct* implementation
        strategies -- these become the choices a budget-aware search explores. Fill
        each field as its schema describes.

        <task>{task}</task>
        """


class Implementer(Agent):
    """You are the Implement agent, an expert Python programmer. You answer by writing
    code, not prose: you turn one module of the design into a function that transforms
    a tour, and the harness compiles and runs it. You read the module's intent and the
    chosen strategy, and you apply any lessons learned from earlier attempts."""

    @Skill.define
    def implement(self, module: ModuleSpec, strategy: str, lessons: str) -> Stage:
        """Write ``stage``: a function ``stage(cities, tour)`` that takes the list of
        ``City`` points and the current ``tour`` (a list of city indices) and RETURNS
        an improved tour -- a list containing every index ``0..len(cities)-1`` exactly
        once. Implement the module's intent using the chosen strategy.

        Read everything from the ``cities`` and ``tour`` arguments (use
        ``len(cities)`` for the size, ``city.x`` / ``city.y`` for coordinates, and
        ``math`` if you need it) -- do NOT hardcode the number of cities or any
        coordinates, so the same code works for any instance. Return a valid
        permutation; never drop, duplicate, or invent an index.

        Module: {module.name} -- {module.intent}
        Strategy to implement: {strategy}

        Lessons learned from earlier attempts (apply any that are relevant):
        {lessons}

        The doctest runs the synthesized stage on a tiny four-city instance, while
        the real evaluation runs it on the demo task (much larger), so a stage that
        hardcodes the problem size fails the doctest and is corrected -- the
        anti-hardcode trick of ``illustration``. The recursive
        ``Implementer().implement`` call is routed to your own submission.

        >>> _module = ModuleSpec(
        ...     name="reorder",
        ...     intent="reorder the incoming tour to shorten the round trip",
        ...     strategies=["greedy nearest-neighbour", "swap crossing edges"],
        ... )
        >>> _cities = [City(0.0, 0.0), City(1.0, 0.0), City(1.0, 1.0), City(0.0, 1.0)]
        >>> _stage = Implementer().implement(_module, "greedy nearest-neighbour", "")
        >>> _out = _stage(_cities, [0, 1, 2, 3])
        >>> sorted(_out) == [0, 1, 2, 3]
        True
        """


class Reflector(Agent):
    """You are the Comparative Reflection agent. You look at two finished branches --
    one that scored well and one that scored poorly -- and you diagnose *why* the good
    one won, distilling a single transferable lesson a future implementer can reuse.
    You solve credit assignment by comparison, not by guessing."""

    @Skill.define
    def reflect(self, better: RolloutSummary, worse: RolloutSummary) -> Reflection:
        """Compare these two branches of the search. The first achieved a higher reward
        (a shorter tour, accounting for its execution cost) than the second. Identify
        the concrete difference in their strategy choices or implementation that most
        plausibly explains the gap, and state one transferable lesson for implementing
        such a stage better next time. Ground the lesson in the comparison -- what the
        better branch did that the worse one did not -- not in generic advice.

        <better_branch>{better}</better_branch>

        <worse_branch>{worse}</worse_branch>
        """


# ---------------------------------------------------------------------------
# Budget-aware MCTS. The tree's actions are the Design's per-module strategies; a
# leaf is a full pipeline, scored by the real evaluator. Cost-aware UCT and a rollout
# budget make the search prefer high-quality, low-cost designs.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Node:
    """One MCTS node: a partial pipeline. ``path`` is the strategy chosen for each
    module decided so far (module ``len(path)`` is decided by this node's children).
    ``untried`` holds the strategies for the next module not yet expanded."""

    path: tuple[str, ...]
    untried: list[str]
    children: list["Node"] = dataclasses.field(default_factory=list)
    visits: int = 0
    reward_sum: float = 0.0
    cost_sum: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.reward_sum / self.visits if self.visits else 0.0

    @property
    def avg_cost(self) -> float:
        return self.cost_sum / self.visits if self.visits else 0.0


@dataclasses.dataclass
class Rollout:
    """The record of one evaluated pipeline: its choices, the compiled stages, and the
    real outcome. ``ok`` is False if synthesis or evaluation failed (that branch scores
    nothing). ``lessons_available`` records how many reflections existed when it ran --
    used to measure cross-branch transfer afterwards."""

    path: tuple[str, ...]
    stages: list[Stage]
    length: float
    cost: float
    reward: float
    ok: bool
    lessons_available: int


@dataclasses.dataclass
class Search:
    """A budget-aware MCTS over the Design's action space, with comparative reflective
    memory. Holds everything one task's search threads together, so no ambient/global
    state is needed (contrast the ContextVar examples): the evaluator is handed its
    stages explicitly and the doctest certifies a structural invariant."""

    task: Task
    design: Design
    exploration: float  # UCT exploration constant C
    cost_weight: float  # UCT cost coefficient alpha
    rng: random.Random
    memory: Memory = dataclasses.field(default_factory=Memory)
    root: Node = dataclasses.field(init=False)
    # A synthesized stage is expensive to produce, so cache it by (generation, module
    # index, strategy). Adding a lesson bumps ``generation``, invalidating the cache so
    # later branches re-implement with the lesson -- how cross-branch transfer bites.
    _cache: dict[tuple[int, int, str], Stage] = dataclasses.field(default_factory=dict)
    generation: int = 0
    baseline: float = 0.0  # length of the identity tour -- the reward's zero point
    max_cost: float = 1e-9  # running max rollout cost, to normalize the UCT penalty
    rollouts: list[Rollout] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Node(path=(), untried=list(self.design.modules[0].strategies))
        self.baseline = tour_length(
            self.task.cities, list(range(len(self.task.cities)))
        )

    # --- reward -----------------------------------------------------------

    def reward_of(self, length: float) -> float:
        """Fractional improvement of a tour over the identity ordering, clamped to
        [0, 1] -- a bounded reward so the UCT exploration term stays well-scaled."""
        return max(0.0, min(1.0, (self.baseline - length) / self.baseline))

    def score(self, reward: float, cost: float) -> float:
        """The budget-aware value of a branch: reward minus a normalized cost penalty.
        This is the UCT *exploitation* term and the final selection key -- the paper's
        'favor high-reward, low-cost branches'."""
        return reward - self.cost_weight * (cost / self.max_cost)

    def uct(self, child: Node, parent: Node) -> float:
        """Cost-aware UCT: exploitation + exploration - alpha * cost/budget (the
        paper's selection criterion). Unvisited children sort first."""
        if child.visits == 0:
            return math.inf
        explore = self.exploration * math.sqrt(math.log(parent.visits) / child.visits)
        return self.score(child.avg_reward, child.avg_cost) + explore

    # --- tree policy ------------------------------------------------------

    def select(self) -> list[Node]:
        """Descend from the root by cost-aware UCT until reaching a node that can be
        expanded (has untried strategies) or is terminal (a full pipeline). Returns the
        path of nodes visited, for backpropagation."""
        node = self.root
        path = [node]
        while not node.untried and node.children:  # fully expanded, non-terminal
            node = max(node.children, key=lambda c: self.uct(c, node))
            path.append(node)
        return path

    def expand(self, node: Node) -> Node:
        """Add one child for an untried strategy of the next module."""
        strategy = node.untried.pop(0)
        depth = len(node.path) + 1
        untried = (
            list(self.design.modules[depth].strategies)
            if depth < len(self.design.modules)
            else []
        )
        child = Node(path=node.path + (strategy,), untried=untried)
        node.children.append(child)
        return child

    def complete(self, path: tuple[str, ...]) -> tuple[str, ...]:
        """Finish a partial path into a full pipeline by choosing a random strategy for
        each remaining module (the default rollout policy)."""
        full = list(path)
        full.extend(
            self.rng.choice(self.design.modules[depth].strategies)
            for depth in range(len(path), len(self.design.modules))
        )
        return tuple(full)

    # --- rollout ----------------------------------------------------------

    def build(self, path: tuple[str, ...]) -> list[Stage]:
        """Synthesize (or reuse) the stage for each chosen module. Caching keys on the
        current ``generation`` so a new lesson forces re-implementation."""
        stages: list[Stage] = []
        for depth, strategy in enumerate(path):
            key = (self.generation, depth, strategy)
            if key not in self._cache:
                self._cache[key] = Implementer().implement(
                    self.design.modules[depth], strategy, self.memory.digest()
                )
            stages.append(self._cache[key])
        return stages

    def rollout(self, node: Node) -> Rollout:
        """Complete the node's path to a full pipeline, synthesize it, and evaluate --
        the real quality and cost. A synthesis or evaluation failure scores nothing."""
        full = self.complete(node.path)
        try:
            stages = self.build(full)
            length, cost = evaluate(self.task, stages)
            reward, ok = self.reward_of(length), True
        except Exception as exc:  # retries exhausted, or an invalid-tour stage
            print(f"    [rollout] {full} failed: {type(exc).__name__}")
            stages, length, cost, reward, ok = [], math.inf, 0.0, 0.0, False
        self.max_cost = max(self.max_cost, cost)
        r = Rollout(
            path=full,
            stages=stages,
            length=length,
            cost=cost,
            reward=reward,
            ok=ok,
            lessons_available=len(self.memory.reflections),
        )
        self.rollouts.append(r)
        return r

    def backpropagate(self, path: list[Node], reward: float, cost: float) -> None:
        for node in path:
            node.visits += 1
            node.reward_sum += reward
            node.cost_sum += cost

    # --- comparative reflection ------------------------------------------

    def reflect(self) -> None:
        """Compare the best and worst distinct successful branches so far and store a
        lesson, then bump the generation so later branches re-implement with it. This
        is the paper's cross-path credit assignment feeding the reflective memory."""
        ok = [r for r in self.rollouts if r.ok]
        if len(ok) < 2:
            return
        best = max(ok, key=lambda r: r.reward)
        # Tie-break away from `best`: `max` and `min` both return the *first*
        # element among equals, so on a run where every rollout scores the same
        # -- common early, before the reward spreads out -- `worst` would be the
        # very object `best` is, the guard below would fire, and comparative
        # reflection would be skipped in silence for the whole search.
        worst = min(ok, key=lambda r: (r.reward, r.path == best.path))
        if best.path == worst.path:
            return
        reflection = Reflector().reflect(self._summ(best), self._summ(worst))
        self.memory.reflections.append(reflection)
        self.generation += (
            1  # invalidate the synthesis cache: re-implement with the lesson
        )
        print(f"    [reflect] lesson: {reflection}")

    def _summ(self, r: Rollout) -> RolloutSummary:
        choices = [
            PipelineChoice(self.design.modules[d].name, s) for d, s in enumerate(r.path)
        ]
        return RolloutSummary(choices, r.length, r.cost, r.reward)

    # --- driver -----------------------------------------------------------

    def run(self, *, max_rollouts: int, reflect_every: int) -> Rollout:
        """The MCTS loop under a rollout budget: select -> expand -> rollout ->
        backpropagate, reflecting every few rollouts. Returns the best actually-
        evaluated pipeline (balancing quality and cost) -- MARS's best-path extraction."""
        for i in range(1, max_rollouts + 1):
            path = self.select()
            leaf = path[-1]
            if leaf.untried:  # expand a new action
                leaf = self.expand(leaf)
                path.append(leaf)
            result = self.rollout(leaf)
            self.backpropagate(path, result.reward, result.cost)
            print(
                f"  rollout {i}/{max_rollouts}: {self._summ(result)}"
                if result.ok
                else f"  rollout {i}/{max_rollouts}: (failed)"
            )
            if reflect_every and i % reflect_every == 0:
                self.reflect()

        succeeded = [r for r in self.rollouts if r.ok]
        if not succeeded:
            raise RuntimeError("every rollout failed to produce a valid pipeline")
        return max(succeeded, key=lambda r: self.score(r.reward, r.cost))

    def cross_branch_gain(self) -> tuple[int, float, float]:
        """A simple post-hoc read on whether lessons helped later branches: the best
        reward reached *before* any lesson existed, versus how many later branches beat
        it. A nod to the paper's cross-branch-transfer analysis, not its statistic."""
        pre = [r.reward for r in self.rollouts if r.ok and r.lessons_available == 0]
        post = [r.reward for r in self.rollouts if r.ok and r.lessons_available > 0]
        best_pre = max(pre, default=0.0)
        best_post = max(post, default=0.0)
        improved = sum(1 for r in post if r > best_pre)
        return improved, best_pre, best_post


# ---------------------------------------------------------------------------
# The pipeline: Design -> (budget-aware MCTS over Decompose/Implement) with reflection.
# ---------------------------------------------------------------------------


def implement(
    task: Task,
    *,
    max_rollouts: int,
    reflect_every: int,
    exploration: float,
    cost_weight: float,
    seed: int,
) -> tuple[Search, Rollout]:
    """Design the pipeline, then run the cost-constrained MCTS over its modules and
    strategies -- synthesizing and evaluating each explored pipeline, and reflecting
    across branches -- and return the search and the best pipeline found."""
    print("[design] decomposing the task into a modular pipeline ...")
    design = Designer().design(task)
    print(design)

    search = Search(
        task=task,
        design=design,
        exploration=exploration,
        cost_weight=cost_weight,
        rng=random.Random(seed),
    )
    print(
        f"\n[search] budget-aware MCTS: {max_rollouts} rollouts, "
        f"baseline tour length {search.baseline:.1f}\n"
    )
    best = search.run(max_rollouts=max_rollouts, reflect_every=reflect_every)
    return search, best


# ---------------------------------------------------------------------------
# Demo task: a planted set of cities. Generated deterministically from a seed so the
# instance is fixed, while the search (and its real, noisy cost signal) does the work.
# ---------------------------------------------------------------------------


def make_task(num_cities: int, seed: int) -> Task:
    rng = random.Random(seed)
    cities = tuple(
        City(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(num_cities)
    )
    return Task(cities=cities)


def _print_stage_source(stages: list[Stage]) -> None:
    """Show the code MARS actually wrote for the winning pipeline, when the synthesized
    source is recoverable (the eval provider registers it with ``linecache``)."""
    for i, stage in enumerate(stages):
        try:
            src = inspect.getsource(stage)
        except (OSError, TypeError):
            continue
        print(f"\n--- stage {i} ---\n{src.rstrip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-cities", type=int, default=40, help="Cities in the planted TSP task"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the task and rollout policy"
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=6,
        help="Rollout budget for the cost-aware MCTS",
    )
    parser.add_argument(
        "--reflect-every",
        type=int,
        default=3,
        help="Run comparative reflection every N rollouts (0 disables it)",
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=1.4,
        help="UCT exploration constant C",
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.3,
        help="UCT cost coefficient alpha (how much execution cost is penalized)",
    )
    args = parser.parse_args()

    task = make_task(args.num_cities, args.seed)
    print(f"Task: shortest closed tour over {args.num_cities} cities\n")

    search, best = implement(
        task,
        max_rollouts=args.max_rollouts,
        reflect_every=args.reflect_every,
        exploration=args.exploration,
        cost_weight=args.cost_weight,
        seed=args.seed,
    )

    print("\n" + "=" * 72)
    summary = search._summ(best)
    print(f"Best pipeline: {summary}")
    print(
        f"  improvement over baseline: "
        f"{(search.baseline - best.length) / search.baseline * 100:.1f}%"
    )

    if search.memory.reflections:
        print("\nLessons learned (comparative reflective memory):")
        for r in search.memory.reflections:
            print(f"  - {r}")
        improved, best_pre, best_post = search.cross_branch_gain()
        print(
            f"\nCross-branch transfer: best reward before any lesson {best_pre:.3f}; "
            f"{improved} later branch(es) beat it (best after {best_post:.3f})."
        )

    _print_stage_source(best.stages)


if __name__ == "__main__":
    main()
