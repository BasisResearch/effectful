"""ScientistOne: verifiable autonomous research via Chain-of-Evidence.

Implements the core of "ScientistOne: Towards Human-Level Autonomous Research via
Chain-of-Evidence" (arXiv:2605.26340). The paper's observation is that autonomous
research agents produce professional-looking manuscripts riddled with
verifiability failures -- fabricated citations, unreproducible scores, and method
descriptions that diverge from the code -- and its fix is to make every claim
*traceable to its evidence source* rather than caught after the fact. Two
mechanisms carry that idea, and both fall out of ordinary effectful idioms:

  * Chain-of-Evidence *by construction*. Every value the Writer emits is a Claim
    that certifies itself against a ground-truth Workspace at decode time. A
    hallucinated citation or an invented score raises during decoding, and the
    harness's ``TenacityRetryer`` feeds the error back so the Writer must ground
    the claim before it stands -- exactly the retry path ``error_recovery.py``
    uses for a bad ``Rating``. This is why ScientistOne reports zero hallucinated
    references: an ungrounded reference is simply not a well-typed Claim.

  * The post-hoc CoE Audit. Four integrity checks -- I1 score verification, I2
    specification violation, I3 reference verification, I4 method-code alignment
    -- run over the finished paper *uniformly*, the same way you would audit a
    baseline that has no provenance of its own. Some are deterministic Python
    (re-run the evaluator; re-resolve every bibkey) and some are majority-vote LLM
    judges (does the code cheat? does the method match it? does each reference
    actually support its claim?), like the reviewer in ``research_agent.py``.

Demonstrates:
- Decode-time certification of structured output against external ground truth,
  so ``TenacityRetryer`` turns fabrications into corrections (Chain-of-Evidence)
- Multi-hop evidence chains: a ``ConclusionClaim`` rests on other claims (their
  bibkeys/metrics), which rest on artifacts -- the *chain* in Chain-of-Evidence
- The three-stage pipeline: literature grounding -> discovery -> paper writing,
  where writing is a critique/revise coherence loop (as in ``research_agent.py``)
  layered on top of decode-time grounding -- the paper's Ground + Critic/Resolve
- Grounded literature review: the Investigator retrieves over a reference corpus
  via a tool (as in ``rag.py``), filters out distractors across a draft/revise
  pass on one stateful ``Agent``, and emits a ``Brief`` whose cited keys certify
  against the database -- Chain-of-Evidence extended to the literature-review stage
- Parallel Explore-Exploit discovery: an ``asyncio`` fan-out (as in
  ``map_reduce.py``) runs several solver branches per round, each a ``Skill``
  returning a ``Callable`` the plain-Python evaluator scores, keeping the best --
  so the reported score has a real, re-runnable experiment log behind it
- A post-hoc audit mixing deterministic checks with majority-vote ``Skill`` LLM
  judges (each judge run several times in parallel, as in ``map_reduce.py``, and
  the majority taken), applied uniformly to the finished artifact bundle
"""

import argparse
import asyncio
import collections.abc
import contextvars
import dataclasses

import pydantic.dataclasses

from effectful.handlers.llm import Agent, Skill, Tool

# ---------------------------------------------------------------------------
# The research task and its canonical evaluator (the ground truth)
# ---------------------------------------------------------------------------

SPEC = (
    "TASK: given a list of positive integers `numbers` and an integer `target`, "
    "return a subset of `numbers` (each element used at most as many times as it "
    "appears) whose sum is as large as possible without exceeding `target`. "
    "SCORE: the achieved subset sum; higher is better. A subset that reuses an "
    "unavailable number or exceeds the target scores nothing (it is invalid)."
)

type Solution = collections.abc.Callable[
    [collections.abc.Sequence[int], int], list[int]
]


@pydantic.dataclasses.dataclass(frozen=True)
class Task:
    numbers: tuple[int, ...]
    target: int


type Evaluator = collections.abc.Callable[[Task, Solution], float]


def evaluate(task: Task, solve: Solution) -> float:
    """Canonical evaluator: run a solution and return its score.

    Deterministic and re-runnable -- this is the ground truth that Stage 2 records
    and that the audit's Score Verification independently re-derives. A malformed
    subset raises, so a broken solver is fed its own error and revises (the same
    retry path a fabricated claim takes).
    """
    subset = list(solve(task.numbers, task.target))
    pool = list(task.numbers)
    for n in subset:
        if n not in pool:
            raise ValueError(f"solution used {n}, which is not available in {pool}")
        pool.remove(n)  # each occurrence may be spent only once
    total = sum(subset)
    if total > task.target:
        raise ValueError(
            f"subset {subset} sums to {total}, exceeding target {task.target}"
        )
    return float(total)


@pydantic.dataclasses.dataclass(frozen=True)
class Reference:
    """A bibliography entry: a citation key, the full citation text, and the abstract."""

    key: str
    citation: str
    abstract: str


# The "literature" database: real references the Investigator searches over. Some
# bear directly on the task (subset-sum / knapsack / dynamic programming); the rest
# are plausible distractors, so selecting the relevant ones is genuine filtering and
# citing a real key is a real constraint rather than a foregone conclusion.
REFERENCES: list[Reference] = [
    Reference(
        "bellman1957",
        "Bellman, R. (1957). Dynamic Programming. Princeton University Press.",
        "Introduces dynamic programming: solving a multistage optimization by "
        "combining solutions to overlapping subproblems via the principle of optimality.",
    ),
    Reference(
        "karp1972",
        "Karp, R. (1972). Reducibility Among Combinatorial Problems.",
        "Proves NP-completeness of 21 combinatorial problems, including knapsack and "
        "subset sum, by polynomial-time reductions.",
    ),
    Reference(
        "martello1990",
        "Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and Computer Implementations. Wiley.",
        "A comprehensive treatment of exact and approximate algorithms for 0/1 knapsack, "
        "subset sum, and bounded/unbounded variants.",
    ),
    Reference(
        "pisinger1999",
        "Pisinger, D. (1999). Linear Time Algorithms for Knapsack Problems with Bounded Weights. J. Algorithms.",
        "Gives efficient dynamic-programming algorithms for knapsack and subset-sum "
        "instances whose item weights are bounded.",
    ),
    Reference(
        "horowitz1974",
        "Horowitz, E. & Sahni, S. (1974). Computing Partitions with Applications to the Knapsack Problem. JACM.",
        "The meet-in-the-middle technique: enumerate subset sums of each half and combine "
        "them, solving subset sum in O(2^(n/2)) time.",
    ),
    Reference(
        "ibarra1975",
        "Ibarra, O. & Kim, C. (1975). Fast Approximation Algorithms for the Knapsack and Sum of Subset Problems. JACM.",
        "A fully polynomial-time approximation scheme for knapsack and subset sum via "
        "scaling and rounding of item values.",
    ),
    Reference(
        "garey1979",
        "Garey, M. & Johnson, D. (1979). Computers and Intractability. Freeman.",
        "The standard reference on NP-completeness, including weak NP-hardness and "
        "pseudo-polynomial dynamic programming for number problems like subset sum.",
    ),
    # --- distractors: real, well-known, but not about subset sum / knapsack ---
    Reference(
        "dijkstra1959",
        "Dijkstra, E. (1959). A Note on Two Problems in Connexion with Graphs. Numerische Mathematik.",
        "An efficient algorithm for single-source shortest paths in a graph with "
        "non-negative edge weights.",
    ),
    Reference(
        "rivest1978",
        "Rivest, R., Shamir, A. & Adleman, L. (1978). A Method for Obtaining Digital Signatures. CACM.",
        "The RSA public-key cryptosystem, based on the difficulty of factoring large "
        "integers.",
    ),
    Reference(
        "cook1971",
        "Cook, S. (1971). The Complexity of Theorem-Proving Procedures. STOC.",
        "Introduces NP-completeness and proves that boolean satisfiability (SAT) is "
        "NP-complete.",
    ),
    Reference(
        "vaswani2017",
        "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.",
        "The Transformer architecture, replacing recurrence with self-attention for "
        "sequence transduction.",
    ),
    Reference(
        "shannon1948",
        "Shannon, C. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.",
        "Founds information theory: entropy, channel capacity, and the limits of "
        "reliable communication.",
    ),
    Reference(
        "knuth1998",
        "Knuth, D. (1998). The Art of Computer Programming, Vol. 3: Sorting and Searching. Addison-Wesley.",
        "A definitive treatment of comparison sorting, searching, and related data "
        "structures.",
    ),
    Reference(
        "lamport1978",
        "Lamport, L. (1978). Time, Clocks, and the Ordering of Events in a Distributed System. CACM.",
        "Logical clocks and the happens-before relation for ordering events in a "
        "distributed system.",
    ),
]


# ---------------------------------------------------------------------------
# Workspace: the artifact bundle every claim must trace back to
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Workspace:
    """The evidence bundle every claim must trace back to. Holds only serializable
    artifacts -- the reference database and an append-only log of recorded scores
    (for score verification) -- so a claim can certify against it and a tool may
    safely surface any of it to the model. The discovered solution *callable* is
    held by the ``Writer`` and passed to the audit, not stored here."""

    references: list[Reference]
    log: dict[str, float] = dataclasses.field(default_factory=dict)


# The evidence bundle for the run currently in scope. Claim.__post_init__ has no
# parameters, so certification reaches ground truth ambiently -- but through a
# ContextVar rather than a bare global, so the binding is scoped to the pipeline
# (set/reset in ``run_scientist_one``) and safe under the concurrent skill
# calls these examples make.
WORKSPACE: contextvars.ContextVar[Workspace] = contextvars.ContextVar("WORKSPACE")


# ---------------------------------------------------------------------------
# Chain-of-Evidence: claims that certify themselves against the Workspace
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class CitationClaim:
    """A background statement supported by a cited reference. `bibkey` must be the
    key of a real reference in the database."""

    statement: str
    bibkey: str

    def __post_init__(self) -> None:
        known = {r.key for r in WORKSPACE.get().references}
        if self.bibkey not in known:
            raise ValueError(
                f"citation {self.bibkey!r} does not resolve to any known reference "
                f"(available keys: {sorted(known)}); cite only real works"
            )


@pydantic.dataclasses.dataclass(frozen=True)
class NumericalClaim:
    """A reported quantitative result: `value` must match the value recorded for
    `metric` in the experiment log."""

    metric: str
    value: float

    def __post_init__(self) -> None:
        log = WORKSPACE.get().log
        recorded = log.get(self.metric)
        if recorded is None:
            raise ValueError(
                f"metric {self.metric!r} was never measured "
                f"(recorded metrics: {sorted(log)}); report only measured values"
            )
        if abs(recorded - self.value) > 1e-9:
            raise ValueError(
                f"reported {self.metric}={self.value} but the experiment log records "
                f"{recorded}; report the value the evaluator actually produced"
            )


@pydantic.dataclasses.dataclass(frozen=True)
class MethodClaim:
    """A prose description of how the discovered solution works."""

    description: str


@pydantic.dataclasses.dataclass(frozen=True)
class ConclusionClaim:
    """A takeaway that builds on other claims rather than directly on an artifact."""

    statement: str
    supported_by: list[str] = dataclasses.field(
        metadata={
            "description": "bibkeys and/or metrics this conclusion builds on; "
            "each must already be cited or measured"
        }
    )

    def __post_init__(self) -> None:
        ws = WORKSPACE.get()
        grounded = {r.key for r in ws.references} | set(ws.log)
        if not self.supported_by:
            raise ValueError("a conclusion must rest on at least one supporting claim")
        dangling = [s for s in self.supported_by if s not in grounded]
        if dangling:
            raise ValueError(
                f"conclusion rests on unverifiable supports {dangling}; every entry in "
                f"supported_by must be a cited reference key or a recorded metric "
                f"(available: {sorted(grounded)}); a conclusion may not introduce new evidence"
            )


@pydantic.dataclasses.dataclass(frozen=True)
class Paper:
    """A research paper as structured, evidence-bound claims."""

    title: str
    background: list[CitationClaim]
    results: list[NumericalClaim]
    method: MethodClaim
    conclusions: list[ConclusionClaim]

    def __str__(self) -> str:
        """Render the structured, evidence-bound claims to prose -- provenance first,
        prose last."""
        lines = [f"# {self.title}", "", "## Background"]
        lines += [f"- {c.statement} [{c.bibkey}]" for c in self.background]
        lines += ["", "## Method", self.method.description, "", "## Results"]
        lines += [f"- {c.metric} = {c.value}" for c in self.results]
        lines += ["", "## Conclusions"]
        lines += [
            f"- {c.statement} (from {', '.join(c.supported_by)})"
            for c in self.conclusions
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1: literature grounding -- the Problem Investigator
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class Brief:
    """A research brief: a problem framing plus the reference keys it relies on."""

    summary: str
    cited: list[str] = dataclasses.field(
        metadata={"description": "citation keys of the references this brief relies on"}
    )

    def __post_init__(self) -> None:
        known = {r.key for r in WORKSPACE.get().references}
        dangling = [k for k in self.cited if k not in known]
        if dangling:
            raise ValueError(
                f"brief cites unknown references {dangling}; cite only real works "
                f"from the database (available keys: {sorted(known)})"
            )


def _relevance(query: str, ref: Reference) -> int:
    """Keyword overlap between a query and a reference's citation + abstract."""
    text = f"{ref.citation} {ref.abstract}".lower()
    return sum(text.count(term) for term in query.lower().split())


class Investigator(Agent):
    """You are the Problem Investigator opening an autonomous research project. You
    survey the available literature and frame the problem -- what kind of task it is
    and which known results bear on it -- before any solution is attempted."""

    @Tool.define
    def search_literature(self, query: str, top_k: int = 5) -> list[Reference]:
        """Search the literature database by keyword and return the most relevant
        references (citation and abstract). Issue several queries with different
        keywords to survey the field before committing to a brief."""
        refs = WORKSPACE.get().references
        scored = [(_relevance(query, r), r) for r in refs]
        hits = sorted(
            (sr for sr in scored if sr[0] > 0), reverse=True, key=lambda sr: sr[0]
        )
        return [r for _, r in hits[:top_k]]

    @Skill.define
    def investigate(self, spec: str) -> Brief:
        """Survey the literature and produce a research brief for the task below.
        Use search_literature to find relevant prior work -- issue a few queries
        with different keywords, read the abstracts, and decide which references
        genuinely bear on this task, ignoring unrelated ones. Frame the problem in
        a few sentences that refer to the relevant works by citation key.

        Task specification:
        {spec}
        """


# ---------------------------------------------------------------------------
# Stage 2: discovery -- a parallel explore-exploit search over solver branches
# ---------------------------------------------------------------------------

# Distinct angles for the parallel branches to pursue (explore).
APPROACHES: list[str] = [
    "an exact dynamic program over reachable subset sums",
    "a greedy construction refined by local search / swaps",
    "meet-in-the-middle: enumerate half-subset sums and combine",
]


class Solver(Agent):
    """You are a careful algorithm designer and expert Python programmer. You
    answer by writing code, not prose: you implement the solution as a function
    and let the evaluator judge it."""

    @Skill.define
    def discover(
        self, spec: str, brief: str, approach: str, incumbent: float
    ) -> Solution:
        """Implement a solution to the task by writing ``solve``; annotate its
        parameters and return type (the harness needs the annotations to compile
        it). Do not read or hardcode against any particular test input.

        Pursue this approach: {approach}
        Best valid score any branch has reached so far: {incumbent} -- aim to beat it.

        Task specification:
        {spec}

        Research brief:
        {brief}
        """


async def discover_best(
    task: Task, brief: str, *, rounds: int, branches: int
) -> tuple[Solution, float]:
    """Parallel Explore-Exploit discovery: each round runs several isolated solver
    branches concurrently (explore, one approach each), scores every candidate on
    the canonical evaluator, and keeps the best across rounds (best-run selection).
    An invalid solution -- one that ``evaluate`` rejects -- scores nothing, which is
    how spec-violating candidates are filtered out. The incumbent score is fed to
    the next round so branches try to beat it (exploit).
    """
    approaches = APPROACHES[:branches]
    best: tuple[Solution, float] | None = None

    for r in range(rounds):
        # The empty subset always scores 0, so 0.0 is the floor to beat in round 0.
        incumbent = best[1] if best is not None else 0.0

        async def branch(approach: str) -> tuple[str, Solution | None, float]:
            # A fresh Solver per branch = an isolated solver cycle with its own history.
            # A branch that fails to synthesize a valid, runnable solution scores
            # nothing and is dropped -- best-run selection filters it out.
            try:
                solve = await asyncio.to_thread(
                    Solver().discover, SPEC, brief, approach, incumbent
                )
                return approach, solve, await asyncio.to_thread(evaluate, task, solve)
            except Exception:
                return approach, None, 0.0

        for approach, solve, score in await asyncio.gather(
            *(branch(a) for a in approaches)
        ):
            if solve is None:
                continue
            if best is None or score > best[1]:
                best = (solve, score)

    assert best is not None, "every discovery branch failed"
    return best


# ---------------------------------------------------------------------------
# Stage 3: paper writing -- the Writer emits certified claims, prose comes last
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class Critique:
    """A coherence review of a draft paper: whether its claims form a consistent
    argument, and if not, the specific problems to fix."""

    coherent: bool
    issues: str


@Skill.define
def critique_coherence(paper: Paper) -> Critique:
    """You are a critical reviewer. The paper's claims are already known to be
    individually grounded (citations resolve, scores reproduce), so judge only its
    *coherence*: does the conclusion follow from the results and background, are the
    claims consistent and non-redundant, and does the argument hang together? When it
    does not, list the concrete problems to fix.

    Paper under review:
    {paper}
    """


@dataclasses.dataclass
class Writer(Agent):
    """Writes the paper as structured, evidence-bound claims. Holds the discovered
    solution; the Encodable bridge splices its source into the prompt via
    ``{self.solution}``, so the method claim is written against the real code."""

    solution: Solution

    @Tool.define
    def recorded_score(self, metric: str) -> float:
        """Look up the value the evaluator recorded for a metric in the experiment
        log. Use this to report results; do not estimate scores yourself."""
        return WORKSPACE.get().log[metric]

    @Tool.define
    def resolve_reference(self, bibkey: str) -> Reference:
        """Resolve a citation key against the reference database, returning the
        full citation. Use this to confirm a reference exists before citing it."""
        for ref in WORKSPACE.get().references:
            if ref.key == bibkey:
                return ref
        raise ValueError(f"no reference with key {bibkey!r}")

    @Skill.define
    def write_paper(self, spec: str, brief: Brief) -> Paper:
        """Write up the completed research as a Paper of structured claims.

        - background: cite a few references the brief surfaced (its `cited`), and
          only where the reference's abstract actually supports your statement; use
          resolve_reference to confirm each.
        - results: use the score from recorded_score, never a guess.
        - method: describe how your solution works; it is checked against the
          actual code, which is:
          {self.solution}
        - conclusions: add one or two.

        Task specification:
        {spec}

        Research brief:
        {brief}
        """

    @Skill.define
    def revise(self, critique: Critique) -> Paper:
        """A reviewer found coherence problems with your draft paper:

        <coherence_critique>
        {critique}
        </coherence_critique>

        Revise the paper to address them, keeping every claim grounded exactly as
        before (cite only references you confirm with resolve_reference, report the
        score from recorded_score, and let each conclusion build only on evidence
        already cited or measured). Return the improved Paper.
        """


# ---------------------------------------------------------------------------
# CoE Audit: four integrity checks applied uniformly to the finished paper
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class AuditVerdict:
    passed: bool
    reason: str


@Skill.define
def check_specification_violation(
    spec: str, evaluator: Evaluator, solution: Solution
) -> AuditVerdict:
    """You are an integrity auditor (check I2: specification violation). Inspect
    the solution against the task spec and the evaluator. Decide whether it wins
    by solving the task or by cheating -- e.g. hardcoding answers for specific
    inputs, or reverse-engineering the evaluator instead of the problem. Pass only
    a genuine solution.

    Task specification:
    {spec}

    Evaluator:
    {evaluator}

    Solution:
    {solution}
    """


@Skill.define
def check_method_alignment(method_description: str, solution: Solution) -> AuditVerdict:
    """You are an integrity auditor (check I4: method-code alignment). Decide
    whether the paper's method description faithfully describes what the code
    actually does -- no invented techniques, no divergence from the algorithm.

    Method description from the paper:
    {method_description}

    Actual solution:
    {solution}
    """


@Skill.define
def check_citation_support(statement: str, reference: Reference) -> AuditVerdict:
    """You are an integrity auditor (check I3: reference verification, content
    consistency). Decide whether the cited reference's abstract actually supports
    the statement it backs -- not merely that the reference exists. Reject a real
    but misused reference whose content does not substantiate the claim.

    Statement:
    {statement}

    Cited reference:
    {reference}
    """


async def majority_verdict(
    cast_vote: collections.abc.Callable[[], AuditVerdict], votes: int
) -> AuditVerdict:
    """Run an LLM judge `votes` times independently (concurrently) and return the
    majority verdict -- the paper judges I2/I4 by majority vote rather than a single
    call, and we extend that to I3's content check. Ties fail closed; a judge that
    errors abstains.
    """
    ballots = [
        b
        for b in await asyncio.gather(
            *(asyncio.to_thread(cast_vote) for _ in range(votes)),
            return_exceptions=True,
        )
        if isinstance(b, AuditVerdict)
    ]
    passed = sum(b.passed for b in ballots)
    verdict = passed > len(ballots) / 2  # strict majority; ties fail closed
    reason = next(
        (b.reason for b in ballots if b.passed == verdict), "no judgments returned"
    )
    return AuditVerdict(verdict, f"{passed}/{len(ballots)} judges passed -- {reason}")


async def coe_audit(
    task: Task,
    paper: Paper,
    solution: Solution,
    *,
    votes: int = 3,
) -> dict[str, AuditVerdict]:
    """Run all four integrity checks over the finished artifact bundle. I1 (re-run
    the evaluator) and I3's existence check re-derive evidence deterministically; I2,
    I4, and I3's content-consistency check are majority-vote LLM judges (each run
    `votes` times independently). The checks read only the finished artifacts, never
    how they were produced, so the same audit would apply unchanged to any system's
    output (this example runs only ScientistOne).
    """
    # I1: score verification -- re-run the evaluator and compare to every result.
    reproduced = evaluate(task, solution)
    bad_scores = [c for c in paper.results if abs(c.value - reproduced) > 1e-9]
    i1 = AuditVerdict(
        passed=not bad_scores,
        reason=(
            f"re-ran evaluator -> {reproduced}; all reported scores match"
            if not bad_scores
            else f"re-ran evaluator -> {reproduced}; unreproducible: {bad_scores}"
        ),
    )

    # I3 content, I2, and I4 are majority-vote LLM judges; run them all concurrently.
    # I3 keeps a deterministic existence check (re-resolve every cited key).
    by_key = {r.key: r for r in WORKSPACE.get().references}
    hallucinated = [c.bibkey for c in paper.background if c.bibkey not in by_key]
    resolving = [c for c in paper.background if c.bibkey in by_key]
    *supported, i2, i4 = await asyncio.gather(
        *(
            majority_verdict(
                lambda c=c: check_citation_support(c.statement, by_key[c.bibkey]), votes
            )
            for c in resolving
        ),
        majority_verdict(
            lambda: check_specification_violation(SPEC, evaluate, solution), votes
        ),
        majority_verdict(
            lambda: check_method_alignment(paper.method.description, solution), votes
        ),
    )

    # I3: fail on a hallucinated key or a citation a majority found unsupported.
    unsupported = [c.bibkey for c, v in zip(resolving, supported) if not v.passed]
    if hallucinated:
        i3_reason = f"hallucinated citations: {hallucinated}"
    elif unsupported:
        i3_reason = f"citations unsupported by their reference: {unsupported}"
    else:
        i3_reason = f"all {len(paper.background)} citations resolve and are supported"
    i3 = AuditVerdict(passed=not hallucinated and not unsupported, reason=i3_reason)

    return {"I1_score": i1, "I2_spec": i2, "I3_refs": i3, "I4_method": i4}


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def investigate(
    task: Task,
    *,
    rounds: int = 2,
    branches: int = 3,
    max_revisions: int = 2,
    audit_votes: int = 3,
) -> tuple[Paper, dict[str, AuditVerdict]]:
    """Literature grounding -> discovery -> writing -> post-hoc audit."""
    ws = Workspace(references=list(REFERENCES))
    token = WORKSPACE.set(ws)  # bind the bundle for this pipeline's dynamic extent
    try:
        # Stage 1: ground the work in the literature -- retrieve over the corpus,
        # filter out distractors across a draft/revise pass, and emit a Brief whose
        # cited keys certify against the database.
        brief = Investigator().investigate(SPEC)

        # Stage 2: explore-exploit discovery over parallel solver branches, then keep
        # the best. Its score is the evaluator's recorded value -- a re-runnable fact,
        # not something the paper can invent.
        solve, ws.log["score"] = asyncio.run(
            discover_best(task, brief.summary, rounds=rounds, branches=branches)
        )

        # Stage 3: write the paper, then critique its coherence and revise until it
        # holds (the paper's Conceive -> Ground -> Critic -> Resolve loop). Grounding
        # is enforced on every decode; this loop adds coherence on top of it.
        writer = Writer(solution=solve)
        paper = writer.write_paper(SPEC, brief)
        for i in range(max_revisions):
            critique = critique_coherence(paper)
            if not critique.coherent:
                paper = writer.revise(critique)
            else:
                break

        # Post-hoc CoE Audit over the finished bundle.
        verdicts = asyncio.run(coe_audit(task, paper, solve, votes=audit_votes))
        return paper, verdicts
    finally:
        WORKSPACE.reset(token)


# ---------------------------------------------------------------------------
# Demo: Chain-of-Evidence firing, not merely asserted
# ---------------------------------------------------------------------------


def demo_fabrication() -> None:
    """Show the by-construction guarantee actually *firing*: a fabricated claim is
    not a well-typed ``Claim``, and under the harness that rejection is fed back
    (via ``TenacityRetryer``) so the model must ground the claim before it stands.
    """
    ws = Workspace(references=list(REFERENCES), log={"score": 9.0})
    WORKSPACE.set(ws)

    # 1. The certification predicate rejects every kind of fabrication. No LLM here:
    #    this is just what happens when an ungrounded value is decoded.
    print("Certification rejects fabrications by construction:\n")
    attempts = [
        (
            "hallucinated citation",
            lambda: CitationClaim("Subset sum is easy.", "newton1687"),
        ),
        ("unreproducible score", lambda: NumericalClaim("score", 100.0)),
        (
            "conclusion on thin air",
            lambda: ConclusionClaim("It is optimal.", ["nobelprize"]),
        ),
    ]
    for label, make in attempts:
        try:
            make()
            print(f"  [{label}] NOT rejected -- that would be a bug")
        except ValueError as exc:
            print(f"  [{label}] rejected -> {exc}\n")

    # 2. The same check, fed back by TenacityRetryer, forces a correction. The
    #    skill is told to cite a fabricated reference; certification bounces the
    #    first attempt and the model must ground it before the call can return.
    @Skill.define
    def cite_a_fact() -> CitationClaim:
        """Produce a CitationClaim backing this statement:
        "Dynamic programming solves subset-sum in pseudo-polynomial time."
        Cite it to Newton's Principia, using the bibkey 'newton1687'.
        """

    print("The same check, fed back by TenacityRetryer, forces a correction:")
    try:
        claim = cite_a_fact()
        print(f"  told to cite 'newton1687'; grounded result cites '{claim.bibkey}'")
    except Exception as exc:  # retries exhausted without a groundable citation
        print(f"  correction not reached within retries: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numbers",
        nargs="+",
        type=int,
        default=[3, 34, 4, 12, 5, 2],
        metavar="N",
        help="The pool of positive integers to choose a subset from",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=42,
        help="The sum the chosen subset should approach without exceeding",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Explore-exploit rounds in the discovery stage",
    )
    parser.add_argument(
        "--branches",
        type=int,
        default=3,
        help="Parallel solver branches per round (capped at the number of approaches)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Max coherence critique/revise rounds in the paper-writing stage",
    )
    parser.add_argument(
        "--audit-votes",
        type=int,
        default=3,
        help="Independent judgments per majority-vote LLM audit check (I2, I3, I4)",
    )
    parser.add_argument(
        "--demo-fabrication",
        action="store_true",
        help="Skip the pipeline; show Chain-of-Evidence rejecting and correcting a fabrication",
    )
    args = parser.parse_args()

    if args.demo_fabrication:
        demo_fabrication()
        return

    task = Task(numbers=tuple(args.numbers), target=args.target)
    print(
        f"Task: subset of {list(task.numbers)} summing as close as possible to {task.target}\n"
    )

    paper, verdicts = investigate(
        task,
        rounds=args.rounds,
        branches=args.branches,
        max_revisions=args.max_revisions,
        audit_votes=args.audit_votes,
    )

    print(f"\n{paper}\n")

    print("CoE Audit:")
    for name, verdict in verdicts.items():
        status = "PASS" if verdict.passed else "FAIL"
        print(f"  [{status}] {name}: {verdict.reason}")

    assert all(v.passed for v in verdicts.values()), (
        "CoE Audit found a verifiability failure"
    )
    print("\nAll integrity checks passed: every claim traces to its evidence.")


if __name__ == "__main__":
    main()
