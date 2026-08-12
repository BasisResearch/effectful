"""ScholarPeer: context-aware peer review as a multi-agent pipeline.

Implements the core of "ScholarPeer: A Context-Aware Multi-Agent Framework for
Automated Peer Review" (arXiv:2601.22638). The paper's diagnosis is that
automated reviewers write "surface-level" critiques because they judge a paper
*in a vacuum* -- frozen parametric knowledge can't place a contribution in its
field or notice a missing comparison. Its fix is a two-stream pipeline that first
*acquires context* (summarize the paper, retrieve and historicize the
literature, scout for omitted baselines) and then *actively verifies* it
(interrogate the claims against that context) before a guidelines-driven
synthesis writes the review. Most of the paper's named agents map to one
``Agent`` here (its Literature Review and Expansion agents are folded into the
Historian's tool-use loop), and the architecture falls out of ordinary effectful
idioms:

  * Tool visibility is decided by class, not by prompt. The single ``search``
    Tool lives on a ``Scholar`` base class, so it reaches the Historian and
    Baseline Scout that subclass it (via the Agent MRO) but is *structurally
    invisible* to the toolless Summarizer, Question/Answer Generators, and
    Reviewer -- they hold no ``Scholar`` instance, so nothing in their lexical
    scope offers them a tool. This is the paper's split between "context
    acquisition" (search-enabled) and "verification/synthesis" (closed-book),
    enforced by where a method is defined rather than by instructions to behave.

  * The agentic tool-use loop *is* iterative retrieval expansion. The Historian
    calls ``search`` several times -- initial query, then temporal/concurrent
    expansion -- and compresses the hits into a chronological domain narrative,
    all inside one Skill call.

  * Grounded critique by construction. A ``MissingBaseline`` the Scout emits
    certifies at decode time that the omitted work it names is a paper ``search``
    actually returned this run; a hallucinated omission raises and
    ``RetryLLMHandler`` feeds it back, so the Scout can only accuse authors of
    skipping work it truly retrieved. This is an addition, not a mechanism the
    paper describes -- its Scout searches but does no such check -- made in the
    spirit of its aim to ground critiques in verified flaws rather than generic
    complaints (the same decode-time certification ``scientist_one.py`` uses for
    citations).

  * Fan-out verification. The Multi-Aspect Q&A engine generates probing
    questions and then answers each independently against the domain narrative --
    a map over questions via ``asyncio.gather`` + ``asyncio.to_thread``, like
    ``map_reduce.py``.

  * Guidelines-driven synthesis. The Reviewer is an ``Agent`` whose
    ``{self.guidelines}`` decouples investigation from reporting: swap the venue
    (ICLR emphasizes novelty, NeurIPS rigor) and only the final synthesis shifts.

Demonstrates:
- A shared Tool on a base ``Agent`` class, offered to subclass skills via the
  MRO but invisible to the sibling toolless agents -- tool scoping as
  encapsulation, so no skill needs a "do not use tools" instruction
- Decode-time certification of structured output against a ground-truth index,
  turning a fabricated finding into a retry (grounded critique)
- Fan-out map over LLM calls with ``asyncio.gather`` + ``asyncio.to_thread``
- An ``Agent`` whose instance field reshapes a Skill prompt (venue guidelines)
- Structured, typed review output (an illustrative ICLR-style schema: per-dimension
  1-10 scores, a recommendation enum, and author-facing suggestions)
- Per-field guidance carried on the types as ``field(metadata={"description": ...})``,
  reaching the model through each schema so no prompt has to restate it
"""

# Simplifications vs. the source:
# - Corpus by default, real search opt-in. Runs default to a tiny in-memory
#   LITERATURE index so they are deterministic; ``--source semanticscholar`` swaps
#   in the live Semantic Scholar Graph API. That is a structured academic database,
#   so it reproduces the paper's grounded, ID-stable retrieval but not its
#   Google-Search reach into grey literature (blogs, GitHub, workshop papers); a
#   fuller reproduction would add a second, open-web search tool whose results have
#   no stable key to certify against.
# - Retrieval and compression are merged. The paper separates a Literature Review
#   & Expansion agent (k retrieval rounds) from the Historian (compression into a
#   narrative); here the Historian's own tool-use loop does both.
# - One review, no metrics. The paper's H-Max score (vs. a human-review ceiling)
#   and Review Diversity score (dissimilarity across N=3 sampled reviews) need a
#   human-review corpus and an embedding model; this produces a single review.
# - A single verification pass. The Answer Generator self-answers and checks
#   against the narrative, but omits the paper's cross-section consistency probing.
# - Consolidated, smaller Q&A. The paper generates N_QA=10 questions via two
#   aspect-specialized calls (one for novelty, one for soundness); here a single
#   QuestionGenerator call emits a handful, each tagged by aspect. Interrogation
#   is otherwise the same.
# - Illustrative review schema. The paper fixes no output schema (it mentions a
#   single 1-10 decision score plus author-facing suggestions); the Review
#   dataclass is an ICLR-style stand-in with per-dimension scores, a
#   recommendation, and suggestions -- shaped to be a typed return value, not
#   transcribed from the paper. Its three dimensions are not the paper's H-Max
#   evaluation axes.

import argparse
import asyncio
import collections.abc
import dataclasses
import datetime
import enum
import os
import typing

import pydantic
import requests

from effectful.handlers.llm import Agent, Skill, Tool

type Score = typing.Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# A field's ``metadata={"description": ...}`` is inlined by pydantic into that
# field's JSON schema, which the harness renders into the system prompt as part of
# a skill's argument (and structured-output) spec. So per-field guidance reaches
# the model *through the type* -- used below only where the field name and type
# don't already say it, so no prompt has to repeat it.


# ---------------------------------------------------------------------------
# The literature the agents read -- the ground truth "context" the frozen model
# lacks. Two backends supply it (chosen by main()): an offline corpus, so runs
# are deterministic, and the live Semantic Scholar Graph API, closer to the
# paper's live search. Both hand back the same LitEntry keyed by a stable citation
# key, so a retrieved paper can be cited and a finding certified against it.
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class LitEntry:
    title: str
    date: datetime.date
    venue: str
    abstract: str


LITERATURE: dict[str, LitEntry] = {
    "spectralgnn2018": LitEntry(
        "Spectral Graph Neural Networks",
        datetime.date(2018, 1, 1),
        "ICLR",
        "Foundational spectral-convolution GNN for graph classification; "
        "O(n^2) per graph in the number of nodes n.",
    ),
    "sketching2019": LitEntry(
        "Fast Matrix Sketching for Kernels",
        datetime.date(2019, 1, 1),
        "NeurIPS",
        "Randomized sketches that approximate large kernel matrices in "
        "sub-quadratic time; a general linear-algebra primitive.",
    ),
    "graphbench2020": LitEntry(
        "GraphBench: A Benchmark for Graph Classification",
        datetime.date(2020, 1, 1),
        "NeurIPS",
        "Standard graph-classification benchmark suite and leaderboard; "
        "reports accuracy with standard deviation over 10 folds.",
    ),
    "randprop2021": LitEntry(
        "RandProp: Sub-Quadratic Graph Classification by Random Projection",
        datetime.date(2021, 1, 1),
        "ICML",
        "A random-projection graph classifier running in sub-quadratic time; "
        "current state of the art on GraphBench. The go-to fast-classification baseline.",
    ),
    "quadgnn2022": LitEntry(
        "QuadGNN: Accurate Quadratic-Time Message Passing",
        datetime.date(2022, 1, 1),
        "ICLR",
        "High-accuracy but O(n^2) message-passing GNN; strong but slow on GraphBench.",
    ),
    "graphtransformer2023": LitEntry(
        "Graph Transformers at Scale",
        datetime.date(2023, 1, 1),
        "NeurIPS",
        "Attention over graphs; accurate but quadratic, motivating faster methods.",
    ),
}


# ----------------------------------------------------------------------------
# Retrieval backends. ``search`` delegates to whichever backend main() selects
# ----------------------------------------------------------------------------


def _corpus_search(query: str, limit: int) -> dict[str, LitEntry]:
    """Keyword match against the in-memory LITERATURE corpus (offline, deterministic)."""
    terms = query.lower().split()
    hits = {
        key: entry
        for key, entry in LITERATURE.items()
        if any(t in f"{key} {entry.title} {entry.abstract}".lower() for t in terms)
    }
    return dict(list(hits.items())[:limit])


def _semanticscholar_search(
    query: str,
    limit: int,
    fields: tuple[str, ...] = ("title", "abstract", "year", "venue"),
) -> dict[str, LitEntry]:
    """Live search via the Semantic Scholar Graph API; each paper's stable
    ``paperId`` becomes its citation key. Set SEMANTIC_SCHOLAR_API_KEY to raise the
    rate limit -- the endpoint also works unauthenticated, just slower."""
    headers = {"User-Agent": "effectful-example/1.0"}
    if api_key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = api_key
    resp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": limit, "fields": ",".join(fields)},
        headers=headers,
        timeout=20,
    )
    resp.raise_for_status()  # a 429/5xx surfaces as a tool error the model retries around
    out: dict[str, LitEntry] = {}
    for p in resp.json().get("data", []):
        if not (pid := p.get("paperId")):
            continue
        year = p.get("year")
        out[pid] = LitEntry(
            title=p.get("title") or "(untitled)",
            date=datetime.date(year, 1, 1) if year else datetime.date.min,
            venue=p.get("venue") or "",
            abstract=(p.get("abstract") or "")[:600],  # S2 abstracts are often null
        )
    return out


# Selected by main(); the search tool reads it at call time.
SEARCH_BACKEND: collections.abc.Callable[[str, int], dict[str, LitEntry]] = (
    _corpus_search
)


# ---------------------------------------------------------------------------
# Structured types crossing the model boundary
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass
class PaperSummary:
    """The Summary Agent's internal compression (the paper's ``S-hat``): dense
    submission text reduced to what a reviewer reasons over."""

    title: str
    core_claims: list[str]
    method: str
    evidence: str


@pydantic.dataclasses.dataclass(frozen=True)
class MissingBaseline:
    """A prior method the submission should have compared against but did not.

    ``paper_key`` MUST be a paper ``search`` actually returned this run (recorded
    in ``RETRIEVED``) or the finding is rejected at decode time as a hallucinated
    omission -- so the Scout can only accuse the authors of skipping work it truly
    retrieved. This is an addition beyond the paper, in the spirit of its aim to
    ground critiques in verified flaws, enforced by construction.
    """

    method: str
    benchmark: str
    paper_key: str = dataclasses.field(
        metadata={
            "description": "A citation key for a paper ``search`` returned (a corpus "
            "key or a Semantic Scholar paperId); a key not among the retrieved "
            "papers is rejected at decode time."
        }
    )
    reason: str = dataclasses.field(
        metadata={
            "description": "Why this omitted comparison matters -- what the missing "
            "baseline would have tested that the submission leaves unchecked."
        }
    )


@pydantic.dataclasses.dataclass(frozen=True)
class Question:
    """A probing question targeting one review aspect."""

    aspect: typing.Literal["novelty", "soundness"]
    text: str


@pydantic.dataclasses.dataclass(frozen=True)
class Interrogation:
    """One entry of the interrogation log: a claim self-answered, then verified
    against the domain narrative. ``discrepancy`` is empty when they agree."""

    question: str
    answer: str
    verification: str
    discrepancy: str = dataclasses.field(
        metadata={
            "description": "Where the paper's self-answer diverges from the domain "
            "narrative; empty when they agree."
        }
    )


class Recommendation(enum.StrEnum):
    REJECT = "reject"
    WEAK_REJECT = "weak reject"
    WEAK_ACCEPT = "weak accept"
    ACCEPT = "accept"


@pydantic.dataclasses.dataclass
class Review:
    """The final review, formatted to a venue's standards."""

    summary: str
    strengths: list[str]
    weaknesses: list[str]
    questions: list[str]
    suggestions: list[str] = dataclasses.field(
        metadata={"description": "Concrete, actionable improvements for the authors."}
    )
    soundness: Score
    novelty: Score
    significance: Score
    recommendation: Recommendation
    confidence: typing.Literal[1, 2, 3, 4, 5]

    def __str__(self) -> str:
        """Render the structured review to a conference-style report body. The venue
        is runtime context, not review data, so the caller prints the header."""
        lines = [
            f"**Recommendation:** {self.recommendation.value}  "
            f"(confidence {self.confidence}/5)",
            f"**Scores:** soundness {self.soundness}/10 · "
            f"novelty {self.novelty}/10 · significance {self.significance}/10",
            "",
            "## Summary",
            self.summary,
            "",
            "## Strengths",
            *(f"- {s}" for s in self.strengths),
            "",
            "## Weaknesses",
            *(f"- {w}" for w in self.weaknesses),
            "",
            "## Questions",
            *(f"- {q}" for q in self.questions),
            "",
            "## Suggestions",
            *(f"- {s}" for s in self.suggestions),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stream 1a -- internal compression. A toolless Agent: nothing in its scope is a
# Tool, so it is closed-book by construction, no "do not use tools" needed.
# ---------------------------------------------------------------------------


class Summarizer(Agent):
    """You are the Summary Agent. You compress a dense submission into the
    review-oriented structure a reviewer actually reasons over, mitigating the
    "lost in the middle" effect by keeping claims, method, and evidence and
    dropping prose."""

    @Skill.define
    def summarize(self, paper_text: str) -> PaperSummary:
        """Compress this submission into a structured summary: its core claims,
        its method, and the evidence it reports.

        <submission>
        {paper_text}
        </submission>
        """


# ---------------------------------------------------------------------------
# Stream 1b -- context acquisition. The `search` tool lives on this base, so it
# is offered to Scholar subclasses' skills and to no one else.
# ---------------------------------------------------------------------------


class Scholar(Agent):
    """Base for agents that read the literature. The ``search`` tool defined here
    is inherited (via the Agent MRO) by every ``Scholar`` subclass's skills,
    and by nothing else: the closed-book agents hold no ``Scholar`` instance, so
    it never enters their lexical scope. One shared tool, scoped to exactly the
    agents that should search."""

    @Tool.define
    def search(self, query: str, limit: int = 5) -> list[str]:
        """Search the scholarly literature for papers relevant to a query (a
        topic, method, or benchmark name). Returns matching entries, each tagged
        with a citation key you can cite as ``paper_key``."""
        found = SEARCH_BACKEND(query, limit)
        results = [
            f"[{key}] {e.title} ({e.venue} {e.date.year}) -- {e.abstract}"
            for key, e in found.items()
        ]
        return results or [f"No papers found for {query!r}; try broader terms."]


class Historian(Scholar):
    """You are the Sub-Domain Historian. You retrieve prior work and compress it
    into a chronological narrative that positions the submission in the arc of
    its field, so significance can be judged against history rather than in a
    vacuum."""

    @Skill.define
    def survey(self, summary: PaperSummary) -> str:
        """Using the search tool, retrieve the relevant prior work for this
        submission -- search more than once to widen coverage (the method, the
        task, the benchmark, concurrent work). Then write a short chronological
        "domain narrative": how the field arrived here, and whether this
        contribution looks incremental or paradigm-shifting against that arc.
        Refer to retrieved papers by their citation key.

        <summary>
        {summary}
        </summary>
        """


class BaselineScout(Scholar):
    """You are the Baseline Scout, an adversarial auditor. You search for the
    state of the art on a submission's benchmarks and report the strong
    comparisons its authors left out."""

    @Skill.define
    def audit(self, summary: PaperSummary) -> list[MissingBaseline]:
        """Identify the submission's task and benchmark, then use the search tool
        to find state-of-the-art methods on that benchmark and closely related
        work. Report every strong baseline the submission should have compared
        against but did not, filling each finding as its schema describes. If the
        comparisons look complete, return an empty list.

        <summary>
        {summary}
        </summary>
        """


class QuestionGenerator(Agent):
    """You are the Question Generator. You turn the gathered context into a few
    sharp, specific probing questions aimed at a submission's weakest points."""

    @Skill.define
    def generate(
        self, summary: PaperSummary, narrative: str, missing: list[MissingBaseline]
    ) -> list[Question]:
        """Given the paper summary, the historian's domain narrative, and the
        baseline scout's findings, write a handful (about four) probing questions
        targeting the paper's weakest points on two aspects: ``novelty`` (does the
        narrative or a missing baseline undercut the claimed contribution?) and
        ``soundness`` (do the reported evidence and method actually support the
        claims?).

        <summary>{summary}</summary>
        <narrative>{narrative}</narrative>
        <missing_baselines>{missing}</missing_baselines>
        """


class AnswerGenerator(Agent):
    """You are the Answer Generator, interrogating one claim like a skeptical
    reviewer: self-answer from the paper, then check that answer against the
    external context and record where they diverge."""

    @Skill.define
    def interrogate(
        self, question: Question, summary: PaperSummary, narrative: str
    ) -> Interrogation:
        """First self-answer the question from the paper summary alone. Then
        verify that answer against the domain narrative (the external context),
        recording where they diverge as the ``discrepancy`` field's schema
        describes. Be concrete; ground any doubt in the narrative, not in generic
        worry.

        <question>{question}</question>
        <summary>{summary}</summary>
        <narrative>{narrative}</narrative>
        """


# ---------------------------------------------------------------------------
# Synthesis -- guidelines-driven Review Generator
# ---------------------------------------------------------------------------

GUIDELINES: dict[str, str] = {
    "ICLR": (
        "ICLR values novelty and significance. Weight the contribution's "
        "originality against the field's trajectory most heavily; an incremental "
        "delta over existing work is grounds for rejection even if technically "
        "sound."
    ),
    "NeurIPS": (
        "NeurIPS values technical rigor. Weight correctness, complete and fair "
        "baseline comparisons, and statistical significance (variance, error "
        "bars) most heavily; missing baselines or unsupported numbers are grounds "
        "for rejection even if the idea is novel."
    ),
}


@dataclasses.dataclass
class Reviewer(Agent):
    """You are the Review Generator. ``guidelines`` decouples investigation from
    reporting: you write up the same gathered evidence under whichever venue's
    emphasis is in scope, so swapping the venue reweights the review without
    re-running the pipeline."""

    guidelines: str

    @Skill.define
    def write_review(
        self,
        summary: PaperSummary,
        narrative: str,
        missing: list[MissingBaseline],
        interrogation_log: list[Interrogation],
    ) -> Review:
        """Write the final peer review, grounding every strength and weakness in
        the evidence gathered by the pipeline: the domain narrative, the scout's
        missing baselines, and above all the interrogation log's recorded
        discrepancies. Do not raise generic concerns; cite the specific verified
        flaw. Fill each field as its schema describes.

        Follow this venue's guidelines, which set what to weight:
        <guidelines>
        {self.guidelines}
        </guidelines>

        <summary>{summary}</summary>
        <narrative>{narrative}</narrative>
        <missing_baselines>{missing}</missing_baselines>
        <interrogation_log>{interrogation_log}</interrogation_log>
        """


# ---------------------------------------------------------------------------
# The dual-stream pipeline
# ---------------------------------------------------------------------------


async def review(paper_text: str, guidelines: str) -> Review:
    """Acquire context, actively verify it, then synthesize -- the two streams."""
    # Internal compression first: everything downstream reasons over the summary.
    summary = Summarizer().summarize(paper_text)

    # Stream 1 (context acquisition): the two search-enabled agents are
    # independent, so run them concurrently (each drives its own tool-use loop).
    narrative = await asyncio.to_thread(Historian().survey, summary)
    missing = await asyncio.to_thread(BaselineScout().audit, summary)

    # Stream 2 (active verification): generate probing questions, then answer each
    # independently against the narrative -- a fan-out map over the questions. A
    # fresh AnswerGenerator per question keeps their histories from colliding as
    # the calls run concurrently in threads.
    questions = QuestionGenerator().generate(summary, narrative, missing)
    interrogations = await asyncio.gather(
        *(
            asyncio.to_thread(AnswerGenerator().interrogate, q, summary, narrative)
            for q in questions
        )
    )

    # Synthesis: write the review under the venue's guidelines.
    return Reviewer(guidelines).write_review(
        summary, narrative, missing, list(interrogations)
    )


# ---------------------------------------------------------------------------
# Sample submission: sub-quadratic graph classification that (deliberately)
# overclaims novelty and omits the obvious fast baseline -- flaws the pipeline's
# context (RandProp, 2021) is meant to surface.
# ---------------------------------------------------------------------------

SUBMISSION = """\
Title: LinearGraphNet: The First Sub-Quadratic Method for Graph Classification

Abstract. We introduce LinearGraphNet, the first graph classifier to run in
sub-quadratic time, using a novel spectral-sketching layer. On the GraphBench
suite LinearGraphNet reaches 82.4% accuracy, beating the quadratic-time QuadGNN
(81.9%) while being an order of magnitude faster.

Method. We approximate the graph's spectral convolution with a randomized sketch
of the Laplacian, avoiding the full O(n^2) eigendecomposition and yielding an
O(n log n) forward pass. This is the first application of sketching to graph
classification.

Experiments. We report a single accuracy number per dataset on GraphBench,
comparing only against QuadGNN. LinearGraphNet is faster and slightly more
accurate, establishing a new state of the art.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venue",
        type=str,
        choices=list(GUIDELINES),
        default="NeurIPS",
        help="Which venue's guidelines to weight the review by",
    )
    parser.add_argument(
        "--submission",
        type=str,
        default=SUBMISSION,
        help="The submission text to review",
    )
    parser.add_argument(
        "--source",
        choices=["corpus", "semanticscholar"],
        default="corpus",
        help="Literature backend: the offline corpus (default, deterministic) or "
        "the live Semantic Scholar API (set SEMANTIC_SCHOLAR_API_KEY to raise limits)",
    )
    args = parser.parse_args()

    if args.source == "semanticscholar":
        global SEARCH_BACKEND
        SEARCH_BACKEND = _semanticscholar_search

    paper_review = asyncio.run(review(args.submission, GUIDELINES[args.venue]))
    print(f"\n# Review ({args.venue})\n\n{paper_review}")


if __name__ == "__main__":
    main()
