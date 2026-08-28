"""PaperOrchestra: raw materials to a submission-ready manuscript, as a pipeline.

Implements the core of "PaperOrchestra: A Multi-Agent Framework for Automated AI
Research Paper Writing" (arXiv:2604.05018). The paper's diagnosis is that existing
autonomous writers are *rigidly coupled to their own experimental loops* -- they
cannot take a human's unstructured pre-writing materials and draft from them --
and that, relying on keyword search, they "produce superficial literature reviews
with insufficient citations." Its fix is to treat writing as an *orchestration*
problem: one agent first synthesizes the materials into a structured outline (the
"score"), and that outline then drives a fan-out of specialists -- a plotter, a
literature reviewer, a section writer -- whose assembled draft is finally
hill-climbed against a simulated reviewer. Each of the paper's named agents becomes
one ``Agent`` here, and the five-step architecture falls out of ordinary effectful
idioms:

  * The outline *is* the orchestration. The Outline Agent emits a typed ``Outline``
    -- a visualization plan, a targeted literature-search strategy, and a
    section-level writing plan -- and every downstream Skill is parameterized by
    it. Coordination lives in a piece of structured data passed between agents, not
    in prose instructions or a control-flow-heavy conductor; the pipeline is a
    handful of ordinary calls threading that plan through.

  * Steps 2 and 3 run concurrently. Plotting and literature review are independent
    given the outline, so they run as two streams via ``asyncio.gather`` +
    ``asyncio.to_thread`` (each drives its own work), exactly the parallel-streams
    shape of ``scholar_peer.py``.

  * Grounded citations by construction, with a temporal cutoff. The Literature
    Review Agent's ``Identify -> Verify`` loop (web search proposes, a Semantic
    Scholar lookup authenticates) ends in a ``Citation`` that certifies *at decode
    time* both that its key resolves to a real indexed paper and that the paper
    predates the venue's cutoff. A hallucinated reference or a leaked
    future-dated one is not a well-typed ``Citation``; it raises, and the harness's
    ``TenacityRetryer`` feeds the error back -- the same decode-time certification
    ``scientist_one.py`` uses for citations, plus the paper's anti-leakage cutoff.

  * Tools are scoped by class. Only the Literature Review Agent holds the
    ``web_search`` Tool; the Outline, Plotting, Section, and
    Refinement agents define no Tool at all and are closed-book by
    construction -- no "do not search" instruction needed, because nothing in their
    lexical scope is a Tool (the encapsulation idiom of ``scholar_peer.py``).

  * Accept-or-revert hill climbing. The Content Refinement Agent optimizes against
    an ``AgentReview`` LLM judge under the paper's exact rule: keep a revision only
    if it raises the overall score, or ties it with a non-negative net sub-axis
    gain; otherwise revert to the previous version and halt. Monotone improvement
    as a plain Python loop over Skill calls -- distinct from the boolean-accept
    refinement loop of ``research_agent.py`` in that it keeps the *best* draft and
    stops the moment a revision fails to earn its place.

Demonstrates:
- A typed *plan* (the ``Outline``) emitted by one agent that parameterizes every
  downstream Skill -- orchestration encoded as data threaded between agents
- Two independent streams run concurrently (plotting || literature review) via
  ``asyncio.gather`` + ``asyncio.to_thread``
- Decode-time certification of a ``Citation`` against a ground-truth index *and* a
  temporal cutoff, so ``TenacityRetryer`` turns a fabricated or leaked reference
  into a correction (Identify -> Verify, grounded by construction)
- Class-scoped search Tools: only one agent can search; the writing agents are
  closed-book by construction, no instruction required
- An accept-or-revert hill-climbing loop against an LLM reviewer that keeps the
  best draft and halts on the first non-improving revision
"""

# Simplifications vs. the source:
# - Static index, not live search. PaperOrchestra's Literature Review Agent hits a
#   live LLM web search and the real Semantic Scholar API; here ``web_search`` and
#   ``Citation.__post_init__`` both read a tiny in-memory INDEX, so a retrieved paper
#   has a stable key a Citation can certify against. The authentication half of the
#   loop is therefore a decode-time check on the type rather than a second Tool call.
#   This shows the Identify->Verify *shape*, not real
#   retrieval, and the anti-leakage cutoff -- a real ``datetime.date`` submission
#   deadline the Citation checks against -- filters a planted future-dated entry
#   rather than genuinely unseen work. The paper's Semantic Scholar ID dedup and its
#   auto-generated BibTeX (.bib) registry collapse to a keyed ``list[Citation]``.
# - No pixels, no VLM. PaperBanana's closed-loop visual refinement (a VLM critic
#   scoring rendered images and regenerating them) becomes a single structured call:
#   the Plotting Agent emits self-contained LaTeX figure stubs (a caption + body)
#   from the visualization plan and the experimental log. The manuscript integrates
#   them as text; nothing is rendered.
# - Numbers are not re-certified. The Section Writer builds tables from the
#   experimental log as prose; unlike ``scientist_one.py`` there is no NumericalClaim
#   re-run of an evaluator (that example owns that idiom), so table values are
#   trusted rather than reproduced.
# - AgentReview is one LLM judge, not the full peer-review simulation, and the
#   pipeline emits a structured manuscript rather than compiling real LaTeX to PDF.
#   The template T and pre-existing figures F are elided: a ``Venue`` supplies only
#   guidelines and the cutoff, not a real conference LaTeX template to fill.
# - No evaluation. PaperWritingBench (200 papers) and the autorater suite (Citation
#   F1 over P0/P1, the multi-axis lit-review judge, the AI-Scientist-v2 / ScholarPeer
#   reviewers, SxS and human studies) are all out of scope; this composes one
#   manuscript from one planted submission, as the sibling examples also do.

import argparse
import asyncio
import contextvars
import dataclasses
import datetime
import typing

import pydantic

from effectful.handlers.llm import Skill, Tool

type Score = typing.Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# A field's ``metadata={"description": ...}`` is inlined by pydantic into that
# field's JSON schema, which the harness renders into the system prompt as part of
# a skill's argument (and structured-output) spec. So per-field guidance reaches
# the model *through the type* -- used below only where the field name and type
# don't already say it, so no prompt has to repeat it.


# ---------------------------------------------------------------------------
# The literature index -- the ground truth every citation is certified against.
# In PaperOrchestra this is the live web + Semantic Scholar; here it is a small
# keyed corpus so a cited paper has a stable key and a publication date the cutoff
# can test.
# Two entries are traps: ``hyperattn2026`` postdates every venue cutoff (a leakage
# test), and any key not in this dict is a hallucination.
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class IndexedPaper:
    title: str
    date: datetime.date
    venue: str
    abstract: str


INDEX: dict[str, IndexedPaper] = {
    "attention2017": IndexedPaper(
        "Attention Is All You Need",
        datetime.date(2017, 6, 12),
        "NeurIPS",
        "Introduces the Transformer; self-attention is O(n^2) in sequence length n, "
        "the quadratic cost every efficient-attention method sets out to reduce.",
    ),
    "longformer2020": IndexedPaper(
        "Longformer: The Long-Document Transformer",
        datetime.date(2020, 4, 10),
        "arXiv",
        "Sparse local+global attention scaling linearly with sequence length for "
        "long documents.",
    ),
    "linformer2020": IndexedPaper(
        "Linformer: Self-Attention with Linear Complexity",
        datetime.date(2020, 6, 8),
        "arXiv",
        "Low-rank projection of keys and values gives linear-time, linear-memory "
        "attention -- prior work on linear attention.",
    ),
    "performer2021": IndexedPaper(
        "Rethinking Attention with Performers",
        datetime.date(2021, 3, 9),
        "ICLR",
        "FAVOR+ approximates softmax attention with random features in linear time; "
        "a canonical linear-attention baseline.",
    ),
    "flashattention2022": IndexedPaper(
        "FlashAttention: Fast and Memory-Efficient Exact Attention",
        datetime.date(2022, 5, 27),
        "NeurIPS",
        "IO-aware exact attention; the standard strong efficiency baseline for "
        "long-context training and inference.",
    ),
    "retnet2023": IndexedPaper(
        "Retentive Network: A Successor to Transformer",
        datetime.date(2023, 7, 17),
        "arXiv",
        "A retention mechanism with a parallel form for training and a recurrent "
        "form for O(1)-per-step inference; the direct methodological ancestor of "
        "block-recurrent retention.",
    ),
    "mamba2023": IndexedPaper(
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        datetime.date(2023, 12, 1),
        "arXiv",
        "Selective state-space model with linear-time long-context modeling; a "
        "leading efficient-attention competitor.",
    ),
    "longbench2023": IndexedPaper(
        "LongBench: A Bilingual, Multitask Benchmark for Long-Context Understanding",
        datetime.date(2023, 8, 28),
        "arXiv",
        "A standard long-context evaluation suite reporting per-task scores; the "
        "benchmark this submission's numbers are measured on.",
    ),
    "hyperattn2026": IndexedPaper(
        "HyperAttention: Near-Linear Attention at Scale",
        datetime.date(2026, 1, 22),
        "ICLR",
        "A 2026 near-linear attention method -- postdates the 2025 venue cutoffs, "
        "so citing it would leak future work.",
    ),
}


# ---------------------------------------------------------------------------
# Inputs: the unstructured pre-writing materials, and the venue (which fixes the
# guidelines and the temporal cutoff). In the paper these are I, E, T, G, F.
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class RawMaterials:
    """The pre-writing bundle W maps to a manuscript: a sparse idea summary (I) and
    a de-contextualized experimental log (E). The LaTeX template (T) and figures (F)
    are elided; the guidelines (G) and cutoff come from the venue."""

    idea_summary: str
    experimental_log: str


@pydantic.dataclasses.dataclass(frozen=True)
class Venue:
    name: str
    guidelines: str
    # Citations must predate the submission deadline (strictly) -- the anti-leakage
    # cutoff. A date, not a year, so it is a real deadline the model can be held to.
    cutoff: datetime.date


VENUES: dict[str, Venue] = {
    "ICLR": Venue(
        "ICLR 2025",
        "ICLR values novelty and clear positioning against prior work. Weight "
        "originality and honest placement in the literature most heavily; an "
        "overclaimed contribution that ignores close prior art is a rejection.",
        cutoff=datetime.date(2024, 10, 1),
    ),
    "CVPR": Venue(
        "CVPR 2025",
        "CVPR values technical rigor and complete comparison. Weight soundness, "
        "fair baselines, and presentation most heavily; missing comparisons or "
        "unsupported numbers are grounds for rejection.",
        cutoff=datetime.date(2024, 11, 15),
    ),
}


# The venue cutoff date for the manuscript currently under composition. ``Citation``
# reads it in __post_init__, exactly as ``scientist_one``'s claims read the
# ``WORKSPACE`` bundle -- through a ContextVar rather than a bare global, so the
# binding is scoped to the pipeline (set/reset in ``compose``) and safe under the
# concurrent skill calls that plotting and literature review make.
CUTOFF: contextvars.ContextVar[datetime.date] = contextvars.ContextVar("CUTOFF")


# ---------------------------------------------------------------------------
# The Outline -- the "score" the whole orchestra plays from. One structured value,
# emitted by Step 1, that parameterizes every downstream Skill.
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class FigurePlan:
    """One entry of the visualization plan: a ``plot`` of the log's numbers or a
    conceptual ``diagram`` of the method."""

    figure_id: str
    kind: typing.Literal["plot", "diagram"]
    intent: str
    data_source: str = dataclasses.field(
        metadata={
            "description": "For a plot, the part of the experimental log whose "
            "numbers it draws from; empty for a diagram."
        }
    )


@pydantic.dataclasses.dataclass(frozen=True)
class SearchStrategy:
    """The targeted literature-search strategy."""

    macro_context: list[str] = dataclasses.field(
        metadata={"description": "Broad themes that frame the Introduction."}
    )
    method_clusters: list[str] = dataclasses.field(
        metadata={
            "description": "Specific method families and baselines to search for and "
            "position Related Work against."
        }
    )


@pydantic.dataclasses.dataclass(frozen=True)
class SectionPlan:
    """A section's writing plan."""

    section: str
    bullets: list[str]
    citation_hints: list[str] = dataclasses.field(
        metadata={
            "description": "Baselines, datasets, and metrics this section must cite."
        }
    )


@pydantic.dataclasses.dataclass(frozen=True)
class Outline:
    """The paper's JSON outline"""

    title: str
    figures: list[FigurePlan]
    search: SearchStrategy
    sections: list[SectionPlan]


# Every field below that carries LaTeX carries this too. The harness answers a
# Skill by having the model write a Python function body, so a control sequence in
# an ordinary string literal is escape-processed before it is ever a value:
# ``"\texttt"`` is a TAB followed by ``exttt``, and ``"$3.1\times$"`` loses the
# ``\t`` the same way. The damage is silent -- the manuscript simply comes out with
# a tab in the middle of a word -- so the guidance goes on the fields themselves,
# where the model reads it as part of the schema it is filling.
_LATEX_ESCAPING = (
    "Contains LaTeX control sequences. When writing this as a Python string "
    "literal, use a raw string (r'...') or double every backslash: in an ordinary "
    r"literal ``\texttt`` and ``\times`` collapse to a TAB character."
)


# ---------------------------------------------------------------------------
# Artifacts crossing between agents
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class Citation:
    """A reference the manuscript cites, bound to the claim it supports."""

    key: str = dataclasses.field(
        metadata={
            "description": """
            ``key`` MUST resolve to a real entry in ``INDEX`` *and* the entry must predate
            the venue ``CUTOFF`` date, or the citation is rejected at decode time -- as a
            hallucination (no such paper) or as leakage (future-dated work).
            """
        }
    )
    claim: str

    def __post_init__(self) -> None:
        entry = INDEX.get(self.key)
        if entry is None:
            raise ValueError(
                f"citation {self.key!r} does not resolve to any indexed paper "
                f"(available: {sorted(INDEX)}); cite only papers found via web_search"
            )
        cutoff = CUTOFF.get()
        if entry.date >= cutoff:
            raise ValueError(
                f"citation {self.key!r} is dated {entry.date.isoformat()}, at or "
                f"after the venue cutoff {cutoff.isoformat()}; citing it would leak "
                f"future work"
            )


@pydantic.dataclasses.dataclass(frozen=True)
class RelatedWork:
    """The Literature Review Agent's output: the drafted Introduction and Related
    Work prose, plus the verified citation bank (the paper's .bib)."""

    introduction: str
    related_work: str
    citations: list[Citation]


@pydantic.dataclasses.dataclass(frozen=True)
class Figure:
    """A generated visual the Section Writer embeds. (In the paper, PaperBanana
    renders real images; here the body is LaTeX text.)"""

    figure_id: str = dataclasses.field(
        metadata={"description": "Matches the FigurePlan.figure_id this realizes."}
    )
    caption: str = dataclasses.field(metadata={"description": _LATEX_ESCAPING})
    latex: str = dataclasses.field(
        metadata={
            "description": "Self-contained LaTeX for the figure: a pgfplots axis or "
            "tabular for a plot, TikZ for a diagram. " + _LATEX_ESCAPING
        }
    )


@pydantic.dataclasses.dataclass(frozen=True)
class Section:
    """One body section of the manuscript (Method, Experiments, ...)."""

    name: str
    body: str = dataclasses.field(metadata={"description": _LATEX_ESCAPING})


@pydantic.dataclasses.dataclass
class Manuscript:
    """The assembled paper the refinement loop revises: an abstract, the ordered
    body sections, the figures, and the citation bank. Not frozen -- the Section
    Writer produces one and each accepted revision replaces it wholesale."""

    title: str
    abstract: str
    sections: list[Section]
    figures: list[Figure]
    citations: list[Citation]

    def __str__(self) -> str:
        """Render the manuscript to a readable Markdown report. Citations resolve
        their key against ``INDEX`` for the title/venue/date, so the reference list
        carries the full bibliographic entry, not just the key."""
        lines = [f"# {self.title}", "", "## Abstract", self.abstract]
        for section in self.sections:
            lines += ["", f"## {section.name}", section.body]
        lines += ["", "## Figures"]
        lines += [f"- **{f.figure_id}**: {f.caption}" for f in self.figures]
        lines += ["", "## References"]
        lines += [
            f"- [{c.key}] {INDEX[c.key].title} ({INDEX[c.key].venue} "
            f"{INDEX[c.key].date.year}) -- {c.claim}"
            for c in self.citations
        ]
        return "\n".join(lines)


@pydantic.dataclasses.dataclass(frozen=True)
class Review:
    """AgentReview's verdict: per-axis 1-10 scores, an overall 1-10, and the single
    highest-impact weakness for the next revision to address (the paper's simulated
    peer-review feedback)."""

    soundness: Score
    presentation: Score
    clarity: Score
    contribution: Score
    overall: Score
    weakness: str = dataclasses.field(
        metadata={
            "description": "The single highest-impact weakness for the next revision "
            "to fix -- specific and grounded in the manuscript."
        }
    )

    @property
    def sub_total(self) -> int:
        """Sum of the per-axis scores -- the tie-breaker when two overall scores are
        equal. The sub-axes are every ``Review`` field except the ``overall`` score
        itself and the written ``weakness``, read off the dataclass so adding an axis to
        ``Review`` extends the tie-breaker automatically."""
        return sum(
            getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("overall", "weakness")
        )

    def __str__(self) -> str:
        """One-line score summary."""
        return (
            f"**Final review:** overall {self.overall}/10 · soundness "
            f"{self.soundness} · presentation {self.presentation} · clarity "
            f"{self.clarity} · contribution {self.contribution}"
        )


class OutlineAgent:
    """You are the Outline Agent that opens the pipeline. You synthesize
    unstructured pre-writing materials into one structured outline that every other
    agent will play from: a visualization plan, a targeted literature-search
    strategy, and a section-level writing plan."""

    @Skill.define
    def plan(self, materials: RawMaterials) -> Outline:
        """Read the pre-writing materials and produce the ``Outline`` that drives the
        rest of the pipeline: a visualization plan, a targeted literature-search
        strategy, and a section-level writing plan. Fill each field as its schema
        describes.

        <materials>{materials}</materials>
        """


class LiteratureReviewAgent:
    """You are the Literature Review Agent. You run a two-move discovery loop --
    *identify* candidate prior work with web search, and *verify* it by citing it:
    every ``Citation`` you build is authenticated as it is constructed, and one that
    names no indexed paper, or one published on or after the cutoff, is rejected and
    returned to you. You draft the Introduction and Related Work grounded in verified
    references, not keyword-matched guesses."""

    @Tool.define
    def web_search(self, query: str) -> list[IndexedPaper]:
        """Identify prior work: search the literature for papers relevant
        to a query (a method family, task, or benchmark)."""
        terms = query.lower().split()
        hits = [
            e
            for key, e in INDEX.items()
            if any(t in f"{key} {e.title} {e.abstract}".lower() for t in terms)
        ]
        return hits

    @Skill.define
    def review(self, outline: Outline, cutoff: datetime.date) -> RelatedWork:
        """Execute the outline's search strategy: for each theme and method cluster,
        use ``web_search`` to identify candidate prior work. Then draft the Introduction
        and Related Work, positioning the contribution honestly against the verified
        prior work, and collect every ``Citation`` into the bank.

        The cutoff is {cutoff}: cite only papers published strictly before it (see
        the ``Citation`` type for the grounding rule it is checked against).

        <outline>{outline}</outline>
        """


class PlottingAgent:
    """You are the Plotting Agent. You execute a visualization plan, turning each
    planned figure into a self-contained LaTeX figure with a context-aware caption:
    statistical plots grounded in the experimental log's numbers, and conceptual
    diagrams that convey the method."""

    @Skill.define
    def draw(self, figures: list[FigurePlan], experimental_log: str) -> list[Figure]:
        """Produce one ``Figure`` per plan entry, realizing each ``FigurePlan``: a
        statistical plot of the numbers named in its ``data_source``, or a conceptual
        diagram of the method. Fill each ``Figure`` field as its schema describes.

        <figure_plans>{figures}</figure_plans>

        <experimental_log>
        {experimental_log}
        </experimental_log>
        """


class SectionWriter:
    """You are the Section Writing Agent. You draft the remaining core sections on
    top of the literature reviewer's Introduction and Related Work, build tables
    from the experimental log, integrate the generated figures, and assemble a
    coherent full manuscript."""

    @Skill.define
    def write(
        self,
        outline: Outline,
        materials: RawMaterials,
        related: RelatedWork,
        figures: list[Figure],
    ) -> Manuscript:
        """Write the complete manuscript. Start from the reviewer's Introduction and
        Related Work, then draft the sections in the outline's writing plan (Method,
        Experiments, Conclusion, ...) following their bullets. Build the experiments
        tables from the experimental log's numbers, and reference each generated
        figure by its ``figure_id`` where the writing plan calls for it. Carry the
        reviewer's citation bank through unchanged.

        <outline>{outline}</outline>
        <materials>{materials}</materials>
        <related_work>{related}</related_work>
        <figures>{figures}</figures>
        """


@dataclasses.dataclass
class Reviewer:
    """You are AgentReview, a simulated peer reviewer who scores one manuscript on
    its own merits. A method on an ``Agent`` rather than a module-level Skill: a
    module-level ``@Skill.define`` lands in every other skill's lexical scope
    and is offered to those agents as a callable tool, but a Skill *method* is
    reached only through its own class, so the writing agents never see it. The
    ``refine`` loop makes a fresh instance per call, so the judge stays stateless --
    no memory of earlier verdicts to anchor the score the accept/revert rule reads."""

    guidelines: str

    @Skill.define
    def review(self, manuscript: Manuscript) -> Review:
        """Score this manuscript and name the single highest-impact weakness for the
        next revision to fix, filling the ``Review`` as its schema describes. Ground
        every score and the weakness in the manuscript.

        Judge under this venue's guidelines:
        <guidelines>{self.guidelines}</guidelines>

        <manuscript>{manuscript}</manuscript>
        """


class ContentRefiner:
    """You are the Content Refinement Agent. Given a reviewer's verdict, you revise
    the manuscript to address the one named weakness -- and only that -- changing as
    little else as possible so the revision is a targeted improvement, not a
    rewrite."""

    @Skill.define
    def revise(self, manuscript: Manuscript, review: Review) -> Manuscript:
        """Return a revised manuscript that fixes the reviewer's named weakness and
        nothing else: preserve everything the reviewer did not fault, keep the
        citation bank grounded (cite only verified, in-cutoff papers), and make the
        smallest change that resolves the weakness.

        <review>{review}</review>

        <manuscript>{manuscript}</manuscript>
        """


def refine(
    draft: Manuscript, guidelines: str, *, max_iters: int
) -> tuple[Manuscript, Review, list[Review]]:
    """Hill-climb the draft against AgentReview: propose a revision, re-score, keep
    it only if it earns its place, else revert to the last accepted version and
    halt. Returns the best manuscript, its review, and the score trace -- every
    review taken along the way, starting with the draft's, so the caller can see
    the climb (and the one rejected step that ends it)."""
    manuscript = draft
    # A fresh Reviewer per call keeps the judge stateless: every version is scored
    # independently, which is what the accept/revert comparison relies on. (The
    # refiner, by contrast, is reused, so it remembers what it already tried.)
    review = Reviewer(guidelines).review(manuscript)
    refiner = ContentRefiner()
    trace = [review]

    for i in range(max_iters):
        candidate = refiner.revise(manuscript, review)
        candidate_review = Reviewer(guidelines).review(candidate)
        trace.append(candidate_review)
        if review.overall < candidate_review.overall or (
            review.overall == candidate_review.overall
            and review.sub_total < candidate_review.sub_total
        ):
            manuscript, review = candidate, candidate_review
        else:
            break

    return manuscript, review, trace


async def _write(
    materials: RawMaterials, venue: Venue, *, max_iters: int
) -> tuple[Manuscript, Review, list[Review]]:
    """Outline -> (plot || review) -> write -> refine: the five steps."""
    # Step 1: synthesize the materials into the plan the rest of the pipeline plays.
    outline = OutlineAgent().plan(materials)

    # Steps 2 & 3 run concurrently: given the outline, plotting and literature
    # review are independent, each driving its own work (the reviewer its tool loop).
    figures, related = await asyncio.gather(
        asyncio.to_thread(
            PlottingAgent().draw, outline.figures, materials.experimental_log
        ),
        asyncio.to_thread(LiteratureReviewAgent().review, outline, venue.cutoff),
    )

    # Step 4: assemble the full draft from the plan, the lit-review sections, and
    # the figures.
    draft = SectionWriter().write(outline, materials, related, figures)

    # Step 5: hill-climb the draft against the simulated reviewer.
    return refine(draft, venue.guidelines, max_iters=max_iters)


def write(
    materials: RawMaterials, venue: Venue, *, max_iters: int
) -> tuple[Manuscript, Review, list[Review]]:
    """The full pipeline: synthesize the outline, run plotting and literature review
    concurrently, assemble the draft, and hill-climb it against the simulated
    reviewer. Returns the best manuscript, its review, and the score trace.

    This is the synchronous entry point: it owns the ``asyncio.run``, so callers
    (and the doctests) drive the whole pipeline with an ordinary call.
    """
    token = CUTOFF.set(venue.cutoff)
    try:
        return asyncio.run(_write(materials, venue, max_iters=max_iters))
    finally:
        CUTOFF.reset(token)


# ---------------------------------------------------------------------------
# Sample materials: a sparse idea + a de-contextualized experimental log for an
# efficient-attention method that (deliberately) overclaims novelty -- the
# literature reviewer's job is to position it honestly against RetNet, Linformer,
# and Performer, and to resist citing the post-cutoff HyperAttention.
# ---------------------------------------------------------------------------

MATERIALS = RawMaterials(
    idea_summary="""\
We propose BlockRetention, the first linear-time attention mechanism for
long-context language modeling. The core idea is a block-recurrent retention layer:
the sequence is split into fixed blocks, attention runs in full within a block, and
a learned exponential decay carries a compressed state across blocks. This gives
O(n) memory in sequence length n while keeping a parallel training form. We claim
this is the first method to combine intra-block full attention with cross-block
recurrence.""",
    experimental_log="""\
Setup: decoder-only LM, 350M params, trained on 8k-token contexts, evaluated up to
32k on the LongBench suite.
Quality: perplexity 8.9 at 32k context; the FlashAttention baseline reaches 9.4 at
32k; Mamba reaches 9.1.
Efficiency: 3.1x higher decoding throughput than FlashAttention at 32k; peak memory
flat in context length (O(n)), vs. FlashAttention growing linearly in the KV cache.
Ablation: removing the learned cross-block decay raises perplexity from 8.9 to 9.7.
Ablation: block size 256 vs 512 vs 1024 -> perplexity 9.0 / 8.9 / 8.9 (512 chosen).""",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venue",
        type=str,
        choices=list(VENUES),
        default="ICLR",
        help="Which venue fixes the guidelines and the citation cutoff",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="Maximum refinement iterations before the hill-climb halts",
    )
    args = parser.parse_args()

    venue = VENUES[args.venue]
    manuscript, review, trace = write(MATERIALS, venue, max_iters=args.max_iters)
    print(f"\n[refine] overall-score trace: {[r.overall for r in trace]}")
    print(f"\n{review}   (venue: {venue.name})")
    print(f"\n{manuscript}")


if __name__ == "__main__":
    main()
