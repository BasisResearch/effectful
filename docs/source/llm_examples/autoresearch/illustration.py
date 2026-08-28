"""PaperBanana: reference-driven academic illustration as a 5-agent pipeline.

Implements the core of "PaperBanana: Automating Academic Illustration for AI
Scientists" (arXiv:2601.23265). The paper maps a *source context* S (a methodology
or plot description, including the data) and a *communicative intent* C (a figure
caption) to an illustration ``I = f(S, C, E)``, optionally guided by a reference
set E, via five specialized agents in two phases: a *Linear Planning Phase*
(Retriever -> Planner -> Stylist) that synthesizes a stylistically optimized
description P*, and an *Iterative Refinement Loop* (T=3) in which a Visualizer
renders P and a Critic inspects the render and refines it.

We implement PaperBanana's 5-agent architecture on its **code-based
statistical-plot path** (paper Sec. 5.5), which makes the Visualizer<->Critic loop
*real*: the Visualizer writes executable Matplotlib code, ``matplotlib`` renders it
to a PNG, and a vision model critiques that actual PNG against S and C. The raster
methodology-diagram path needs an image-generation model (Nano-Banana-Pro /
GPT-Image) and is out of scope. Each agent falls out of an ordinary effectful idiom:

  * Retriever -- generative retrieval as decode-time certification. A ``Retrieval``
    names keys of exemplars from the fixed reference set R; ``__post_init__`` rejects
    any key that does not resolve in the immutable ``REFERENCES`` constant, so a
    hallucinated selection is fed back by ``TenacityRetryer`` (the certification
    idiom of ``scholar_peer.py``/``scientist_one.py``). Because R is a module
    constant, not per-run mutable state, the check reads it directly -- no ContextVar
    needed. A retrieval ``Tool`` surfaces the candidate metadata as a strong
    ``list[PlotExemplar]``, the way ``scientist_one``'s tools return domain types.

  * Planner -- in-context learning, toolless. It reads the retrieved exemplars and
    transcribes S's data table into a structured, data-bearing ``PlotDescription``,
    so the numbers the Visualizer draws are carried explicitly and stay checkable.

  * Stylist -- a plan threaded through the pipeline. It synthesizes an
    ``AestheticGuideline`` G from R, then restyles P into P* (a ``StyledPlot``
    bundling the data-bearing description with G), exactly the typed-plan-as-
    orchestration shape of ``paper_orchestra.py``'s Outline.

  * Visualizer -- code synthesis that must actually render. Its Skill returns a
    ``Plot`` (a ``Callable`` the harness compiles, as in ``scientist_one``'s
    ``Solver -> Solution``): a pure nullary closure that builds and returns a fresh
    ``Figure`` via matplotlib's object-oriented API. A render-doctest in its docstring
    calls the synthesized ``plot()``, so a plot whose code raises when it runs fails
    the doctest and is fed back by ``TenacityRetryer`` -- "the plot must render" is
    grounding by construction, and the doctest runs the code on a *different* plan,
    forcing it to read its data from the closed-over plan rather than
    hardcode.

  * Critic -- the loop made real, multimodally. It receives the rendered
    ``PIL.Image.Image`` (the image-input idiom of ``image_input.py``), inspects it
    against S and C for factual misalignments and visual glitches, and returns a
    refined ``StyledPlot`` plus the concrete issues it saw. The Visualizer<->Critic
    loop is a plain Python ``for`` loop over these two Skills.

Demonstrates:
- Code synthesis whose return ``Callable`` must render: the model writes a pure
  nullary closure that builds and returns a ``Figure`` via matplotlib's OO API, and a
  doctest turns "the plot actually renders" into a decode-time contract fed back by
  ``TenacityRetryer``
- A real multimodal refinement loop: matplotlib renders a PNG a vision model
  critiques, then the plan is regenerated -- the Visualizer<->Critic loop, not simulated
- Decode-time certification of a retrieval selection against an immutable reference
  set, read directly (no ContextVar) because the set is a module constant
- A typed plan (``StyledPlot``) threaded through the pipeline as orchestration data
- Per-field guidance carried on the types via ``field(metadata={"description": ...})``
"""

# Simplifications vs. the source:
# - The raster methodology-diagram path -- PaperBanana's headline -- is out of scope:
#   it needs an image-generation model (Nano-Banana-Pro / GPT-Image). We implement the
#   paper's own code-based statistical-plot path (Sec. 5.5), where the Visualizer emits
#   Matplotlib and the loop is a real render+critique cycle rather than image gen.
# - Static reference corpus, not live/web-scale retrieval. ``REFERENCES`` is a tiny
#   in-memory set of *textual* structure/style descriptors (no exemplar images), so
#   the Retriever ranks over metadata; this shows the pipeline's shape, not retrieval
#   at scale, and the "prioritize visual structure over topic" instruction is only
#   gestured at without real reference images.
# - One task, not PaperBananaBench. The paper curates 292 evaluation cases; here a
#   single planted illustration task runs end to end, as the sibling examples do.
# - No evaluation. The paper's VLM-as-a-Judge scores a render against a human-drawn
#   figure on four dimensions and aggregates them hierarchically; scoring the pipeline
#   is out of scope here, as it is in the sibling examples. ``refine`` still returns
#   both ends of the paper's Critic-on/off ablation -- the round-0 and final renders --
#   and every round is written to ``outdir``, so the loop's effect can be read off the
#   PNGs directly.

import argparse
import collections.abc
import dataclasses
import pathlib
import tempfile
import typing

import pydantic
from matplotlib.figure import Figure
from PIL import Image

from effectful.handlers.llm import Skill, Tool

# A field's ``metadata={"description": ...}`` is inlined by pydantic into that
# field's JSON schema, which the harness renders into the system prompt as part of
# a skill's argument (and structured-output) spec. So per-field guidance reaches
# the model *through the type* -- used below only where the field name and type
# don't already say it, so no prompt has to repeat it.

type ChartType = typing.Literal[
    "grouped_bar", "stacked_bar", "line", "scatter", "heatmap"
]

# A plotting function the Visualizer writes: a pure nullary closure that builds and
# returns a fresh matplotlib ``Figure`` via the object-oriented API (no pyplot, no
# global figure registry, no side effects). The harness compiles the model's code
# into one of these; the closure captures the round's ``plan``.
type PlottingFn = collections.abc.Callable[[], Figure]


# ---------------------------------------------------------------------------
# The reference set R -- the fixed corpus of exemplars the Retriever ranks over.
# In PaperBanana each exemplar is a triplet (S, C, I) with a real reference image;
# here it is textual structure/style metadata (no images), so an exemplar has a
# stable key a selection can be certified against. R is an immutable module
# constant, so the certification reads it directly (contrast ``scientist_one``,
# whose per-run mutable Workspace needs a ContextVar).
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class PlotExemplar:
    """One reference exemplar: its chart type, research domain, and -- kept separate
    so the Retriever can weight them as the paper does -- its *visual structure*
    (prioritized) versus its *topic* and aesthetic style."""

    key: str
    chart_type: ChartType
    domain: str
    caption: str
    structure_notes: str = dataclasses.field(
        metadata={
            "description": "The visual/structural composition (axes, grouping, marks, "
            "legend, layout) independent of subject matter -- what the Retriever "
            "weights above topic when matching."
        }
    )
    style_notes: str


REFERENCES: dict[str, PlotExemplar] = {
    e.key: e
    for e in [
        PlotExemplar(
            key="grouped_bar_benchmark",
            chart_type="grouped_bar",
            domain="ML benchmarking",
            caption="Accuracy of several methods across benchmarks.",
            structure_notes="Bars clustered by benchmark on the x-axis, one colored "
            "bar per method within each cluster; shared y-axis starting at 0; a legend "
            "keys color to method.",
            style_notes="Categorical palette, one hue per method; light horizontal "
            "gridlines; top and right spines removed; value labels above bars.",
        ),
        PlotExemplar(
            key="line_scaling",
            chart_type="line",
            domain="scaling laws",
            caption="A metric as a function of training scale.",
            structure_notes="Several monotone lines share x (steps/size, often log) "
            "and y (the metric); one line per method, markers at measured points.",
            style_notes="Distinct hue+marker per line; faint grid; legend inside the "
            "plot; no chartjunk.",
        ),
        PlotExemplar(
            key="scatter_tradeoff",
            chart_type="scatter",
            domain="efficiency analysis",
            caption="Accuracy vs. cost trade-off across methods.",
            structure_notes="Points in an x=cost / y=quality plane, one marker per "
            "method; a Pareto frontier implied toward the upper-left.",
            style_notes="One hue per method, labeled points; equal-weight axes; "
            "minimal grid.",
        ),
        PlotExemplar(
            key="heatmap_ablation",
            chart_type="heatmap",
            domain="ablation study",
            caption="A metric over a grid of two design choices.",
            structure_notes="A matrix of cells indexed by two categorical axes, cell "
            "color encoding the metric; a colorbar legend; cells annotated with values.",
            style_notes="Sequential colormap; annotated cells; square aspect.",
        ),
        PlotExemplar(
            key="stacked_bar_composition",
            chart_type="stacked_bar",
            domain="component analysis",
            caption="Contribution of components to a total per setting.",
            structure_notes="One bar per setting on x, segments stacked to a total on "
            "y, each segment a component; a legend keys color to component.",
            style_notes="Sequential/categorical stack palette; legend outside; totals "
            "labeled atop each bar.",
        ),
    ]
}


# ---------------------------------------------------------------------------
# Inputs: the task the illustration must satisfy.
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class IllustrationTask:
    """The task ``(S, C)``: a source context and a communicative intent. ``S``
    embeds the actual data (as a small text table) so faithfulness is checkable;
    ``C`` is the figure caption that fixes the illustration's scope and focus."""

    source_context: str
    intent: str


# ---------------------------------------------------------------------------
# Structured artifacts crossing between agents.
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class Retrieval:
    """The Retriever's selection E: keys of the exemplars from R that best match the
    task by diagram type and domain (visual structure weighted over topic)."""

    selected: list[str] = dataclasses.field(
        metadata={
            "description": "Keys of the chosen exemplars; each MUST resolve in the "
            "reference set R (certified at decode time), so a hallucinated key is "
            "rejected and fed back."
        }
    )
    rationale: str

    def __post_init__(self) -> None:
        unknown = [k for k in self.selected if k not in REFERENCES]
        if unknown:
            raise ValueError(
                f"retrieval selected unknown exemplar keys {unknown}; choose only "
                f"keys that exist in the reference set (available: {sorted(REFERENCES)})"
            )
        if not self.selected:
            raise ValueError("select at least one exemplar from the reference set")


@pydantic.dataclasses.dataclass(frozen=True)
class DataSeries:
    """One data series (a method / line / stack): a name and its numeric values."""

    name: str
    values: list[float] = dataclasses.field(
        metadata={
            "description": "One value per category, in the SAME order as the "
            "description's ``categories`` -- the transcribed numbers from S the plot "
            "must reproduce exactly."
        }
    )


@pydantic.dataclasses.dataclass(frozen=True)
class PlotDescription:
    """The Planner's description P of the target plot: its type, labels, and -- carried
    explicitly so the Visualizer's code stays faithful -- the exact data from S."""

    chart_type: ChartType
    title: str
    x_label: str
    y_label: str
    categories: list[str] = dataclasses.field(
        metadata={
            "description": "The groups along the x-axis (e.g. datasets); the "
            "Visualizer's code iterates these and each series aligns to them by order."
        }
    )
    series: list[DataSeries]
    notes: str


@pydantic.dataclasses.dataclass(frozen=True)
class AestheticGuideline:
    """The Stylist's synthesized guideline G, one directive per aesthetic dimension
    (the paper's palette / shapes / lines / layout / typography / icons), read off R
    and specialized to statistical plots."""

    palette: list[str] = dataclasses.field(
        metadata={
            "description": "Ordered color specs (hex like '#4C72B0' or matplotlib "
            "names), one per series, applied in order."
        }
    )
    marks_and_containers: str
    lines_and_arrows: str
    layout: str
    typography: str
    icons: str = dataclasses.field(
        metadata={
            "description": "Any small glyphs/markers or annotation style; 'none' for a "
            "plain statistical plot."
        }
    )


@pydantic.dataclasses.dataclass(frozen=True)
class StyledPlot:
    """P* -- the plan the Visualizer renders and the Critic refines, threaded through
    the refinement loop. Bundles the data-bearing description with the aesthetic
    guideline G and the concrete directives that restyle P into P*."""

    description: PlotDescription
    guideline: AestheticGuideline
    directives: str = dataclasses.field(
        metadata={
            "description": "Concrete restyling instructions applying G to this "
            "description -- what colors/spines/gridlines/labels the Visualizer should use."
        }
    )

    def __str__(self) -> str:
        """Render the plan to a compact, exact brief -- data first, so the Visualizer
        (and a human) sees the precise numbers it must draw."""
        d = self.description
        lines = [
            f"{d.chart_type} titled {d.title!r}",
            f"  x-axis ({d.x_label}): {d.categories}",
            f"  y-axis: {d.y_label}",
            "  series:",
        ]
        lines += [f"    - {s.name}: {s.values}" for s in d.series]
        g = self.guideline
        lines += [
            f"  planner notes: {d.notes}",
            f"  palette: {g.palette}",
            f"  marks/containers: {g.marks_and_containers}",
            f"  lines/arrows: {g.lines_and_arrows}",
            f"  layout: {g.layout}",
            f"  typography: {g.typography}",
            f"  icons: {g.icons}",
            f"  style directives: {self.directives}",
        ]
        return "\n".join(lines)


@pydantic.dataclasses.dataclass(frozen=True)
class Critique:
    """The Critic's verdict on one rendered plot: the concrete problems it saw and a
    refined plan addressing them."""

    issues: list[str] = dataclasses.field(
        metadata={
            "description": "What the Critic targets in the RENDERED plot, per the "
            "paper: factual misalignments (wrong/missing numbers or labels vs. S and "
            "C), visual glitches, OR areas for improvement (readability/aesthetics). "
            "Leave empty ONLY if the plot is already publication-ready with nothing to "
            "improve."
        }
    )
    refined: StyledPlot


# ---------------------------------------------------------------------------
# The Visualizer's render path: compile the model's code, draw it to a real PNG.
# ---------------------------------------------------------------------------


def render(plot: PlottingFn, path: pathlib.Path) -> Image.Image:
    """Render a plot to a real PNG at ``path`` and load it back as a PIL image -- the
    actual pixels the Critic inspects. A drawing error propagates (the
    Visualizer's render-doctest already guards against non-rendering code).

    ``savefig(bbox_inches="tight")`` trims margins during the save itself -- safe on a
    canvas-less OO figure, unlike a separate ``tight_layout()`` call.
    """
    fig = plot()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return Image.open(path)


# ---------------------------------------------------------------------------
# Agent 1 -- the Retriever. Holds the retrieval Tool (scoped to this class), ranks
# the candidate metadata, and emits a selection certified against R.
# ---------------------------------------------------------------------------


class Retriever:
    """You are the Retriever Agent that opens the pipeline. You perform generative
    retrieval: rank the reference exemplars by how well their *visual structure* and
    research domain match the task -- prioritizing diagram structure over topic
    similarity -- and select the few that will best guide the downstream agents."""

    @Tool.define
    def reference_catalog(self) -> list[PlotExemplar]:
        """Return the full reference set R -- every candidate exemplar's key, chart
        type, domain, caption, and structure/style notes -- to rank over before
        selecting."""
        return list(REFERENCES.values())

    @Skill.define
    def retrieve(self, task: IllustrationTask) -> Retrieval:
        """Inspect the reference set via ``reference_catalog``, then select the two or
        three exemplars whose visual structure and domain best fit the task. Weight
        structural/diagram-type match above topic similarity. Return their keys and a
        one-line rationale; select only keys that exist in R.

        <task>{task}</task>
        """


# ---------------------------------------------------------------------------
# Agent 2 -- the Planner. Toolless: it in-context-learns from the retrieved
# exemplars and turns S + C into a structured, data-bearing description P.
# ---------------------------------------------------------------------------


class Planner:
    """You are the Planner Agent, the cognitive core. By in-context learning from the
    retrieved exemplars, you translate the source context and caption into a detailed,
    structured description of the target plot -- transcribing the data exactly so the
    figure will be faithful."""

    @Skill.define
    def plan(
        self, task: IllustrationTask, exemplars: list[PlotExemplar]
    ) -> PlotDescription:
        """Produce the ``PlotDescription`` for this task, learning the appropriate
        chart type and composition from the retrieved exemplars. Transcribe the data
        table in the source context into ``categories`` and ``series`` exactly -- every
        number must come from S, in order. Fill each field as its schema describes.

        <task>{task}</task>

        <retrieved_exemplars>{exemplars}</retrieved_exemplars>
        """


# ---------------------------------------------------------------------------
# Agent 3 -- the Stylist. Synthesizes the aesthetic guideline G from R, then
# restyles the description P into the stylistically optimized plan P*.
# ---------------------------------------------------------------------------


class Stylist:
    """You are the Stylist Agent, a design consultant. You first distill a reusable
    aesthetic guideline from the reference set, then apply it to restyle the planner's
    description into a publication-quality, stylistically optimized plan."""

    @Skill.define
    def synthesize_guideline(self, exemplars: list[PlotExemplar]) -> AestheticGuideline:
        """Traverse the reference exemplars' style notes and synthesize one reusable
        ``AestheticGuideline`` for academic statistical plots.

        <reference_exemplars>{exemplars}</reference_exemplars>
        """

    @Skill.define
    def restyle(
        self, description: PlotDescription, guideline: AestheticGuideline
    ) -> StyledPlot:
        """Restyle the planner's description into the optimized plan P* by bundling it
        with the aesthetic guideline and writing concrete ``directives`` that apply the
        guideline to this specific plot. Carry the description's data through
        unchanged -- restyling never alters the numbers.

        <description>{description}</description>

        <guideline>{guideline}</guideline>
        """


# ---------------------------------------------------------------------------
# Agent 4 -- the Visualizer. Writes Matplotlib code (a Plot callable); the doctest
# makes "the plot must actually render" a decode-time contract.
# ---------------------------------------------------------------------------


class Visualizer:
    """You are the Visualizer Agent, an expert Matplotlib programmer. You answer by
    writing code: you turn a plan into a function that draws the plot, and the harness
    renders it. You never reason the figure out in prose -- you draw it."""

    @Skill.define
    def visualize(self, plan: StyledPlot) -> PlottingFn:
        """Write ``plot``: a nullary function that BUILDS and RETURNS a fresh
        matplotlib ``Figure`` via the object-oriented API. Inside, do:
        ``fig = Figure(figsize=(8, 5)); ax = fig.subplots()``, draw onto ``ax``, and
        ``return fig``. ``Figure`` is available in scope (or ``from matplotlib.figure
        import Figure`` inside the function). Do NOT use ``pyplot``/``plt`` and do NOT
        call ``savefig`` -- the harness saves the returned figure.

        <plan>
        {plan}
        </plan>

        Read ALL data and labels from the ``plan`` object, which is in scope
        (``plan.description.categories``, ``plan.description.series``, etc.) -- do not
        hardcode values, so the same code draws any plan. Apply the plan's palette and
        style directives.

        Example usage:

        >>> _SMOKE_PLAN = StyledPlot(
        ...     description=PlotDescription(
        ...         chart_type="grouped_bar",
        ...         title="smoke",
        ...         x_label="group",
        ...         y_label="value",
        ...         categories=["p", "q"],
        ...         series=[DataSeries("m1", [1.0, 2.0]), DataSeries("m2", [3.0, 4.0])],
        ...         notes="two groups, two series",
        ...     ),
        ...     guideline=AestheticGuideline(
        ...         palette=["#4C72B0", "#DD8452"],
        ...         marks_and_containers="plain bars",
        ...         lines_and_arrows="none",
        ...         layout="grouped",
        ...         typography="default",
        ...         icons="none",
        ...     ),
        ...     directives="grouped bars, legend, y from 0",
        ... )
        >>> isinstance(Visualizer().visualize(_SMOKE_PLAN)(), Figure)
        True
        """


# ---------------------------------------------------------------------------
# Agent 5 -- the Critic. Sees the rendered PNG and refines the plan. A stateless
# Agent method (a fresh instance per loop iteration), never a module-level Skill.
# ---------------------------------------------------------------------------


class Critic:
    """You are the Critic Agent. You close the refinement loop: you look at the
    actually-rendered plot, judge it against the source context and caption, and hand
    the Visualizer a refined plan that fixes what you saw."""

    @Skill.define
    def critique(
        self, image: Image.Image, task: IllustrationTask, plan: StyledPlot
    ) -> Critique:
        """Here is the plot rendered from the current plan. Inspect the IMAGE against
        the source context S and the caption C. Following the paper, target three
        things: (1) factual misalignments -- are the numbers, categories, and labels
        correct and complete vs. S and C?; (2) visual glitches -- overlap, clipping,
        missing legend, unreadable text, clutter; and (3) areas for improvement --
        concrete readability/aesthetic upgrades even when nothing is strictly wrong
        (clearer emphasis of the proposed method, better label/legend placement,
        gridline and spine styling, value labels, headroom). List what you actually
        see and return a refined plan that applies it, always keeping the data true to
        S. Leave issues empty only if the plot is already publication-ready.

        <rendered_plot>{image}</rendered_plot>

        <task>{task}</task>

        <current_plan>
        {plan}
        </current_plan>
        """


# ---------------------------------------------------------------------------
# The iterative refinement loop -- the Visualizer<->Critic cycle, made real.
# ---------------------------------------------------------------------------


def refine(
    task: IllustrationTask,
    plan: StyledPlot,
    *,
    max_iter: int,
    outdir: pathlib.Path,
) -> tuple[Image.Image, Image.Image, StyledPlot]:
    """Run the T-round Visualizer<->Critic loop. I_0 = render(P*); each round the
    Critic inspects the current render and refines the plan, which the Visualizer
    re-renders (final output I_T). Returns (round-0 image, final image, final plan).

    Runs the paper's *fixed* T rounds: the Critic always emits a refined
    description P_{t+1}, and it is always re-rendered. Halting early on an empty
    ``issues`` list would make the demonstration conditional on the Critic's
    mood -- a model that reports no issues on I_0 (the common case on a simple
    plot) would leave ``round_1.png`` unwritten and the refinement loop, the
    thing this function exists to show, entirely unexercised.

    A fresh Visualizer/Critic per iteration keeps them stateless, so each render is
    judged on its own (nothing anchors on an earlier round's verdict).
    """
    round0 = img = render(Visualizer().visualize(plan), outdir / "round_0.png")
    for t in range(max_iter):
        critique = Critic().critique(img, task, plan)
        plan = critique.refined
        img = render(Visualizer().visualize(plan), outdir / f"round_{t + 1}.png")

    return round0, img, plan


# ---------------------------------------------------------------------------
# The pipeline -- the two phases threaded together.
# ---------------------------------------------------------------------------


def illustrate(
    task: IllustrationTask,
    *,
    max_iter: int,
    outdir: pathlib.Path,
) -> StyledPlot:
    """Linear Planning Phase (Retriever -> Planner -> Stylist) then the Iterative
    Refinement Loop (Visualizer <-> Critic). Returns the final plan; both rounds'
    renders are left on disk under ``outdir``."""
    # Linear Planning Phase. The Retriever certifies its selection as *keys*, so
    # resolve them against R here -- the Planner learns from the exemplars'
    # structure and style notes, which a bare key does not carry.
    retrieval = Retriever().retrieve(task)
    exemplars = [REFERENCES[key] for key in retrieval.selected]
    description = Planner().plan(task, exemplars)

    stylist = Stylist()
    guideline = stylist.synthesize_guideline(list(REFERENCES.values()))
    p_star = stylist.restyle(description, guideline)

    # Iterative Refinement Loop: the real render+critique cycle.
    _round0, _final, plan = refine(task, p_star, max_iter=max_iter, outdir=outdir)
    return plan


# ---------------------------------------------------------------------------
# Demo task: a grouped-bar comparison whose data lives in S, so faithfulness is crisp.
# ---------------------------------------------------------------------------

DEMO_TASK = IllustrationTask(
    source_context="""\
We evaluate three methods -- Baseline, Ours, and Ours+Aug -- on three image
classification benchmarks, reporting top-1 accuracy (%). The measured results:

  Method     | CIFAR-10 | SVHN  | STL-10
  -----------+----------+-------+-------
  Baseline   |   71.2   | 88.4  |  64.9
  Ours       |   78.5   | 91.2  |  70.3
  Ours+Aug   |   82.1   | 92.8  |  73.6

Ours improves over Baseline on every benchmark, and adding augmentation (Ours+Aug)
improves further; the ordering Baseline < Ours < Ours+Aug holds on all three.""",
    intent="Overall comparison of the three methods across the three benchmarks.",
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Visualizer<->Critic refinement rounds (the paper's fixed T=3)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for rendered PNGs (defaults to a fresh temp dir)",
    )
    args = parser.parse_args()

    outdir = (
        pathlib.Path(args.outdir)
        if args.outdir is not None
        else pathlib.Path(tempfile.mkdtemp(prefix="paperbanana_"))
    )
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Task caption: {DEMO_TASK.intent}")

    plan = illustrate(DEMO_TASK, max_iter=args.rounds, outdir=outdir)

    print("\n[final plan]")
    print(plan)


if __name__ == "__main__":
    main()
