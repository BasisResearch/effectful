"""LEAP: blueprint-driven formal theorem proving over a *real* Lean compiler.

Implements the core of "LEAP: Supercharging LLMs for Formal Mathematics with
Agentic Frameworks" (arXiv:2606.03303). The paper's diagnosis is that general
LLMs reason well informally but "struggle to generate mechanically verifiable
proofs in formal languages like Lean" -- one-shot formalization of a hard theorem
essentially never compiles. Its fix is to treat proving as an *orchestration*
problem: register the theorem as the root of an AND-OR DAG, attempt a *direct*
proof with compiler-feedback revision, and on failure *decompose* it -- draft an
informal blueprint proposing intermediate lemmas, translate that into a Lean
*sketch* that proves the goal assuming the lemmas (``sorry`` placeholders), have an
LLM reviewer judge the decomposition, and recurse on the subgoals -- reusing proved
lemmas across branches via hierarchical memoization.

Unlike the sibling examples, which fake their environment (a static in-memory
index instead of live web search), LEAP's environment *is* the load-bearing part,
so we do not fake it: the ``VERIFIER (LEAN)`` of the paper's Figure 1 is a real
Lean 4 + Mathlib toolchain, invoked as a subprocess. A proof that does not compile
raises with the actual Lean error, and the harness's ``RetryLLMHandler`` feeds that
error back -- the paper's "continuous interaction with the Lean compiler" is a real
compile loop, not a simulation. Each of the paper's named components falls out of
an ordinary effectful idiom:

  * Grounded proofs by construction. A ``LeanProof`` certifies *at decode time*
    that ``<goal> := by <tactics>`` compiles under Lean with no errors and no
    ``sorry`` -- the same decode-time certification ``scientist_one.py`` uses for
    citations, except the ground truth is a theorem prover rather than an index.
    An uncompilable proof is not a well-typed ``LeanProof``; it raises, and
    ``RetryLLMHandler`` feeds the compiler diagnostic back (the paper's ``REVISER``).

  * The sketch is the same certification with a richer preamble. A decomposition's
    sketch proves the goal *assuming* its proposed lemmas: the search installs the
    lemmas as ``sorry`` stubs in the compile preamble, so the sketch's own tactics
    must be ``sorry``-free (checked) while depending on the sorried lemmas -- exactly
    the paper's "main theorem body is ``sorry``-free, ``sorry`` permitted in the
    proposed lemma statements".

  * Interleaved informal->formal planning. Both paths pass through an informal
    step before Lean: the ``NLProver`` writes an informal argument the
    ``FormalProver`` formalizes, and the ``BlueprintAgent`` drafts an informal
    decomposition the ``SketchAgent`` turns into a Lean sketch (the two-stream
    shape of ``scholar_peer.py``).

  * Tools scoped by class: only the formalizing agents subclass ``LeanAgent`` and
    hold the ``check`` tool that compiles a candidate against the live goal state
    and returns Lean's messages -- the compiler-in-the-loop. The planning and
    reviewing agents are closed-book by construction, no "do not compile"
    instruction needed (the encapsulation idiom of ``scholar_peer.py``).

  * Verification-guided proof search. Compiler verification is necessary but not
    sufficient: a sketch can compile while introducing a subgoal no simpler than
    its parent (paper Figure 3). The ``Reviewer`` LLM acts as a search filter that
    rejects such decompositions, and the ``state_writer`` refuses any subgoal that
    would reintroduce an ancestor -- preserving the DAG's acyclicity. Search is a
    DFS with backtracking over blueprints.

  * Hierarchical memoization via the AND-OR DAG. Goals are OR nodes keyed by their
    (normalized) statement; a decomposition is an AND node whose parent is proved
    once all its child subgoals are. A lemma proved in one branch is stored as a
    real Lean declaration and (a) reused verbatim if the same statement resurfaces
    in another branch -- turning a would-be decomposition into a direct proof --
    and (b) carried in every downstream compile preamble, so the final assembled
    proof of the root is one real Lean file that compiles end-to-end with no
    ``sorry``.

Demonstrates:
- Decode-time certification against a *real external tool* (the Lean compiler),
  so ``RetryLLMHandler`` turns an uncompilable proof into a compiler-feedback
  revision -- the certification idiom of ``scientist_one.py`` with a prover as
  ground truth
- A ContextVar carrying per-goal compile state (preamble + goal), read ambiently
  by ``LeanProof.__post_init__`` and the ``check`` tool, scoped to the pipeline
  (the ``WORKSPACE``/``CUTOFF`` idiom of ``scientist_one``/``paper_orchestra``)
- A class-scoped compiler tool offered to the formalizing agents via the Agent MRO
  and invisible to the closed-book planning/review agents (``scholar_peer.py``)
- An AND-OR DAG with hierarchical memoization, DFS backtracking, an LLM reviewer
  as a search filter, and a state-writer acyclicity guard -- the paper's Figure 1
- End-to-end verification: the assembled proof tree is emitted as one Lean file and
  compiled with no ``sorry``, the way ``scientist_one``'s audit re-derives its
  evidence
"""

# Simplifications vs. the source:
# - No Lean-IMO-Bench / Putnam. The paper proves olympiad-level theorems; this
#   composes a proof of a small, self-contained target so the example runs in
#   minutes, not a leaderboard. The architecture -- direct-then-decompose over an
#   AND-OR DAG with memoization -- is the same.
# - LeanSearch is a compile loop, not premise retrieval. The paper retrieves premises
#   with LeanSearch; here the ``check`` tool compiles a candidate against the live
#   goal and returns Lean's messages (errors / remaining goals), which is the
#   compiler-interaction half of that loop. Mathlib's own ``exact?``/``apply?`` remain
#   available to the model *inside* a proof, so premise search still happens -- in Lean.
# - One reviewer pass, single-vote. The decomposition reviewer judges once rather
#   than by majority vote (contrast ``scientist_one``'s majority-vote audit); the
#   acyclicity guard is deterministic Python.
# - Memoization is textual. Two lemmas are "the same" node when their normalized
#   statements match textually (whitespace-collapsed), not up to Lean-level
#   defeq/alpha -- enough to share the reusable-lemma story without an elaboration
#   check on every pair.

import argparse
import collections.abc
import contextvars
import dataclasses
import hashlib
import os
import re
import shutil
import subprocess
import textwrap

import pydantic.dataclasses

from effectful.handlers.llm import Agent, Skill, Tool

# ---------------------------------------------------------------------------
# The Lean compiler -- the ground truth every proof is certified against. This is
# the paper's ``VERIFIER (LEAN)``: a real Lean 4 + Mathlib toolchain shelled out to,
# not a stand-in. A proof is valid iff Lean accepts the file with no error message.
# ---------------------------------------------------------------------------

# Where the Mathlib lake project lives. Built once (elan + `lake exe cache get`);
# see this module's header. Override with LEAP_LEAN_PROJECT.
LEAN_PROJECT = os.environ.get(
    "LEAP_LEAN_PROJECT", os.path.expanduser("~/.cache/leap-lean/leapproj")
)
# Every compiled fragment opens with this; `import Mathlib` pulls the whole library
# so the model may use any tactic/lemma it knows (`ring`, `omega`, `simp`, `exact?`).
PRELUDE = "import Mathlib\n"


@dataclasses.dataclass(frozen=True)
class LeanResult:
    """The outcome of compiling a Lean fragment: ``ok`` is true iff Lean reported no
    error (``sorry`` warnings are not errors). ``messages`` is Lean's stdout+stderr,
    fed back to the model verbatim on failure -- the raw compiler diagnostic."""

    ok: bool
    messages: str


def _lake_bin() -> str:
    """Locate the ``lake`` executable, tolerating a not-yet-on-PATH elan install."""
    for cand in (
        os.environ.get("LAKE"),
        shutil.which("lake"),
        os.path.expanduser("~/.elan/bin/lake"),
    ):
        if cand and os.path.exists(cand):
            return cand
    return "lake"


class LeanKernel:
    """Compiles Lean source via ``lake env lean`` in the Mathlib project, with an
    in-memory cache keyed by source text so identical fragments (retries, repeated
    tool calls, the same proved lemma seen twice) compile at most once. Importing
    all of Mathlib per check is slow; the cache is what keeps the search tractable."""

    def __init__(self, project: str = LEAN_PROJECT, timeout: float = 120.0) -> None:
        self.project = project
        self.timeout = timeout
        self._cache: dict[str, LeanResult] = {}

    def available(self) -> bool:
        return os.path.isdir(os.path.join(self.project, ".lake"))

    def compile(self, source: str) -> LeanResult:
        """Compile a full Lean source string and return the result (cached)."""
        key = hashlib.sha256(source.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        env = dict(os.environ)
        env["PATH"] = (
            os.path.expanduser("~/.elan/bin") + os.pathsep + env.get("PATH", "")
        )
        # A scratch file inside the project's build dir so `lake env` resolves imports.
        scratch = os.path.join(self.project, f".leap_scratch_{key[:12]}.lean")
        try:
            with open(scratch, "w") as fh:
                fh.write(source)
            proc = subprocess.run(
                [_lake_bin(), "env", "lean", scratch],
                cwd=self.project,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            out = (proc.stdout + proc.stderr).strip()
            # `lean` exits non-zero on error; `sorry` and linter notes are warnings.
            ok = proc.returncode == 0 and "error:" not in out
        except subprocess.TimeoutExpired:
            ok, out = False, f"Lean timed out after {self.timeout}s (proof too slow)."
        finally:
            if os.path.exists(scratch):
                os.remove(scratch)
        result = LeanResult(ok, out or ("no output" if ok else "unknown error"))
        self._cache[key] = result
        return result


# The compile context for the goal currently being worked. ``LeanProof`` and the
# ``check`` tool read it ambiently -- through a ContextVar rather than a bare global,
# so it is scoped to the pipeline and safe if goals are ever worked concurrently.
# Exactly ``scientist_one``'s WORKSPACE / ``paper_orchestra``'s CUTOFF pattern.
@dataclasses.dataclass(frozen=True)
class LeanContext:
    kernel: LeanKernel
    preamble: str  # PRELUDE + proved lemmas + (for a sketch) the sorry-stub lemmas
    decl: (
        str  # the goal declaration header, e.g. "theorem leap_goal (n : ℕ) : n + 0 = n"
    )


LEAN_CTX: contextvars.ContextVar[LeanContext] = contextvars.ContextVar("LEAN_CTX")

# A proof body may not smuggle in `sorry` (or its cousins): the main goal must be
# genuinely closed. `sorry` is legitimate only in the search-generated lemma stubs,
# which live in the preamble, never in model-authored tactics.
_SORRY = re.compile(r"\b(sorry|admit|sorryAx)\b")


def assemble(decl: str, tactics: str, preamble: str) -> str:
    """Build the full Lean source for ``decl := by <tactics>`` under ``preamble``."""
    body = textwrap.indent(tactics.strip(), "  ")
    return f"{preamble}\n\n{decl} := by\n{body}\n"


def with_ctx[T](ctx: "LeanContext", fn: collections.abc.Callable[[], T]) -> T:
    """Run ``fn`` with ``LEAN_CTX`` bound to ``ctx`` for exactly that call, so the
    ``check`` tool and the decode-time certifications read the right goal/preamble.
    One balanced set/reset per call -- no fragile nesting across a whole loop body."""
    token = LEAN_CTX.set(ctx)
    try:
        return fn()
    finally:
        LEAN_CTX.reset(token)


# ---------------------------------------------------------------------------
# Types crossing the model boundary
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class LeanProof:
    """A tactic-block proof of the goal currently in scope, certified at decode time.

    ``__post_init__`` assembles ``<goal> := by <tactics>`` under the in-scope
    preamble and compiles it with the real Lean kernel; an uncompilable proof (or one
    that tries to use ``sorry``) raises, and ``RetryLLMHandler`` feeds Lean's own
    error message back so the model revises against the compiler. Used for both the
    direct proof and the decomposition sketch -- they differ only in the preamble the
    search installs (a sketch's preamble carries the proposed lemmas as ``sorry``
    stubs, so the goal may lean on them while its own tactics stay ``sorry``-free)."""

    tactics: str = dataclasses.field(
        metadata={
            "description": "The tactic block that proves the goal, i.e. what follows "
            "`:= by`. Do not include the theorem signature or the word `by`, and do "
            "not use `sorry`/`admit`: the goal must be fully closed."
        }
    )

    def __post_init__(self) -> None:
        if _SORRY.search(self.tactics):
            raise ValueError(
                "the proof uses `sorry`/`admit`; the goal must be closed for real "
                "(sorry is only allowed for the separately-proposed lemmas)"
            )
        ctx = LEAN_CTX.get()
        result = ctx.kernel.compile(assemble(ctx.decl, self.tactics, ctx.preamble))
        if not result.ok:
            raise ValueError(
                "Lean rejected this proof. Fix it against the compiler output below "
                f"(you may call `check` to iterate):\n{result.messages}"
            )


@pydantic.dataclasses.dataclass(frozen=True)
class ProposedLemma:
    """One intermediate lemma a blueprint proposes: a Lean declaration header the
    sketch may assume. ``__post_init__`` certifies the *statement* type-checks (as a
    ``sorry`` stub) so a malformed or ill-typed lemma is fed back before it becomes a
    subgoal -- the statement must at least be a well-formed proposition, even though
    its proof is deferred."""

    name: str = dataclasses.field(
        metadata={
            "description": "A fresh Lean identifier for the lemma (snake_case, unique "
            "within this decomposition), referenced by name from the sketch."
        }
    )
    decl: str = dataclasses.field(
        metadata={
            "description": "The lemma's Lean declaration header WITHOUT the name or "
            "`:= ...`, i.e. the binders and proposition: e.g. `(n : ℕ) : 0 < n + 1`. "
            "It must type-check as a standalone statement."
        }
    )
    rationale: str = dataclasses.field(
        metadata={
            "description": "Why proving this lemma helps -- what it lets the sketch do, "
            "and why it is strictly simpler / more general than the goal."
        }
    )

    def header(self) -> str:
        """The full stub header ``theorem <name> <decl>`` for the compile preamble."""
        return f"theorem {self.name} {self.decl}"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", self.name):
            raise ValueError(f"lemma name {self.name!r} is not a valid Lean identifier")
        ctx = LEAN_CTX.get()
        # The statement must type-check; its proof may be deferred (`sorry`).
        stub = f"{ctx.preamble}\n\n{self.header()} := sorry\n"
        result = ctx.kernel.compile(stub)
        if not result.ok:
            raise ValueError(
                f"the lemma statement `{self.name} {self.decl}` does not type-check. "
                f"Fix the statement against Lean's output:\n{result.messages}"
            )


@pydantic.dataclasses.dataclass(frozen=True)
class Blueprint:
    """The informal decomposition: a natural-language plan plus the intermediate
    lemmas it proposes. The sketch (a ``LeanProof``) is the formal counterpart that
    proves the goal assuming these lemmas."""

    plan: str = dataclasses.field(
        metadata={
            "description": "The informal proof blueprint in a few sentences: how the "
            "goal reduces to the proposed lemmas."
        }
    )
    lemmas: list[ProposedLemma]

    def __post_init__(self) -> None:
        if not self.lemmas:
            raise ValueError(
                "a decomposition must propose at least one lemma; if the goal needs "
                "no lemmas it should be proved directly, not decomposed"
            )
        names = [lm.name for lm in self.lemmas]
        if len(set(names)) != len(names):
            raise ValueError(f"proposed lemma names are not unique: {names}")


@pydantic.dataclasses.dataclass(frozen=True)
class ReviewVerdict:
    """The decomposition reviewer's judgment -- the paper's planning-level search
    filter (Sec. 2.5 / Figure 3)."""

    simplifies: bool = dataclasses.field(
        metadata={
            "description": "True only if every proposed lemma is genuinely simpler or "
            "more general than the goal and plausibly provable; false if any lemma "
            "merely restates the goal or is no easier than it."
        }
    )
    reason: str


# ---------------------------------------------------------------------------
# The compiler-in-the-loop base class. The `check` tool is defined here, so it
# reaches the FormalProver and SketchAgent (via the Agent MRO) and is invisible to
# the closed-book NLProver, BlueprintAgent, and Reviewer.
# ---------------------------------------------------------------------------


class LeanAgent(Agent):
    """Base for agents that write Lean against the live compiler. The ``check`` tool
    defined here compiles a candidate tactic block for the goal in scope and returns
    Lean's messages, so a formalizing agent can iterate against real compiler feedback
    before committing an answer -- the paper's continuous compiler interaction. Agents
    that only plan or review do not subclass ``LeanAgent``, so the tool never enters
    their lexical scope."""

    @Tool.define
    def check(self, tactics: str) -> str:
        """Compile ``<current goal> := by <tactics>`` with Lean and return the
        compiler's output: an empty/clean result means it is accepted; otherwise the
        errors and the remaining goal state. Use this to test tactics and read the
        goal before you commit a final proof. (You may also use Mathlib's own
        ``exact?`` / ``apply?`` inside ``tactics`` to search for premises.)"""
        ctx = LEAN_CTX.get()
        if _SORRY.search(tactics):
            return (
                "Refused: `tactics` contains `sorry`/`admit`; the goal must be closed."
            )
        result = ctx.kernel.compile(assemble(ctx.decl, tactics, ctx.preamble))
        if result.ok:
            return "Lean accepts this proof (no errors)."
        return f"Lean output:\n{result.messages}"


# ---------------------------------------------------------------------------
# Stage 1 -- direct formalization: NL prover (informal) -> formal prover (Lean).
# ---------------------------------------------------------------------------


class NLProver(Agent):
    """You are the informal reasoner. You write a short, rigorous natural-language
    proof of a statement -- the mathematical argument, not Lean code -- for a
    formalizer to translate. Closed-book: you hold no compiler tool."""

    @Skill.define
    def argue(self, goal: str, context: str) -> str:
        """Give a concise but rigorous informal proof of the following statement.
        State the key steps a formal proof would need (case splits, inductions,
        lemmas invoked). Do not write Lean.

        Statement:
        {goal}

        Context that may help (available lemmas already proved, and the shape of the
        problem):
        {context}
        """


class FormalProver(LeanAgent):
    """You are the formal prover. You translate an informal argument into a Lean 4
    tactic proof and make it compile, using the ``check`` tool to iterate against the
    real compiler. You prefer short, robust proofs (``simp``, ``omega``, ``ring``,
    ``induction``, ``exact?``) and you never leave a ``sorry``."""

    @Skill.define
    def formalize(self, goal: str, informal: str, context: str) -> LeanProof:
        """Prove the goal below in Lean 4 with Mathlib by returning the tactic block
        (what follows ``:= by``). Translate the informal argument, then use ``check``
        to compile and fix it against Lean's output until it is accepted. The proof
        must fully close the goal -- no ``sorry``.

        Goal declaration (your tactics complete ``<this> := by ...``):
        {goal}

        Informal argument to formalize:
        {informal}

        Context (lemmas already proved and in scope; you may cite them by name):
        {context}
        """


# ---------------------------------------------------------------------------
# Stage 2 -- decomposition: blueprint (informal) -> reviewer -> sketch (Lean).
# ---------------------------------------------------------------------------


class BlueprintAgent(Agent):
    """You are the blueprint planner. When a goal resists direct proof, you propose
    a decomposition: intermediate lemmas that are each strictly simpler or more
    general than the goal, such that the goal follows easily once they hold. Closed-
    book: you plan in mathematics, not against the compiler."""

    @Skill.define
    def draft(self, goal: str, context: str, feedback: str) -> Blueprint:
        """The goal below could not be proved directly within budget. Draft a proof
        blueprint: an informal plan plus a small set of intermediate lemmas that make
        the goal easy to prove. Each lemma must be genuinely simpler or more general
        than the goal -- never a restatement of it -- and should be broadly useful.
        Give each lemma a fresh Lean identifier and a well-formed statement.

        Goal declaration:
        {goal}

        Context (lemmas already proved and in scope -- prefer reusing these to
        proposing new ones):
        {context}

        Feedback from prior attempts (empty on the first try):
        {feedback}
        """


class SketchAgent(LeanAgent):
    """You are the sketch formalizer. Given a blueprint, you write a Lean tactic
    proof of the goal that *assumes the proposed lemmas* (they are in scope as
    hypotheses you may cite by name). Your tactics themselves must be ``sorry``-free:
    the goal must reduce to the lemmas. Use ``check`` to compile against the real
    Lean, where the proposed lemmas are present as stubs."""

    @Skill.define
    def sketch(self, goal: str, blueprint: str, context: str) -> LeanProof:
        """Prove the goal below assuming the blueprint's lemmas. Return the tactic
        block (what follows ``:= by``); you may reference each proposed lemma by its
        name as an already-proved fact. Your tactics must not use ``sorry`` -- only the
        lemmas are deferred. Use ``check`` to compile and fix against Lean's output.

        Goal declaration:
        {goal}

        Blueprint (plan and the lemmas now in scope, by name):
        {blueprint}

        Context (other lemmas already proved and in scope):
        {context}
        """


class Reviewer(Agent):
    """You are the decomposition reviewer -- a planning-level search filter. Compiler
    verification only checks that a sketch is well-typed, not that its decomposition
    makes progress: a sketch can compile while proposing a subgoal no simpler than the
    goal (e.g. one syntactically equivalent to it). You reject such non-simplifying
    decompositions so search does not waste effort on them."""

    @Skill.define
    def review(self, goal: str, blueprint: str) -> ReviewVerdict:
        """Judge whether this decomposition genuinely simplifies proving the goal.
        Reject it if any proposed lemma merely restates the goal, is no easier than
        it, or does not plausibly advance the proof. Accept only a decomposition whose
        lemmas are each strictly simpler or more general than the goal and together
        make it easy.

        Goal declaration:
        {goal}

        Proposed decomposition:
        {blueprint}
        """


# ---------------------------------------------------------------------------
# The AND-OR DAG -- proof progress and hierarchical memoization (paper Sec. 2.3).
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    """Collapse whitespace so two statements that differ only in spacing share a
    memoization key. (Textual, not Lean-defeq: enough for the reuse story.)"""
    return " ".join(text.split())


@dataclasses.dataclass
class GoalNode:
    """An OR node: a goal (or lemma) to prove. ``decl`` is its Lean header
    ``theorem <name> <sig>``; once ``proved``, ``tactics`` is the accepted tactic
    block, and the node is a reusable Lean declaration in every downstream preamble."""

    name: str
    decl: str  # "theorem <name> <binders> : <prop>"
    proved: bool = False
    attempted: bool = False
    tactics: str | None = None
    reused: bool = False

    def declaration(self) -> str:
        """The full proved Lean declaration, for the reuse preamble and final file."""
        assert self.proved and self.tactics is not None
        body = textwrap.indent(self.tactics.strip(), "  ")
        return f"{self.decl} := by\n{body}"


@dataclasses.dataclass
class ProofDAG:
    """The proof graph: OR nodes keyed by normalized statement (memoization), plus a
    monotonically-growing preamble of proved lemma declarations that every subsequent
    compile reuses. The ``state_reader``/``state_writer`` of the paper are this
    object's read/commit methods."""

    kernel: LeanKernel
    nodes: dict[str, GoalNode] = dataclasses.field(default_factory=dict)
    # Proved nodes in completion order. A node is proved only after its children, so
    # this order is topological (dependencies first) -- the order the reuse preamble
    # and the final assembly must emit declarations in.
    proof_order: list[GoalNode] = dataclasses.field(default_factory=list)
    _counter: int = 0

    def fresh_name(self, hint: str) -> str:
        self._counter += 1
        slug = re.sub(r"[^A-Za-z0-9_]", "_", hint).strip("_")[:24] or "lemma"
        return f"leap_{self._counter}_{slug}"

    def get_or_add(self, sig: str, name_hint: str) -> tuple[GoalNode, bool]:
        """Look a goal up by normalized signature; create its OR node if new. Returns
        (node, is_new). A hit on a proved node is the memoization payoff."""
        key = _norm(sig)
        if key in self.nodes:
            return self.nodes[key], False
        name = self.fresh_name(name_hint)
        node = GoalNode(name=name, decl=f"theorem {name} {sig}")
        self.nodes[key] = node
        return node, True

    def mark_proved(self, node: GoalNode, tactics: str) -> None:
        """Commit a node's accepted proof and record it in topological order."""
        node.proved, node.tactics = True, tactics
        self.proof_order.append(node)

    def proved_preamble(self) -> str:
        """PRELUDE + every proved lemma's real declaration in dependency order, so any
        compile reuses the whole proved library (real Lean-level lemma sharing)."""
        decls = [n.declaration() for n in self.proof_order]
        return PRELUDE + ("\n\n".join(decls) + "\n\n" if decls else "")

    def context_digest(self) -> str:
        """A short human/model-readable list of proved lemmas in scope (state_reader)."""
        if not self.proof_order:
            return "(no lemmas proved yet)"
        return "\n".join(f"- {n.decl}" for n in self.proof_order)


# ---------------------------------------------------------------------------
# Verification-guided proof search: direct proof, else decompose. DFS + backtrack.
# ---------------------------------------------------------------------------


def _sig_of_lemma(lm: ProposedLemma) -> str:
    """A proposed lemma's signature (binders : prop) -- its memoization key."""
    return lm.decl.strip()


def try_direct(dag: ProofDAG, node: GoalNode, sig: str, depth: int) -> bool:
    """Attempt a direct proof: informal argument -> Lean formalization, certified by
    the compiler on decode. Returns True and records the tactics on success; on
    failure (retries exhausted without a compiling proof) returns False so the caller
    decomposes. This is the paper's direct-formalization path with REVISER feedback
    (here, ``RetryLLMHandler``)."""
    ind = "  " * depth
    ctx = LeanContext(dag.kernel, dag.proved_preamble(), node.decl)
    informal = with_ctx(ctx, lambda: NLProver().argue(node.decl, dag.context_digest()))
    try:
        proof = with_ctx(
            ctx,
            lambda: FormalProver().formalize(node.decl, informal, dag.context_digest()),
        )
    except Exception as exc:  # retries exhausted without a compiling proof
        print(f"{ind}[direct] no compiling proof ({type(exc).__name__}); decomposing")
        return False
    dag.mark_proved(node, proof.tactics)
    print(
        f"{ind}[direct] proved `{node.name}` ({len(proof.tactics.splitlines())} tactic lines)"
    )
    return True


def decompose(
    dag: ProofDAG, node: GoalNode, sig: str, ancestors: frozenset[str], depth: int
) -> bool:
    """Blueprint -> review -> sketch -> recurse. A decomposition is committed only if
    the reviewer finds it simplifying, the state_writer finds it acyclic, the sketch
    compiles (assuming the lemmas), and every child subgoal is then proved. On any
    failure it backtracks and re-drafts, up to a bound (DFS with backtracking)."""
    ind = "  " * depth
    feedback = ""
    for attempt in range(1, MAX_BLUEPRINTS + 1):
        # Blueprint (informal plan + proposed lemma statements). ProposedLemma decode
        # type-checks each statement against the proved preamble, so it needs the ctx.
        base = LeanContext(dag.kernel, dag.proved_preamble(), node.decl)
        try:
            blueprint = with_ctx(
                base,
                lambda: BlueprintAgent().draft(
                    node.decl, dag.context_digest(), feedback
                ),
            )
        except Exception as exc:
            print(f"{ind}[blueprint {attempt}] draft failed ({type(exc).__name__})")
            continue
        bp_text = _render_blueprint(blueprint)

        # Reviewer: reject a decomposition that does not simplify (Figure 3).
        verdict = Reviewer().review(node.decl, bp_text)
        if not verdict.simplifies:
            print(f"{ind}[review {attempt}] rejected: {verdict.reason}")
            feedback = (
                f"A reviewer rejected the previous decomposition: {verdict.reason}"
            )
            continue

        # state_writer: reject any subgoal that would reintroduce an ancestor -- keep
        # the DAG acyclic (also catches the Figure-3 "subgoal == parent" pathology).
        cyclic = [
            lm.name for lm in blueprint.lemmas if _norm(_sig_of_lemma(lm)) in ancestors
        ]
        if cyclic:
            print(f"{ind}[state_writer {attempt}] rejected cyclic subgoals {cyclic}")
            feedback = (
                f"These proposed lemmas restate an ancestor goal (a cycle): {cyclic}. "
                "Propose strictly simpler, non-circular lemmas."
            )
            continue

        # Bind each proposed lemma to a DAG node (memoization-aware): a lemma whose
        # statement is already a node reuses that node -- and its name -- so the sketch,
        # the stubs, and the eventually-stored proof all agree on one identifier. This
        # is what makes the assembled proof reference real, in-scope declarations.
        lemma_nodes = [
            dag.get_or_add(_sig_of_lemma(lm), lm.name)[0] for lm in blueprint.lemmas
        ]

        # Sketch: prove the goal assuming the lemmas. Already-proved lemmas are in the
        # proved preamble; the rest are installed as `sorry` stubs under their DAG names.
        stubs = "\n\n".join(
            f"{nd.decl} := sorry" for nd in lemma_nodes if not nd.proved
        )
        sketch_ctx = LeanContext(
            dag.kernel,
            dag.proved_preamble() + (stubs + "\n\n" if stubs else ""),
            node.decl,
        )
        sketch_bp = _render_blueprint(blueprint, lemma_nodes)
        try:
            sketch = with_ctx(
                sketch_ctx,
                lambda: SketchAgent().sketch(
                    node.decl, sketch_bp, dag.context_digest()
                ),
            )
        except Exception as exc:
            print(f"{ind}[sketch {attempt}] no compiling sketch ({type(exc).__name__})")
            feedback = "The sketch did not compile even assuming the lemmas; simplify the plan."
            continue
        print(
            f"{ind}[sketch {attempt}] compiles assuming {len(blueprint.lemmas)} lemma(s)"
        )

        # Recurse on each subgoal, sharing the DAG (memoization across branches).
        child_ancestors = ancestors | {_norm(sig)}
        all_proved = True
        for lm in blueprint.lemmas:
            if not prove(dag, _sig_of_lemma(lm), lm.name, child_ancestors, depth + 1):
                all_proved = False
                break
        if not all_proved:
            print(f"{ind}[decompose {attempt}] a subgoal failed; backtracking")
            feedback = "A proposed lemma could not be proved; propose different lemmas."
            continue

        # All children proved -> the AND node succeeds -> the parent is proved. Its
        # reusable proof is the sketch over the now-real (not sorry) lemma preamble.
        dag.mark_proved(node, sketch.tactics)
        print(f"{ind}[decompose {attempt}] all subgoals proved -> `{node.name}` proved")
        return True

    print(f"{ind}[decompose] exhausted {MAX_BLUEPRINTS} blueprints for `{node.name}`")
    return False


def prove(
    dag: ProofDAG, sig: str, name_hint: str, ancestors: frozenset[str], depth: int
) -> bool:
    """Prove a goal (statement ``sig`` = ``binders : prop``): memo hit, else direct,
    else decompose. Shared ``dag`` gives hierarchical memoization; ``ancestors``
    enforces acyclicity."""
    ind = "  " * depth
    node, _ = dag.get_or_add(sig, name_hint)
    if node.proved:  # memoization hit: a lemma already proved in another branch
        node.reused = True
        print(f"{ind}[memo] reuse proved lemma `{node.name}`: {_norm(sig)[:70]}")
        return True
    if _norm(sig) in ancestors:  # this goal is its own ancestor -> a cycle
        print(f"{ind}[cycle] `{node.name}` restates an ancestor; abandoning branch")
        return False
    if node.attempted:  # tried before and not proved; don't loop on it again
        print(f"{ind}[skip] `{node.name}` was already attempted and failed")
        return False
    node.attempted = True
    print(f"{ind}[goal] {node.name}: {_norm(sig)[:80]}")

    # Normally: direct first, decompose on failure. ``--decompose-root`` forces the
    # root to decompose so the blueprint/reviewer/sketch/memoization path is exercised
    # even when a strong model could one-shot it (a labeled demo, not the paper's flow).
    force = FORCE_DECOMPOSE_ROOT and depth == 0
    if not force and try_direct(dag, node, sig, depth):
        return True
    if depth >= MAX_DEPTH:
        print(f"{ind}[depth] max decomposition depth reached for `{node.name}`")
        return False
    return decompose(dag, node, sig, ancestors, depth)


def _render_blueprint(bp: Blueprint, nodes: list[GoalNode] | None = None) -> str:
    """Render a blueprint for a prompt. When ``nodes`` is given (one per lemma, in
    order), lemmas are named by their DAG identifier -- the name the sketch must cite
    and under which the proof is stored -- so the sketch references real declarations."""
    names = (
        [nd.name for nd in nodes]
        if nodes is not None
        else [lm.name for lm in bp.lemmas]
    )
    lines = [bp.plan, "", "Proposed lemmas (in scope, cite by the name shown):"]
    for name, lm in zip(names, bp.lemmas):
        lines.append(f"- {name} : {lm.decl}   -- {lm.rationale}")
    return "\n".join(lines)


# Search bounds.
MAX_BLUEPRINTS = 3  # decomposition re-drafts before a node is abandoned (backtracking)
MAX_DEPTH = 3  # deepest decomposition nesting
FORCE_DECOMPOSE_ROOT = False  # set by --decompose-root: skip the root's direct attempt


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LeapResult:
    proved: bool
    dag: ProofDAG
    root: GoalNode


def run_leap(root_sig: str, kernel: LeanKernel) -> LeapResult:
    """Register the root theorem and drive the DFS: direct-then-decompose over the
    shared AND-OR DAG. Returns the DAG so the caller can inspect memoization and emit
    the assembled proof."""
    dag = ProofDAG(kernel=kernel)
    proved = prove(dag, root_sig, "root", frozenset(), 0)
    root = dag.nodes[_norm(root_sig)]
    return LeapResult(proved=proved, dag=dag, root=root)


def assemble_full_proof(dag: ProofDAG) -> str:
    """Emit the whole proof tree as one Lean file: PRELUDE + every proved lemma +
    the root, ordered so dependencies precede uses (proved-order suffices since a
    lemma is proved before the parent that uses it). Compiling this with no ``sorry``
    is the end-to-end check -- like ``scientist_one``'s audit re-deriving its
    evidence."""
    parts = [PRELUDE.strip(), ""]
    for n in dag.proof_order:  # topological: dependencies precede uses; root is last
        parts += [n.declaration(), ""]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Targets: small theorems whose proof benefits from a lemma decomposition. The
# statement is the Lean signature `(binders) : proposition`; LEAP supplies the name.
# ---------------------------------------------------------------------------

TARGETS: dict[str, str] = {
    # Sum of the first n odd numbers is n^2. Direct: induction + `Finset.sum_range_succ`
    # + `ring`. A natural decomposition proves the successor step as its own lemma.
    "odd_sum": r"(n : ℕ) : (∑ i ∈ Finset.range n, (2 * i + 1)) = n ^ 2",
    # Gauss sum, doubled to stay in ℕ. Induction; the step is a clean sub-lemma.
    "gauss": r"(n : ℕ) : (2 * ∑ i ∈ Finset.range (n + 1), i) = n * (n + 1)",
    # A divisibility fact that invites a two-lemma decomposition (parity of n*(n+1)
    # feeding 6 ∣ n*(n+1)*(n+2)); harder, exercises deeper decomposition.
    "div6": r"(n : ℕ) : 6 ∣ n * (n + 1) * (n + 2)",
}

# A trivial theorem used to validate the toolchain without any LLM.
SANITY = r"(n : ℕ) : n + 0 = n"


def check_toolchain(kernel: LeanKernel) -> None:
    """Compile a trivial theorem (no LLM) to confirm Lean+Mathlib is wired up."""
    if not kernel.available():
        print(
            f"Lean project not found/built at {kernel.project!r}. Build it once:\n"
            "  elan default stable  # if elan is installed\n"
            f"  cd {kernel.project} && lake exe cache get && lake build"
        )
        return
    print(f"Compiling a trivial theorem via {kernel.project} ...")
    src = f"{PRELUDE}\ntheorem leap_sanity {SANITY} := by simp\n"
    result = kernel.compile(src)
    print("Toolchain OK." if result.ok else f"Toolchain FAILED:\n{result.messages}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=list(TARGETS),
        default="odd_sum",
        help="Which theorem to prove",
    )
    parser.add_argument(
        "--statement",
        type=str,
        default=None,
        help="A custom Lean signature `(binders) : prop` to prove instead of --target",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=LEAN_PROJECT,
        help="Path to the Lean+Mathlib lake project",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-compile timeout in seconds",
    )
    parser.add_argument(
        "--check-toolchain",
        action="store_true",
        help="Skip the pipeline; compile a trivial theorem to validate Lean+Mathlib",
    )
    parser.add_argument(
        "--decompose-root",
        action="store_true",
        help="Force the root goal to decompose (skip its direct proof), to demonstrate "
        "the blueprint/reviewer/sketch/memoization path even on an easy target",
    )
    args = parser.parse_args()

    global FORCE_DECOMPOSE_ROOT
    FORCE_DECOMPOSE_ROOT = args.decompose_root
    kernel = LeanKernel(project=args.project, timeout=args.timeout)

    if args.check_toolchain:
        check_toolchain(kernel)
        return

    if not kernel.available():
        raise SystemExit(
            f"Lean project not built at {args.project!r}; run with --check-toolchain "
            "for build instructions."
        )

    sig = args.statement or TARGETS[args.target]
    print(f"Proving: {sig}\n")

    result = run_leap(sig, kernel)

    print("\n" + "=" * 72)
    proved = [n for n in result.dag.nodes.values() if n.proved]
    reused = [n for n in result.dag.nodes.values() if n.reused]
    print(
        f"DAG: {len(result.dag.nodes)} goal node(s), {len(proved)} proved, "
        f"{len(reused)} reused via memoization."
    )
    for n in result.dag.nodes.values():
        mark = "proved" if n.proved else "OPEN"
        extra = " (reused)" if n.reused else ""
        print(f"  [{mark}]{extra} {n.decl}")

    if not result.proved:
        raise SystemExit(
            "\nLEAP did not close the root goal (as the paper notes, "
            "one-shot formal proving is hard; try another --target)."
        )

    # End-to-end verification: compile the whole assembled proof tree with no sorry.
    full = assemble_full_proof(result.dag)
    print("\nAssembled proof; recompiling the whole tree end-to-end ...")
    final = kernel.compile(full)
    if final.ok:
        print("VERIFIED: the complete proof compiles under Lean with no `sorry`.\n")
        print(full)
    else:
        raise SystemExit(f"Assembled proof failed to recompile:\n{final.messages}")


if __name__ == "__main__":
    main()
