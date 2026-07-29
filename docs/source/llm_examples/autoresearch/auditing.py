"""ClaimCheck: auditing whether a proved theorem is the theorem you meant.

Implements the core of ClaimCheck ("Narrowing the Gap Between Proof and Intent",
https://midspiral.com/blog/claimcheck-narrowing-the-gap-between-proof-and-intent/,
reference implementation at https://github.com/metareflection/claimcheck, MIT).
Its diagnosis is that a verifier proves code matches its *specification* and says
nothing about whether the specification matches your *intent*. The motivating case
is a Dafny election tally whose ``TallyMonotonic`` lemma was supposed to say
"adding a ballot can't decrease a tally" and in fact said ``Count(...) >= 0`` --
trivially true, since counts are naturals. Dafny reported 14 verified, 0 errors.
The proof was correct and the guarantee was not the one anybody wanted.

Its fix is *round-trip informalization*. Pass 1 translates the formal statement,
**and nothing else**, back into English; pass 2 compares that back-translation
against the requirement it was meant to formalize and classifies any divergence.
The load-bearing part is what pass 1 does not get: having never seen the
requirement, it cannot parrot it back, so agreement in pass 2 is evidence and not
an echo. The blog measures this -- across 5 domains and 36 requirement/lemma pairs,
the two-pass split scored 96.3% against a single combined prompt's 86.1%.

That is a claim about *scope*, which is what makes it an effectful example. The
reference implementation enforces it by hand-assembling prompt strings, under a
comment reading ``CRITICAL: This prompt must NOT include the original
requirements``. Here the comment is a signature:

  * Blindness by construction. ``Informalizer.informalize`` takes one argument, a
    Lean statement. There is no parameter through which a requirement could
    arrive, its ``Agent`` history holds no turn in which one appeared, and nothing
    in its lexical scope can fetch one. The instruction "do not look at the
    requirement" is not in the prompt because it does not need to be -- the same
    encapsulation ``review.py`` uses for tools, applied to data.

  * ...and the ablation is visible in the signature too. ``--strategy
    single-pass`` asks one template to informalize-then-compare, so
    ``requirement`` is a parameter of the call doing the analysis; ``--strategy
    naive`` just asks "does this match?". What separates the three is precisely
    what is in scope for the model at the moment it commits to a reading of the
    formal statement -- a difference in the type signature, not in the wording.

    On *this* corpus the three do not separate: gpt-5.5 and gpt-5-mini both score
    17/17 under all three strategies (measured, 2026-07). The traps here are
    detectable even naively, so the structural separation is not load-bearing at
    this scale, and the blog's ordering is not reproduced -- see the note in the
    simplifications below. The ablation is wired up as instrumentation for a
    harder corpus, not as a result this file demonstrates.

  * Coherent verdicts by construction. A ``Comparison`` certifies at decode time
    that ``match`` and ``weakening`` agree and that a mismatch names its
    discrepancy; an incoherent verdict raises and ``RetryLLMHandler`` feeds it
    back. Upstream's JSON schema admits ``match: true`` alongside
    ``weakeningType: "tautology"``, and nothing catches it.

  * The deterministic half stays in Python. The blog's between-pass checks --
    flag a back-translation rated ``trivial``, flag two requirements whose
    theorems have the same conclusion -- are a loop over typed
    ``Informalization`` values, not a model call.

  * The premise is checked by a real prover. Under ``--verify`` the whole corpus
    is compiled by the Lean 4 + Mathlib toolchain ``formalization.py`` already
    shells out to. It reports 16 theorems, 0 errors, no ``sorry`` -- and then the
    audit finds six of them that do not mean what they were written to mean. The
    verifier's clean bill of health is the setup, not the punchline.

Demonstrates:
- Structural separation as *lexical scope*: an agent that cannot see a value
  because its template's signature has no parameter for it, contrasted against
  two ablations that can
- Decode-time certification of a structured verdict's internal coherence, turning
  a self-contradictory answer into a `RetryLLMHandler` retry
- Reuse of a sibling example's real external verifier (`formalization.py`'s
  `LeanKernel`) to establish a premise, rather than asserting it
- A labelled corpus and an accuracy report that separates the two error
  directions, since waving a weak theorem through is the failure that matters
- Fan-out over independent audits with ``asyncio.gather`` + ``asyncio.to_thread``
- Per-field guidance carried on the types as ``field(metadata={"description": ...})``
"""

# Simplifications vs. the source:
# - Lean 4 + Mathlib, not Dafny. The corpus is a re-authored port of upstream's
#   election-tally demo (`demo/election0.dfy`, `demo/election.dfy`) plus three
#   planted variants adapted from its `counter` test domain. Lean was chosen
#   because a real, already-built toolchain is reachable from this repo, so
#   "every one of these theorems is proved" is a compile and not a claim. The
#   mapping is close: Dafny's `requires`/`ensures` split becomes hypotheses and
#   conclusion, and `nat` becoming `ℕ` preserves the original tautology traps
#   (`0 ≤ count ...` is as vacuous in Lean as `Count(...) >= 0` was in Dafny).
# - Statements are sent, proofs are not. Upstream sends the Dafny lemma body
#   along with its contract; here only the statement crosses the boundary, as
#   upstream's own `lemmascript` preset does ("only the signature + requires +
#   ensures is sent, never a body"). In Lean the meaning is entirely in the
#   statement, and ClaimCheck explicitly does not audit the proof.
# - One domain, one model. The blog runs 5 domains and splits the passes across
#   two models (Haiku informalizes, Sonnet compares), reporting that the weaker
#   informalizer is sufficient. Both passes here run on whatever model the
#   harness was given, so that model-asymmetry result is *not* reproduced -- only
#   the structural separation is.
# - One call per claim, not one batched call per pass. Upstream batches every
#   lemma into a single request for throughput; auditing each claim separately
#   keeps the blindness argument obvious and lets the fan-out be the demo.
# - No coverage check. Both here and upstream, a requirement that no theorem
#   addresses at all goes undetected: the audit only ever judges pairs it is
#   handed. Upstream lists this as a known limitation.
# - The corpus is saturated, so the ablation does not reproduce the blog's
#   ordering. 17 claims over one small self-contained domain is a demonstration,
#   not a benchmark: every planted flaw here is coarse enough to be caught
#   without the two-pass split, and both models tried score 17/17 under all three
#   strategies. The blog's gap came from five domains of invariant-laden Dafny,
#   where a statement is long enough that a model shown the requirement first can
#   read the formal text through it. Making these traps subtle enough to separate
#   the strategies would mean planting divergences whose *labels* are arguable,
#   which would make the answer key worse, not the example better. If you want
#   the ablation to bite, extend MAPPING with a harder domain rather than
#   sharpening these.

import argparse
import asyncio
import dataclasses
import enum
import pathlib
import re
import sys
import textwrap
import typing

import pydantic.dataclasses

from effectful.handlers.llm import Agent, Template

# ---------------------------------------------------------------------------
# The corpus. A verified election tally: `count` tallies the ballots cast for a
# candidate, and each theorem below was written to formalize one plain-English
# requirement. Every one of them compiles (see `--verify`).
#
# The two namespaces are two versions of the same specification file, the way
# upstream's demo has `election0.dfy` and the `election.dfy` that replaced it.
# `Draft` is what was shipped; `Audited` is what survived review. They use the
# same theorem names deliberately -- a developer names a theorem after the
# property they *meant* to prove, which is exactly why the name is no evidence
# that they proved it. Only the statement is ever shown to the model, so neither
# the namespace nor this commentary reaches it.
# ---------------------------------------------------------------------------

CORPUS = r"""import Mathlib

/-- Tally: how many of the ballots in `bs` were cast for candidate `c`. -/
def count : List ℕ → ℕ → ℕ
  | [], _ => 0
  | b :: bs, c => (if b = c then 1 else 0) + count bs c

namespace Audited

theorem empty_election (c : ℕ) : count [] c = 0 := by
  simp [count]

theorem single_ballot_for (c : ℕ) : count [c] c = 1 := by
  simp [count]

theorem single_ballot_against (c d : ℕ) (h : c ≠ d) : count [c] d = 0 := by
  simp [count, h]

theorem count_bounded (bs : List ℕ) (c : ℕ) : count bs c ≤ bs.length := by
  induction bs with
  | nil => simp [count]
  | cons b bs ih =>
    simp only [count, List.length_cons]
    split <;> omega

theorem combine_tallies (a b : List ℕ) (c : ℕ) :
    count (a ++ b) c = count a c + count b c := by
  induction a with
  | nil => simp [count]
  | cons x xs ih =>
    simp only [List.cons_append, count, ih]
    omega

theorem vote_increment (bs : List ℕ) (c : ℕ) :
    count (bs ++ [c]) c = count bs c + 1 := by
  rw [combine_tallies, single_ballot_for]

theorem vote_no_effect (bs : List ℕ) (c d : ℕ) (h : c ≠ d) :
    count (bs ++ [d]) c = count bs c := by
  rw [combine_tallies, single_ballot_against d c (Ne.symm h)]
  omega

theorem order_irrelevant {a b : List ℕ} (h : a.Perm b) (c : ℕ) :
    count a c = count b c := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp only [count, ih]
  | swap x y l => simp only [count]; omega
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

theorem unanimous_tally (bs : List ℕ) (c : ℕ) (h : ∀ x ∈ bs, x = c) :
    count bs c = bs.length := by
  induction bs with
  | nil => simp [count]
  | cons b bs ih =>
    have hb : b = c := h b (by simp)
    have hrest : ∀ x ∈ bs, x = c := fun x hx => h x (by simp [hx])
    simp [count, hb, ih hrest]
    omega

theorem unanimous_exclusion (bs : List ℕ) (c d : ℕ) (h : ∀ x ∈ bs, x = c)
    (hne : d ≠ c) : count bs d = 0 := by
  induction bs with
  | nil => simp [count]
  | cons b bs ih =>
    have hb : b = c := h b (by simp)
    have hrest : ∀ x ∈ bs, x = c := fun x hx => h x (by simp [hx])
    simp [count, hb, Ne.symm hne, ih hrest]

theorem tally_monotonic (bs : List ℕ) (v c : ℕ) :
    count bs c ≤ count (bs ++ [v]) c := by
  rw [combine_tallies]
  omega

end Audited

namespace Draft

theorem count_bounded (bs : List ℕ) (c : ℕ) : count bs c ≤ bs.length + 1 := by
  have := Audited.count_bounded bs c
  omega

theorem vote_increment (bs : List ℕ) (c : ℕ) :
    count (bs ++ [c]) c = count (bs ++ [c]) c := rfl

theorem order_irrelevant (bs : List ℕ) (v c : ℕ) :
    count ([v] ++ bs) c = count (bs ++ [v]) c := by
  rw [Audited.combine_tallies, Audited.combine_tallies]
  omega

set_option linter.unusedVariables false in
theorem unanimous_tally (bs : List ℕ) (c : ℕ) (hbig : 100 < bs.length)
    (h : ∀ x ∈ bs, x = c) : count bs c = bs.length :=
  Audited.unanimous_tally bs c h

theorem tally_monotonic (bs : List ℕ) (v c : ℕ) : 0 ≤ count (bs ++ [v]) c :=
  Nat.zero_le _

end Draft
"""


def statement_of(corpus: str, qualified: str) -> str:
    """Extract the *statement* of ``<namespace>.<theorem>`` from Lean source: the
    text from ``theorem <name>`` up to the ``:=`` that begins its proof.

    Only this crosses the model boundary. The proof is dropped because ClaimCheck
    assumes it correct and audits the claim, and the enclosing namespace is
    dropped because it says which version of the file a theorem came from --
    which the auditor is precisely not entitled to know.
    """
    namespace, _, name = qualified.rpartition(".")
    section = corpus
    if namespace:
        start = section.index(f"namespace {namespace}")
        end = section.index(f"end {namespace}", start)
        section = section[start:end]
    match = re.search(rf"^theorem {re.escape(name)}\b", section, re.MULTILINE)
    if match is None:
        raise KeyError(f"no theorem {qualified!r} in the corpus")
    # The proof begins at the first `:=` at or after the statement; no statement
    # in this corpus contains one, so the first occurrence is the right one.
    body = section[match.start() :]
    return textwrap.dedent(body[: body.index(":=")]).strip()


# ---------------------------------------------------------------------------
# The mapping: which theorem was written to formalize which requirement, plus
# the ground truth. Upstream's `test/integration/mappings/*.json` carry exactly
# these `expected`/`reason` labels, which is what makes the audit scoreable
# rather than merely demonstrable.
# ---------------------------------------------------------------------------


class Verdict(enum.StrEnum):
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"


class Weakening(enum.StrEnum):
    """How a theorem can fail to mean its requirement -- the blog's taxonomy."""

    NONE = "none"
    TAUTOLOGY = "tautology"
    WEAKENED_CONCLUSION = "weakened-conclusion"
    NARROWED_SCOPE = "narrowed-scope"
    MISSING_CASE = "missing-case"
    WRONG_PROPERTY = "wrong-property"


@dataclasses.dataclass(frozen=True)
class Claim:
    """A requirement, the theorem said to formalize it, and the labelled truth.

    This is Python-side bookkeeping: ``expected`` and ``why`` are the answer key
    and never cross the model boundary.
    """

    requirement: str
    theorem: str
    expected: Verdict
    why: str = ""


MAPPING: tuple[Claim, ...] = (
    # The eleven audited theorems: each says what it was written to say.
    Claim(
        "In an empty election with no ballots, every candidate has zero votes",
        "Audited.empty_election",
        Verdict.CONFIRMED,
    ),
    Claim(
        "A single ballot cast for a candidate gives that candidate exactly one vote",
        "Audited.single_ballot_for",
        Verdict.CONFIRMED,
    ),
    Claim(
        "A single ballot cast for one candidate gives every other candidate zero votes",
        "Audited.single_ballot_against",
        Verdict.CONFIRMED,
    ),
    Claim(
        "No candidate can receive more votes than the total number of ballots cast",
        "Audited.count_bounded",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Merging two ballot boxes produces a tally equal to the sum of the "
        "individual tallies",
        "Audited.combine_tallies",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Casting a ballot for a candidate increases that candidate's tally by "
        "exactly one",
        "Audited.vote_increment",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Casting a ballot for one candidate does not change any other candidate's "
        "tally",
        "Audited.vote_no_effect",
        Verdict.CONFIRMED,
    ),
    Claim(
        "The order in which ballots are counted does not affect the final tally",
        "Audited.order_irrelevant",
        Verdict.CONFIRMED,
    ),
    Claim(
        "In a unanimous election, the winning candidate's tally equals the total "
        "number of ballots",
        "Audited.unanimous_tally",
        Verdict.CONFIRMED,
    ),
    Claim(
        "In a unanimous election, every other candidate receives zero votes",
        "Audited.unanimous_exclusion",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Adding a ballot to the election cannot cause any candidate's tally to "
        "decrease",
        "Audited.tally_monotonic",
        Verdict.CONFIRMED,
    ),
    # A faithful theorem against a requirement that asks for strictly more: the
    # theorem is true and proved, and covers half of what was asked for.
    Claim(
        "Casting a ballot for a candidate increases that candidate's tally by "
        "exactly one and leaves every other candidate's tally unchanged",
        "Audited.vote_increment",
        Verdict.DISPUTED,
        "missing-case: covers the chosen candidate only, and says nothing about "
        "the others",
    ),
    # The five drafted theorems: all proved, none of them the claim intended.
    Claim(
        "No candidate can receive more votes than the total number of ballots cast",
        "Draft.count_bounded",
        Verdict.DISPUTED,
        "weakened-conclusion: bounds the tally by one more than the number of "
        "ballots, permitting a candidate to exceed it",
    ),
    Claim(
        "Casting a ballot for a candidate increases that candidate's tally by "
        "exactly one",
        "Draft.vote_increment",
        Verdict.DISPUTED,
        "tautology: both sides of the equation are the same term, so nothing is "
        "claimed about the increment",
    ),
    Claim(
        "The order in which ballots are counted does not affect the final tally",
        "Draft.order_irrelevant",
        Verdict.DISPUTED,
        "narrowed-scope: only one reordering -- moving a single ballot from the "
        "end to the front -- rather than any permutation",
    ),
    Claim(
        "In a unanimous election, the winning candidate's tally equals the total "
        "number of ballots",
        "Draft.unanimous_tally",
        Verdict.DISPUTED,
        "narrowed-scope: an extra hypothesis restricts the guarantee to elections "
        "of more than 100 ballots",
    ),
    Claim(
        "Adding a ballot to the election cannot cause any candidate's tally to "
        "decrease",
        "Draft.tally_monotonic",
        Verdict.DISPUTED,
        "wrong-property: says the new tally is non-negative, which is trivially "
        "true of a natural number, instead of comparing it to the old one",
    ),
)


# ---------------------------------------------------------------------------
# Types crossing the model boundary. A field's ``metadata={"description": ...}``
# is inlined by pydantic into that field's JSON schema and rendered into the
# system prompt, so per-field guidance reaches the model through the type and no
# prompt has to restate it.
# ---------------------------------------------------------------------------


class Strength(enum.StrEnum):
    TRIVIAL = "trivial"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


@pydantic.dataclasses.dataclass(frozen=True)
class Informalization:
    """Pass 1's output: what a Lean statement says, read on its own terms.

    Produced without sight of the requirement the theorem was written for, which
    is the whole mechanism -- a back-translation that agreed with the requirement
    because it had been shown the requirement would be worth nothing.
    """

    natural_language: str = dataclasses.field(
        metadata={
            "description": "One sentence of plain English for what this theorem "
            "guarantees. Be literal: describe what the statement says, not what "
            "you suppose its author was aiming at."
        }
    )
    hypotheses: str = dataclasses.field(
        metadata={
            "description": "What must hold for the guarantee to apply, in English; "
            "'none' if the statement holds unconditionally."
        }
    )
    conclusion: str = dataclasses.field(
        metadata={"description": "What is guaranteed, in English."}
    )
    scope: str = dataclasses.field(
        metadata={
            "description": "What the guarantee ranges over: all ballot sequences, "
            "one particular sequence, sequences satisfying some restriction, etc."
        }
    )
    strength: Strength = dataclasses.field(
        metadata={
            "description": "'trivial' if the conclusion restates a hypothesis or "
            "holds for every value of the types involved regardless (e.g. a "
            "natural number being non-negative, or both sides of an equation "
            "being the same term); 'weak' if it says very little; 'moderate' if "
            "it is a substantive claim; 'strong' if it constrains behaviour "
            "sharply."
        }
    )
    confidence: float = dataclasses.field(
        metadata={"description": "0-1, how sure you are this reading is faithful."}
    )


@pydantic.dataclasses.dataclass(frozen=True)
class Comparison:
    """Pass 2's verdict on one requirement/theorem pair.

    ``__post_init__`` certifies the verdict is internally coherent before it is
    ever returned: a match is exactly a `Weakening.NONE`, and a mismatch has to
    say what is wrong. An incoherent answer raises, and `RetryLLMHandler` hands
    the message back to the model as the next turn -- so "matches, but it's a
    tautology" is not a verdict this pipeline can emit.
    """

    match: bool = dataclasses.field(
        metadata={
            "description": "True only if the theorem expresses the whole of the "
            "requirement. A theorem that is stronger than the requirement still "
            "matches; one that is weaker, narrower, or about something else "
            "does not."
        }
    )
    weakening: Weakening = dataclasses.field(
        metadata={
            "description": "The category of divergence: 'none' when and only when "
            "match is true."
        }
    )
    discrepancy: str = dataclasses.field(
        metadata={
            "description": "What the requirement asks for that the theorem does "
            "not deliver. Empty when match is true."
        }
    )
    explanation: str = dataclasses.field(
        metadata={"description": "Brief reasoning for the verdict."}
    )

    def __post_init__(self) -> None:
        if self.match and self.weakening is not Weakening.NONE:
            raise ValueError(
                f"incoherent verdict: match is true but weakening is "
                f"{self.weakening.value!r}. If the theorem really expresses the "
                "requirement the weakening is 'none'; otherwise match is false."
            )
        if not self.match and self.weakening is Weakening.NONE:
            raise ValueError(
                "incoherent verdict: match is false but weakening is 'none'. "
                "Name the category of the divergence."
            )
        if not self.match and not self.discrepancy.strip():
            raise ValueError(
                "match is false but no discrepancy is given; say what the "
                "requirement asks for that the theorem does not deliver."
            )

    @property
    def verdict(self) -> Verdict:
        return Verdict.CONFIRMED if self.match else Verdict.DISPUTED


# ---------------------------------------------------------------------------
# Pass 1. `informalize` takes a Lean statement and nothing else: there is no
# parameter for a requirement, this Agent's history contains no turn in which
# one appeared, and no Tool in scope can go and find one. That is the entire
# separation -- not an instruction the model is trusted to obey.
# ---------------------------------------------------------------------------


class Informalizer(Agent):
    """You read Lean 4 theorem statements and say, in plain English, exactly what
    they guarantee. You are a translator, not a sympathetic reader: you report
    what the statement says, never what you imagine it was for. You are not shown
    why any theorem was written, and you should not speculate about it."""

    @Template.define
    def informalize(self, statement: str) -> Informalization:
        """Translate this Lean 4 theorem statement into English, as literally as
        you can.

        ```lean
        {statement}
        ```

        Separate what is assumed (the hypotheses) from what is guaranteed (the
        conclusion), and say what the guarantee ranges over. Then rate how much
        the statement actually claims -- be blunt about this. A conclusion that
        holds for every value of the types involved, or that merely repeats a
        hypothesis, is trivial no matter how substantial the theorem's name
        makes it sound.

        Read only the statement in front of you. Do not guess at intent.
        """


# ---------------------------------------------------------------------------
# Pass 2. This agent does see the requirement -- comparing is its job. What it
# gets from pass 1 is a reading of the formal statement produced in ignorance of
# that requirement, so agreement between them is evidence.
# ---------------------------------------------------------------------------


class Comparator(Agent):
    """You check whether a formal theorem carries the weight a natural-language
    requirement puts on it. You assume the proof is correct: you are not auditing
    the proof, you are auditing the claim. You are strict -- a theorem that is
    true, proved, and beside the point is a finding -- but not pedantic about
    wording, since only the meaning has to survive."""

    @Template.define
    def compare(
        self, requirement: str, statement: str, back_translation: Informalization
    ) -> Comparison:
        """Decide whether this theorem expresses this requirement.

        **Requirement, as written by the person who asked for it:**
        {requirement}

        **The theorem said to formalize it:**
        ```lean
        {statement}
        ```

        **Back-translation** -- what the statement says, according to a reader
        who was shown the statement alone and never saw the requirement above:
        {back_translation}

        Watch for the ways a proved theorem can still miss:

        1. **tautology** -- the conclusion restates a hypothesis, or holds for
           every value of the types involved, so nothing is established.
        2. **weakened-conclusion** -- the theorem guarantees less than was asked
           (a looser bound, a weaker relation).
        3. **narrowed-scope** -- an extra hypothesis, or a less general shape,
           restricts the guarantee to a subset of the cases the requirement
           covers.
        4. **missing-case** -- the requirement asks for several things and the
           theorem delivers some of them.
        5. **wrong-property** -- the theorem is about something else, however
           adjacent.

        A theorem *stronger* than the requirement still matches; do not flag
        rephrasing. But if the back-translation rates the statement trivial, the
        requirement had better be trivial too. Judge the statement, not its name:
        a theorem called after the property it was meant to prove is no evidence
        that it proves it.
        """


# ---------------------------------------------------------------------------
# The ablations. Both collapse the two passes into one call -- so the
# requirement is in scope for the model at the moment it reads the formal
# statement, and its reading of that statement can be shaped by what it already
# knows the answer is supposed to be. This is the failure mode the split exists
# to prevent, and `--strategy` is what makes it measurable rather than assumed --
# on this corpus it measures no difference at all, which is reported and not
# hidden.
# ---------------------------------------------------------------------------


class SinglePassAuditor(Agent):
    """You audit whether a formal theorem expresses a natural-language
    requirement, informalizing the theorem and then comparing, in one pass. You
    assume the proof is correct and audit the claim."""

    @Template.define
    def audit(self, requirement: str, statement: str) -> Comparison:
        """Check whether this theorem expresses this requirement.

        **The theorem:**
        ```lean
        {statement}
        ```

        First, state to yourself what the theorem guarantees and under what
        hypotheses, in plain English, before you read any further.

        **Requirement:** {requirement}

        Now compare the two, watching for a conclusion that restates a
        hypothesis or holds trivially (**tautology**), one that guarantees less
        than was asked (**weakened-conclusion**), an extra hypothesis or less
        general shape (**narrowed-scope**), a requirement only partly delivered
        (**missing-case**), or a theorem about something else entirely
        (**wrong-property**).

        A theorem stronger than the requirement still matches. Judge the
        statement, not its name.
        """


class NaiveAuditor(Agent):
    """You check whether formal theorems match the requirements they are said to
    formalize."""

    @Template.define
    def audit(self, requirement: str, statement: str) -> Comparison:
        """Does this Lean theorem faithfully capture the requirement below?

        **Requirement:** {requirement}

        ```lean
        {statement}
        ```

        Answer with a match verdict; if it does not match, categorize how and say
        what is missing.
        """


# ---------------------------------------------------------------------------
# Driving one claim through a mode. Each claim gets its own agent instances, so
# nothing an audit learns can leak into the next one through a shared history.
# ---------------------------------------------------------------------------


class Mode(enum.StrEnum):
    TWO_PASS = "two-pass"
    SINGLE_PASS = "single-pass"
    NAIVE = "naive"


@dataclasses.dataclass(frozen=True)
class Audit:
    """One claim's result: what the pipeline decided, and what it read on the way."""

    claim: Claim
    statement: str
    comparison: Comparison
    back_translation: Informalization | None  # None outside two-pass mode

    @property
    def correct(self) -> bool:
        return self.comparison.verdict is self.claim.expected


def audit_claim(claim: Claim, mode: Mode) -> Audit:
    """Audit one requirement/theorem pair under the given mode."""
    statement = statement_of(CORPUS, claim.theorem)

    if mode is Mode.TWO_PASS:
        # Pass 1 receives `statement`. There is nowhere in this call for
        # `claim.requirement` to go.
        back = Informalizer().informalize(statement)
        comparison = Comparator().compare(claim.requirement, statement, back)
        return Audit(claim, statement, comparison, back)

    auditor = SinglePassAuditor() if mode is Mode.SINGLE_PASS else NaiveAuditor()
    return Audit(claim, statement, auditor.audit(claim.requirement, statement), None)


async def audit_all(claims: typing.Sequence[Claim], mode: Mode) -> list[Audit]:
    """Audit every claim concurrently -- they are independent by construction."""
    return list(
        await asyncio.gather(
            *(asyncio.to_thread(audit_claim, claim, mode) for claim in claims)
        )
    )


# ---------------------------------------------------------------------------
# The deterministic half. The blog's between-pass checks are diagnostics over
# typed values, so they are a loop in Python rather than another model call:
# code does what code can decide, and the model is asked only what needs
# judgment.
# ---------------------------------------------------------------------------


def pre_checks(audits: typing.Sequence[Audit]) -> list[str]:
    """Flag back-translations rated trivial, and distinct requirements whose
    theorems were read as guaranteeing the same thing."""
    notes: list[str] = []
    seen: dict[str, Claim] = {}
    for audit in audits:
        if (back := audit.back_translation) is None:
            continue
        if back.strength is Strength.TRIVIAL:
            notes.append(
                f"{audit.claim.theorem} was read as a trivial claim ({back.conclusion})"
            )
        key = " ".join(back.conclusion.lower().split())
        if (earlier := seen.get(key)) is not None:
            if earlier.requirement != audit.claim.requirement:
                notes.append(
                    f"{audit.claim.theorem} and {earlier.theorem} were read as "
                    "guaranteeing the same thing, but formalize different "
                    "requirements"
                )
        else:
            seen[key] = audit.claim
    return notes


# ---------------------------------------------------------------------------
# Reporting. The two error directions are reported apart: a missed dispute is a
# weak theorem waved through, which is the failure ClaimCheck exists to prevent,
# while a false dispute costs a developer an argument with the tool.
# ---------------------------------------------------------------------------


def report(audits: typing.Sequence[Audit], mode: Mode) -> None:
    print(f"\n{'=' * 78}\nClaimCheck audit -- strategy: {mode.value}\n{'=' * 78}\n")

    for audit in audits:
        got = audit.comparison.verdict
        mark = "ok " if audit.correct else "MISS"
        category = (
            ""
            if audit.comparison.weakening is Weakening.NONE
            else f" [{audit.comparison.weakening.value}]"
        )
        print(f"[{mark}] {audit.claim.theorem}: {got.value}{category}")
        print(f"       requirement: {audit.claim.requirement}")
        print(f"       statement:   {' '.join(audit.statement.split())}")
        if audit.back_translation is not None:
            back = audit.back_translation
            print(
                f"       read as:     {back.natural_language} "
                f"(strength: {back.strength.value})"
            )
        if audit.comparison.discrepancy:
            print(f"       discrepancy: {audit.comparison.discrepancy}")
        if not audit.correct:
            print(f"       EXPECTED {audit.claim.expected.value}: {audit.claim.why}")
        print()

    if notes := pre_checks(audits):
        print("Pre-check diagnostics (deterministic, no model involved):")
        for note in notes:
            print(f"  - {note}")
        print()

    # A "missed dispute" is an unfaithful theorem the audit confirmed.
    missed = [
        a for a in audits if a.claim.expected is Verdict.DISPUTED and not a.correct
    ]
    false_alarms = [
        a for a in audits if a.claim.expected is Verdict.CONFIRMED and not a.correct
    ]
    correct = sum(a.correct for a in audits)
    print(
        f"Accuracy: {correct}/{len(audits)} "
        f"({correct / len(audits):.1%})\n"
        f"  unfaithful theorems waved through: {len(missed)}"
        + (f" ({', '.join(a.claim.theorem for a in missed)})" if missed else "")
        + f"\n  faithful theorems disputed:        {len(false_alarms)}"
        + (
            f" ({', '.join(a.claim.theorem for a in false_alarms)})"
            if false_alarms
            else ""
        )
    )


# ---------------------------------------------------------------------------
# The premise, checked. ClaimCheck is only interesting if the formal artifacts
# really are proved -- otherwise a disputed theorem might just be a broken one.
# `formalization.py` (LEAP) already drives a real Lean 4 + Mathlib toolchain, so
# the check reuses its kernel rather than restating it. Imported inside the
# function, as `world_model_agent.py` imports `gridworlds`, so the example has no
# Lean dependency unless the check is asked for.
# ---------------------------------------------------------------------------


def verify_corpus() -> bool:
    """Compile the whole corpus with Lean, and report what it proves."""
    # The examples are importable as ``docs.source.llm_examples...`` from the
    # repository root, which is on ``sys.path`` under the harness but not when
    # this file is run directly; add it so both invocations work.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
    from docs.source.llm_examples.autoresearch.formalization import (
        _SORRY,
        LeanKernel,
    )

    kernel = LeanKernel()
    if not kernel.available():
        print(
            f"Lean project not built at {kernel.project!r}; skipping verification.\n"
            "Build it once (see autoresearch/formalization.py --check-toolchain):\n"
            "  elan default stable\n"
            f"  cd {kernel.project} && lake exe cache get && lake build"
        )
        return False

    theorems = re.findall(r"^theorem (\w+)", CORPUS, re.MULTILINE)
    print(f"Compiling {len(theorems)} theorems with Lean 4 + Mathlib ...")
    result = kernel.compile(CORPUS)
    if not result.ok:
        raise SystemExit(f"The corpus does not compile:\n{result.messages}")
    if _SORRY.search(CORPUS):
        raise SystemExit("The corpus contains `sorry`; its theorems are not proved.")
    print(
        f"VERIFIED: {len(theorems)} theorems, 0 errors, no `sorry`. Every claim "
        "below is proved.\n"
        "The audit that follows is not about whether they are true.\n"
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Named `--strategy`, not `--mode`: the harness parses its own flags with
    # argparse's prefix matching on, so a script flag named `--mode` is swallowed
    # as an abbreviation of the harness's `--model`.
    parser.add_argument(
        "--strategy",
        type=Mode,
        choices=list(Mode),
        default=Mode.TWO_PASS,
        help="Audit strategy: the two-pass split (informalizer never sees the "
        "requirement), one combined call, or a bare 'does this match?'",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compile the corpus with a real Lean toolchain first, establishing "
        "that every theorem audited below is actually proved",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Compile the corpus and exit, without calling any model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit only the first N claims (a cheap smoke test)",
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_corpus()
        return
    if args.verify:
        verify_corpus()

    claims = MAPPING[: args.limit] if args.limit else MAPPING
    report(asyncio.run(audit_all(claims, args.strategy)), args.strategy)


if __name__ == "__main__":
    main()
