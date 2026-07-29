"""Agents and model-boundary types for the ClaimCheck audit in `auditing.py`.

Four agents read Lean theorem statements and judge whether they express a
natural-language requirement, plus the typed values that cross the model
boundary.

**This module is deliberately small, and that is the point.** The harness builds
a template's system prompt partly from the source of the module the template is
defined in (see the prompt-assembly table in
`effectful.handlers.llm.types.Template`), so everything sharing a file with an
`Agent` is shown to it verbatim. Keeping these agents in a module of their own is
what makes it true that the informalizer sees a Lean statement and nothing else.
Nothing here knows anything about the material being audited; `auditing.py`
imports this module and is never imported by it.
"""

import dataclasses
import enum

import pydantic.dataclasses

from effectful.handlers.llm import Agent, Template


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
            "description": "What the guarantee ranges over: every state of the "
            "system, one particular state, states satisfying some restriction, "
            "etc."
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
        # No invariant caveat here, deliberately: upstream's
        # ROUNDTRIP_COMPARE_PROMPT has none, while its NAIVE_PROMPT and
        # CLAIMCHECK_PROMPT both do. That asymmetry is upstream's, and handing
        # this arm a caveat it was never given would be scoring a different
        # experiment. The blind pass is supposed to be what makes the caveat
        # unnecessary -- that is the claim under test.


# ---------------------------------------------------------------------------
# The ablations. Both collapse the two passes into one call, so the requirement
# is in scope at the moment the model reads the formal statement and its reading
# of that statement can be shaped by it. That is the failure mode the split
# exists to prevent; whether it actually costs anything is what `--strategy`
# measures.
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

        An invariant hypothesis (a named predicate such as ``Inv m``, whose
        definition you have not been shown) is expected and normal -- do not flag
        it as a narrowing. A theorem that extracts a concrete consequence from an
        invariant is useful, not vacuous: ``(h : Inv m) : 0 ≤ m`` is a real
        guarantee. But do flag a hypothesis that restricts *when* the property
        holds.
        """
        # The caveat above is upstream's CLAIMCHECK_PROMPT, kept because this arm
        # is a port of that prompt. `Comparator` gets none, matching upstream.
