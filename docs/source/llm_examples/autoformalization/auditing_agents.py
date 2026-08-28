"""Agents and model-boundary types for the ClaimCheck audit in `auditing.py`.

Two agents read Lean theorem statements and judge whether they express a
natural-language requirement, plus the typed values that cross the model
boundary.
"""

import dataclasses
import enum
import typing

import pydantic.dataclasses

from effectful.handlers.llm import Skill


class Verdict(enum.StrEnum):
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"


class Weakening(enum.StrEnum):
    """
    How a theorem can fail to mean its requirement -- the blog's taxonomy.
    'none' when and only when the theorem expresses the requirement.
    Others are the ways a proved theorem can still miss:

        1. **tautology** -- the conclusion restates a hypothesis, or holds for
           every value of the types involved, so nothing is established.
        2. **weakened-conclusion** -- the theorem guarantees less than was asked
           (a looser bound, a weaker relation).
        3. **narrowed-scope** -- the theorem only covers a subset of the
           cases the requirement describes.
        4. **missing-case** -- the requirement asks for several things and the
           theorem delivers some of them.
        5. **wrong-property** -- the theorem is about something else, however
           adjacent.
    """

    NONE = "none"
    TAUTOLOGY = "tautology"
    WEAKENED_CONCLUSION = "weakened-conclusion"
    NARROWED_SCOPE = "narrowed-scope"
    MISSING_CASE = "missing-case"
    WRONG_PROPERTY = "wrong-property"


class Strength(enum.StrEnum):
    """
    'trivial' if the conclusion restates a hypothesis or
    holds for every value of the types involved regardless (e.g. a
    natural number being non-negative, or both sides of an equation
    being the same term); 'weak' if it says very little; 'moderate' if
    it is a substantive claim; 'strong' if it constrains behaviour sharply.
    """

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
    strength: Strength
    confidence: typing.Annotated[float, pydantic.Field(ge=0, le=1)]


@pydantic.dataclasses.dataclass(frozen=True)
class Comparison:
    """Pass 2's verdict on one requirement/theorem pair.

    ``match`` is True only if the theorem expresses the whole of the requirement.
    A theorem that is stronger than the requirement still matches;
    one that is weaker, narrower, or about something else does not.

    ``__post_init__`` certifies the verdict is internally coherent before it is
    ever returned: a match is exactly a `Weakening.NONE`, and a mismatch has to
    say what is wrong. An incoherent answer raises.
    """

    match: bool
    weakening: Weakening
    discrepancy: str = dataclasses.field(
        metadata={
            "description": "What the requirement asks for that the theorem does "
            "not deliver. Empty when match is true."
        }
    )
    explanation: str

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


class Informalizer:
    """You read Lean 4 theorem statements and say, in plain English, exactly what
    they guarantee. You are a translator, not a sympathetic reader: you report
    what the statement says, never what you imagine it was for. You are not shown
    why any theorem was written, and you should not speculate about it."""

    @Skill.define
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


class Comparator:
    """You check whether a formal theorem carries the weight a natural-language
    requirement puts on it. You assume the proof is correct: you are not auditing
    the proof, you are auditing the claim. You are strict -- a theorem that is
    true, proved, and beside the point is a finding -- but not pedantic about
    wording, since only the meaning has to survive."""

    @Skill.define
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

        A theorem *stronger* than the requirement still matches; do not flag
        rephrasing. But if the back-translation rates the statement trivial, the
        requirement had better be trivial too. Judge the statement, not its name:
        a theorem called after the property it was meant to prove is no evidence
        that it proves it.
        """
