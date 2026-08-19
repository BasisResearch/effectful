"""The naive audit arm: one call, a verdict, and a sentence.

Alone in a module because a skill's system prompt includes the source of its
defining module, so agents sharing a file are shown each other's prompts and
types. The arms differ, so they do not share a file.
"""

import dataclasses
import enum

import pydantic.dataclasses

from effectful.handlers.llm import Agent, Skill


class NaiveVerdict(enum.StrEnum):
    JUSTIFIED = "JUSTIFIED"
    NOT_JUSTIFIED = "NOT_JUSTIFIED"


@pydantic.dataclasses.dataclass(frozen=True)
class NaiveJudgement:
    """A verdict and a sentence of justification."""

    verdict: NaiveVerdict = dataclasses.field(
        metadata={
            "description": "JUSTIFIED if the theorem captures the requirement, "
            "NOT_JUSTIFIED if there is a meaningful discrepancy."
        }
    )
    explanation: str = dataclasses.field(
        metadata={"description": "Brief explanation of your verdict."}
    )

    @property
    def match(self) -> bool:
        return self.verdict is NaiveVerdict.JUSTIFIED


class NaiveAuditor(Agent):
    """You check whether verified Lean theorems correctly formalize the natural
    language requirements they are said to capture."""

    @Skill.define
    def audit(self, requirement: str, statement: str) -> NaiveJudgement:
        """Does this Lean theorem faithfully capture the requirement below?

        ## Natural Language Requirement

        > {requirement}

        ## Lean Theorem

        ```lean
        {statement}
        ```

        ## Instructions

        - **JUSTIFIED** if the theorem's statement expresses the requirement (it
          may be stronger, that's fine).
        - **NOT_JUSTIFIED** if there is a meaningful discrepancy: the theorem is
          weaker, proves something different, is vacuous, or misses key aspects.

        Invariant hypotheses (e.g. ``Inv m``) are expected and normal -- don't
        count them as discrepancies. A theorem that extracts a concrete
        consequence from an invariant is useful, not vacuous.
        """
