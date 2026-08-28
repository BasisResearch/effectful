import enum

import pydantic.dataclasses

from effectful.handlers.llm import Skill


class NaiveVerdict(enum.StrEnum):
    """
    - **JUSTIFIED** if a theorem's statement expresses the requirement (it
        may be stronger, that's fine).
    - **NOT_JUSTIFIED** if there is a meaningful discrepancy: a theorem is
        weaker, proves something different, is vacuous, or misses key aspects.
    """

    JUSTIFIED = "JUSTIFIED"
    NOT_JUSTIFIED = "NOT_JUSTIFIED"


@pydantic.dataclasses.dataclass(frozen=True)
class NaiveJudgement:
    """A verdict and a sentence of justification."""

    verdict: NaiveVerdict
    explanation: str

    @property
    def match(self) -> bool:
        return self.verdict is NaiveVerdict.JUSTIFIED


class NaiveAuditor:
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

        Invariant hypotheses (e.g. ``Inv m``) are expected and normal -- don't
        count them as discrepancies. A theorem that extracts a concrete
        consequence from an invariant is useful, not vacuous.
        """
