"""The naive audit arm, deliberately alone in its own module.

This arm is the ablation's floor: one call, a yes/no, and a sentence. Upstream's
`NAIVE_TOOL` is exactly that -- no weakening taxonomy, no "a stronger theorem
still matches" rule.

It lives apart from `auditing_compare` because a template's system prompt
includes the source of its defining module. Sharing a file with `Comparison` and
the richer agents would hand this arm the five-category taxonomy and the other
arms' instructions through the module source, however impoverished its own
return type and prompt were -- and the ablation would be comparing two spellings
of the same guidance instead of a structural difference. Measured: with the
naive agent in the shared module, its prompt still contained every category
name.
"""

import dataclasses
import enum

import pydantic.dataclasses

from effectful.handlers.llm import Agent, Template


class NaiveVerdict(enum.StrEnum):
    JUSTIFIED = "JUSTIFIED"
    NOT_JUSTIFIED = "NOT_JUSTIFIED"


@pydantic.dataclasses.dataclass(frozen=True)
class NaiveJudgement:
    """The naive arm's entire output: a yes/no and a sentence.

    Deliberately impoverished, mirroring upstream's ``NAIVE_TOOL``. The richer
    `Comparison` carries the five-category weakening taxonomy and the rule that a
    stronger theorem still matches -- and a return type is part of the prompt,
    since its JSON schema is rendered into the system message. Letting the naive
    arm return a `Comparison` would hand it the very guidance the elaborate arms
    exist to supply, so the ablation would be comparing two spellings of the same
    instructions rather than a structural difference.
    """

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

    @Template.define
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

        Invariant hypotheses (e.g. ``Wf p``) are expected and normal -- don't
        count them as discrepancies. A theorem that extracts a concrete
        consequence from an invariant is useful, not vacuous.
        """
