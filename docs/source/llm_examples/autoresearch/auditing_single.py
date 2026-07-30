"""The single-pass ablation: informalize and compare in one call.

Alone in a module because the harness builds a template's system prompt from the
source of the module the template is defined in, so agents sharing a file are
shown each other's prompts. The arms differ, so they do not share a file.
"""

from auditing_agents import Comparison

from effectful.handlers.llm import Agent, Template


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
        than was asked (**weakened-conclusion**), a theorem covering only a subset of
        the cases described (**narrowed-scope**), a requirement only partly delivered
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
