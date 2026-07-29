"""The single-pass ablation: informalize and compare in one call.

Alone in a module for the same reason `auditing_naive` is. The harness builds a
template's system prompt from the source of the module the template is defined
in, so an agent defined beside this one would be shown this one's prompt --
including the invariant caveat below, which upstream gives to its single-pass
and naive prompts and pointedly *not* to its two-pass comparator. Keeping that
asymmetry is the whole point of running the arms against each other, and a
shared module silently destroys it, in a way nothing in the prompts would show.
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
        # The caveat above is upstream's CLAIMCHECK_PROMPT, kept because this arm
        # is a port of that prompt. `Comparator` gets none, matching upstream.
