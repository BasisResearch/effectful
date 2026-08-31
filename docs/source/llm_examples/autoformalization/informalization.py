"""ClaimCheck: auditing whether a proved theorem is the theorem you meant.

Implements the core of ClaimCheck ("Narrowing the Gap Between Proof and Intent",
https://midspiral.com/blog/claimcheck-narrowing-the-gap-between-proof-and-intent/,
reference implementation at https://github.com/metareflection/claimcheck, MIT).
Its diagnosis is that a verifier proves code matches its *specification* and says
nothing about whether the specification matches your *intent*. The motivating
case is a Dafny election tally whose ``TallyMonotonic`` lemma was supposed to say
"adding a ballot can't decrease a tally" and in fact said ``Count(...) >= 0`` --
trivially true, since counts are naturals. Dafny reported 14 verified, 0 errors.

Its fix is *round-trip informalization*. Pass 1 translates the formal statement,
and nothing else, back into English; pass 2 compares that back-translation
against the requirement it was meant to formalize. The load-bearing part is what
pass 1 does not get: having never seen the requirement, it cannot parrot it back,
so agreement in pass 2 is evidence rather than an echo.

That is a claim about *scope*, which is what makes it an effectful example. The
reference implementation enforces it by hand-assembling prompt strings, under a
comment reading ``CRITICAL: This prompt must NOT include the original
requirements``. Here it is enforced by the code's shape:

  * ``Informalizer.informalize(statement)`` has no parameter through which a
    requirement could arrive, its ``Agent`` history holds no turn in which one
    appeared, and no ``Tool`` in scope can fetch one.

  * ``Comparison`` certifies at decode time that a verdict is coherent: a match
    is exactly a `Weakening.NONE`, and a mismatch must name its discrepancy. An
    incoherent answer raises and `TenacityRetryer` hands it back as the next
    turn. Upstream's schema admits ``match: true`` alongside
    ``weakeningType: "tautology"`` with nothing to catch it.

  * The premise is checked by a real prover. Under ``--verify`` the corpora are
    compiled, 0 errors and no ``sorry``, by the Lean 4 + Mathlib toolchain that
    `verification.py` (LEAP) drives too -- so every theorem the audit questions
    is proved before it is questioned, and no disputed verdict below can be
    blamed on a broken proof.

**The corpus**, the requirements written against it, and the labelled ground
truth this run is scored against are `library.py`'s, and stay there. The harness
builds a skill's system prompt partly from the source of the module the skill is
*defined* in, so what an auditor must not see has to live in another file --
including prose about the benchmark, which is why the description of it is in
that module's docstring rather than this one.

Demonstrates:
- Structural separation as *lexical scope*, with the module rather than the
  signature as the boundary the framework actually respects
- Decode-time certification of a structured verdict's internal coherence, turning
  a self-contradictory answer into a `TenacityRetryer` retry
- Establishing a premise with a real external verifier (`library.py`'s `compile_lean`,
  the compiler `verification.py` proves against) rather than asserting it
- Labelled corpora and an accuracy report separating the two error directions
- Fan-out over independent audits with ``asyncio.gather`` + ``asyncio.to_thread``
"""

import argparse
import asyncio
import collections.abc
import enum
import pprint
import typing

import pydantic.dataclasses

from docs.source.llm_examples.autoformalization.library import Claim, Domain, Verdict
from effectful.handlers.llm import Skill


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

    natural_language: typing.Annotated[
        str,
        pydantic.Field(
            description="One sentence of plain English for what this theorem "
            "guarantees. Be literal: describe what the statement says, not what "
            "you suppose its author was aiming at."
        ),
    ]
    hypotheses: typing.Annotated[
        str,
        pydantic.Field(
            description="What must hold for the guarantee to apply, in English; "
            "'none' if the statement holds unconditionally."
        ),
    ]
    conclusion: typing.Annotated[
        str, pydantic.Field(description="What is guaranteed, in English.")
    ]
    scope: typing.Annotated[
        str,
        pydantic.Field(
            description="What the guarantee ranges over: every state of the system,"
            "one particular state, states satisfying some restriction, etc."
        ),
    ]
    strength: Strength
    confidence: typing.Annotated[float, pydantic.Field(ge=0, le=1)]


@pydantic.dataclasses.dataclass(frozen=True)
class Comparison:
    """Pass 2's verdict on one requirement/theorem pair.

    ``__post_init__`` certifies the verdict is internally coherent before it is
    ever returned: a match is exactly a `Weakening.NONE`, and a mismatch has to
    say what is wrong. An incoherent answer raises.
    """

    verdict: Verdict
    weakening: Weakening
    discrepancy: typing.Annotated[
        str,
        pydantic.Field(
            description="What the requirement asks for that the theorem does "
            "not deliver. Empty when match is true."
        ),
    ]

    def __post_init__(self) -> None:
        if self.verdict is Verdict.CONFIRMED and self.weakening is not Weakening.NONE:
            raise ValueError(
                f"incoherent verdict: match is true but weakening is "
                f"{self.weakening.value!r}. If the theorem really expresses the "
                "requirement the weakening is 'none'; otherwise match is false."
            )
        if self.verdict is Verdict.DISPUTED and self.weakening is Weakening.NONE:
            raise ValueError(
                "incoherent verdict: match is false but weakening is 'none'. "
                "Name the category of the divergence."
            )
        if self.verdict is Verdict.DISPUTED and not self.discrepancy.strip():
            raise ValueError(
                "match is false but no discrepancy is given; say what the "
                "requirement asks for that the theorem does not deliver."
            )


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


type Audit = tuple[Claim, Informalization, Comparison]


async def audit_domain(domain: Domain) -> collections.abc.Sequence[Audit]:
    """Audit all requirement/theorem pairs in a domain, concurrently."""

    async def audit_claim(domain: Domain, claim: Claim) -> Audit:
        statement = domain.statement_of(claim.theorem)
        informalization = await asyncio.to_thread(Informalizer().informalize, statement)
        comparison = await asyncio.to_thread(
            Comparator().compare, claim.requirement, statement, informalization
        )
        return claim, informalization, comparison

    return await asyncio.gather(
        *(audit_claim(domain, claim) for claim in domain.claims)
    )


def _report(domain: Domain, audits: collections.abc.Sequence[Audit]) -> None:
    print(f"\n{'=' * 78}\nClaimCheck audit: {domain.name}\n{'=' * 78}\n")

    for audit in audits:
        pprint.pprint(audit)

    seen: dict[str, Claim] = {}
    missed = []
    false_alarms = []
    correct = []
    for claim, back, comparison in audits:
        if back.strength is Strength.TRIVIAL:
            print(f"{claim.theorem} was read as a trivial claim ({back.conclusion})")
        key = " ".join(back.conclusion.lower().split())
        if (earlier := seen.get(key)) is not None:
            if earlier.requirement != claim.requirement:
                print(
                    f"{claim.theorem} and {earlier.theorem} were read as "
                    "guaranteeing the same thing, but formalize different "
                    "requirements"
                )
        else:
            seen[key] = claim

        if comparison.verdict is Verdict.DISPUTED:
            print(
                f"{claim.theorem} does not express its requirement: "
                f"{comparison.discrepancy}"
            )
            if claim.expected is Verdict.DISPUTED:
                correct.append((claim, back, comparison))
            else:
                false_alarms.append((claim, back, comparison))
        elif comparison.verdict is Verdict.CONFIRMED:
            if claim.expected is Verdict.DISPUTED:
                print(
                    f"{claim.theorem} expresses its requirement, but it was "
                    "expected to be disputed"
                )
                missed.append((claim, back, comparison))
            else:
                correct.append((claim, back, comparison))

    print(
        f"Accuracy: {len(correct)}/{len(audits)}"
        f"\n  unfaithful theorems waved through: {len(missed)}"
        f"\n  faithful theorems disputed:        {len(false_alarms)}"
    )


def main() -> None:
    from docs.source.llm_examples.autoformalization.library import DOMAINS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=[*DOMAINS, "all"],
        default="all",
        help="Which of upstream's five benchmark domains to audit, or all of them",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compile the corpus with a real Lean toolchain first, establishing "
        "that every theorem audited below is actually proved",
    )
    args = parser.parse_args()

    domains = list(DOMAINS.values()) if args.domain == "all" else [DOMAINS[args.domain]]
    if args.verify:
        if all(domain.verify_corpus for domain in domains):
            print(
                f"VERIFIED: {sum(len(d.theorems) for d in domains)} theorems, 0 errors, no "
                "`sorry`. Every claim below is proved.\nThe audit that follows is not "
                "about whether they are true.\n"
            )

    for domain in domains:
        print(f"Auditing {domain.name} ({len(domain.claims)} claims)...")
        audits = asyncio.run(audit_domain(domain))
        _report(domain, audits)


if __name__ == "__main__":
    main()
