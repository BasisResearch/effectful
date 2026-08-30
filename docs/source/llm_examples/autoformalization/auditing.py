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

  * The premise is checked by a real prover. Under ``--verify`` all five corpora
    are compiled by the Lean 4 + Mathlib toolchain `formalization.py` already
    shells out to: 36 theorems, 0 errors, no ``sorry``. Then the audit
    finds nine claims that do not mean what they were written to mean.

**The corpus** transliterates upstream's benchmark item for item from
`test/integration/claims/*.dfy` and `test/integration/mappings/*.json`: five
domains, 36 requirement/theorem pairs, the same 27 faithful / 9 planted split,
upstream's requirement sentences verbatim and its lemma names in snake_case.

**The ablation** is ``--strategy``: the two-pass pipeline above, against
``naive`` -- one call, requirement and statement together, a yes/no and a
sentence, ported from upstream's `NAIVE_PROMPT`. That is the right comparator
for upstream's published middle rung rather than a soft target, since
`CLAIMCHECK_PROMPT` and `NAIVE_PROMPT` score identically on the same 36 items
(86.1% each).

Demonstrates:
- Structural separation as *lexical scope*, with the module rather than the
  signature as the boundary the framework actually respects
- Decode-time certification of a structured verdict's internal coherence, turning
  a self-contradictory answer into a `TenacityRetryer` retry
- Reuse of a sibling example's real external verifier (`formalization.py`'s
  `compile_lean`) to establish a premise, rather than asserting it
- Labelled corpora and an accuracy report separating the two error directions
- Fan-out over independent audits with ``asyncio.gather`` + ``asyncio.to_thread``
- Per-field guidance carried on the types as ``field(metadata={"description": ...})``
"""

import argparse
import asyncio
import collections.abc
import dataclasses
import enum
import pprint

from auditing_agents import (
    Comparator,
    Comparison,
    Informalization,
    Informalizer,
    Strength,
    Verdict,
)
from auditing_corpora import DOMAINS, Claim, Domain
from auditing_naive import NaiveAuditor


class Mode(enum.StrEnum):
    TWO_PASS = "two-pass"
    NAIVE = "naive"


@dataclasses.dataclass(frozen=True)
class Audit:
    """
    One claim's result: what the pipeline decided, and what it read on the way.
    """

    domain: str
    claim: Claim
    statement: str
    verdict: Verdict
    explanation: str
    comparison: Comparison | None = None
    back_translation: Informalization | None = None

    @property
    def correct(self) -> bool:
        return self.verdict is self.claim.expected

    @property
    def label(self) -> str:
        return f"{self.domain}/{self.claim.theorem}"


def audit_claim(domain: Domain, claim: Claim, mode: Mode) -> Audit:
    """Audit one requirement/theorem pair under the given mode.

    A claim the model cannot produce a decodable verdict for becomes an ``error``
    result rather than an exception: one intractable item should cost one item,
    not the other thirty-nine.
    """
    statement = domain.statement_of(claim.theorem)
    if mode is Mode.NAIVE:
        judgement = NaiveAuditor().audit(claim.requirement, statement)
        match, explanation = judgement.match, judgement.explanation
        verdict = Verdict.CONFIRMED if match else Verdict.DISPUTED
        return Audit(domain.name, claim, statement, verdict, explanation)
    else:
        back = Informalizer().informalize(statement)
        comparison = Comparator().compare(claim.requirement, statement, back)
        match, explanation = comparison.match, comparison.explanation
        verdict = Verdict.CONFIRMED if match else Verdict.DISPUTED
        return Audit(
            domain.name, claim, statement, verdict, explanation, comparison, back
        )


async def audit_all(
    domains: collections.abc.Sequence[Domain], mode: Mode
) -> list[Audit]:
    """Audit every claim in every domain concurrently -- independent by
    construction, since each gets its own agent instances."""
    return list(
        await asyncio.gather(
            *(
                asyncio.to_thread(audit_claim, domain, claim, mode)
                for domain in domains
                for claim in domain.claims
            )
        )
    )


def pre_checks(audits: collections.abc.Sequence[Audit]) -> list[str]:
    """Flag back-translations rated trivial, and distinct requirements whose
    theorems were read as guaranteeing the same thing."""
    notes: list[str] = []
    seen: dict[str, Claim] = {}
    for audit in audits:
        if (back := audit.back_translation) is None:
            continue
        if back.strength is Strength.TRIVIAL:
            notes.append(
                f"{audit.label} was read as a trivial claim ({back.conclusion})"
            )
        key = f"{audit.domain}: {' '.join(back.conclusion.lower().split())}"
        if (earlier := seen.get(key)) is not None:
            if earlier.requirement != audit.claim.requirement:
                notes.append(
                    f"{audit.label} and {earlier.theorem} were read as "
                    "guaranteeing the same thing, but formalize different "
                    "requirements"
                )
        else:
            seen[key] = audit.claim
    return notes


def report(audits: collections.abc.Sequence[Audit], mode: Mode) -> None:
    print(f"\n{'=' * 78}\nClaimCheck audit -- strategy: {mode.value}\n{'=' * 78}\n")

    for audit in audits:
        pprint.pprint(audit)

    if notes := pre_checks(audits):
        print("Pre-check diagnostics (deterministic, no model involved):")
        for note in notes:
            print(f"  - {note}")

    errored = [a for a in audits if a.verdict is None]
    missed = [
        a
        for a in audits
        if a.verdict is not None
        and a.claim.expected is Verdict.DISPUTED
        and not a.correct
    ]
    false_alarms = [
        a
        for a in audits
        if a.verdict is not None
        and a.claim.expected is Verdict.CONFIRMED
        and not a.correct
    ]
    correct = sum(a.correct for a in audits)

    by_domain: dict[str, list[Audit]] = {}
    for audit in audits:
        by_domain.setdefault(audit.domain, []).append(audit)
    if len(by_domain) > 1:
        for name, group in by_domain.items():
            hits = sum(a.correct for a in group)
            print(f"  {name:12} {hits}/{len(group)} ({hits / len(group):.1%})")

    print(
        f"Accuracy: {correct}/{len(audits)} "
        f"({correct / len(audits):.1%})\n"
        f"  unfaithful theorems waved through: {len(missed)}"
        + (f" ({', '.join(a.label for a in missed)})" if missed else "")
        + f"\n  faithful theorems disputed:        {len(false_alarms)}"
        + (f" ({', '.join(a.label for a in false_alarms)})" if false_alarms else "")
        + f"\n  no verdict (retries exhausted):    {len(errored)}"
        + (f" ({', '.join(a.label for a in errored)})" if errored else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        type=Mode,
        choices=list(Mode),
        default=Mode.TWO_PASS,
        help="Audit strategy: the two-pass split, in which the informalizer "
        "never sees the requirement, or the naive floor -- one call, "
        "'does this match?'",
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
        "--domain",
        choices=[*DOMAINS, "all"],
        default="all",
        help="Which of upstream's five benchmark domains to audit, or all of them",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit only the first N claims of each domain (a cheap smoke test)",
    )
    args = parser.parse_args()

    domains = list(DOMAINS.values()) if args.domain == "all" else [DOMAINS[args.domain]]
    if args.limit:
        domains = [
            dataclasses.replace(d, claims=d.claims[: args.limit]) for d in domains
        ]

    if args.verify_only or args.verify:
        if all(domain.verify_corpus for domain in domains):
            print(
                f"VERIFIED: {sum(len(d.theorems) for d in domains)} theorems, 0 errors, no "
                "`sorry`. Every claim below is proved.\nThe audit that follows is not "
                "about whether they are true.\n"
            )

    if not args.verify_only:
        report(asyncio.run(audit_all(domains, args.strategy)), args.strategy)


if __name__ == "__main__":
    main()
