"""LEAP: blueprint-driven formal theorem proving against a real Lean compiler.

Implements the core of "LEAP: Supercharging LLMs for Formal Mathematics with
Agentic Frameworks" (arXiv:2606.03303). One-shot formalization of a hard theorem
essentially never compiles, so LEAP treats proving as orchestration: register the
theorem as the root of an AND-OR DAG, plan a proof informally, formalize it into
Lean against the compiler, and recurse on whatever intermediate lemmas the plan
wanted to assume, reusing proved lemmas across branches.

The paper's two paths are one loop here. Its direct proof is a plan that assumes no
lemmas, its decomposition a plan that assumes some, and what separates them is a
judgment the planner makes rather than a branch the search takes: a plan is drafted,
a reviewer vets any lemmas in it, the formalizer proves the goal against them as
``sorry`` stubs, and each lemma is then proved the same way. Every failure -- a
rejected plan, a proof Lean will not accept, a lemma that cannot be proved -- comes
back as the feedback the next attempt is drafted against.

Demonstrates:
- Decode-time certification against a real external tool, so a retry becomes a
  compiler-feedback revision (``autoresearch/investigation.py``'s idiom, with a
  theorem prover as ground truth rather than a citation index)
- Certification through the validation context: a skill's arguments, the agent it
  was called on among them, are what its answer is decoded under, so the goal and
  the DAG reach the certification with nothing ambient and nothing closed over
  (``basics/flight_booking.py``)
- An AND-OR DAG with hierarchical memoization, DFS backtracking, an LLM reviewer as
  a search filter, and an acyclicity guard on every proposed subgoal -- built by one
  pure recursive function returning an immutable DAG

Needs a built Lean 4 + Mathlib project; run ``--check-toolchain`` for instructions.
"""

import argparse
import collections.abc
import dataclasses
import hashlib
import os
import re
import textwrap
import typing

import pydantic

from docs.source.llm_examples.autoformalization.library import (
    LEAN_PROJECT,
    PRELUDE,
    LeanError,
    compile_lean,
)
from effectful.handlers.llm import Skill, Tool
from effectful.handlers.llm.harness.hooks import DecodingError


class Unproved(Exception):
    """A goal the search could not close, and why."""


@pydantic.dataclasses.dataclass(frozen=True)
class GoalNode:
    """A goal, and -- once proved -- the tactics that close it and the lemmas they
    cite. An OR node of the paper's AND-OR graph; `lemmas` are the AND node under it.

    Two goals are the same node when their statements match up to whitespace, which
    is what lets a lemma proved in one branch be reused in every other.
    """

    name: str
    sig: str
    tactics: str = ""
    lemmas: tuple[str, ...] = ()

    @classmethod
    def named(cls, sig: str, hint: str) -> "GoalNode":
        """A goal for `sig`, under a unique identifier readable from `hint`"""
        slug = re.sub(r"[^A-Za-z0-9_]", "_", hint).strip("_")[:24] or "lemma"
        digest = hashlib.sha256(cls("", sig).key.encode()).hexdigest()[:6]
        return cls(f"leap_{slug}_{digest}", sig)

    @property
    def key(self) -> str:
        return " ".join(self.sig.split())

    @property
    def decl(self) -> str:
        """This goal's Lean declaration header."""
        return f"theorem {self.name} {self.sig}"


@pydantic.dataclasses.dataclass(frozen=True)
class ProofDAG:
    """Proved goals under the prelude they compile against, in the order they were
    proved -- which is a dependency order, since a goal is proved only once the
    lemmas it cites are.

    Immutable: `extend` returns a larger DAG rather than growing this one, so which
    of a failed branch's proofs survive is decided by which value is kept. `lower` is
    the only place any of this becomes Lean source.
    """

    nodes: collections.abc.Mapping[str, GoalNode] = dataclasses.field(
        default_factory=dict
    )
    prelude: str = PRELUDE

    def extend(self, *nodes: GoalNode) -> "ProofDAG":
        """This DAG plus `nodes`, whose lemmas it must already contain."""
        return dataclasses.replace(
            self, nodes={**self.nodes, **{n.key: n for n in nodes}}
        )

    @property
    def lemmas(self) -> tuple[GoalNode, ...]:
        """Its goals, in the order they were proved."""
        return tuple(self.nodes.values())

    def lower(self) -> str:
        """The Lean file this DAG denotes."""
        bodies = (textwrap.indent(n.tactics.strip(), "  ") for n in self.nodes.values())
        decls = (f"{n.decl} := by\n{b}" for n, b in zip(self.nodes.values(), bodies))
        return self.prelude + "\n" + "".join(f"{decl}\n\n" for decl in decls)


def _proved(goal: GoalNode, tactics: str, *lemmas: str) -> GoalNode:
    """`goal` closed by `tactics`, which cite the goals keyed by `lemmas`."""
    return dataclasses.replace(goal, tactics=tactics, lemmas=lemmas)


def _certified(tactics: str, info: pydantic.ValidationInfo) -> str:
    """Certify that `tactics` close the goal of the agent"""
    (info.context or {})["self"].compile(tactics)
    return tactics


def _type_checks(sig: str, info: pydantic.ValidationInfo) -> str:
    """Certify that `sig` is a well-formed proposition, its proof deferred."""
    dag = (info.context or {})["dag"]
    compile_lean(dag.extend(GoalNode("leap_stub", sig, tactics="sorry")).lower())
    return sig


@pydantic.dataclasses.dataclass(frozen=True)
class LeanProof:
    """A tactic-block proof of a goal, certified by Lean at decode time."""

    tactics: typing.Annotated[str, pydantic.AfterValidator(_certified)] = (
        dataclasses.field(
            metadata={
                "description": "The tactic block that proves the goal: what follows "
                "`:= by`, with no signature and no `by`."
            }
        )
    )


@pydantic.dataclasses.dataclass(frozen=True)
class ProposedLemma:
    """
    One lemma a blueprint proposes, its statement certified to type-check as a
    ``sorry`` stub, so a malformed one is fed back before it becomes a subgoal.
    """

    name: str = dataclasses.field(
        metadata={
            "description": "A short snake_case label for this lemma, unique within "
            "the decomposition."
        }
    )
    sig: typing.Annotated[
        str, pydantic.AfterValidator(str.strip), pydantic.AfterValidator(_type_checks)
    ] = dataclasses.field(
        metadata={
            "description": "The lemma's Lean signature -- binders and proposition, "
            "with no name and no `:= ...`, e.g. `(n : ℕ) : 0 < n + 1`."
        }
    )
    rationale: str = dataclasses.field(
        metadata={
            "description": "What proving this lets the sketch do, and why it is "
            "strictly simpler or more general than the goal."
        }
    )


@pydantic.dataclasses.dataclass(frozen=True)
class Blueprint:
    """
    How to prove a goal: an informal argument outlining the proof
    in a few English sentences, and any lemmas it wants to assume.

    Proposing no lemmas is proposing to prove the goal outright, which is the paper's
    direct path; proposing some is its decomposition. They are the same answer to
    the same question, so they are one type and one skill.
    """

    plan: str
    lemmas: list[ProposedLemma]

    def __post_init__(self) -> None:
        sigs = [lm.sig for lm in self.lemmas]
        if len(set(sigs)) != len(sigs):
            raise ValueError(f"two proposed lemmas have the same statement: {sigs}")


@pydantic.dataclasses.dataclass(frozen=True)
class ReviewVerdict:
    """The reviewer's judgment -- the paper's planning-level search filter."""

    simplifies: bool = dataclasses.field(
        metadata={
            "description": "True only if every proposed lemma is genuinely simpler "
            "or more general than the goal and plausibly provable."
        }
    )
    reason: str


@pydantic.dataclasses.dataclass
class Formalizer:
    """You are the formalizer. You turn an informal argument into a Lean 4 tactic
    proof and make it compile, preferring short robust proofs (`simp`, `omega`,
    `ring`, `induction`, `exact?`). You never leave a `sorry`.
    """

    goal: GoalNode
    dag: ProofDAG = dataclasses.field(default_factory=ProofDAG)
    assume: tuple[GoalNode, ...] = ()

    @property
    def scope(self) -> ProofDAG:
        """The DAG its Lean compiles against: everything proved, plus whatever it
        assumes and nobody has proved yet, as ``sorry`` stubs."""
        stubs = (
            _proved(lemma, "sorry")
            for lemma in self.assume
            if lemma.key not in self.dag.nodes
        )
        return self.dag.extend(*stubs)

    @property
    def lemmas(self) -> tuple[GoalNode, ...]:
        """Everything the goal may cite by name."""
        return self.scope.lemmas

    @Tool.define
    def compile(self, tactics: str) -> str:
        """Compile ``<current goal> := by <tactics>``. Raises with Lean's errors and
        remaining goal state if it does not compile, so call it to read the goal and
        test tactics before committing an answer. Mathlib's ``exact?``/``apply?``
        work inside ``tactics`` for premise search."""
        if re.fullmatch(r".*\b(sorry|admit|sorryAx)\b.*", tactics, re.DOTALL):
            raise LeanError("`sorry`/`admit` is refused here: close the goal for real.")
        compile_lean(self.scope.extend(_proved(self.goal, tactics)).lower())
        return "Lean accepts this proof (no errors)."

    @Skill.define
    def formalize(self, argument: str) -> LeanProof:
        """Complete `{self.goal.decl} := by ...` in Lean 4 with Mathlib, returning
        the tactic block that closes the goal. Use `compile` to check it against the
        real compiler and fix it against Lean's output before answering.

        The argument to formalize:
        {argument}

        Lemmas in scope, citable by name: {self.lemmas}

        Of those, the ones this proof is meant to rest on -- empty unless the goal
        was decomposed, and stubbed rather than proved, so `compile` accepts a proof
        that leans on them but not one that leaves the goal itself open:
        {self.assume}
        """


@pydantic.dataclasses.dataclass
class Planner:
    """You are the planner. You say how a goal should be proved: the mathematical
    argument, and -- when the goal is too hard to argue outright -- the intermediate
    lemmas to assume, each strictly simpler or more general than the goal, and each
    worth having on its own. You plan in mathematics, not in Lean."""

    goal: GoalNode

    @Skill.define
    def draft(self, dag: ProofDAG, feedback: str) -> Blueprint:
        """Say how to prove `{self.goal.decl}`: the informal argument, and the
        lemmas a formalizer should be allowed to assume while doing it.

        Propose no lemmas if you can argue the goal outright -- that is the cheaper
        proof and the one to prefer. Propose some when it is too hard, but never a
        lemma that restates the goal, and reuse what is already proved before
        proposing anything new.

        Already proved and in scope: {dag.lemmas}

        What went wrong with the previous attempts, empty on the first: {feedback}
        """


@pydantic.dataclasses.dataclass
class Reviewer:
    """You are the decomposition reviewer, a planning-level search filter. Compiler
    verification only says a sketch is well-typed: a sketch can compile while
    proposing a subgoal no simpler than the goal. You reject those, so the search
    does not spend itself on decompositions that make no progress."""

    goal: GoalNode

    @Skill.define
    def review(self, blueprint: Blueprint) -> ReviewVerdict:
        """Judge whether this decomposition genuinely simplifies proving
        `{self.goal.decl}`. Accept only if each proposed lemma is strictly simpler or more
        general than the goal and together they make it easy; reject one that merely
        restates the goal, is no easier than it, or does not advance the proof.

        {blueprint}
        """


def prove(
    goal: GoalNode,
    dag: ProofDAG = ProofDAG(),
    ancestors: frozenset[str] = frozenset(),
    *,
    attempts: int = 3,
    depth: int = 3,
) -> ProofDAG:
    """
    `dag` extended with a proof of `goal` and every lemma that proof rests on.
    """
    log = "  " * len(ancestors)
    if dag.nodes.get(goal.key):
        return dag
    if goal.key in ancestors:
        raise Unproved(f"`{goal.sig}` restates an ancestor goal: a cycle")
    print(f"{log}[goal] {goal.name}: {goal.key[:80]}")

    planner, reviewer = Planner(goal), Reviewer(goal)

    feedback = ""
    for attempt in range(1, attempts + 1):
        try:
            blueprint = planner.draft(dag, feedback)

            subgoals = tuple(
                dag.nodes.get(fresh.key, fresh)
                for lm in blueprint.lemmas
                for fresh in [GoalNode.named(lm.sig, lm.name)]
            )
            if subgoals:
                if depth <= 0:
                    raise Unproved("this goal may not be decomposed any further")
                verdict = reviewer.review(blueprint)
                if not verdict.simplifies:
                    raise Unproved(f"a reviewer rejected it: {verdict.reason}")

            proof = Formalizer(goal, dag, subgoals).formalize(blueprint.plan)
            for sub in subgoals:
                dag = prove(
                    sub,
                    dag,
                    ancestors | {goal.key},
                    attempts=attempts,
                    depth=depth - 1,
                )
            print(f"{log}[proved] `{goal.name}`")
            return dag.extend(_proved(goal, proof.tactics, *(s.key for s in subgoals)))
        except (Unproved, DecodingError) as exc:
            feedback = f"[attempt {attempt}] {exc}"
            print(f"{log}{feedback}")

    raise Unproved(f"no proof of `{goal.sig}` in {attempts} attempts")


TARGETS: dict[str, str] = {
    "odd_sum": r"(n : ℕ) : (∑ i ∈ Finset.range n, (2 * i + 1)) = n ^ 2",
    "gauss": r"(n : ℕ) : (2 * ∑ i ∈ Finset.range (n + 1), i) = n * (n + 1)",
    "div6": r"(n : ℕ) : 6 ∣ n * (n + 1) * (n + 2)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=list(TARGETS),
        default="odd_sum",
        help="Which theorem to prove",
    )
    parser.add_argument(
        "--statement",
        type=str,
        default=None,
        help="A custom Lean signature `(binders) : prop` to prove instead of --target",
    )
    parser.add_argument(
        "--check-toolchain",
        action="store_true",
        help="Skip the pipeline; compile a trivial theorem to validate Lean+Mathlib",
    )
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(LEAN_PROJECT, ".lake")):
        raise RuntimeError(f"Lean project not built at {LEAN_PROJECT!r}.")

    if args.check_toolchain:
        compile_lean(f"{PRELUDE}\ntheorem leap_sanity (n : ℕ) : n + 0 = n := by simp\n")
        print("Toolchain OK.")
        return

    sig = args.statement or TARGETS[args.target]
    print(f"Proving: {sig}\n")

    root = GoalNode.named(sig, "root")
    dag = prove(root)

    cited = [key for n in dag.nodes.values() for key in n.lemmas]
    print(f"{len(dag.nodes)} theorem(s) proved, {len(cited)} used")

    full = dag.lower()
    print(f"\nAssembled proof: \n{full} \nrecompiling end-to-end ...")
    compile_lean(full)
    print("VERIFIED: the complete proof compiles under Lean with no `sorry`.\n")


if __name__ == "__main__":
    main()
