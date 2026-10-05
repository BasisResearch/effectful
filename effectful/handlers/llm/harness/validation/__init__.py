"""Operations and handlers that type-check and doctest model-authored Python before it runs.

.. rubric:: Validation is a feedback architecture

A Skill turn finishes only when its reply decodes and satisfies the installed
contracts. With retries installed, a failed model answer becomes another
observation in another round of that same turn::

    model round -> candidate reply -> installed decoding and checks
                                      |
                             failure  |  success
                                      |
                 feedback -> next     +-> Python return: this Skill turn ends
                 model round              |
                 (same Skill turn)        +-> Python-side domain evaluator
                                               |
                                               +-> a later Skill turn, if passed back

Feedback exists only because of the retryer: without `TenacityRetryer`
(``num_retries=0``) a decode or Tool error fails the turn. Use this loop for formatting
constraints, compiler feedback, smoke tests, and repair. Match each check to the
defect it can actually detect: a type checker does not discover a wrong physical
sign, and a coarse-grid finiteness predicate does not establish accuracy. Leave
domain-quality failures to an appropriate evaluator and feed its diagnostics into
a later Skill turn when revision is intended.

An evaluator exposed as a Tool can run before acceptance, so its result can
inform a later round of the current turn. An evaluator called by ordinary Python
after the Skill returns is outside that turn and can influence only a later
Skill turn.

Preconditions on Skill arguments are enforced on every Skill call when contract
checking is enabled, with a known gap for metadata on plain Tool parameters
reached by expression calls: :mod:`~effectful.handlers.llm.harness.validation.pydantic`. A return
predicate certifies the property it encodes before the caller receives the
value; an evaluation metric describes what happened after acceptance.

Retries are part of the program's behavior: what the model sees of a rejected
attempt and what joins committed history is in :mod:`~effectful.handlers.llm.harness.durability.retrying`,
and how tool output is bounded before it enters history in
:mod:`~effectful.handlers.llm.harness.durability.truncation`. Choose retry limits and error messages as
deliberately as the Skill prompt; both affect the artifact returned to Python.

Put machine-checkable properties in the declared return contract::

    import dataclasses
    from typing import Annotated

    import annotated_types

    from effectful.handlers.llm import Skill


    @dataclasses.dataclass(frozen=True)
    class GroundedAnswer:
        text: str
        sources: list[str]


    def cites_a_source(answer: GroundedAnswer) -> bool:
        return bool(answer.sources)


    @Skill.define
    def answer(question: str) -> Annotated[
        GroundedAnswer, annotated_types.Predicate(cites_a_source)
    ]:
        \"\"\"Answer {question} and identify at least one source.\"\"\"

With retries enabled, a failed predicate is fed back within the same Skill turn.
This guarantees only that ``sources`` is nonempty; it does not prove retrieval or
semantic support. Same-module validators are visible to the model and therefore
constitute pre-encoded knowledge as well as enforcement. See
:mod:`effectful.handlers.llm.examples.basics.guardrails` for argument and return predicates,
:mod:`effectful.handlers.llm.examples.basics.flight_booking` for a contextual postcondition, and
:mod:`effectful.handlers.llm.examples.basics.error_recovery` for Tool and decode failures returning as
repair feedback. For deterministic external verification, see
:mod:`effectful.handlers.llm.examples.autoformalization.verification`.
"""
