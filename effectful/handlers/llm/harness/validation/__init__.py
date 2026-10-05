"""Check Skill inputs and model-authored answers.

The Skill's annotations define its input and return types. With
``check_contracts=True`` (the default),
:mod:`~effectful.handlers.llm.harness.validation.pydantic` applies constraints
on Skill arguments before the model sees them. The decoder checks constraints
on the return value before Python receives it.

With retries enabled, a failed answer becomes feedback for another model round
in the same Skill turn. Without retries, it raises an error. A Tool exposed to
the model can evaluate work during the turn; an evaluator called by Python
afterward can affect only a later Skill call.

.. code-block:: text

    model round -> candidate reply -> installed decoding and checks
                                      |
                             failure  |  success
                                      |
                 feedback -> next     +-> Python return: this Skill turn ends
                 model round              |
                 (same Skill turn)        +-> Python-side domain evaluator
                                               |
                                               +-> a later Skill turn, if passed back

For a Python-side evaluator, pass feedback into a new turn explicitly. Here
``writer.write``, ``writer.revise``, and ``judge`` are Skills::

    draft = writer.write(request)
    for _ in range(3):
        review = judge(request, draft)
        if review.accepted:
            break
        draft = writer.revise(request, draft, review.feedback)

Reusing ``writer`` carries its successful history; a free ``judge`` starts
fresh each time. Approval and side effects can be placed in the same ordinary
Python control flow.

For model-authored Python, :mod:`~effectful.handlers.llm.harness.validation.ty`
and :mod:`~effectful.handlers.llm.harness.validation.mypy` provide static type
checks. :mod:`~effectful.handlers.llm.harness.synthesis` explains which code
paths use these checks and whose doctests they run. A passing type check or
schema confirms only the property it tests; use domain-specific checks for
behavior that types cannot establish.

For a return contract, attach a predicate to the annotated type::

    from dataclasses import dataclass
    from typing import Annotated
    from annotated_types import Predicate
    from effectful.handlers.llm import Skill

    @dataclass
    class Answer:
        text: str
        sources: list[str]

    def has_source(answer: Answer) -> bool:
        return bool(answer.sources)

    @Skill.define
    def answer(question: str) -> Annotated[Answer, Predicate(has_source)]:
        \"\"\"Answer {question} and name a source.\"\"\"

This checks that ``sources`` is nonempty; it does not prove the source supports
the answer.

See :mod:`effectful.handlers.llm.examples.basics.guardrails` for annotated
contracts and :mod:`effectful.handlers.llm.examples.basics.error_recovery` for
retry feedback.
"""
