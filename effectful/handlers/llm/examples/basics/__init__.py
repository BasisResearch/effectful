"""One pattern per script, each small enough to read whole.

- :mod:`~effectful.handlers.llm.examples.basics.conversation`: a Skill method on a reused object, so
  successive calls share history (:mod:`~effectful.handlers.llm.harness.durability.transaction`).
- :mod:`~effectful.handlers.llm.examples.basics.rag`: retrieval as a ``Tool`` the Skill is told to call,
  published explicitly with ``Tool.define`` (:mod:`~effectful.handlers.llm.harness.legibility.lexical`).
- :mod:`~effectful.handlers.llm.examples.basics.lexical_scope`: sub-skills captured from lexical scope and
  invoked two ways, as tools and from a returned function
  (:mod:`~effectful.handlers.llm.harness.legibility`, :mod:`~effectful.handlers.llm.harness.synthesis.function`).
- :mod:`~effectful.handlers.llm.examples.basics.flight_booking`: composing agents by passing one's typed
  output to the next, with a contextual postcondition
  (:mod:`~effectful.handlers.llm.harness.validation`).
- :mod:`~effectful.handlers.llm.examples.basics.guardrails`: argument and return predicates as input
  guardrails (:mod:`~effectful.handlers.llm.harness.validation.pydantic`).
- :mod:`~effectful.handlers.llm.examples.basics.error_recovery`: a flaky tool and invalid structured output
  returning to the model as repair feedback (:mod:`~effectful.handlers.llm.harness.durability.retrying`).
- :mod:`~effectful.handlers.llm.examples.basics.text2sql`: SQL generated, executed against SQLite, and fixed
  from the database's own errors in an explicit Python loop.
- :mod:`~effectful.handlers.llm.examples.basics.research_agent`: a writer receiver kept across turns and a
  free judge Skill, the evaluator loop from the package guide.
- :mod:`~effectful.handlers.llm.examples.basics.hitl`: human approval between proposal and side effect,
  with the executor out of the proposing Skill's reach.
- :mod:`~effectful.handlers.llm.examples.basics.map_reduce`: fan-out with ``asyncio.gather`` over
  ``asyncio.to_thread``, each call its own turn.
- :mod:`~effectful.handlers.llm.examples.basics.image_input` and :mod:`~effectful.handlers.llm.examples.basics.image_tool`: PIL images
  as Skill arguments and as tool inputs and results (``Encodable``,
  :mod:`~effectful.handlers.llm.harness.serialization`).
"""
