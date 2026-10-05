"""Several Skill-owning roles coordinated by a choreography.

- :mod:`~effectful.handlers.llm.examples.choreographies.library`: choreographic programming for multi-agent
  systems. A choreography is one async function over all roles; endpoint
  projection runs each role's part, with ``step`` for a Skill call and
  ``scatter`` to distribute items over a pool. Implemented as handlers of
  ``call_tool``, ``call_system`` and ``call_user`` (:mod:`~effectful.handlers.llm.harness.hooks`), with a
  log that lets an interrupted build resume.
- :mod:`~effectful.handlers.llm.examples.choreographies.multi_agent_choreography`: an architect, coders and
  reviewers building a small library, resumable after Ctrl-C, with
  ``--persist-db`` checkpointing each agent's history
  (:mod:`~effectful.handlers.llm.harness.durability.persistence`).

The roles subclass ``Agent`` explicitly because the library's API is typed on it.
"""
