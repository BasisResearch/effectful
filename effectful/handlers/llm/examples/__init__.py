"""Complete, runnable example programs, grouped by what they teach.

Run one with ``python -m effectful.handlers.llm.harness -m effectful.handlers.llm.examples.<group>.<name>``;
each prints ``--help``. No example brings a handler stack of its own: the
launcher supplies one, so every example runs unchanged under a different stack,
and the stack it runs under is whatever :func:`~effectful.handlers.llm.harness.harness` was told.

- :mod:`~effectful.handlers.llm.examples.basics`: one pattern per script -- conversation, tools, structured
  values, repair, approval, fan-out, images. Start here.
- :mod:`~effectful.handlers.llm.examples.reasoning`: answering by writing code -- synthesized bodies,
  returned programs, games against hidden rules, and a self-improving receiver.
- :mod:`~effectful.handlers.llm.examples.optimization`: improving an artifact by search or by textual
  gradients, with a cheaper worker model and an evaluator the model can query.
- :mod:`~effectful.handlers.llm.examples.choreographies`: several Skill-owning roles coordinated by a
  choreography, projected onto each role's endpoint.
- :mod:`~effectful.handlers.llm.examples.autoformalization`: formal claims checked by a real Lean
  toolchain, and an audit of whether a proved theorem is the intended one.
- :mod:`~effectful.handlers.llm.examples.autoresearch`: research pipelines adapted from named papers.
- :mod:`~effectful.handlers.llm.examples.acp`: an Agent served to editors over the Agent Client Protocol
  and to web frontends over AG-UI, with persistence, MCP tools and autoreload.

Each group's docstring says what every module in it demonstrates and which part
of :mod:`effectful.handlers.llm.harness` it exercises. Examples adapted from papers have additional
fidelity questions; see the source-derived example review guide in the
documentation.
"""
