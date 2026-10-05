"""Answering by writing code, and learning across calls.

Synthesized bodies, judged by caller-authored doctests
(:mod:`~effectful.handlers.llm.harness.synthesis.body`):

- :mod:`~effectful.handlers.llm.examples.reasoning.countdown`: in-context learning to solve arithmetic
  puzzles with code across a conversation.
- :mod:`~effectful.handlers.llm.examples.reasoning.fix_typos`: typo correction whose docstring examples are
  the standard a synthesized body must meet.
- :mod:`~effectful.handlers.llm.examples.reasoning.aime2024`: competition mathematics solved by writing and
  running Python.

Constrained generation, puzzles and games:

- :mod:`~effectful.handlers.llm.examples.reasoning.constrained_paragraph`: writing under exact lexical
  constraints, checked by Python.
- :mod:`~effectful.handlers.llm.examples.reasoning.lineup`: a logic puzzle solved by writing and running
  code.
- :mod:`~effectful.handlers.llm.examples.reasoning.theory_of_mind`: belief-tracking questions answered by
  simulating the scenario in code.
- :mod:`~effectful.handlers.llm.examples.reasoning.hanoi`: Towers of Hanoi with tools created in a closure
  around task state, two strategies (:mod:`~effectful.handlers.llm.harness.legibility`).
- :mod:`~effectful.handlers.llm.examples.reasoning.taboo`: a multi-agent word-guessing game.

Learning a hidden environment (:mod:`~effectful.handlers.llm.examples.reasoning.gridworlds` holds the
environments and an expert strategy, kept out of the model's view):

- :mod:`~effectful.handlers.llm.examples.reasoning.world_model_agent`: a receiver that learns a game's
  rules as executable code and returns the model as a function
  (:mod:`~effectful.handlers.llm.harness.synthesis.function`).
- :mod:`~effectful.handlers.llm.examples.reasoning.continual`: a self-improving agent whose harness lives
  on ``self`` while the transcript is disposable: dynamic tools, notes, and
  compaction (:mod:`~effectful.handlers.llm.harness.legibility`, :mod:`~effectful.handlers.llm.harness.durability.compaction`).
"""
