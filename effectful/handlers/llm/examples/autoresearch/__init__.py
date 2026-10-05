"""Research pipelines adapted from named papers.

Each module implements the core of one paper as ordinary Skill calls, and its
docstring says what was kept, what was substituted, and why:

- :mod:`~effectful.handlers.llm.examples.autoresearch.review`: ScholarPeer, context-aware peer review, with
  tool visibility decided by class rather than by prompt
  (:mod:`~effectful.handlers.llm.harness.legibility`).
- :mod:`~effectful.handlers.llm.examples.autoresearch.investigation`: ScientistOne, claims that certify
  themselves against a ground-truth workspace at decode time
  (:mod:`~effectful.handlers.llm.harness.validation`).
- :mod:`~effectful.handlers.llm.examples.autoresearch.writing`: PaperOrchestra, an outline that drives a
  fan-out of specialist agents.
- :mod:`~effectful.handlers.llm.examples.autoresearch.illustration`: PaperBanana, a five-agent illustration
  pipeline whose critic inspects a real rendered figure.
- :mod:`~effectful.handlers.llm.examples.autoresearch.implementation`: MARS, budget-aware modular ML
  engineering as a cost-constrained tree search with a real evaluator.

Fidelity to the source papers is reviewed with the source-derived example
review guide in the documentation.
"""
