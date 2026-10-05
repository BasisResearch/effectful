"""Improving an artifact by search or by textual gradients.

Two engines, both ordinary programs over Skill calls:

- :mod:`~effectful.handlers.llm.examples.optimization.library`: the optimize_anything engine, a reflective
  Pareto search over a typed artifact, with ``worker()`` scoping a cheaper model
  to the evaluation (:mod:`~effectful.handlers.llm.harness.provision`).
- :mod:`~effectful.handlers.llm.examples.optimization.textgrad`: textual gradients, backprop-style credit
  assignment recorded live by a handler of ``call_agent`` (:mod:`~effectful.handlers.llm.harness.hooks`).

Search (optimize_anything), one mode each:

- :mod:`~effectful.handlers.llm.examples.optimization.prompting`: a system prompt optimized for a cheaper
  model against held-out instances (generalization mode).
- :mod:`~effectful.handlers.llm.examples.optimization.kernels`: code-generation instructions over a shared
  frontier across tasks (multi-task mode).
- :mod:`~effectful.handlers.llm.examples.optimization.packing`: circle packing as a single code artifact
  scored directly (single-task mode).
- :mod:`~effectful.handlers.llm.examples.optimization.avo`: agentic variation, where the evaluator moves
  inside the model's reach as a Tool it can query before committing.

Textual gradients:

- :mod:`~effectful.handlers.llm.examples.optimization.guidelines`: two parameters learned from one sentence
  of feedback on the composed result.
- :mod:`~effectful.handlers.llm.examples.optimization.ds1000`: training on DS-1000 scipy problems with the
  benchmark's execution oracle as feedback; :mod:`~effectful.handlers.llm.examples.optimization.ds1000_data`
  holds the problems and the oracle.

The optimizer iterations (``--budget``) default to research runs; the live test
uses two.
"""
