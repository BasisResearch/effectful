"""Formal claims checked by a real Lean toolchain.

- :mod:`~effectful.handlers.llm.examples.autoformalization.verification`: LEAP, blueprint-driven theorem
  proving against a real Lean compiler, the deterministic external verifier of
  the package guide (:mod:`~effectful.handlers.llm.harness.validation`).
- :mod:`~effectful.handlers.llm.examples.autoformalization.informalization`: ClaimCheck, auditing whether a
  proved theorem is the theorem that was meant.
- :mod:`~effectful.handlers.llm.examples.autoformalization.library`: the ClaimCheck benchmark corpora,
  claims and labelled ground truth, and the Lean toolchain that checks the
  corpora are proved.

Both scripts need Lean 4 with Mathlib installed; the live test skips
verification when it is absent.
"""
