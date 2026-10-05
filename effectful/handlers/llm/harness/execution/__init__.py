"""Operations and handlers for parsing, compiling and running model-authored Python.

Either with the builtins or under a `RestrictedPython
<https://restrictedpython.readthedocs.io/>`_ policy.

The executor is the authority boundary for model-authored code, not tool
advertisement: the builtin executor gives generated code the process's Python
authority, and the restricted one narrows builtins, imports and attribute access
without being a complete sandbox. What that implies for claims about what the
model could reach is in :mod:`~effectful.handlers.llm.harness.legibility`.

:mod:`~effectful.handlers.llm.harness.execution.hooks` defines the operations (``parse``, ``compile``,
``eval``, ``exec``), which have no default rule. ``harness(eval_provider=...)``
installs :mod:`~effectful.handlers.llm.harness.execution.builtin` (``"builtin"``, the default),
:mod:`~effectful.handlers.llm.harness.execution.restricted` (``"restricted"``), or nothing (``"none"``,
which also removes the REPL, body synthesis and code tool-calling that would
need one). Type checking is a separate layer, :mod:`~effectful.handlers.llm.harness.validation`, installed
alongside whichever executor is chosen.
"""
