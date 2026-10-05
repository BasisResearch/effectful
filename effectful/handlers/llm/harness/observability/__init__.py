"""Handlers that expose what happened during a call without changing it.

None is a `PromptInjectingInterpretation`, so none appears under ``# Harness``
and the model cannot tell they are installed. Each is opt-in through one
:func:`~effectful.handlers.llm.harness.harness` argument or launcher flag:

- :mod:`~effectful.handlers.llm.harness.observability.rich` -- ``render=True`` / ``--render``: stream each
  model round and print the conversation as panels. The only one that alters
  the request (it forces streaming).
- :mod:`~effectful.handlers.llm.harness.observability.dump` -- ``dump_system_prompt=PATH`` /
  ``--dump-system-prompt``: overwrite a Markdown file with the assembled system
  message on every ``call_system``.
- :mod:`~effectful.handlers.llm.harness.observability.langfuse` -- ``langfuse=True`` / ``--langfuse``:
  record Skill calls, tool calls and completions as nested Langfuse
  observations with token usage.

`LangfuseTracer` is installed last, so its observations enclose everything else,
retries included.
"""
