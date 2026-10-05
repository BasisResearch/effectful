"""An Agent served to editors and web frontends.

- :mod:`~effectful.handlers.llm.examples.acp.assistant`: the application, a coding assistant with a
  ``prompt`` Skill and four editor tools, run by the launcher with
  ``--persist-db`` so sessions survive restarts
  (:mod:`~effectful.handlers.llm.harness.durability.persistence`) and the editor's MCP servers forwarded as
  tools (:mod:`~effectful.handlers.llm.harness.legibility.mcp`).
- :mod:`~effectful.handlers.llm.examples.acp.library`: the reloadable half of the server: editor
  capabilities as ``Tool``\\ s, streaming by a handler of ``completion``,
  permission requests by a handler of ``call_tool``, and slash commands
  (:mod:`~effectful.handlers.llm.harness.hooks`).
- :mod:`~effectful.handlers.llm.examples.acp.server`: the long-lived half: the connection, the sessions and
  their state. Opted out of ``--autoreload`` with ``__autoreload__ = False`` and
  restarted with ``/restart`` instead (:mod:`~effectful.handlers.llm.harness.autoreload`).
- :mod:`~effectful.handlers.llm.examples.acp.client`: a bridge that runs an ACP agent as a subprocess and
  serves each AG-UI run over HTTP, for frontends such as CopilotKit.

These need the ``docs`` extra's protocol libraries, and the live test drives
them offline rather than from the launcher.
"""
