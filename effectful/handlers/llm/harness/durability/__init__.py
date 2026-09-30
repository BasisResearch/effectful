"""Handlers that make a call survive failure.

Message-history accumulation, transactional rollback, tool-output truncation,
retrying on malformed model output, and checkpointing a persisted
:class:`~effectful.handlers.llm.types.Agent` to SQLite.
"""
