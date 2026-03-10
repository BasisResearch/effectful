import dataclasses
import inspect
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from effectful.handlers.llm.template import Agent, Template
from effectful.ops.types import NotHandled


@Template.define
def summarize_context(transcript: str) -> str:
    """Summarise the following conversation transcript into a concise
    context summary. Preserve key facts, decisions, and any
    information the agent would need to continue working.

    Transcript:
    {transcript}"""
    raise NotHandled


class _PersistentTemplateDescriptor:
    """Descriptor that wraps a :class:`Template` on a :class:`PersistentAgent`
    subclass to provide automatic checkpointing and context compaction.

    Only the outermost template call triggers persistence logic; nested
    template calls (e.g. a tool calling another template on the same agent)
    are passed through without additional checkpointing.

    The descriptor returns the bound template as-is for attribute access
    (preserving all Template semantics for tool discovery, etc.), but
    intercepts ``__call__`` by running the template inside a handler that
    wraps ``Template.__apply__`` — mirroring how Agent's own metaclass
    scopes handler contexts.
    """

    def __init__(self, template: Template, name: str):
        self._template = template
        self._name = name

    def __set_name__(self, owner: type, name: str):
        self._name = name

    def __get__(self, instance: Any, owner: type | None = None):
        if instance is None:
            return self._template
        bound_template = self._template.__get__(instance, owner)
        if not isinstance(instance, PersistentAgent):
            return bound_template
        agent: PersistentAgent = instance
        name = self._name
        return _BoundPersistentTemplate(bound_template, agent, name)


class _BoundPersistentTemplate:
    """A bound template that adds persistence around calls.

    This wraps a bound :class:`Template` and delegates attribute access
    to it (so tool discovery, prompt introspection, etc. work unchanged),
    but intercepts ``__call__`` to add checkpointing and compaction.
    """

    def __init__(self, template: Template, agent: "PersistentAgent", name: str):
        self._template = template
        self._agent = agent
        self._name = name

    def __getattr__(self, item: str) -> Any:
        return getattr(self._template, item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        agent = self._agent
        tls = agent._persistent_tls
        depth = getattr(tls, "depth", 0)
        tls.depth = depth + 1
        is_outermost = depth == 0

        try:
            if is_outermost:
                arg_repr = repr(args)[:200]
                agent.set_handoff(f"Executing {self._name} with args={arg_repr}")
                agent.save_checkpoint()

            result = self._template(*args, **kwargs)

            if is_outermost:
                agent.clear_handoff()
                if len(agent.__history__) > agent.max_history_len:
                    agent._compact_history()
                agent.save_checkpoint()

            return result
        except BaseException:
            if is_outermost:
                agent.save_checkpoint()
            raise
        finally:
            tls.depth = depth


class PersistentAgent(Agent):
    """An :class:`Agent` that persists its state to disk across sessions.

    :class:`PersistentAgent` extends :class:`Agent` with:

    - **Automatic checkpointing**: history, handoff notes, and subclass
      state are saved to ``persist_dir`` before and after every
      top-level :class:`Template` call.
    - **Context compaction**: when the conversation history exceeds
      ``max_history_len`` messages, older messages are summarised into a
      compact form using an LLM call, keeping the context window
      manageable.
    - **Crash recovery**: if the process is interrupted mid-call, the
      checkpoint includes a handoff note describing what was in progress
      so the next session can resume.

    ``persist_dir`` is a required argument — there is no default. This
    follows the convention of libraries like ``shelve`` and ``sqlite3``
    that require callers to specify storage locations explicitly.

    **Subclass state persistence**: if the subclass is a
    :func:`dataclasses.dataclass`, all its fields are automatically
    serialised into the checkpoint.  For custom serialisation, override
    :meth:`checkpoint_state` and :meth:`restore_state`.

    **Plain subclass** (no dataclass)::

        from pathlib import Path
        from effectful.handlers.llm import PersistentAgent, Template
        from effectful.handlers.llm.completions import LiteLLMProvider
        from effectful.ops.semantics import handler
        from effectful.ops.types import NotHandled

        class ResearchBot(PersistentAgent):
            \"""You are a research assistant that remembers prior sessions.\"""

            @Template.define
            def ask(self, question: str) -> str:
                \"""Answer: {question}\"""
                raise NotHandled

        bot = ResearchBot(persist_dir=Path("./research_bot_state"))

        with handler(LiteLLMProvider()):
            bot.ask("What is the capital of France?")
            # Kill process here, restart, and the bot resumes with context.

    **Dataclass subclass** (fields automatically persisted)::

        import dataclasses
        from pathlib import Path

        @dataclasses.dataclass
        class StatefulBot(PersistentAgent):
            \"""You are a bot that remembers patterns.\"""

            persist_dir: Path = Path(".")
            learned_patterns: list[str] = dataclasses.field(default_factory=list)

            @Template.define
            def ask(self, question: str) -> str:
                \"""Answer: {question}\"""
                raise NotHandled

        bot = StatefulBot(persist_dir=Path("./bot_state"))
        bot.learned_patterns.append("User prefers concise answers")
        # learned_patterns is automatically saved/restored across sessions.
    """

    persist_dir: Path
    max_history_len: int
    _handoff: str
    _persistent_tls: threading.local

    def __init__(
        self,
        persist_dir: Path,
        *,
        max_history_len: int = 50,
        agent_id: str | None = None,
    ):
        self._init_persistent(
            persist_dir, max_history_len=max_history_len, agent_id=agent_id
        )

    def _init_persistent(
        self,
        persist_dir: Path,
        *,
        max_history_len: int = 50,
        agent_id: str | None = None,
    ) -> None:
        """Shared initialisation logic.

        Called from ``__init__`` for plain subclasses and from the
        auto-injected ``__post_init__`` for :func:`dataclasses.dataclass`
        subclasses.
        """
        self.persist_dir = Path(persist_dir)
        self.max_history_len = max_history_len
        self._agent_id = agent_id or type(self).__name__
        self._handoff = ""
        self._persistent_tls = threading.local()
        self.load_checkpoint()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in list(vars(cls)):
            attr = vars(cls)[name]
            if isinstance(attr, Template):
                setattr(cls, name, _PersistentTemplateDescriptor(attr, name))
        # When a subclass is decorated with @dataclasses.dataclass, the
        # generated __init__ replaces PersistentAgent.__init__.  Install a
        # __post_init__ so that _init_persistent still runs.  If the user
        # defines their own __post_init__, it is called afterwards.
        existing_post_init = vars(cls).get("__post_init__")

        def _auto_post_init(self: "PersistentAgent") -> None:
            if hasattr(self, "persist_dir"):
                self._init_persistent(
                    self.persist_dir,
                    max_history_len=getattr(self, "max_history_len", 50),
                    agent_id=getattr(self, "agent_id", None),
                )
            if existing_post_init is not None:
                existing_post_init(self)

        cls.__post_init__ = _auto_post_init  # type: ignore[attr-defined]

    # -- subclass state hooks ----------------------------------------------

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of subclass state to persist.

        The default implementation serialises all
        :func:`dataclasses.dataclass` fields.  Override this (and
        :meth:`restore_state`) for custom serialisation.
        """
        if not dataclasses.is_dataclass(self):
            return {}
        state: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            try:
                json.dumps(val)
                state[f.name] = val
            except (TypeError, ValueError):
                pass  # skip non-serialisable fields
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore subclass state from *state* dict.

        The default implementation sets each key as an attribute.
        Override this (and :meth:`checkpoint_state`) for custom
        deserialisation.
        """
        for key, value in state.items():
            setattr(self, key, value)

    # -- persistence API (public) ------------------------------------------

    @property
    def _checkpoint_path(self) -> Path:
        return self.persist_dir / f"{self._agent_id}.json"

    def save_checkpoint(self) -> Path:
        """Write current state to *persist_dir* and return the path."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "agent_id": self._agent_id,
            "handoff": self._handoff,
            "state": self.checkpoint_state(),
            "history": list(self.__history__.values()),
        }
        tmp = self._checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(self._checkpoint_path)
        return self._checkpoint_path

    def load_checkpoint(self) -> bool:
        """Restore state from *persist_dir*. Returns ``True`` if a
        checkpoint was found and loaded."""
        path = self._checkpoint_path
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self._handoff = data.get("handoff", "")
        self.restore_state(data.get("state", {}))
        history = OrderedDict()
        for msg in data.get("history", []):
            history[msg["id"]] = msg
        self.__history__.clear()
        self.__history__.update(history)
        return True

    def set_handoff(self, note: str) -> None:
        """Record a note describing work in progress so a future session
        can resume."""
        self._handoff = note

    def clear_handoff(self) -> None:
        """Clear the in-progress handoff note."""
        self._handoff = ""

    @property
    def handoff(self) -> str:
        return self._handoff

    # -- system prompt augmentation ----------------------------------------

    @property
    def __system_prompt__(self) -> str:  # type: ignore[override]
        base = inspect.getdoc(type(self)) or ""
        parts = [base]
        if self._handoff:
            parts.append(f"[HANDOFF FROM PRIOR SESSION] {self._handoff}")
        return "\n\n".join(p for p in parts if p)

    # -- compaction (internal) ---------------------------------------------

    def _compact_history(self) -> None:
        """Summarise older messages, keeping recent ones intact.

        Compaction replaces messages older than ``keep_recent`` with a
        single summary message produced by an LLM call executed in its
        own handler context (mirroring how ``Agent``'s metaclass scopes
        handler contexts for nested template calls).
        """

        keep_recent = max(self.max_history_len // 2, 4)
        items = list(self.__history__.items())
        if len(items) <= keep_recent:
            return

        old_items = items[:-keep_recent]
        recent_items = items[-keep_recent:]

        # Build a text summary of old messages for the compaction prompt
        old_text_parts: list[str] = []
        for _, msg in old_items:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                content = " ".join(text_parts)
            if content:
                old_text_parts.append(f"[{role}]: {content}")
        old_transcript = "\n".join(old_text_parts)

        if not old_transcript.strip():
            return
        #
        summary = summarize_context(old_transcript)

        # Replace old messages with the summary
        self.__history__.clear()
        summary_msg: dict[str, Any] = {
            "id": f"compaction-{id(self)}",
            "role": "user",
            "content": f"[CONTEXT SUMMARY FROM PRIOR CONVERSATION]\n{summary}",
        }
        self.__history__[summary_msg["id"]] = summary_msg
        for key, msg in recent_items:
            self.__history__[key] = msg
