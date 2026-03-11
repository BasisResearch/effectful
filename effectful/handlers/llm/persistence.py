import dataclasses
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from effectful.handlers.llm.completions import get_agent_history, set_agent_history
from effectful.handlers.llm.template import Agent, Template, get_bound_agent
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled


@Template.define
def summarize_context(transcript: str) -> str:
    """Summarise the following conversation transcript into a concise
    context summary. Preserve key facts, decisions, and any
    information the agent would need to continue working.

    Transcript:
    {transcript}"""
    raise NotHandled


class PersistentAgent(Agent):
    """An :class:`Agent` whose history can be persisted by :class:`PersistenceHandler`.

    This is a lightweight marker class. All persistence *behaviour*
    (checkpointing, handoff, file I/O) lives in :class:`PersistenceHandler`,
    a composable handler following the same pattern as
    :class:`~effectful.handlers.llm.completions.RetryLLMHandler`.

    Unlike plain :class:`Agent` (which uses ``id(self)`` by default),
    ``PersistentAgent`` **requires** a stable ``agent_id`` so that
    checkpoints can be matched across process restarts.

    Override :meth:`checkpoint_state` and :meth:`restore_state` to persist
    custom subclass state alongside the message history.

    **Usage**::

        from pathlib import Path
        from effectful.handlers.llm.persistence import PersistentAgent, PersistenceHandler
        from effectful.handlers.llm import Template
        from effectful.handlers.llm.completions import LiteLLMProvider
        from effectful.ops.semantics import handler
        from effectful.ops.types import NotHandled

        class ResearchBot(PersistentAgent):
            \"""You are a research assistant that remembers prior sessions.\"""

            @Template.define
            def ask(self, question: str) -> str:
                \"""Answer: {question}\"""
                raise NotHandled

        bot = ResearchBot(agent_id="research-bot")

        with handler(LiteLLMProvider()), handler(PersistenceHandler(Path("./state"))):
            bot.ask("What is the capital of France?")
            # Kill process here, restart, and the bot resumes with context.
    """

    def __init__(self, *, agent_id: str):
        self.__agent_id__ = agent_id

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


class PersistenceHandler(ObjectInterpretation):
    """Handler that persists :class:`PersistentAgent` history to disk.

    All persistence state (handoff notes, loaded flags, file paths) lives
    here, not on the agent.  Install alongside
    :class:`~effectful.handlers.llm.completions.LiteLLMProvider`::

        with handler(LiteLLMProvider()), handler(PersistenceHandler(Path("./state"))):
            bot.ask("question")

    **Automatic checkpointing**:

    - **Before** each top-level template call: saves a checkpoint with a
      handoff note describing the in-progress work.
    - **After** each successful call: clears the handoff and saves again.
    - **On failure**: saves the checkpoint (with handoff) so the next
      session can resume.

    **Crash recovery**: on the next run, the handoff note from the prior
    crash is injected into the system prompt so the LLM can resume.

    **Nested calls** (e.g. a tool calling another template on the same
    agent) are passed through without additional checkpointing.

    Composes with :class:`~effectful.handlers.llm.completions.RetryLLMHandler`
    and :class:`CompactionHandler`::

        with (
            handler(LiteLLMProvider()),
            handler(RetryLLMHandler()),
            handler(CompactionHandler()),
            handler(PersistenceHandler(Path("./state"))),
        ):
            bot.ask("question")
    """

    def __init__(self, persist_dir: Path) -> None:
        self._persist_dir = Path(persist_dir)
        self._handoffs: dict[str, str] = {}
        self._loaded: set[str] = set()
        self._tls = threading.local()

    def _get_depths(self) -> dict[str, int]:
        if not hasattr(self._tls, "depths"):
            self._tls.depths = {}
        return self._tls.depths

    def _checkpoint_path(self, agent_id: str) -> Path:
        return self._persist_dir / f"{agent_id}.json"

    def ensure_loaded(self, agent: PersistentAgent) -> bool:
        """Load an agent's checkpoint from disk if not already loaded.

        Returns ``True`` if a checkpoint was found and loaded.
        """
        agent_id = agent.__agent_id__
        if agent_id in self._loaded:
            return False
        self._loaded.add(agent_id)
        path = self._checkpoint_path(agent_id)
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self._handoffs[agent_id] = data.get("handoff", "")
        agent.restore_state(data.get("state", {}))
        history = OrderedDict()
        for msg in data.get("history", []):
            history[msg["id"]] = msg
        set_agent_history(agent_id, history)
        return True

    def save(self, agent: PersistentAgent) -> Path:
        """Write an agent's current state to disk and return the path."""
        agent_id = agent.__agent_id__
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        history = get_agent_history(agent_id)
        data = {
            "agent_id": agent_id,
            "handoff": self._handoffs.get(agent_id, ""),
            "state": agent.checkpoint_state(),
            "history": list(history.values()),
        }
        path = self._checkpoint_path(agent_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(path)
        return path

    def get_handoff(self, agent_id: str) -> str:
        """Return the current handoff note for *agent_id*."""
        return self._handoffs.get(agent_id, "")

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        agent = get_bound_agent(template)
        if not isinstance(agent, PersistentAgent):
            return fwd(template, *args, **kwargs)

        agent_id = agent.__agent_id__
        self.ensure_loaded(agent)

        # Nesting: only checkpoint for outermost call per agent
        depths = self._get_depths()
        depth = depths.get(agent_id, 0)
        depths[agent_id] = depth + 1
        is_outermost = depth == 0

        try:
            if is_outermost:
                # Inject prior-session handoff into system prompt
                prior_handoff = self._handoffs.get(agent_id, "")
                if prior_handoff:
                    template.__system_prompt__ = (
                        f"{template.__system_prompt__}\n\n"
                        f"[HANDOFF FROM PRIOR SESSION] {prior_handoff}"
                    )

                # Record current call as handoff for crash recovery
                self._handoffs[agent_id] = (
                    f"Executing {template.__name__} with args={repr(args)[:200]}"
                )
                self.save(agent)

            result = fwd(template, *args, **kwargs)

            if is_outermost:
                self._handoffs[agent_id] = ""
                self.save(agent)

            return result
        except BaseException:
            if is_outermost:
                self.save(agent)
            raise
        finally:
            depths[agent_id] = depth


class CompactionHandler(ObjectInterpretation):
    """Handler that compacts agent history when it exceeds a threshold.

    After each top-level template call on an :class:`Agent`, if the
    message history exceeds ``max_history_len``, older messages are
    summarised into a single context-summary message via an LLM call.::

        with handler(LiteLLMProvider()), handler(CompactionHandler(max_history_len=20)):
            agent.ask("question")  # history auto-compacted after call
    """

    def __init__(self, max_history_len: int = 50) -> None:
        self._max_history_len = max_history_len
        self._tls = threading.local()

    def _get_depths(self) -> dict[str, int]:
        if not hasattr(self._tls, "depths"):
            self._tls.depths = {}
        return self._tls.depths

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        agent = get_bound_agent(template)
        if not isinstance(agent, Agent):
            return fwd(template, *args, **kwargs)

        agent_id = agent.__agent_id__
        depths = self._get_depths()
        depth = depths.get(agent_id, 0)
        depths[agent_id] = depth + 1
        is_outermost = depth == 0

        try:
            result = fwd(template, *args, **kwargs)

            if is_outermost:
                history = get_agent_history(agent_id)
                if len(history) > self._max_history_len:
                    self._compact(agent_id, history)

            return result
        finally:
            depths[agent_id] = depth

    def _compact(self, agent_id: str, history: OrderedDict[str, Any]) -> None:
        keep_recent = max(self._max_history_len // 2, 4)
        items = list(history.items())
        if len(items) <= keep_recent:
            return

        old_items = items[:-keep_recent]
        recent_items = items[-keep_recent:]

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

        summary = summarize_context(old_transcript)

        new_history: OrderedDict[str, Any] = OrderedDict()
        summary_msg: dict[str, Any] = {
            "id": f"compaction-{agent_id}",
            "role": "user",
            "content": f"[CONTEXT SUMMARY FROM PRIOR CONVERSATION]\n{summary}",
        }
        new_history[summary_msg["id"]] = summary_msg
        for key, msg in recent_items:
            new_history[key] = msg
        set_agent_history(agent_id, new_history)
