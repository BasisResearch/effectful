import abc
import functools
import typing
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from typing import Any

from effectful.handlers.llm.completions import (
    DecodedToolCall,
    _make_message,
    call_assistant,
    call_tool,
    get_message_sequence,
    history_checkpoint,
    history_rollback,
    record_message,
)
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements

from .template import Template


class Agent(abc.ABC):
    """Mixin that gives each instance a persistent LLM message history.

    Subclass and decorate methods with :func:`Template.define`.
    Each instance accumulates messages across calls so the LLM sees
    prior conversation context.

    Agents compose freely with :func:`dataclasses.dataclass` and other
    base classes.  Instance attributes are available in template
    docstrings via ``{self.attr}``.

    Example::

        import dataclasses
        from effectful.handlers.llm import Agent, Template
        from effectful.handlers.llm.completions import LiteLLMProvider
        from effectful.ops.semantics import handler
        from effectful.ops.types import NotHandled

        @dataclasses.dataclass
        class ChatBot(Agent):
            bot_name: str = dataclasses.field(default="ChatBot")

            @Template.define
            def send(self, user_input: str) -> str:
                \"""Friendly bot named {self.bot_name}. User writes: {user_input}\"""
                raise NotHandled

        provider = LiteLLMProvider()
        chatbot = ChatBot()

        with handler(provider):
            chatbot.send("Hi! How are you? I am in France.")
            chatbot.send("Remind me again, where am I?")  # sees prior context

    """

    __history__: OrderedDict[str, Mapping[str, Any]]
    __history_state__: "_AgentHistoryState"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prop = functools.cached_property(lambda _: OrderedDict())
        prop.__set_name__(cls, "__history__")
        cls.__history__ = prop
        state_prop = functools.cached_property(
            lambda self: _AgentHistoryState(self.__history__)
        )
        state_prop.__set_name__(cls, "__history_state__")
        cls.__history_state__ = state_prop

        for name, attr in list(cls.__dict__.items()):
            if not isinstance(attr, Template) or isinstance(
                attr.__default__, staticmethod | classmethod
            ):
                continue

            def _template_prop_fn[T: Template](self, *, template: T) -> T:
                inst_template = template.__get__(self, type(self))
                setattr(inst_template, "__history__", self.__history_state__)
                return inst_template

            _template_property = functools.cached_property(
                functools.partial(_template_prop_fn, template=attr)
            )
            _template_property.__set_name__(cls, name)
            setattr(cls, name, _template_property)


class _AgentHistoryState(ObjectInterpretation):
    """Parent/child scoped history state for nested tool/template calls."""

    def __init__(
        self,
        root: OrderedDict[str, Mapping[str, Any]] | None = None,
        *,
        parent: "_AgentHistoryState | None" = None,
    ):
        super().__init__()
        self._parent = parent
        if parent is None:
            assert root is not None
            self._owner = self
            self._messages = root
            self._active: _AgentHistoryState = self
            self._sequence_view = _AgentHistorySequenceView(self)
        else:
            self._owner = parent._owner
            self._messages = OrderedDict()
        self._pending_tool_call_ids: dict[str, str] = {}

    def _active_state(self) -> "_AgentHistoryState":
        return self._owner._active

    def _active_lineage(self) -> list["_AgentHistoryState"]:
        lineage: list[_AgentHistoryState] = []
        node: _AgentHistoryState | None = self._active_state()
        while node is not None:
            lineage.append(node)
            node = node._parent
        lineage.reverse()
        return lineage

    def merged(self) -> OrderedDict[str, Mapping[str, Any]]:
        merged: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
        for state in self._active_lineage():
            merged.update(state._messages)
        return merged

    def merged_for_llm(self) -> OrderedDict[str, Mapping[str, Any]]:
        """Merged view with in-flight tool call sequences stripped.

        Pending ``__PENDING_TOOL_RESULT__`` placeholders must never reach the
        LLM.  An assistant message whose tool_calls include *any* still-pending
        id is also removed (together with every tool response belonging to that
        same assistant turn), because an incomplete tool-call sequence is
        invalid for the OpenAI message format.
        """
        all_pending: set[str] = set()
        for state in self._active_lineage():
            all_pending.update(state._pending_tool_call_ids.keys())

        if not all_pending:
            return self.merged()

        raw = self.merged()

        # Collect ALL tool_call_ids from assistant messages that have at least
        # one pending tool_call – both completed and pending ids are "tainted".
        tainted: set[str] = set()
        for msg in raw.values():
            if msg.get("role") != "assistant":
                continue
            tc_ids = {
                typing.cast(str, tc.get("id"))
                for tc in (msg.get("tool_calls") or ())
                if isinstance(tc, Mapping) and isinstance(tc.get("id"), str)
            }
            if tc_ids & all_pending:
                tainted |= tc_ids

        filtered: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
        for key, msg in raw.items():
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                tc_ids = {
                    typing.cast(str, tc.get("id"))
                    for tc in (msg.get("tool_calls") or ())
                    if isinstance(tc, Mapping) and isinstance(tc.get("id"), str)
                }
                if tc_ids & tainted:
                    continue
            elif role == "tool":
                if msg.get("tool_call_id") in tainted:
                    continue
            filtered[key] = msg
        return filtered

    @implements(get_message_sequence)
    def _get_message_sequence(self):
        return self._owner._sequence_view

    @implements(record_message)
    def _record_message(self, message):
        role = message.get("role")
        if role == "tool" and self._replace_pending_placeholder(message):
            return
        self._active_state()._messages[typing.cast(str, message["id"])] = message

    @implements(call_assistant)
    def _call_assistant(self, *args, **kwargs):
        message, tool_calls, result = fwd(*args, **kwargs)
        if message.get("role") == "assistant" and tool_calls:
            self._append_pending_placeholders(
                self._active_state(), [tool_call.id for tool_call in tool_calls]
            )
        return (message, tool_calls, result)

    @implements(call_tool)
    def _call_tool(self, tool_call: DecodedToolCall):
        child = _AgentHistoryState(parent=self._active_state())
        self._owner._active = child
        try:
            with handler(child):
                return fwd(tool_call)
        finally:
            if self._owner._active is child:
                parent = child._parent
                assert parent is not None
                self._owner._active = parent

    @implements(history_checkpoint)
    def _history_checkpoint(self):
        return len(self.merged())

    @implements(history_rollback)
    def _history_rollback(self, checkpoint):
        while len(self.merged()) > checkpoint:
            merged = self.merged()
            if not merged:
                raise KeyError("dictionary is empty")
            self.delete(next(reversed(merged)))

    def _append_pending_placeholders(
        self, state: "_AgentHistoryState", tool_call_ids: typing.Sequence[str]
    ) -> None:
        for tool_call_id in tool_call_ids:
            pending_message_id = f"pending_tool_{tool_call_id}"
            state._messages[pending_message_id] = _make_message(
                {
                    "id": pending_message_id,
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "__PENDING_TOOL_RESULT__",
                }
            )
            state._pending_tool_call_ids[tool_call_id] = pending_message_id

    def _replace_pending_placeholder(self, tool_message: Mapping[str, Any]) -> bool:
        tool_call_id = tool_message.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            return False
        for state in reversed(self._active_lineage()[:-1]):
            pending_message_id = state._pending_tool_call_ids.get(tool_call_id)
            if pending_message_id is None:
                continue
            state._messages[pending_message_id] = tool_message
            del state._pending_tool_call_ids[tool_call_id]
            return True
        return False

    def delete(self, key: str) -> Mapping[str, Any]:
        for state in reversed(self._active_lineage()):
            if key not in state._messages:
                continue
            deleted_value = state._messages[key]
            del state._messages[key]
            self._cleanup_after_delete(state, key, deleted_value)
            return deleted_value
        raise KeyError(key)

    def lookup(self, key: str) -> Mapping[str, Any]:
        for state in reversed(self._active_lineage()):
            if key in state._messages:
                return state._messages[key]
        raise KeyError(key)

    def _cleanup_after_delete(
        self, state: "_AgentHistoryState", key: str, value: Mapping[str, Any]
    ) -> None:
        for tcid, pending_id in tuple(state._pending_tool_call_ids.items()):
            if pending_id == key:
                del state._pending_tool_call_ids[tcid]

        if value.get("role") == "assistant":
            tool_calls = value.get("tool_calls") or []
            tool_call_ids: list[str] = []
            for tc in tool_calls:
                if not isinstance(tc, Mapping):
                    continue
                raw_tcid = tc.get("id")
                if isinstance(raw_tcid, str):
                    tool_call_ids.append(raw_tcid)
            for tcid in tool_call_ids:
                if tcid in state._pending_tool_call_ids:
                    pending_id = state._pending_tool_call_ids.pop(tcid)
                    state._messages.pop(pending_id, None)


class _AgentHistorySequenceView(MutableMapping[str, Mapping[str, Any]]):
    """Mutable mapping view backed by `_AgentHistoryState`."""

    def __init__(self, state: _AgentHistoryState):
        self._state = state

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        return self._state.lookup(key)

    def __setitem__(self, key: str, value: Mapping[str, Any]) -> None:
        if value.get("id") != key:
            value = dict(value)
            value["id"] = key
        role = value.get("role")
        if role == "tool" and self._state._replace_pending_placeholder(value):
            return

        current = self._state._active_state()
        current._messages[typing.cast(str, value["id"])] = value
        if role == "assistant" and value.get("tool_calls"):
            tool_call_ids = [
                typing.cast(str, tc["id"])
                for tc in typing.cast(
                    typing.Sequence[Mapping[str, Any]], value["tool_calls"]
                )
            ]
            self._state._append_pending_placeholders(current, tool_call_ids)

    def __delitem__(self, key: str) -> None:
        self._state.delete(key)

    def __iter__(self):
        return iter(self._state.merged_for_llm())

    def __len__(self) -> int:
        return len(self._state.merged_for_llm())

    def copy(self):
        return self._state.merged_for_llm().copy()

    def values(self):
        return self._state.merged_for_llm().values()

    def items(self):
        return self._state.merged_for_llm().items()

    def keys(self):
        return self._state.merged_for_llm().keys()

    def popitem(self, last: bool = True):
        merged = self._state.merged()
        if not merged:
            raise KeyError("dictionary is empty")
        key = next(reversed(merged)) if last else next(iter(merged))
        value = self._state.delete(key)
        return (key, value)
