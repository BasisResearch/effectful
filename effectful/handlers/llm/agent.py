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
    """Mutable stacked history state shared across nested template calls."""

    def __init__(self, root: OrderedDict[str, Mapping[str, Any]]):
        super().__init__()
        self._frames: list[OrderedDict[str, Mapping[str, Any]]] = [root]
        self._pending_by_frame: list[dict[str, str]] = [{}]
        self._sequence_view = _AgentHistorySequenceView(self)

    def _current_idx(self) -> int:
        return len(self._frames) - 1

    def _push_frame(self) -> None:
        self._frames.append(OrderedDict())
        self._pending_by_frame.append({})

    def _pop_frame(self) -> None:
        if len(self._frames) > 1:
            self._frames.pop()
            self._pending_by_frame.pop()

    def merged(self) -> OrderedDict[str, Mapping[str, Any]]:
        merged: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
        for frame in self._frames:
            merged.update(frame)
        return merged

    @implements(get_message_sequence)
    def _get_message_sequence(self):
        return self._sequence_view

    @implements(record_message)
    def _record_message(self, message):
        role = message.get("role")
        if role == "tool" and self._replace_pending_placeholder(message):
            return

        frame_idx = self._current_idx()
        self._frames[frame_idx][typing.cast(str, message["id"])] = message

    @implements(call_assistant)
    def _call_assistant(self, *args, **kwargs):
        message, tool_calls, result = fwd(*args, **kwargs)
        if message.get("role") == "assistant" and tool_calls:
            frame_idx = self._current_idx()
            self._append_pending_placeholders(
                frame_idx, [tool_call.id for tool_call in tool_calls]
            )
        return (message, tool_calls, result)

    @implements(call_tool)
    def _call_tool(self, tool_call: DecodedToolCall):
        self._push_frame()
        try:
            with handler({get_message_sequence: lambda: self._sequence_view}):
                return fwd(tool_call)
        finally:
            self._pop_frame()

    @implements(history_checkpoint)
    def _history_checkpoint(self):
        return len(self.merged())

    @implements(history_rollback)
    def _history_rollback(self, checkpoint):
        while len(self.merged()) > checkpoint:
            merged = self.merged()
            if not merged:
                raise KeyError("dictionary is empty")
            key = next(reversed(merged))
            self.delete(key)

    def _append_pending_placeholders(
        self, frame_idx: int, tool_call_ids: typing.Sequence[str]
    ) -> None:
        pending_for_frame = self._pending_by_frame[frame_idx]
        for tool_call_id in tool_call_ids:
            pending_message_id = f"pending_tool_{tool_call_id}"
            self._frames[frame_idx][pending_message_id] = _make_message(
                {
                    "id": pending_message_id,
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "__PENDING_TOOL_RESULT__",
                }
            )
            pending_for_frame[tool_call_id] = pending_message_id

    def _replace_pending_placeholder(self, tool_message: Mapping[str, Any]) -> bool:
        tool_call_id = tool_message.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            return False

        for frame_idx in range(self._current_idx() - 1, -1, -1):
            pending_for_frame = self._pending_by_frame[frame_idx]
            pending_message_id = pending_for_frame.get(tool_call_id)
            if pending_message_id is None:
                continue

            self._frames[frame_idx][pending_message_id] = tool_message
            del pending_for_frame[tool_call_id]
            return True
        return False

    def delete(self, key: str) -> Mapping[str, Any]:
        for frame_idx in range(self._current_idx(), -1, -1):
            frame = self._frames[frame_idx]
            if key not in frame:
                continue
            deleted_value = frame[key]
            del frame[key]
            self._cleanup_after_delete(frame_idx, key, deleted_value)
            return deleted_value
        raise KeyError(key)

    def get(self, key: str) -> Mapping[str, Any]:
        for frame in reversed(self._frames):
            if key in frame:
                return frame[key]
        raise KeyError(key)

    def _cleanup_after_delete(
        self, frame_idx: int, key: str, value: Mapping[str, Any]
    ) -> None:
        pending_for_frame = self._pending_by_frame[frame_idx]
        for tcid, pending_id in tuple(pending_for_frame.items()):
            if pending_id == key:
                del pending_for_frame[tcid]

        if value.get("role") == "assistant":
            tool_calls = value.get("tool_calls") or []
            tool_call_ids = {
                tc.get("id")
                for tc in tool_calls
                if isinstance(tc, Mapping) and isinstance(tc.get("id"), str)
            }
            frame = self._frames[frame_idx]
            for tcid in tool_call_ids:
                pending_id = pending_for_frame.pop(tcid, None)
                if pending_id is not None:
                    frame.pop(pending_id, None)



class _AgentHistorySequenceView(MutableMapping[str, Mapping[str, Any]]):
    """Mutable mapping view backed by `_AgentHistoryState`."""

    def __init__(self, state: _AgentHistoryState):
        self._state = state

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        return self._state.get(key)

    def __setitem__(self, key: str, value: Mapping[str, Any]) -> None:
        if value.get("id") != key:
            value = dict(value)
            value["id"] = key
        role = value.get("role")
        if role == "tool" and self._state._replace_pending_placeholder(value):
            return

        frame_idx = self._state._current_idx()
        self._state._frames[frame_idx][typing.cast(str, value["id"])] = value
        if role == "assistant" and value.get("tool_calls"):
            tool_call_ids = [
                typing.cast(str, tc["id"])
                for tc in typing.cast(
                    typing.Sequence[Mapping[str, Any]], value["tool_calls"]
                )
            ]
            self._state._append_pending_placeholders(frame_idx, tool_call_ids)

    def __delitem__(self, key: str) -> None:
        self._state.delete(key)

    def __iter__(self):
        return iter(self._state.merged())

    def __len__(self) -> int:
        return len(self._state.merged())

    def copy(self):
        return self._state.merged().copy()

    def values(self):
        return self._state.merged().values()

    def items(self):
        return self._state.merged().items()

    def keys(self):
        return self._state.merged().keys()

    def popitem(self, last: bool = True):
        merged = self._state.merged()
        if not merged:
            raise KeyError("dictionary is empty")
        key = next(reversed(merged)) if last else next(iter(merged))
        value = self._state.delete(key)
        return (key, value)
