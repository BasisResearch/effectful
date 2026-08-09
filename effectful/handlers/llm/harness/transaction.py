import collections
import collections.abc
import contextlib
import copy
import uuid

from effectful.handlers.llm.harness.hooks import (
    Message,
    ResultDecodingError,
    Template,
    ToolCallDecodingError,
    ToolCallExecutionError,
    call_assistant,
    call_system,
    call_tool,
    call_user,
)
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation


class HistoryBuilder(ObjectInterpretation):
    """Ensures that the message history does not end up in a malformed state"""

    @Operation.define
    @classmethod
    def agents_called(cls) -> frozenset[int]:
        return frozenset()

    @Operation.define
    @classmethod
    def get_history(cls) -> collections.OrderedDict[str, Message]:
        return collections.OrderedDict()

    @classmethod
    def append_message(cls, message: Message, last: bool = True) -> None:
        if "id" not in message:
            message = {**message, "id": str(uuid.uuid4())}  # type: ignore
        history = cls.get_history()
        if message["role"] == "tool":
            assert cls._tool_call_answers_request(message, history)
        history[message["id"]] = message  # type: ignore
        if not last:
            history.move_to_end(message["id"], last=False)  # type: ignore

    @implements(call_system)
    def call_system(self, *args, **kwargs):
        message = fwd(*args, **kwargs)
        if not self.get_history():
            self.append_message(message, last=False)
        return message

    @implements(call_user)
    def call_user(self, *args, **kwargs):
        message = fwd(*args, **kwargs)
        self.append_message(message)
        return message

    @implements(call_assistant)
    def call_assistant(self, *args, **kwargs):
        try:
            message, tool_calls, result = fwd(*args, **kwargs)
        except (ToolCallDecodingError, ResultDecodingError) as e:
            self.append_message(e.raw_message)
            self.append_message(e.to_feedback_message(include_traceback=True))
            raise
        self.append_message(message)
        return (message, tool_calls, result)

    @staticmethod
    def _tool_call_answers_request(
        message: Message, history: collections.OrderedDict[str, Message]
    ) -> bool:
        for request_message in reversed(history.values()):
            if request_message["role"] == "assistant":
                for call in request_message.get("tool_calls") or []:
                    if message["tool_call_id"] == call["id"]:  # type: ignore
                        return True
                return False
        raise ValueError("shouldnt be here")

    @implements(call_tool)
    def call_tool(self, *args, **kwargs):
        try:
            message, result, is_final = fwd(*args, **kwargs)
        except ToolCallExecutionError as e:
            self.append_message(e.to_feedback_message(include_traceback=True))
            raise
        self.append_message(message)
        return (message, result, is_final)

    @implements(Template.__apply__)
    def call_template(self, template, *args, **kwargs):
        history = getattr(template, "__history__", collections.OrderedDict())
        with transaction(history, write_back=id(history) not in self.agents_called()):
            with handler({self.agents_called: lambda: fwd() | {id(history)}}):
                return fwd(template, *args, **kwargs)


@contextlib.contextmanager
def transaction(
    prefix: collections.OrderedDict[str, Message] | None = None,
    *,
    write_back: bool = True,
) -> collections.abc.Generator[collections.OrderedDict[str, Message], None, None]:
    """Context manager for a message transaction."""
    prefix = HistoryBuilder.get_history() if prefix is None else prefix
    buffer = copy.copy(prefix)
    with handler({HistoryBuilder.get_history: lambda: buffer}):
        yield buffer

    if write_back:
        prefix.update(buffer)
