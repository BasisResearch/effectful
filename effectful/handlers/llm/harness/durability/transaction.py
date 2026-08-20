import collections.abc
import contextlib

from effectful.handlers.llm.harness.hooks import (
    Message,
    ResultDecodingError,
    ToolCallDecodingError,
    ToolCallExecutionError,
    call_agent,
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
    def get_history(cls) -> collections.abc.MutableSequence[Message]:
        return []

    @classmethod
    def append_message(cls, message: Message) -> None:
        history = cls.get_history()
        if message["role"] == "tool":
            assert cls._tool_call_answers_request(message, history)
        history.append(message)

    @implements(call_system)
    def call_system(self, *args, **kwargs):
        message = fwd(*args, **kwargs)
        if not self.get_history():
            self.append_message(message)
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
            if isinstance(e, ToolCallDecodingError):
                self._answer_abandoned_tool_calls(e)
            raise
        self.append_message(message)
        return (message, tool_calls, result)

    @classmethod
    def _answer_abandoned_tool_calls(cls, e: ToolCallDecodingError) -> None:
        """Answer the sibling calls that decoding abandoned.

        `call_assistant` decodes a turn's tool calls in a loop and raises on the
        first one that fails to validate, so the calls after it never run -- but
        the assistant message it appends advertised all of them, and its feedback
        answers only the one that failed. Both OpenAI APIs require exactly one
        output per advertised call, so resending that buffer earns a
        `BadRequestError` -- which is not in `TenacityRetryer`'s retry set -- in
        place of the retry the feedback exists to inform.

        `ResultDecodingError` needs no such padding: a result is only decoded on a
        turn that requested no tools at all.
        """
        # Narrowing on `role` picks the one arm of the `Message` union that has
        # `tool_calls` at all; the guard is what the type is, not a defensive
        # check, since a decode failure is by construction a failure to decode a
        # call the assistant asked for.
        if e.raw_message["role"] != "assistant":
            return
        for raw_tool_call in e.raw_message.get("tool_calls") or []:
            if raw_tool_call["id"] != e.raw_tool_call.id:
                unanswered: Message = {
                    "role": "tool",
                    "tool_call_id": str(raw_tool_call["id"]),
                    "content": (
                        "Not executed: another tool call in the same turn failed "
                        "to decode. Reissue this call if it is still needed."
                    ),
                }
                cls.append_message(unanswered)

    @staticmethod
    def _tool_call_answers_request(
        message: Message, history: collections.abc.Sequence[Message]
    ) -> bool:
        for request_message in reversed(history):
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

    @implements(call_agent)
    def call_skill(self, skill, *args, **kwargs):
        history: collections.abc.MutableSequence[Message] = getattr(
            skill, "__history__", []
        )
        called = self.agents_called()
        with transaction(history, write_back=id(history) not in called):
            with handler({self.agents_called: lambda: called | {id(history)}}):
                return fwd(skill, *args, **kwargs)


@contextlib.contextmanager
def transaction(
    prefix: collections.abc.MutableSequence[Message] | None = None,
    *,
    write_back: bool = True,
) -> collections.abc.Generator[collections.abc.MutableSequence[Message], None, None]:
    """Context manager for a message transaction.

    The buffer starts as a copy of `prefix` and is only ever appended to, so
    writing back means handing `prefix` the tail the transaction produced. The
    split point is taken on entry, so a `prefix` that grew by some other route
    meanwhile still receives exactly this transaction's messages.
    """
    prefix = HistoryBuilder.get_history() if prefix is None else prefix
    start = len(prefix)
    buffer = list(prefix)
    with handler({HistoryBuilder.get_history: lambda: buffer}):
        yield buffer

    if write_back:
        prefix.extend(buffer[start:])
