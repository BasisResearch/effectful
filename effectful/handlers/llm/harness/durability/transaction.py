import collections.abc
import contextlib
import enum

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
from effectful.handlers.llm.harness.serialization import ToolCallID
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
        """Append `message` to the ambient history, if it is legal where it lands.

        Both checks are about position rather than content, which is why they sit
        here: every message the harness records passes through this method,
        including the ones a failed attempt records on its way out, and those are
        the ones that get a history into a shape no provider will accept.
        """
        history = cls.get_history()
        if message["role"] == "tool":
            assert cls._tool_call_answers_request(message, history)
        elif message["role"] == "assistant":
            assert cls._assistant_speaks_in_turn(history), (
                "an assistant message may not directly follow another; see "
                "`HistoryBuilder._assistant_speaks_in_turn`"
            )
        history.append(message)

    @staticmethod
    def _assistant_speaks_in_turn(
        history: collections.abc.Sequence[Message],
    ) -> bool:
        """Whether the model may speak next: whether `history` is empty or ends
        with something other than the model's own words.

        The harness never prefills, so an assistant message directly following
        another is always a mistake -- and a quiet one, because it is not the
        second message that a provider objects to. Anthropic merges the pair and
        reads the result as a prefill of the reply it is about to write, which
        under a response format is a ``BadRequestError`` and without one is an
        answer that continues whatever the pair ended with (see
        `~effectful.handlers.llm.harness.hooks.ResultDecodingError.to_feedback_message`,
        the feedback path that has to choose a role at all). Neither failure
        names the message that caused it, and both surface a request or two after
        the append that did.

        The check is against the *last* message only, not against a general
        alternation of roles: a turn's tool results are several messages in a row
        (one per call), and a nested call on the same `Agent` opens with a user
        message of its own in the middle of the enclosing turn. Both are
        well-formed, and only the model speaking twice is not.
        """
        return not history or history[-1]["role"] != "assistant"

    @implements(call_system)
    def call_system(self, *args, **kwargs):
        """Record the system message, but only as the first message of a history.

        A history that already has content was restored or inherited, and its
        system message is already at position zero; appending a second one would
        leave two, which no provider accepts.
        """
        message = fwd(*args, **kwargs)
        if not self.get_history():
            self.append_message(message)
        return message

    @implements(call_user)
    def call_user(self, *args, **kwargs):
        """Record the user message that opens a turn."""
        message = fwd(*args, **kwargs)
        self.append_message(message)
        return message

    @implements(call_assistant)
    def call_assistant(self, *args, **kwargs):
        """Record the assistant's reply, including the replies that failed.

        A decoding failure appends the raw message *and* the feedback describing
        what was wrong with it, because that pair is what the next attempt reads;
        the exception then propagates to whatever is retrying. Abandoned sibling
        tool calls are answered first (see `_answer_abandoned_tool_calls`) so the
        buffer stays well-formed enough to resend.
        """
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
        """Record the tool result, including a failed call's traceback.

        Every advertised call must be answered, so a raising tool still appends
        a message before the exception continues outward: the feedback message
        *is* that answer, and leaving the call unanswered would make the
        conversation unresendable.
        """
        try:
            message, result, is_final = fwd(*args, **kwargs)
        except ToolCallExecutionError as e:
            self.append_message(e.to_feedback_message(include_traceback=True))
            raise
        self.append_message(message)
        return (message, result, is_final)

    @implements(call_agent)
    def call_agent(self, skill, *args, **kwargs):
        """Run the call in a transaction over the agent's own history.

        The buffer starts as a copy, so a call that raises leaves the agent's
        history as it found it. ``write_back`` is keyed on the identity of that
        history: the outermost call for a given agent commits, while a nested
        call on the *same* agent -- a tool invoking another of its skills --
        contributes to the same buffer instead of committing a second time.
        Being inside some other agent's transaction does not count, which is
        what keeps a cross-agent nested call from being mistaken for a
        same-agent one.
        """
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

    The buffer starts as a copy of `prefix`, and writing back reconciles the two.
    There are two ways a transaction can end, distinguished by whether the
    inherited prefix survived in the buffer:

    * *Appended to.* The usual case: the buffer still opens with the same message
      objects it was seeded with, so writing back means handing `prefix` the tail
      the transaction produced. The split point is taken on entry, so a `prefix`
      that grew by some other route meanwhile still receives exactly this
      transaction's messages.
    * *Rewritten.* A compaction (see
      `~effectful.handlers.llm.harness.durability.compaction.compact`) drops
      messages the transaction inherited, so the buffer no longer extends its
      seed -- it may even be shorter than it. There is no tail to hand over then;
      the buffer *is* the new history, and appending ``buffer[start:]`` would
      leave `prefix` uncompacted and silently discard everything the transaction
      added. Adopt the buffer wholesale instead, keeping any concurrent growth
      after the split point, which is the same guarantee the appending case
      makes.
    """
    prefix = HistoryBuilder.get_history() if prefix is None else prefix
    start = len(prefix)
    buffer = list(prefix)
    with handler({HistoryBuilder.get_history: lambda: buffer}):
        yield buffer

    if write_back:
        # Identity, not equality: two distinct messages can compare equal (a
        # repeated question), and what decides the case is whether these are the
        # very objects the buffer was seeded with.
        appended = len(buffer) >= start and all(
            a is b for a, b in zip(buffer[:start], prefix[:start])
        )
        if appended:
            prefix.extend(buffer[start:])
        else:
            prefix[:] = [*buffer, *prefix[start:]]


class ClearScope(enum.StrEnum):
    """How much of the conversation a compacting tool call drops.

    ``"none"`` compacts nothing. ``"turn"`` drops the current call's earlier rounds,
    keeping every previous call. ``"conversation"`` additionally drops those previous
    calls, leaving the system message, the request and the asking round.
    """

    NONE = "none"
    TURN = "turn"
    CONVERSATION = "conversation"


def compact_(
    history: collections.abc.MutableSequence[Message],
    tool_call_id: ToolCallID,
    scope: ClearScope,
) -> None:
    """Compact the ambient history, keeping the request and the asking round.

    `tool_call_id` identifies the call that asked, and so the round to keep: the
    assistant message advertising it, and everything after (which is exactly the
    tool messages answering it and its siblings, whether they were appended
    before this one or are still to come -- truncation only ever removes messages
    *ahead* of that assistant message, so no tool message is ever orphaned from
    the call it answers).

    The request kept is the last user message before that round -- the one this
    call opened -- carried over untouched.

    A no-op for ``scope="none"``, and whenever the shape this reads off the
    history is not the one it expects: no assistant message advertising
    `tool_call_id`, or no user message ahead of it. Declining is the right
    failure here; a compaction is a courtesy, and a wrong guess about the shape
    would corrupt the history the call still has to finish over. A conversation
    that opens with something other than a system message simply has no head to
    keep, which is not a failure.
    """
    asking, request = None, None
    for i, message in reversed(list(enumerate(history))):
        if message["role"] == "assistant" and any(
            call["id"] == tool_call_id for call in message.get("tool_calls") or []
        ):
            for j in reversed(range(i)):
                if history[j]["role"] == "user":
                    asking, request = i, j
                    break
            break

    if scope == ClearScope.NONE or asking is None or request is None:
        return
    elif scope == ClearScope.CONVERSATION:
        history[:] = [history[0], history[request], *history[asking:]]
    elif scope == ClearScope.TURN:
        history[:] = [*history[:request], history[request], *history[asking:]]
