import collections.abc
import typing

import tenacity

from effectful.handlers.llm.harness.durability.transaction import (
    HistoryBuilder,
    transaction,
)
from effectful.handlers.llm.harness.hooks import (
    AssistantResult,
    Message,
    PromptInjectingInterpretation,
    ResultDecodingError,
    ToolCallDecodingError,
    ToolCallExecutionError,
    ToolResult,
    call_assistant,
    call_tool,
)
from effectful.handlers.llm.harness.serialization import DecodedToolCall
from effectful.handlers.llm.types import Tool
from effectful.ops.semantics import fwd
from effectful.ops.syntax import implements


class TenacityRetryer(PromptInjectingInterpretation):
    """A reply that cannot be decoded is not the end of the attempt. If your
    answer or a tool call comes back malformed -- wrong shape for the return
    type, a tool call whose arguments do not fit the signature -- you are shown
    the decoding error and asked again, with the failed reply and the error
    visible in the conversation. A malformed tool call is reported as that
    call's result; a malformed *answer* has no call to report against, so it
    comes back as a user message. That message is not a new question, and says
    so: it is the same call, asked again.

    That budget is finite — a handful of attempts, fixed by whoever configured
    this harness — after which the error is raised to the caller and the call
    fails. So a second attempt should not
    resubmit the first one with cosmetic edits. If the same shape has already
    been rejected once, the error is telling you the shape is wrong; change it.
    None of these attempts leave a trace once one succeeds, so you will not see
    the failed exchanges again in later turns.

    A tool that *raises* is different, and not a failure of this kind. The
    traceback comes back as that tool's result and the conversation continues
    normally, without consuming a retry. Read it as data about the call you
    made -- a wrong argument, a missing file -- and make the next call.
    """

    call_assistant_retryer: tenacity.Retrying

    def __init__(
        self,
        catch_tool_errors: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
        stop: tenacity.stop.stop_base = tenacity.stop_after_attempt(4),
        **kwargs,
    ):
        """Configure the retry policy.

        Args:
            catch_tool_errors: Exception type(s) to catch during tool execution.
                Can be a single exception class or a tuple of exception classes.
                Defaults to Exception (catches all exceptions).
            stop: tenacity stop condition for retrying `call_assistant`. Defaults
                to `tenacity.stop_after_attempt(4)`, which stops after 4 attempts.
            **kwargs: Additional keyword arguments forwarded to
                `tenacity.Retrying`.
        """
        self.catch_tool_errors = catch_tool_errors
        assert "retry" not in kwargs, "Cannot override retry logic of TenacityRetryer"
        assert "reraise" not in kwargs, (
            "Cannot override reraise logic of TenacityRetryer"
        )
        self.call_assistant_retryer = tenacity.Retrying(
            retry=tenacity.retry_if_exception_type(
                (ToolCallDecodingError, ResultDecodingError)
            ),
            reraise=True,
            stop=stop,
            **kwargs,
        )

    @implements(call_assistant)
    def call_assistant[T](
        self,
        messages: collections.abc.Sequence[Message],
        response_type: type[T],
        env: collections.abc.Mapping[str, typing.Any],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult[T]:
        """Retry the request while the reply fails to decode.

        `ToolCallDecodingError` and `ResultDecodingError` are the retryable
        failures: both mean the model replied but the reply could not be turned
        into the requested type. Anything else propagates on the first raise.

        Each attempt re-reads `buffer`: the transaction makes it the ambient
        history for the duration, so `HistoryBuilder` appends the failed
        response and its error feedback there, and the next attempt sends them
        to the model. `write_back=False` then discards that scratch work, and
        only the response that finally succeeded joins the real history --
        which is why the caller never sees the malformed attempts.
        """
        with transaction(list(messages), write_back=False) as buffer:
            result = self.call_assistant_retryer(
                lambda: fwd(list(buffer), response_type, env, tools)
            )
        HistoryBuilder.append_message(result[0])
        return result

    @implements(call_tool)
    def call_tool[T](self, tool_call: DecodedToolCall[T]) -> ToolResult[T]:
        """Handle tool execution with runtime error capture.

        Runtime errors from tool execution are captured and returned as
        error messages to the LLM. Only exceptions matching `catch_tool_errors`
        are caught; others propagate up.

        A captured failure is reported as ``is_final=False``, and the error object
        itself is returned in place of a result, so an enclosing handler can see
        that the call failed and decline to finalize on it -- see
        `effectful.handlers.llm.harness.synthesis.body.FinalBodySynthesizer`. The
        completion loop therefore continues: the model sees the error message and
        gets another turn to retry.
        """
        try:
            return fwd(tool_call)
        except ToolCallExecutionError as e:
            if isinstance(e.original_error, self.catch_tool_errors):
                return (e.to_feedback_message(include_traceback=True), e, False)
            else:
                raise
