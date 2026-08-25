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
    """Retries LLM requests if tool call or result decoding fails.

    This handler intercepts `call_assistant` and catches `ToolCallDecodingError`
    and `ResultDecodingError`. When these errors occur, it appends error feedback
    to the messages and retries the request. Malformed messages from retry attempts
    are pruned from the final result.

    For runtime tool execution failures (handled via `call_tool`), errors are
    captured and returned as tool response messages.

    Args:
        catch_tool_errors: Exception type(s) to catch during tool execution.
            Can be a single exception class or a tuple of exception classes.
            Defaults to Exception (catches all exceptions).
        stop: tenacity stop condition for retrying `call_assistant`. Defaults to
            `tenacity.stop_after_attempt(4)`, which stops after 4 attempts.
        **kwargs: Additional keyword arguments forwarded to `tenacity.Retrying`.
    """

    call_assistant_retryer: tenacity.Retrying

    def __init__(
        self,
        catch_tool_errors: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
        stop: tenacity.stop.stop_base = tenacity.stop_after_attempt(4),
        **kwargs,
    ):
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
    def _call_assistant[T](
        self,
        messages: collections.abc.Sequence[Message],
        response_type: type[T],
        env: collections.abc.Mapping[str, typing.Any],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult[T]:
        # Each attempt re-reads `buffer`: the transaction makes it the ambient
        # history for the duration, so `HistoryBuilder` appends the failed
        # response and its error feedback there, and the next attempt sends them
        # to the model. `write_back=False` then discards that scratch work, and
        # only the response that finally succeeded joins the real history.
        with transaction(list(messages), write_back=False) as buffer:
            result = self.call_assistant_retryer(
                lambda: fwd(list(buffer), response_type, env, tools)
            )
        HistoryBuilder.append_message(result[0])
        return result

    @implements(call_tool)
    def _call_tool[T](self, tool_call: DecodedToolCall[T]) -> ToolResult[T]:
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
