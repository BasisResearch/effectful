import collections.abc
import typing

import tenacity

from effectful.handlers.llm.harness.hooks import (
    AssistantResult,
    ResultDecodingError,
    ToolCallDecodingError,
    ToolCallExecutionError,
    ToolResult,
    call_assistant,
    call_tool,
)
from effectful.handlers.llm.harness.serialization import DecodedToolCall
from effectful.handlers.llm.harness.transaction import HistoryBuilder, transaction
from effectful.handlers.llm.types import Template, Tool
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class TenacityRetryer(ObjectInterpretation):
    """Retries LLM requests if tool call or result decoding fails.

    This handler intercepts `call_assistant` and catches `ToolCallDecodingError`
    and `ResultDecodingError`. When these errors occur, it appends error feedback
    to the messages and retries the request. Malformed messages from retry attempts
    are pruned from the final result.

    For runtime tool execution failures (handled via `call_tool`), errors are
    captured and returned as tool response messages.

    Args:
        include_traceback: If True, include full traceback in error feedback
            for better debugging context (default: True).
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
        assert "retry" not in kwargs, "Cannot override retry logic of RetryLLMHandler"
        assert "reraise" not in kwargs, (
            "Cannot override reraise logic of RetryLLMHandler"
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
        env: collections.abc.Mapping[str, typing.Any],
        response_type: type[T],
        tools: collections.abc.Set[Tool] = frozenset(),
        anchor: "Template | None" = None,
        force_tool: bool = False,
    ) -> AssistantResult[T]:
        with transaction(write_back=False):
            result = self.call_assistant_retryer(fwd)
        HistoryBuilder.append_message(result[0])
        return result

    @implements(call_tool)
    def _call_tool[T](self, tool_call: DecodedToolCall[T]) -> ToolResult[T]:
        """Handle tool execution with runtime error capture.

        Runtime errors from tool execution are captured and returned as
        error messages to the LLM. Only exceptions matching `catch_tool_errors`
        are caught; others propagate up.

        A captured failure is reported as ``is_final=False`` so that the
        completion loop continues even when a :class:`FinalTool` call raised:
        the model sees the error message and gets another turn to retry.
        """
        try:
            return fwd(tool_call)
        except ToolCallExecutionError as e:
            if isinstance(e.original_error, self.catch_tool_errors):
                return (e.to_feedback_message(include_traceback=True), None, False)
            else:
                raise
