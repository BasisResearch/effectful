import collections
import collections.abc
import dataclasses
import functools
import inspect
import string
import textwrap
import traceback
import typing
import uuid

import litellm
import pydantic
import tenacity
from litellm import (
    ChatCompletionFunctionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionTextObject,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    OpenAIChatCompletionAssistantMessage,
    OpenAIChatCompletionSystemMessage,
    OpenAIChatCompletionUserMessage,
    OpenAIMessageContentListBlock,
)

from effectful.handlers.llm.encoding import Encodable
from effectful.handlers.llm.template import Template, Tool
from effectful.internals.unification import nested_type
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation


class AssistantMessage(OpenAIChatCompletionAssistantMessage):
    id: str


class ToolMessage(ChatCompletionToolMessage):
    id: str


class FunctionMessage(ChatCompletionFunctionMessage):
    id: str


class SystemMessage(OpenAIChatCompletionSystemMessage):
    id: str


class UserMessage(OpenAIChatCompletionUserMessage):
    id: str


Message = AssistantMessage | ToolMessage | FunctionMessage | SystemMessage | UserMessage


@Operation.define
def _get_history() -> collections.OrderedDict[str, Message]:
    raise NotImplementedError


def append_message(message: Message):
    try:
        _get_history()[message["id"]] = message
    except NotImplementedError:
        pass


def _make_message(content: dict) -> Message:
    m_id = content.get("id") or str(uuid.uuid1())
    message = typing.cast(Message, {**content, "id": m_id})
    return message


type ToolCallID = str


@dataclasses.dataclass
class ToolCallDecodingError(Exception):
    """Error raised when decoding a tool call fails."""

    tool_name: str
    tool_call_id: str
    original_error: Exception
    raw_message: Message

    def __str__(self) -> str:
        return f"Error decoding tool call '{self.tool_name}': {self.original_error}. Please provide a valid response and try again."

    def to_feedback_message(self, include_traceback: bool) -> Message:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return _make_message(
            {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": error_message,
            },
        )


@dataclasses.dataclass
class ResultDecodingError(Exception):
    """Error raised when decoding the LLM response result fails."""

    original_error: Exception
    raw_message: Message

    def __str__(self) -> str:
        return f"Error decoding response: {self.original_error}. Please provide a valid response and try again."

    def to_feedback_message(self, include_traceback: bool) -> Message:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return _make_message(
            {"role": "user", "content": error_message},
        )


@dataclasses.dataclass
class ToolCallExecutionError(Exception):
    """Error raised when a tool execution fails at runtime."""

    tool_name: str
    tool_call_id: str
    original_error: BaseException

    def __str__(self) -> str:
        return f"Tool execution failed: Error executing tool '{self.tool_name}': {self.original_error}"

    def to_feedback_message(self, include_traceback: bool) -> Message:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return _make_message(
            {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": error_message,
            },
        )


@dataclasses.dataclass
class DecodedToolCall[T]:
    tool: Tool[..., T]
    bound_args: inspect.BoundArguments
    id: ToolCallID
    is_final: bool = False


type MessageResult[T] = tuple[Message, typing.Sequence[DecodedToolCall], T | None, bool]


@functools.cache
def _param_model(tool: Tool) -> type[pydantic.BaseModel]:
    sig = inspect.signature(tool)
    return pydantic.create_model(
        "Params",
        __config__={"extra": "forbid"},
        **{
            name: Encodable.define(param.annotation).enc
            for name, param in sig.parameters.items()
        },  # type: ignore
    )


@functools.cache
def _function_model(tool: Tool) -> ChatCompletionToolParam:
    response_format = litellm.utils.type_to_response_format_param(_param_model(tool))
    assert response_format is not None
    assert tool.__default__.__doc__ is not None
    return {
        "type": "function",
        "function": {
            "name": tool.__name__,
            "description": textwrap.dedent(tool.__default__.__doc__),
            "parameters": response_format["json_schema"]["schema"],
            "strict": True,
        },
    }


def decode_tool_call(
    tool_call: ChatCompletionMessageToolCall,
    tools: collections.abc.Mapping[str, Tool],
    raw_message: Message,
) -> DecodedToolCall:
    """Decode a tool call from the LLM response into a DecodedToolCall.

    Args:
        tool_call: The tool call to decode.
        tools: Mapping of tool names to Tool objects.
        raw_message: Optional raw assistant message for error context.

    Raises:
        ToolCallDecodingError: If the tool call cannot be decoded.
    """
    tool_name = tool_call.function.name
    assert tool_name is not None

    try:
        tool = tools[tool_name]
    except KeyError as e:
        raise ToolCallDecodingError(
            tool_name, tool_call.id, e, raw_message=raw_message
        ) from e

    json_str = tool_call.function.arguments
    sig = inspect.signature(tool)

    try:
        # build dict of raw encodable types U
        raw_args = _param_model(tool).model_validate_json(json_str)

        # use encoders to decode Us to python types T
        bound_sig: inspect.BoundArguments = sig.bind(
            **{
                param_name: Encodable.define(
                    sig.parameters[param_name].annotation, {}
                ).decode(getattr(raw_args, param_name))
                for param_name in raw_args.model_fields_set
            }
        )
    except (pydantic.ValidationError, TypeError, ValueError, SyntaxError) as e:
        raise ToolCallDecodingError(
            tool_name, tool_call.id, e, raw_message=raw_message
        ) from e

    from effectful.handlers.llm.template import _is_final_answer_tool

    return DecodedToolCall(
        tool, bound_sig, tool_call.id, is_final=_is_final_answer_tool(tool)
    )


@Operation.define
@functools.wraps(litellm.completion)
def completion(*args, **kwargs) -> typing.Any:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    """
    return litellm.completion(*args, **kwargs)


@Operation.define
def call_assistant[T, U](
    tools: collections.abc.Mapping[str, Tool],
    response_format: Encodable[T, U],
    model: str,
    **kwargs,
) -> MessageResult[T]:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    Raises:
        ToolCallDecodingError: If a tool call cannot be decoded. The error
            includes the raw assistant message for retry handling.
        ResultDecodingError: If the result cannot be decoded. The error
            includes the raw assistant message for retry handling.
    """
    tool_specs = {k: _function_model(t) for k, t in tools.items()}
    response_model = pydantic.create_model(
        "Response", value=response_format.enc, __config__={"extra": "forbid"}
    )

    messages = list(_get_history().values())
    response: litellm.types.utils.ModelResponse = completion(
        model,
        messages=list(messages),
        response_format=response_model,
        tools=list(tool_specs.values()),
        **kwargs,
    )
    choice = response.choices[0]
    assert isinstance(choice, litellm.types.utils.Choices)

    message: litellm.Message = choice.message
    assert message.role == "assistant"

    raw_message = _make_message({**message.model_dump(mode="json")})
    append_message(raw_message)

    tool_calls: list[DecodedToolCall] = []
    raw_tool_calls = message.get("tool_calls") or []
    for raw_tool_call in raw_tool_calls:
        validated_tool_call = ChatCompletionMessageToolCall.model_validate(
            raw_tool_call
        )
        decoded_tool_call = decode_tool_call(validated_tool_call, tools, raw_message)
        tool_calls.append(decoded_tool_call)

    final_tcs = [tc for tc in tool_calls if tc.is_final]
    if final_tcs:
        final_tc = final_tcs[0]
        if len(tool_calls) > 1:
            raise ToolCallDecodingError(
                final_tc.tool.__name__,
                final_tc.id,
                ValueError(
                    f"IsFinal tool '{final_tc.tool.__name__}' must be the "
                    f"only tool call in a round, but {len(tool_calls)} tool calls "
                    f"were generated."
                ),
                raw_message=raw_message,
            )
        # Validate that the tool's return type matches the template's.
        tool_ret = inspect.signature(final_tc.tool).return_annotation
        if typing.get_origin(tool_ret) is typing.Annotated:
            tool_ret = typing.get_args(tool_ret)[0]
        if not issubclass(tool_ret, response_format.base):
            raise ToolCallDecodingError(
                final_tc.tool.__name__,
                final_tc.id,
                TypeError(
                    f"IsFinal tool '{final_tc.tool.__name__}' returns "
                    f"{tool_ret!r}, but the enclosing template expects "
                    f"{response_format.base!r}."
                ),
                raw_message=raw_message,
            )

    result = None
    if not tool_calls:
        # return response
        serialized_result = message.get("content") or message.get("reasoning_content")
        assert isinstance(serialized_result, str), (
            "final response from the model should be a string"
        )
        try:
            raw_result = response_model.model_validate_json(serialized_result)
            result = response_format.decode(raw_result.value)  # type: ignore
        except (pydantic.ValidationError, TypeError, ValueError, SyntaxError) as e:
            raise ResultDecodingError(e, raw_message=raw_message) from e

    is_final = not all(not tc.is_final for tc in tool_calls)
    return (raw_message, tool_calls, result, is_final)


@Operation.define
def call_tool[T](tool_call: DecodedToolCall[T]) -> tuple[Message, T | None, bool]:
    """Execute a tool and return the serialised message, the raw result, and
    whether this result is a final answer.

    Returns:
        A 3-tuple ``(message, result, is_final)``.  ``message`` is appended
        to the conversation history.  When ``is_final`` is ``True`` the
        completion loop uses ``result`` directly as the template return value.
    """
    # call tool with python types
    try:
        result: T = tool_call.tool(
            *tool_call.bound_args.args, **tool_call.bound_args.kwargs
        )
    except Exception as e:
        raise ToolCallExecutionError(tool_call.tool.__name__, tool_call.id, e) from e

    # serialize back to U using encoder for return type
    return_type: Encodable[T, typing.Any] = Encodable.define(
        typing.cast(type[typing.Any], nested_type(result).value)
    )  # type: ignore
    encoded_result = return_type.serialize(return_type.encode(result))
    message = _make_message(
        dict(role="tool", content=encoded_result, tool_call_id=tool_call.id),
    )
    append_message(message)
    return message, result, tool_call.is_final


@Operation.define
def call_user(
    template: str,
    env: collections.abc.Mapping[str, typing.Any],
) -> Message:
    """
    Format a template applied to arguments into a user message.
    """
    formatter = string.Formatter()
    parts: list[OpenAIMessageContentListBlock] = []

    buf: list[str] = []

    def flush_text() -> None:
        if buf:
            parts.append(ChatCompletionTextObject(type="text", text="".join(buf)))
            buf.clear()

    for literal, field_name, format_spec, conversion in formatter.parse(
        textwrap.dedent(template)
    ):
        if literal:
            buf.append(literal)

        if field_name is None:
            continue

        obj, _ = formatter.get_field(field_name, (), env)
        encoder = Encodable.define(
            typing.cast(type[typing.Any], nested_type(obj).value)
        )
        encoded_obj: typing.Sequence[OpenAIMessageContentListBlock] = encoder.serialize(
            encoder.encode(obj)
        )
        for part in encoded_obj:
            if part["type"] == "text":
                text = (
                    formatter.convert_field(part["text"], conversion)
                    if conversion
                    else part["text"]
                )
                buf.append(formatter.format_field(text, format_spec or ""))
            else:
                flush_text()
                parts.append(part)

    flush_text()

    # Note: The OpenAI api only seems to accept images in the 'user' role. The
    # effect of different roles on the model's response is currently unclear.
    message = _make_message(dict(role="user", content=parts))
    append_message(message)
    return message


@Operation.define
def call_system(template: Template) -> Message:
    """Get system instruction message(s) to prepend to all LLM prompts."""
    system_prompt = textwrap.dedent(f"""
    SYSTEM: You are a helpful LLM assistant named {template.__name__}.
    """)

    message = _make_message(dict(role="system", content=system_prompt))
    try:
        history: collections.OrderedDict[str, Message] = _get_history()
        if any(m["role"] == "system" for m in history.values()):
            assert sum(1 for m in history.values() if m["role"] == "system") == 1, (
                "There should be at most one system message in the history"
            )
            assert history[next(iter(history))]["role"] == "system", (
                "The system message should be the first message in the history"
            )
            history.popitem(last=False)  # remove existing system message
        history[message["id"]] = message
        history.move_to_end(message["id"], last=False)
        return message
    except NotImplementedError:
        return message


class RetryLLMHandler(ObjectInterpretation):
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

    _user_before_sleep: collections.abc.Callable[[tenacity.RetryCallState], None] | None

    def __init__(
        self,
        include_traceback: bool = True,
        catch_tool_errors: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
        stop: tenacity.stop.stop_base = tenacity.stop_after_attempt(4),
        **kwargs,
    ):
        self.include_traceback = include_traceback
        self.catch_tool_errors = catch_tool_errors
        assert "retry" not in kwargs, "Cannot override retry logic of RetryLLMHandler"
        assert "reraise" not in kwargs, (
            "Cannot override reraise logic of RetryLLMHandler"
        )
        self._user_before_sleep = kwargs.pop("before_sleep", None)
        self.call_assistant_retryer = tenacity.Retrying(
            retry=tenacity.retry_if_exception_type(
                (ToolCallDecodingError, ResultDecodingError)
            ),
            reraise=True,
            before_sleep=self._before_sleep,
            stop=stop,
            **kwargs,
        )

    def _before_sleep(self, retry_state: tenacity.RetryCallState) -> None:
        e = retry_state.outcome.exception()  # type: ignore
        assert isinstance(e, (ToolCallDecodingError, ResultDecodingError))
        append_message(e.raw_message)
        append_message(e.to_feedback_message(self.include_traceback))
        if self._user_before_sleep is not None:
            self._user_before_sleep(retry_state)

    @implements(call_assistant)
    def _call_assistant[T, U](
        self,
        tools: collections.abc.Mapping[str, Tool],
        response_format: Encodable[T, U],
        model: str,
        **kwargs,
    ) -> MessageResult[T]:
        _message_sequence = _get_history().copy()

        def _attempt() -> MessageResult[T]:
            return fwd(tools, response_format, model, **kwargs)

        with handler({_get_history: lambda: _message_sequence}):
            message, tool_calls, result, is_final = self.call_assistant_retryer(
                _attempt
            )

        append_message(message)
        return (message, tool_calls, result, is_final)

    @implements(call_tool)
    def _call_tool[T](
        self, tool_call: DecodedToolCall[T]
    ) -> tuple[Message, T | None, bool]:
        """Handle tool execution with runtime error capture.

        Runtime errors from tool execution are captured and returned as
        error messages to the LLM. Only exceptions matching `catch_tool_errors`
        are caught; others propagate up.  When an error is caught,
        ``is_final`` is always ``False`` so the error feedback goes back
        to the LLM rather than being mistaken for a final answer.
        """
        try:
            return fwd(tool_call)
        except ToolCallExecutionError as e:
            if isinstance(e.original_error, self.catch_tool_errors):
                message = e.to_feedback_message(self.include_traceback)
                append_message(message)
                return message, None, False
            else:
                raise


class LiteLLMProvider(ObjectInterpretation):
    """Implements templates using the LiteLLM API."""

    config: collections.abc.Mapping[str, typing.Any]

    def __init__(self, model="gpt-4o", **config):
        self.config = {
            "model": model,
            **inspect.signature(litellm.completion).bind_partial(**config).kwargs,
        }

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        # encode arguments
        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(bound_args.arguments)

        # Create response_model with env so tools passed as arguments are available
        response_model = Encodable.define(template.__signature__.return_annotation, env)

        history: collections.OrderedDict[str, Message] = getattr(
            template, "__history__", collections.OrderedDict()
        )  # type: ignore
        history_copy = history.copy()

        with handler({_get_history: lambda: history_copy}):
            call_system(template)

            message: Message = call_user(template.__prompt_template__, env)

            # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
            tool_calls: list[DecodedToolCall] = []
            result: T | None = None
            is_final: bool = False
            while not is_final:
                message, tool_calls, result, is_final = call_assistant(
                    template.tools, response_model, **self.config
                )
                for tool_call in tool_calls:
                    message, result, is_final = call_tool(tool_call)

        try:
            _get_history()
        except NotImplementedError:
            history.update(history_copy)
        return typing.cast(T, result)
