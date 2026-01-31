import collections
import collections.abc
import dataclasses
import functools
import inspect
import string
import textwrap
import traceback
import typing

import litellm
import pydantic
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
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation

Message = (
    OpenAIChatCompletionAssistantMessage
    | ChatCompletionToolMessage
    | ChatCompletionFunctionMessage
    | OpenAIChatCompletionSystemMessage
    | OpenAIChatCompletionUserMessage
)

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


@dataclasses.dataclass
class ResultDecodingError(Exception):
    """Error raised when decoding the LLM response result fails."""

    original_error: Exception
    raw_message: Message

    def __str__(self) -> str:
        return f"Error decoding response: {self.original_error}. Please provide a valid response and try again."


@dataclasses.dataclass
class ToolCallExecutionError(Exception):
    tool_name: str
    e: Exception

    def __str__(self) -> str:
        return (
            f"Tool execution failed: Error executing tool '{self.tool_name}': {self.e}"
        )


class DecodedToolCall[T](typing.NamedTuple):
    tool: Tool[..., T]
    bound_args: inspect.BoundArguments
    id: ToolCallID


type MessageResult[T] = tuple[Message, typing.Sequence[DecodedToolCall], T | None]


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
    except (pydantic.ValidationError, TypeError, ValueError) as e:
        raise ToolCallDecodingError(
            tool_name, tool_call.id, e, raw_message=raw_message
        ) from e

    return DecodedToolCall(tool, bound_sig, tool_call.id)


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
    messages: collections.abc.Sequence[Message],
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

    raw_message = typing.cast(Message, message.model_dump(mode="json"))

    tool_calls: list[DecodedToolCall] = []
    raw_tool_calls = message.get("tool_calls") or []
    for raw_tool_call in raw_tool_calls:
        validated_tool_call = ChatCompletionMessageToolCall.model_validate(
            raw_tool_call
        )
        decoded_tool_call = decode_tool_call(validated_tool_call, tools, raw_message)
        tool_calls.append(decoded_tool_call)

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
        except pydantic.ValidationError as e:
            raise ResultDecodingError(e, raw_message=raw_message) from e

    return (raw_message, tool_calls, result)


@Operation.define
def call_tool(tool_call: DecodedToolCall) -> Message:
    """Implements a roundtrip call to a python function. Input is a json
    string representing an LLM tool call request parameters. The output is
    the serialised response to the model.

    """
    # call tool with python types
    result = tool_call.tool(*tool_call.bound_args.args, **tool_call.bound_args.kwargs)

    # serialize back to U using encoder for return type
    return_type = Encodable.define(type(result))
    encoded_result = return_type.serialize(return_type.encode(result))
    return typing.cast(
        Message, dict(role="tool", content=encoded_result, tool_call_id=tool_call.id)
    )


@Operation.define
def call_user(
    template: str,
    env: collections.abc.Mapping[str, typing.Any],
) -> list[Message]:
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
        encoder = Encodable.define(type(obj))
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
    return [typing.cast(Message, dict(role="user", content=parts))]


@Operation.define
def call_system(template: Template) -> collections.abc.Sequence[Message]:
    """Get system instruction message(s) to prepend to all LLM prompts."""
    return ()


class RetryLLMHandler(ObjectInterpretation):
    """Retries LLM requests if tool call or result decoding fails.

    This handler intercepts `call_assistant` and catches `ToolCallDecodingError`
    and `ResultDecodingError`. When these errors occur, it appends error feedback
    to the messages and retries the request. Malformed messages from retry attempts
    are pruned from the final result.

    For runtime tool execution failures (handled via `call_tool`), errors are
    captured and returned as tool response messages.

    Args:
        num_retries: The maximum number of retries (default: 3).
        include_traceback: If True, include full traceback in error feedback
            for better debugging context (default: False).
    """

    def __init__(self, num_retries: int = 3, include_traceback: bool = False):
        self.num_retries = num_retries
        self.include_traceback = include_traceback

    def _format_error(self, error: Exception) -> str:
        """Format an error message, optionally including traceback."""
        if self.include_traceback:
            tb = traceback.format_exc()
            return f"{error}\n\nTraceback:\n```\n{tb}```"
        return f"{error}"

    @implements(call_assistant)
    def _call_assistant[T, U](
        self,
        messages: collections.abc.Sequence[Message],
        tools: collections.abc.Mapping[str, Tool],
        response_format: Encodable[T, U],
        model: str,
        **kwargs,
    ) -> MessageResult[T]:
        messages_list = list(messages)
        last_attempt = self.num_retries

        for attempt in range(self.num_retries + 1):
            try:
                message, tool_calls, result = fwd(
                    messages_list, tools, response_format, model, **kwargs
                )

                # Success! The returned message is the final successful response.
                # Malformed messages from retries are only in messages_list,
                # not in the returned result.
                return (message, tool_calls, result)

            except ToolCallDecodingError as e:
                # On last attempt, re-raise to preserve full traceback
                if attempt == last_attempt:
                    raise

                # Add the malformed assistant message
                messages_list.append(e.raw_message)

                # Add error feedback as a tool response
                error_feedback: Message = typing.cast(
                    Message,
                    {
                        "role": "tool",
                        "tool_call_id": e.tool_call_id,
                        "content": self._format_error(e),
                    },
                )
                messages_list.append(error_feedback)

            except ResultDecodingError as e:
                # On last attempt, re-raise to preserve full traceback
                if attempt == last_attempt:
                    raise

                # Add the malformed assistant message
                messages_list.append(e.raw_message)
                # Add error feedback as a user message
                result_error_feedback: Message = typing.cast(
                    Message,
                    {
                        "role": "user",
                        "content": self._format_error(e),
                    },
                )
                messages_list.append(result_error_feedback)

        # Should never reach here - either we return on success or raise on final failure
        raise AssertionError("Unreachable: retry loop exited without return or raise")

    @implements(completion)
    def _completion(self, *args, **kwargs) -> typing.Any:
        """Inject num_retries for litellm's built-in network error handling."""
        return fwd(*args, num_retries=self.num_retries, **kwargs)

    @implements(call_tool)
    def _call_tool(self, tool_call: DecodedToolCall) -> Message:
        """Handle tool execution with runtime error capture.

        Runtime errors from tool execution are captured and returned as
        error messages to the LLM.
        """
        try:
            return fwd(tool_call)
        except Exception as e:
            return typing.cast(
                Message,
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": self._format_error(
                        ToolCallExecutionError(f"{tool_call.tool.__name__}", e)
                    ),
                },
            )


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
        messages: list[Message] = [*call_system(template)]

        # encode arguments
        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(bound_args.arguments)

        # Create response_model with env so tools passed as arguments are available
        response_model = Encodable.define(template.__signature__.return_annotation, env)

        user_messages: list[Message] = call_user(template.__prompt_template__, env)
        messages.extend(user_messages)

        # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
        tool_calls: list[DecodedToolCall] = []

        message = messages[-1]
        result: T | None = None
        while message["role"] != "assistant" or tool_calls:
            message, tool_calls, result = call_assistant(
                messages, template.tools, response_model, **self.config
            )
            messages.append(message)
            for tool_call in tool_calls:
                message = call_tool(tool_call)
                messages.append(message)

        assert result is not None, (
            "call_assistant did not produce a result nor tool_calls"
        )
        # return response
        return result
