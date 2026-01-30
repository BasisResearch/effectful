import collections
import collections.abc
import functools
import inspect
import string
import textwrap
import typing

import litellm
import pydantic
from litellm import (
    ChatCompletionFunctionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionTextObject,
    ChatCompletionToolMessage,
    OpenAIChatCompletionAssistantMessage,
    OpenAIChatCompletionSystemMessage,
    OpenAIChatCompletionUserMessage,
    OpenAIMessageContentListBlock,
)

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.encoding import Encodable
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


class DecodedToolCall[T](typing.NamedTuple):
    tool: Tool[..., T]
    bound_args: inspect.BoundArguments
    id: ToolCallID


type MessageResult[T] = tuple[Message, typing.Sequence[DecodedToolCall], T | None]


def decode_tool_call(
    tool_call: ChatCompletionMessageToolCall,
    tools: collections.abc.Mapping[str, Tool],
) -> DecodedToolCall:
    """Decode a tool call from the LLM response into a DecodedToolCall."""
    assert tool_call.function.name is not None
    tool = tools[tool_call.function.name]
    json_str = tool_call.function.arguments

    sig = inspect.signature(tool)

    # build dict of raw encodable types U
    raw_args = tool.param_model.model_validate_json(json_str)

    # use encoders to decode Us to python types T
    bound_sig: inspect.BoundArguments = sig.bind(
        **{
            param_name: Encodable.define(
                sig.parameters[param_name].annotation, {}
            ).decode(getattr(raw_args, param_name))
            for param_name in raw_args.model_fields_set
        }
    )
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

    """
    tool_specs = {k: t.model for k, t in tools.items()}
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

    tool_calls: list[DecodedToolCall] = []
    raw_tool_calls = message.get("tool_calls") or []
    for tool_call in raw_tool_calls:
        tool_call = ChatCompletionMessageToolCall.model_validate(tool_call)
        decoded_tool_call = decode_tool_call(tool_call, tools)
        tool_calls.append(decoded_tool_call)

    result = None
    if not tool_calls:
        # return response
        serialized_result = message.get("content") or message.get("reasoning_content")
        assert isinstance(serialized_result, str), (
            "final response from the model should be a string"
        )
        raw_result = response_model.model_validate_json(serialized_result)
        result = response_format.decode(raw_result.value)  # type: ignore

    return (typing.cast(Message, message.model_dump(mode="json")), tool_calls, result)


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
