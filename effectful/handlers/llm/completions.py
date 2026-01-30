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
    ChatCompletionToolParam,
    OpenAIChatCompletionAssistantMessage,
    OpenAIChatCompletionSystemMessage,
    OpenAIChatCompletionUserMessage,
    OpenAIMessageContentListBlock,
)

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.encoding import Encodable
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


def _parameter_model(sig: inspect.Signature) -> type[pydantic.BaseModel]:
    return pydantic.create_model(
        "Params",
        __config__={"extra": "forbid"},
        **{
            name: Encodable.define(param.annotation).enc
            for name, param in sig.parameters.items()
        },  # type: ignore
    )


def _response_model(sig: inspect.Signature) -> type[pydantic.BaseModel]:
    return pydantic.create_model(
        "Response",
        value=Encodable.define(sig.return_annotation).enc,
        __config__={"extra": "forbid"},
    )


def _tool_model(tool: Tool) -> ChatCompletionToolParam:
    param_model = _parameter_model(inspect.signature(tool))
    response_format = litellm.utils.type_to_response_format_param(param_model)
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


@Operation.define
def call_assistant(
    messages: collections.abc.Sequence[Message],
    response_format: type[pydantic.BaseModel] | None,
    tools: collections.abc.Mapping[str, ChatCompletionToolParam],
    model: str,
    **kwargs,
) -> Message:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    """
    response: litellm.types.utils.ModelResponse = litellm.completion(
        model,
        messages=list(messages),
        response_format=response_format,
        tools=list(tools.values()),
        **kwargs,
    )
    choice = response.choices[0]
    assert isinstance(choice, litellm.types.utils.Choices)
    message: litellm.Message = choice.message
    assert message.role == "assistant"
    return typing.cast(Message, message.model_dump(mode="json"))


@Operation.define
def call_tool(
    tool_call: ChatCompletionMessageToolCall,
    tools: collections.abc.Mapping[str, Tool],
) -> Message:
    """Implements a roundtrip call to a python function. Input is a json
    string representing an LLM tool call request parameters. The output is
    the serialised response to the model.

    """
    assert tool_call.function.name is not None
    tool = tools[tool_call.function.name]
    json_str = tool_call.function.arguments

    sig = inspect.signature(tool)
    param_model = _parameter_model(sig)

    # build dict of raw encodable types U
    raw_args = param_model.model_validate_json(json_str)

    # use encoders to decode Us to python types T
    bound_sig: inspect.BoundArguments = sig.bind(
        **{
            param_name: Encodable.define(
                sig.parameters[param_name].annotation, {}
            ).decode(getattr(raw_args, param_name))
            for param_name in raw_args.model_fields_set
        }
    )

    # call tool with python types
    result = tool(*bound_sig.args, **bound_sig.kwargs)

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

    @implements(call_assistant)
    @functools.wraps(call_assistant)
    def _completion(self, *args, **kwargs):
        return fwd(*args, **{**self.config, **kwargs})

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        response_encoding_type: Encodable = Encodable.define(
            inspect.signature(template).return_annotation, template.__context__
        )
        response_model = _response_model(inspect.signature(template))

        messages: list[Message] = [*call_system(template)]

        # encode arguments
        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(bound_args.arguments)

        user_messages: list[Message] = call_user(template.__prompt_template__, env)
        messages.extend(user_messages)

        tools = {
            **template.tools,
            **{k: t for k, t in bound_args.arguments.items() if isinstance(t, Tool)},
        }
        tool_specs = {k: _tool_model(t) for k, t in tools.items()}

        # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
        tool_calls: list[ChatCompletionMessageToolCall] = []

        message = messages[-1]
        while message["role"] != "assistant" or tool_calls:
            message = call_assistant(messages, response_model, tool_specs)
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            for tool_call in tool_calls:
                tool_call = ChatCompletionMessageToolCall.model_validate(tool_call)
                message = call_tool(tool_call, tools)
                messages.append(message)

        # return response
        serialized_result = message.get("content") or message.get("reasoning_content")
        assert isinstance(serialized_result, str), (
            "final response from the model should be a string"
        )
        encoded_result = (
            serialized_result
            if response_model is None
            else response_model.model_validate_json(serialized_result).value  # type: ignore
        )
        return response_encoding_type.decode(encoded_result)
