import collections
import collections.abc
import functools
import inspect
import string
import textwrap
from typing import Any

import litellm
import pydantic
from litellm import (
    ChatCompletionMessageToolCall,
    ChatCompletionTextObject,
    ChatCompletionToolParam,
    Message,
    OpenAIMessageContentListBlock,
)

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.encoding import Encodable, type_to_encodable_type
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation


def _parameter_model(sig: inspect.Signature) -> type[pydantic.BaseModel]:
    return pydantic.create_model(
        "Params",
        __config__={"extra": "forbid"},
        **{
            name: type_to_encodable_type(param.annotation).t
            for name, param in sig.parameters.items()
        },  # type: ignore
    )


def _response_model(sig: inspect.Signature) -> type[pydantic.BaseModel]:
    return pydantic.create_model(
        "Response",
        value=type_to_encodable_type(sig.return_annotation).t,
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
    *,
    model: str,
    **kwargs,
) -> Message:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    """
    response: litellm.types.utils.ModelResponse = litellm.completion(
        model=model,
        messages=list(messages),
        response_format=response_format,
        tools=list(tools.values()),
        **kwargs,
    )
    choice = response.choices[0]
    assert isinstance(choice, litellm.types.utils.Choices)
    message: Message = choice.message
    assert message.role == "assistant"
    return message


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
    return_type = type_to_encodable_type(sig.return_annotation)

    # build dict of raw encodable types U
    raw_args = param_model.model_validate_json(json_str)

    # use encoders to decode Us to python types T
    bound_sig: inspect.BoundArguments = sig.bind(
        **{
            param_name: type_to_encodable_type(
                sig.parameters[param_name].annotation
            ).decode(getattr(raw_args, param_name))
            for param_name in raw_args.model_fields_set
        }
    )

    # call tool with python types
    result = tool(*bound_sig.args, **bound_sig.kwargs)

    # serialize back to U using encoder for return type
    encoded_result = return_type.serialize(return_type.encode(result))
    return Message.model_validate(dict(role="tool", content=encoded_result))


@Operation.define
def call_user(
    template: str,
    env: collections.abc.Mapping[str, Any],
) -> Message:
    """
    Format a template applied to arguments into a user message.
    """
    formatter: string.Formatter = string.Formatter()
    prompt_parts: list[OpenAIMessageContentListBlock] = []

    for literal, field_name, fspec, cspec in formatter.parse(textwrap.dedent(template)):
        if literal:
            prompt_parts.append(ChatCompletionTextObject(type="text", text=literal))
        if field_name is not None:
            obj, _ = formatter.get_field(field_name, (), env)
            encoder = type_to_encodable_type(type(obj))
            encoded_obj = encoder.serialize(encoder.encode(obj))
            for part in formatter.convert_field(encoded_obj, cspec):
                if part["type"] == "text":
                    part["text"] = formatter.format_field(part["text"], fspec or "")
                prompt_parts.append(part)

    # Note: The OpenAI api only seems to accept images in the 'user' role. The
    # effect of different roles on the model's response is currently unclear.
    return Message.model_validate(dict(role="user", content=prompt_parts), strict=True)


@Operation.define
def call_system(template: Template) -> collections.abc.Sequence[Message]:
    """Get system instruction message(s) to prepend to all LLM prompts."""
    return ()


class LiteLLMProvider(ObjectInterpretation):
    """Implements templates using the LiteLLM API."""

    config: collections.abc.Mapping[str, Any]

    def __init__(self, **config):
        self.config = (
            inspect.signature(litellm.completion).bind_partial(**config).kwargs
        )

    @implements(call_assistant)
    @functools.wraps(call_assistant)
    def _completion(self, *args, **kwargs):
        return fwd(*args, **{**self.config, **kwargs})

    @implements(Template.__apply__)
    @staticmethod
    def _call[**P, T](template: Template[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        response_encoding_type: Encodable[T] = type_to_encodable_type(
            inspect.signature(template).return_annotation
        )
        response_model = _response_model(inspect.signature(template))

        messages: list[Message] = [*call_system(template)]

        # encode arguments
        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(bound_args.arguments)

        message: Message = call_user(template.__prompt_template__, env)
        messages.append(message)

        tools = {
            **template.tools,
            **{k: t for k, t in bound_args.arguments.items() if isinstance(t, Tool)},
        }
        tool_specs = {k: _tool_model(t) for k, t in tools.items()}

        # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
        while message.role != "assistant" or message.tool_calls:
            message = call_assistant(messages, response_model, tool_specs)
            messages.append(message)

            for tool_call in message.tool_calls or []:
                message = call_tool(tool_call, tools)
                messages.append(message)

        # return response
        serialized_result = message.content or message.reasoning_content
        encoded_result = (
            serialized_result
            if response_model is None
            else response_model.model_validate_json(serialized_result).value  # type: ignore
        )
        return response_encoding_type.decode(encoded_result)


class InstructionsHandler(ObjectInterpretation):
    """Implements system instructions using the LiteLLM API."""

    instructions: str | collections.abc.Mapping[Template, str]

    def __init__(self, instructions: str | collections.abc.Mapping[Template, str]):
        if isinstance(instructions, collections.abc.Mapping):
            assert instructions, "Instructions mapping cannot be empty."
            assert all(instr for instr in instructions.values()), (
                "All instructions in the mapping must be non-empty."
            )
        else:
            assert instructions, "Instructions string cannot be empty."
        self.instructions = instructions

    @implements(call_system)
    def _system_instruction(
        self, template: Template
    ) -> collections.abc.Sequence[Message]:
        if isinstance(self.instructions, str):
            return (
                *fwd(),
                Message.model_validate(dict(role="system", content=self.instructions)),
            )
        elif template in self.instructions:
            return (
                *fwd(),
                Message.model_validate(
                    dict(role="system", content=self.instructions[template])
                ),
            )
        else:
            return fwd()
