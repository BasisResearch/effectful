import contextlib
import functools
import inspect
import logging
import string
import traceback
import typing
from collections.abc import Callable, Hashable
from typing import Any

import litellm
import pydantic
from litellm import (
    Choices,
    Message,
    OpenAIChatCompletionToolParam,
    OpenAIMessageContent,
    OpenAIMessageContentListBlock,
)
from litellm.types.utils import ModelResponse

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.encoding import type_to_encodable_type
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import Operation


class _OpenAIPromptFormatter(string.Formatter):
    def format_as_messages(
        self, format_str: str, /, *args, **kwargs
    ) -> OpenAIMessageContent:
        prompt_parts: list[OpenAIMessageContentListBlock] = []
        current_text = ""

        def push_current_text():
            nonlocal current_text
            if current_text:
                prompt_parts.append({"type": "text", "text": current_text})
            current_text = ""

        for literal, field_name, format_spec, conversion in self.parse(format_str):
            current_text += literal

            if field_name is not None:
                obj, _ = self.get_field(field_name, args, kwargs)
                part = self.convert_field(obj, conversion)
                # special casing for text
                if (
                    isinstance(part, list)
                    and len(part) == 1
                    and part[0]["type"] == "text"
                ):
                    current_text += self.format_field(
                        part[0]["text"], format_spec if format_spec else ""
                    )
                elif isinstance(part, list):
                    push_current_text()
                    prompt_parts.extend(part)
                else:
                    prompt_parts.append(part)

        push_current_text()
        return prompt_parts


@defop
@functools.wraps(litellm.completion)
def completion(*args, **kwargs) -> Any:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    """
    return litellm.completion(*args, **kwargs)


class CacheLLMRequestHandler(ObjectInterpretation):
    """Caches LLM requests."""

    def __init__(self):
        self.cache: dict[Hashable, Any] = {}

    def _make_hashable(self, obj: Any) -> Hashable:
        """Recursively convert objects to hashable representations."""
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list | tuple):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, set):
            return frozenset(self._make_hashable(item) for item in obj)
        else:
            # Primitives (int, float, str, bytes, etc.) are already hashable
            return obj

    @implements(completion)
    def _cache_completion(self, *args, **kwargs) -> Any:
        key = self._make_hashable((args, kwargs))
        if key in self.cache:
            return self.cache[key]
        response = fwd()
        self.cache[key] = response
        return response


class LLMLoggingHandler(ObjectInterpretation):
    """Logs completion rounds and tool_call invocations using Python logging.

    Configure with a logger or logger name. By default logs at INFO level.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
    ):
        """Initialize the logging handler.

        Args:
            logger: The logger to use. If None, the logger name will be the name of the class. Note that the logger should have a handler that print out also the extra payload, e.g. `%(payload)s`.
        """
        self.logger = logger or logging.getLogger(__name__)

    @implements(completion)
    def _log_completion(self, *args, **kwargs) -> Any:
        """Log the LLM request and response."""

        response = fwd()
        self.logger.info(
            "llm.request",
            extra={"payload": {"args": args, "kwargs": kwargs, "response": response}},
        )
        return response

    @implements(Tool.__apply__)
    def _log_tool_call(self, tool: Operation, *args, **kwargs) -> Any:
        """Log the tool call and result."""

        tool_name = tool.__name__
        result = fwd()
        self.logger.info(
            "llm.tool_call",
            extra={"payload": {"tool": tool_name, "args": args, "kwargs": kwargs}},
        )
        return result


def parameter_model(tool: Tool) -> type[pydantic.BaseModel]:
    fields = {
        name: type_to_encodable_type(param.annotation).t
        for name, param in tool.__signature__.parameters.items()
    }
    parameter_model = pydantic.create_model(
        "Params",
        __config__={"extra": "forbid"},
        **fields,  # type: ignore
    )
    return parameter_model


def function_definition(tool: Tool) -> OpenAIChatCompletionToolParam:
    param_model = parameter_model(tool)
    response_format = litellm.utils.type_to_response_format_param(param_model)
    description = tool.__default__.__doc__
    assert response_format is not None
    assert description is not None
    return {
        "type": "function",
        "function": {
            "name": tool.__name__,
            "description": description,
            "parameters": response_format["json_schema"]["schema"],
            "strict": True,
        },
    }


def call_with_json_args(tool: Tool, json_str: str) -> OpenAIMessageContent:
    """Implements a roundtrip call to a python function. Input is a json
    string representing an LLM tool call request parameters. The output is
    the serialised response to the model.

    """
    sig = tool.__signature__
    param_model = parameter_model(tool)
    try:
        # build dict of raw encodable types U
        raw_args = param_model.model_validate_json(json_str)

        # use encoders to decode Us to python types T
        params: dict[str, Any] = {
            param_name: type_to_encodable_type(
                sig.parameters[param_name].annotation
            ).decode(getattr(raw_args, param_name))
            for param_name in raw_args.model_fields_set
        }

        # call tool with python types
        result = tool(**params)

        # serialize back to U using encoder for return type
        encoded_ty = type_to_encodable_type(sig.return_annotation)
        encoded_value = encoded_ty.encode(result)

        # serialise back to Json
        return encoded_ty.serialize(encoded_value)
    except Exception as exn:
        return str({"status": "failure", "exception": str(exn)})


@defop
def compute_response(template: Template, model_input: list[Any]) -> ModelResponse:
    """Produce a complete model response for an input message sequence. This may
    involve multiple API requests if tools are invoked by the model.

    """
    ret_type = template.__signature__.return_annotation
    tools = template.tools

    tool_schemas = [function_definition(t) for t in tools.values()]
    response_encoding_type: type | None = type_to_encodable_type(ret_type).t
    if response_encoding_type == str:
        response_encoding_type = None

    # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
    while True:
        response: ModelResponse = completion(
            messages=model_input,
            response_format=pydantic.create_model(
                "Response", value=response_encoding_type, __config__={"extra": "forbid"}
            )
            if response_encoding_type
            else None,
            tools=tool_schemas,
        )

        choice: Choices = typing.cast(Choices, response.choices[0])
        message: Message = choice.message
        if not message.tool_calls:
            return response
        model_input.append(message.to_dict())

        for tool_call in message.tool_calls:
            function = tool_call.function
            function_name = function.name
            assert function_name is not None
            tool = tools[function_name]
            tool_result = call_with_json_args(tool, function.arguments)
            model_input.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )


def decode_response[**P, T](template: Callable[P, T], response: ModelResponse) -> T:
    """Decode an LLM response into an instance of the template return type. This
    operation should raise if the output cannot be decoded.
    """
    assert isinstance(template, Template)
    choice: Choices = typing.cast(Choices, response.choices[0])
    last_resp: Message = choice.message
    assert isinstance(last_resp, Message)
    result_str = last_resp.content or last_resp.reasoning_content
    assert result_str

    ret_type = template.__signature__.return_annotation
    encodable_ty = type_to_encodable_type(ret_type)

    if encodable_ty.t == str:
        # if encoding as a type, value is just directly what the llm returned
        value = result_str
    else:
        Result = pydantic.create_model("Result", value=encodable_ty.t)
        result = Result.model_validate_json(result_str)
        assert isinstance(result, Result)
        value = result.value  # type: ignore

    return encodable_ty.decode(value)  # type: ignore


@defop
def format_model_input[**P, T](
    template: Template[P, T], *args: P.args, **kwargs: P.kwargs
) -> list[Any]:
    """Format a template applied to arguments into a sequence of input
    messages.

    """
    bound_args = template.__signature__.bind(*args, **kwargs)
    bound_args.apply_defaults()
    # encode arguments
    arguments = {}
    for param in bound_args.arguments:
        encoder = type_to_encodable_type(
            template.__signature__.parameters[param].annotation
        )
        encoded = encoder.encode(bound_args.arguments[param])
        arguments[param] = encoder.serialize(encoded)

    prompt = _OpenAIPromptFormatter().format_as_messages(
        template.__prompt_template__, **arguments
    )

    # Note: The OpenAI api only seems to accept images in the 'user' role. The
    # effect of different roles on the model's response is currently unclear.
    messages = [{"type": "message", "content": prompt, "role": "user"}]
    return messages


class InstructionHandler(ObjectInterpretation):
    """Scoped handler that injects additional instructions into model input.

    This handler appends instruction messages to the formatted model input.
    It's designed to be used as a scoped handler within RetryLLMHandler to
    provide error feedback without polluting shared state.
    """

    def __init__(self, instruction: str):
        """Initialize with an instruction message to inject.

        Args:
            instruction: The instruction text to append to model input.
        """
        self.instruction = instruction

    @implements(format_model_input)
    def _inject_instruction(self, template: Template, *args, **kwargs) -> list[Any]:
        """Append instruction message to the formatted model input."""
        messages = fwd()
        return messages + [
            {"type": "message", "content": self.instruction, "role": "user"}
        ]


class RetryLLMHandler(ObjectInterpretation):
    """Retries LLM requests if they fail.

    If the request fails, the handler retries with optional error feedback injected
    into the prompt via scoped InstructionHandler instances. This ensures nested
    template calls maintain independent error tracking.

    Args:
        max_retries: The maximum number of retries.
        add_error_feedback: Whether to add error feedback to the prompt on retry.
        exception_cls: The exception class to catch and retry on.
    """

    def __init__(
        self,
        max_retries: int = 3,
        add_error_feedback: bool = False,
        exception_cls: type[BaseException] = Exception,
    ):
        self.max_retries = max_retries
        self.add_error_feedback = add_error_feedback
        self.exception_cls = exception_cls

    @implements(Template.__apply__)
    def _retry_completion(self, template: Template, *args, **kwargs) -> Any:
        """Retry template execution with error feedback injection via scoped handlers."""
        failures: list[str] = []

        for attempt in range(self.max_retries):
            try:
                # Install scoped handlers for each accumulated failure
                with contextlib.ExitStack() as stack:
                    for failure in failures:
                        stack.enter_context(handler(InstructionHandler(failure)))
                    return fwd()
            except self.exception_cls:
                if attempt == self.max_retries - 1:
                    raise  # Last attempt, re-raise the exception
                if self.add_error_feedback:
                    tb = traceback.format_exc()
                    failures.append(f"\nError from previous attempt:\n```\n{tb}```")

        # This should not be reached, but just in case
        return fwd()


class LiteLLMProvider(ObjectInterpretation):
    """Implements templates using the LiteLLM API."""

    model_name: str
    config: dict[str, Any]

    def __init__(self, model_name: str = "gpt-4o", **config):
        self.model_name = model_name
        self.config = inspect.signature(completion).bind_partial(**config).kwargs

    @implements(completion)
    def _completion(self, *args, **kwargs):
        return fwd(self.model_name, *args, **(self.config | kwargs))

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        model_input = format_model_input(template, *args, **kwargs)
        resp = compute_response(template, model_input)
        return decode_response(template, resp)
