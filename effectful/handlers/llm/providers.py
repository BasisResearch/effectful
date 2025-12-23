import base64
import dataclasses
import functools
import inspect
import io
import logging
import string
import traceback
import typing
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any, get_type_hints

import litellm
import pydantic

from effectful.handlers.llm.encoding import type_to_encodable_type

try:
    from PIL import Image
except ImportError:
    raise ImportError("'pillow' is required to use effectful.handlers.providers")

from litellm import (
    Choices,
    Message,
    OpenAIChatCompletionToolParam,
    OpenAIMessageContent,
    OpenAIMessageContentListBlock,
)
from litellm.types.utils import ModelResponse

from effectful.handlers.llm import Template
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import Operation


def _pil_image_to_base64_data(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_image_to_base64_data_uri(pil_image: Image.Image) -> str:
    return f"data:image/png;base64,{_pil_image_to_base64_data(pil_image)}"


@dataclasses.dataclass
class Tool[**P, T]:
    callable: Operation[P, T] | Template[P, T]
    name: str
    parameter_annotations: dict[str, type]
    description: str

    def serialise_return_value(self, value) -> OpenAIMessageContent:
        """Serializes a value returned by the function into a json format suitable for the OpenAI API."""
        sig = inspect.signature(self.callable)
        encoded_ty = type_to_encodable_type(sig.return_annotation)
        encoded_value = encoded_ty.encode(value)
        return encoded_ty.serialize(encoded_value)

    @functools.cached_property
    def parameter_model(self) -> type[pydantic.BaseModel]:
        fields = {
            param_name: type_to_encodable_type(param_type).t
            for param_name, param_type in self.parameter_annotations.items()
        }
        parameter_model = pydantic.create_model(
            "Params",
            __config__={"extra": "forbid"},
            **fields,  # type: ignore
        )
        return parameter_model

    def call_with_json_args(
        self, template: Template, json_str: str
    ) -> OpenAIMessageContent:
        """Implements a roundtrip call to a python function. Input is a json string representing an LLM tool call request parameters. The output is the serialised response to the model."""
        try:
            # build dict of raw encodable types U
            raw_args = self.parameter_model.model_validate_json(json_str)

            # use encoders to decode Us to python types T
            params: dict[str, Any] = {
                param_name: type_to_encodable_type(
                    self.parameter_annotations[param_name]
                ).decode(getattr(raw_args, param_name))
                for param_name in raw_args.model_fields_set
            }

            # call tool with python types
            result = tool_call(
                template,
                self.callable,
                **params,
            )
            # serialize back to U using encoder for return type
            sig = inspect.signature(self.callable)
            encoded_ty = type_to_encodable_type(sig.return_annotation)
            encoded_value = encoded_ty.encode(result)
            # serialise back to Json
            return encoded_ty.serialize(encoded_value)
        except Exception as exn:
            return str({"status": "failure", "exception": str(exn)})

    @classmethod
    def define(cls, obj: Operation[P, T] | Template[P, T]):
        """Create a Tool from an Operation or Template.

        Returns None if the object cannot be converted to a tool (e.g., missing type annotations).
        """
        sig = inspect.signature(obj)
        tool_name = obj.__name__

        description = (
            obj.__prompt_template__ if isinstance(obj, Template) else obj.__doc__ or ""
        )

        # Try to get type hints, fall back to signature annotations if that fails
        try:
            hints = get_type_hints(obj)
        except Exception:
            hints = {
                p.name: p.annotation
                for p in sig.parameters.values()
                if p.annotation is not inspect.Parameter.empty
            }

        parameter_annotations: dict[str, type] = {}
        for param_name, param in sig.parameters.items():
            # Skip parameters without type annotations
            if param.annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"Parameter '{param_name}' in '{obj.__name__}' "
                    "does not have a type annotation"
                )
            # get_type_hints might not include the parameter if annotation is invalid
            if param_name not in hints:
                raise TypeError(
                    f"Parameter '{param_name}' in '{obj.__name__}' "
                    "does not have a valid type annotation"
                )
            parameter_annotations[param_name] = hints[param_name]

        return cls(
            callable=obj,
            name=tool_name,
            parameter_annotations=parameter_annotations,
            description=description,
        )

    @property
    def function_definition(self) -> OpenAIChatCompletionToolParam:
        response_format = litellm.utils.type_to_response_format_param(
            self.parameter_model
        )
        assert response_format is not None
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": response_format["json_schema"][
                    "schema"
                ],  # extract the schema
                "strict": True,
            },
        }


def _tools_of_operations(
    ops: Iterable[Operation | Template],
) -> Mapping[str, Tool]:
    tools = {}
    for op in ops:
        tool = Tool.define(op)
        # NOTE: Because lexical handling is already guaranteeing unique names, we can just use the tool's name directly.
        tools[tool.name] = tool
    return tools


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


# Emitted for model request/response rounds so handlers can observe/log requests.
@defop
@functools.wraps(litellm.completion)
def completion(*args, **kwargs) -> Any:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd()."""
    return litellm.completion(*args, **kwargs)


# Note: attempting to type the tool arguments causes type-checker failures
@defop
def tool_call[T](
    template: Template, tool: Operation[..., T] | Template[..., T], *args, **kwargs
) -> T:
    """Perform a model-initiated tool call (can be an Operation or another Template)."""
    return tool(*args, **kwargs)


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
            extra={
                "payload": {
                    "args": args,
                    "kwargs": kwargs,
                    "response": response,
                }
            },
        )
        return response

    @implements(tool_call)
    def _log_tool_call(
        self, template: Template, tool: Operation, *args, **kwargs
    ) -> Any:
        """Log the tool call and result."""

        tool_name = tool.__name__
        result = fwd()
        self.logger.info(
            "llm.tool_call",
            extra={
                "payload": {
                    "tool": tool_name,
                    "args": args,
                    "kwargs": kwargs,
                }
            },
        )
        return result


class RetryLLMHandler(ObjectInterpretation):
    """Retries LLM requests if they fail.
    If the request fails, the error is logged and the prompt is updated to include the error.
    If the request fails after the maximum number of retries, an exception is raised.
    Args:
        max_retries: The maximum number of retries.
        add_error_feedback: Whether to add error feedback to the prompt.
        exception_cls: The exception class to raise if the maximum number of retries is reached.
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

    @implements(Template.__call__)
    def _retry_completion(self, template: Template, *args, **kwargs) -> Any:
        max_retries = self.max_retries
        current_template = template
        while max_retries > 0:
            try:
                return fwd(current_template, *args, **kwargs)
            except self.exception_cls as exn:
                max_retries -= 1
                if max_retries == 0:
                    raise exn
                if self.add_error_feedback:
                    # Capture the full traceback for better error context
                    tb = traceback.format_exc()
                    prompt_ext = (
                        f"Retry generating the following prompt: {template.__prompt_template__}\n\n"
                        f"Error from previous generation:\n```\n{tb}```"
                    )
                    current_template = dataclasses.replace(
                        template, __prompt_template__=prompt_ext
                    )
                # Continue the loop to retry
        raise Exception("Max retries reached")


def _pydantic_model_from_type(typ: type):
    return pydantic.create_model("Response", value=typ, __config__={"extra": "forbid"})


@defop
def compute_response(template: Template, model_input: list[Any]) -> ModelResponse:
    """Produce a complete model response for an input message sequence. This may
    involve multiple API requests if tools are invoked by the model.

    """
    ret_type = template.__signature__.return_annotation

    tools = _tools_of_operations(template.tools)
    tool_schemas = [t.function_definition for t in tools.values()]
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
            function_name = typing.cast(str, function.name)
            tool = tools[function_name]
            tool_result = tool.call_with_json_args(template, function.arguments)
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

    @implements(Template.__call__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        model_input = format_model_input(template, *args, **kwargs)
        resp = compute_response(template, model_input)
        return decode_response(template, resp)
