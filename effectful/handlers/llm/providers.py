import base64
import dataclasses
import inspect
import io
import logging
import string
import typing
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, get_type_hints

import pydantic

try:
    import openai
except ImportError:
    raise ImportError("'openai' is required to use effectful.handlers.providers")

try:
    from PIL import Image
except ImportError:
    raise ImportError("'pillow' is required to use effectful.handlers.providers")

from openai.types.responses import FunctionToolParam

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


def _pil_image_to_openai_image_param(
    pil_image: Image.Image,
) -> openai.types.responses.ResponseInputImageParam:
    return openai.types.responses.ResponseInputImageParam(
        type="input_image",
        detail="auto",
        image_url=_pil_image_to_base64_data_uri(pil_image),
    )


OpenAIFunctionOutputParamType = (
    str | list[openai.types.responses.ResponseInputImageParam]
)


@dataclasses.dataclass
class Tool[**P, T]:
    parameter_model: type[pydantic.BaseModel]
    operation: Operation[P, T]
    name: str

    def serialise_return_value(self, value) -> OpenAIFunctionOutputParamType:
        """Serializes a value returned by the function into a json format suitable for the OpenAI API."""
        sig = inspect.signature(self.operation)
        ret_ty = sig.return_annotation
        ret_ty_origin = typing.get_origin(ret_ty) or ret_ty
        ret_ty_args = typing.get_args(ret_ty)

        # special casing for images
        if ret_ty == Image.Image:
            return [_pil_image_to_openai_image_param(value)]

        # special casing for sequences of images (tuple[Image.Image, Image.Image], etc.)
        if issubclass(ret_ty_origin, Sequence) and all(
            arg == Image.Image for arg in ret_ty_args
        ):
            return [_pil_image_to_openai_image_param(image) for image in value]

        # otherwise stringify
        return str({"status": "success", "result": str(value)})

    @classmethod
    def of_operation(cls, op: Operation[P, T], name: str):
        sig = inspect.signature(op)
        hints = get_type_hints(op)
        fields = {
            param_name: hints.get(param_name, str) for param_name in sig.parameters
        }

        parameter_model = pydantic.create_model(
            "Params", __config__={"extra": "forbid"}, **fields
        )

        return cls(
            parameter_model=parameter_model,
            operation=op,
            name=name,
        )

    @property
    def function_definition(self) -> FunctionToolParam:
        return {
            "type": "function",
            "name": self.name,
            "description": self.operation.__doc__,
            "parameters": self.parameter_model.model_json_schema(),
            "strict": True,
        }


def _tools_of_operations(ops: Iterable[Operation]) -> Mapping[str, Tool]:
    tools = {}
    for op in ops:
        name = op.__name__

        # Ensure tool names are unique. Operation names may not be.
        if name in tools:
            suffix = 0
            while f"{name}_{suffix}" in tools:
                suffix += 1
            name = f"{name}_{suffix}"

        tools[name] = Tool.of_operation(op, name)
    return tools


class _OpenAIPromptFormatter(string.Formatter):
    def format_as_messages(
        self, format_str: str, /, *args, **kwargs
    ) -> openai.types.responses.ResponseInputMessageContentListParam:
        prompt_parts = []
        current_text = ""

        def push_current_text():
            nonlocal current_text
            if current_text:
                prompt_parts.append({"type": "input_text", "text": current_text})
            current_text = ""

        for literal, field_name, format_spec, conversion in self.parse(format_str):
            current_text += literal

            if field_name is not None:
                obj, _ = self.get_field(field_name, args, kwargs)
                obj = self.convert_field(obj, conversion)

                if isinstance(obj, Image.Image):
                    assert not format_spec, (
                        "image template parameters cannot have format specifiers"
                    )
                    push_current_text()
                    prompt_parts.append(
                        {
                            "type": "input_image",
                            "image_url": _pil_image_to_base64_data_uri(obj),
                        }
                    )
                else:
                    current_text += self.format_field(
                        obj, format_spec if format_spec else ""
                    )

        push_current_text()
        return prompt_parts


# Emitted for model request/response rounds so handlers can observe/log requests.
@defop
def llm_request(client: openai.OpenAI, *args, **kwargs) -> Any:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd()."""
    return client.responses.create(*args, **kwargs)


# Note: attempting to type the tool arguments causes type-checker failures
@defop
def tool_call[T](template: Template, tool: Operation[..., T], *args, **kwargs) -> T:
    """Perform a model-initiated tool call."""
    return tool(*args, **kwargs)


class CacheLLMRequestHandler(ObjectInterpretation):
    """Caches LLM requests."""

    def __init__(self, silent_failure: bool = True):
        """Initialize the cache handler.

        Args:
            silent_failure: If True, will not raise an error if args and kwargs are not hashable, proceed as normal LLM calls.
        """
        self.cache: dict[tuple[Hashable, Hashable], Any] = {}
        self.silent_failure = silent_failure

    def _make_hashable(self, obj: Any) -> Hashable:
        """Recursively convert objects to hashable representations."""
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list | tuple):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, set):
            return frozenset(self._make_hashable(item) for item in obj)
        elif isinstance(obj, int | float | bool | str | bytes):
            return obj
        else:
            if self.silent_failure:
                return obj
            raise ValueError(f"Object {obj} is not hashable")

    @implements(llm_request)
    def _cache_llm_request(self, client: openai.OpenAI, *args, **kwargs) -> Any:
        hashable_args = self._make_hashable(args)
        hashable_kwargs = self._make_hashable(kwargs)
        key = (hashable_args, hashable_kwargs)
        if key in self.cache:
            return self.cache[key]
        response = fwd()
        self.cache[key] = response
        return response


class LLMLoggingHandler(ObjectInterpretation):
    """Logs llm_request rounds and tool_call invocations using Python logging.

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

    @implements(llm_request)
    def _log_llm_request(self, client: openai.OpenAI, *args, **kwargs) -> Any:
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


def _call_tool_with_json_args(
    template: Template, tool: Tool, json_str_args: str
) -> OpenAIFunctionOutputParamType:
    try:
        args = tool.parameter_model.model_validate_json(json_str_args)
        result = tool_call(
            template, tool.operation, **args.model_dump(exclude_defaults=True)
        )
        return tool.serialise_return_value(result)
    except Exception as exn:
        return str({"status": "failure", "exception": str(exn)})


class OpenAIAPIProvider(ObjectInterpretation):
    """Implements templates using the OpenAI API."""

    def __init__(self, client: openai.OpenAI, model_name: str = "gpt-4o"):
        self._client = client
        self._model_name = model_name

    @implements(Template.__call__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        ret_type = template.__signature__.return_annotation
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        prompt = _OpenAIPromptFormatter().format_as_messages(
            template.__prompt_template__, **bound_args.arguments
        )

        tools = _tools_of_operations(template.tools)
        tool_definitions = [t.function_definition for t in tools.values()]

        response_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "tools": tool_definitions,
            "tool_choice": "auto",
        }

        if ret_type == str:
            result_schema = None
        else:
            Result = pydantic.create_model(
                "Response", value=ret_type, __config__={"extra": "forbid"}
            )
            result_schema = openai.lib._pydantic.to_strict_json_schema(Result)
            response_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": result_schema,
                    "strict": True,
                }
            }

        called_tools = set([])  # tool calls that we have discharged

        # Note: The OpenAI api only seems to accept images in the 'user' role.
        # The effect of different roles on the model's response is currently
        # unclear.
        model_input: list[Any] = [
            {"type": "message", "content": prompt, "role": "user"}
        ]

        while True:
            response = llm_request(self._client, input=model_input, **response_kwargs)

            new_input = []
            for message in response.output:
                if message.type != "function_call":
                    continue

                call_id = message.call_id
                if call_id in called_tools:
                    continue
                called_tools.add(call_id)

                tool = tools[message.name]
                tool_result = _call_tool_with_json_args(
                    template, tool, message.arguments
                )
                tool_response = {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_result,
                }
                new_input.append(tool_response)

            if not new_input:
                break

            model_input += response.output + new_input

        last_resp = response.output[-1]
        assert last_resp.type == "message"
        last_resp_content = last_resp.content[0]
        assert last_resp_content.type == "output_text"
        result_str = last_resp_content.text

        if result_schema is None:
            return result_str

        result = Result.model_validate_json(result_str)
        assert isinstance(result, Result)
        return result.value  # type: ignore[attr-defined]
