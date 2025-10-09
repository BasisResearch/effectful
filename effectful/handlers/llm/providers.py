import base64
import dataclasses
import inspect
import io
import json
import string
from typing import Any, Callable, Iterable, Mapping, Sequence, get_type_hints

import pydantic
from pydantic import TypeAdapter, create_model

try:
    import openai
except ImportError:
    raise ImportError("'openai' is required to use effectful.handlers.providers")

try:
    from PIL import Image
except ImportError:
    raise ImportError("'pillow' is required to use effectful.handlers.providers")

from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.responses import FunctionToolParam
from openai.types.shared.function_definition import FunctionDefinition
from openai.types.shared.function_parameters import FunctionParameters

from effectful.handlers.llm import Template, decode
from effectful.internals.runtime import get_interpretation
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
    parameter_model: type[pydantic.BaseModel]
    result_model: type[pydantic.BaseModel]
    operation: Operation[P, T]
    name: str

    @classmethod
    def of_operation(cls, op: Operation[P, T], name: str):
        sig = inspect.signature(op)
        hints = get_type_hints(op)

        # Build field definitions with defaults
        fields = {}
        for param_name, param in sig.parameters.items():
            field_type = hints.get(param_name, str)
            if param.default == inspect.Parameter.empty:
                field_desc = field_type
            else:
                field_desc = (field_type, param.default)
            fields[param_name] = field_desc

        parameter_model = create_model("Params", **fields)
        result_model = create_model("Result", result=sig.return_annotation)

        return cls(
            parameter_model=parameter_model,
            result_model=result_model,
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
            "strict": False,
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


@defop
def tool_call[**P, T](
    template: Template, tool: Operation[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Perform a model-initiated tool call."""
    return tool(*args, **kwargs)


def _call_tool_with_json_args(
    template: Template, tool: Tool, call_id: str, json_str_args: str
) -> dict:
    def _function_call_output(output):
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": str(output),
        }

    try:
        args = tool.parameter_model.model_validate_json(json_str_args)
        result: Any = tool_call(
            template, tool, **args.model_dump(exclude_defaults=True)
        )
        return _function_call_output({"status": "success", "result": str(result)})
    except Exception as exn:
        return _function_call_output({"status": "failure", "exception": str(exn)})


class OpenAIAPIProvider(ObjectInterpretation):
    """Implements templates using the OpenAI API."""

    def __init__(self, client: openai.OpenAI, model_name: str = "gpt-4o"):
        self._client = client
        self._model_name = model_name

    @implements(Template.__call__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        prompt = _OpenAIPromptFormatter().format_as_messages(
            template.__prompt_template__, **bound_args.arguments
        )

        tools, tool_names = _tools_of_operations(get_interpretation())

        # Note: The OpenAI api only seems to accept images in the 'user' role.
        # The effect of different roles on the model's response is currently
        # unclear.

        called_tools = set([])  # tool calls that we have discharged
        model_input = [{"content": prompt, "role": "user"}]

        while True:
            response = self._client.responses.create(
                model=self._model_name,
                input=model_input,
                tools=tools,
                tool_choice="auto",
            )

            for message in response.output:
                if message.type != "function_call":
                    continue

                call_id = message.call_id
                if call_id in called_tools:
                    continue
                called_tools.add(call_id)

                name = message.name
                tool_result = None
                try:
                    args = json.loads(message.arguments)
                except json.JSONDecodeError as exn:
                    tool_result = exn

                if tool_result is None:
                    tool_names[name]()

        first_response = response.output[0]
        assert first_response.type == "message"
        first_response_content = first_response.content[0]
        assert first_response_content.type == "output_text"

        ret_type = template.__signature__.return_annotation
        return decode(ret_type, first_response_content.text)
