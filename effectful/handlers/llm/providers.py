import base64
import dataclasses
import inspect
import io
import string
from collections.abc import Callable, Iterable, Mapping
from typing import Annotated, Any, Literal, get_type_hints

import pydantic
from openai import BaseModel

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
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import Operation


def _pil_image_to_base64_data(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_image_to_base64_data_uri(pil_image: Image.Image) -> str:
    return f"data:image/png;base64,{_pil_image_to_base64_data(pil_image)}"


# this type is present in OpenAI's python bindings under ResponseInputImageParam but not exported.
class ImageFunctionOutput(BaseModel):
    detail: Literal["auto"] = "auto"
    """The detail level of the image to be sent to the model."""

    type: Literal["input_image"] = "input_image"
    """The type of the input item. Always `input_image`."""

    image_url: str
    """A base64 encoded image in a data URL."""

    @classmethod
    def of_image(cls, image: Image.Image):
        return cls(image_url=_pil_image_to_base64_data_uri(image))


def build_model(ty: type) -> type[pydantic.BaseModel]:
    """Constructs a pydantic model capable of serialising model outputs"""

    # special handling of PIL image types/sequences of image types
    def with_converter[T](ty: type[T], converter: Callable[..., T]):
        return Annotated[
            ty, pydantic.WrapValidator(lambda value, handler: handler(converter(value)))
        ]

    result_ty = with_converter(
        str, lambda vl: str({"status": "success", "result": str(vl)})
    )
    if ty == Image.Image:
        result_ty = with_converter(
            list[ImageFunctionOutput], lambda vl: [ImageFunctionOutput.of_image(vl)]
        )
    elif ty == list[Image.Image]:
        result_ty = with_converter(
            list[ImageFunctionOutput],
            lambda vls: list(map(ImageFunctionOutput.of_image, vls)),
        )

    result_model = pydantic.create_model(
        "Result", __config__={"extra": "forbid"}, result=result_ty
    )
    return result_model


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
        fields = {
            param_name: hints.get(param_name, str) for param_name in sig.parameters
        }

        parameter_model = pydantic.create_model(
            "Params", __config__={"extra": "forbid"}, **fields
        )

        result_model = build_model(sig.return_annotation)

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


# Note: attempting to type the tool arguments causes type-checker failures
@defop
def tool_call[T](template: Template, tool: Operation[..., T], *args, **kwargs) -> T:
    """Perform a model-initiated tool call."""
    return tool(*args, **kwargs)


def _call_tool_with_json_args(
    template: Template, tool: Tool, json_str_args: str
) -> pydantic.JsonValue:
    try:
        args = tool.parameter_model.model_validate_json(json_str_args)
        result = tool_call(
            template, tool.operation, **args.model_dump(exclude_defaults=True)
        )
        result = tool.result_model(result=result)
        result_json = result.model_dump(mode="json")
        return result_json["result"]
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
            response = self._client.responses.create(
                input=model_input, **response_kwargs
            )

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
