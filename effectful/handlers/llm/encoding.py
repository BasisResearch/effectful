import ast
import base64
import dataclasses
import functools
import inspect
import io
import json
import textwrap
import typing
from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
)
from types import CodeType
from typing import Any

import litellm
import pydantic
from litellm import (
    ChatCompletionImageObject,
    ChatCompletionMessageToolCall,
    ChatCompletionTextObject,
    ChatCompletionToolParam,
    OpenAIMessageContent,
    OpenAIMessageContentListBlock,
)
from PIL import Image

import effectful.handlers.llm.evaluation as evaluation
from effectful.handlers.llm.template import Tool
from effectful.internals.unification import GenericAlias, TypeEvaluator, nested_type
from effectful.ops.types import Operation, Term

type ToolCallID = str

CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    literal
    for member in typing.get_args(OpenAIMessageContentListBlock)
    for literal in typing.get_args(typing.get_type_hints(member).get("type", str))
    if isinstance(literal, str)
)


@pydantic.validate_call(validate_return=True)
def to_content_blocks(value: typing.Any) -> list[OpenAIMessageContentListBlock]:
    """Convert an encoded JSON-compatible value into a flat list of content blocks.

    Walks the value tree, extracting content-block-shaped dicts (identified by
    their ``type`` discriminator) and emitting JSON syntax as text around them.

    Top-level strings are emitted bare (for natural template rendering).
    Inside JSON structures, separators match ``json.dumps`` defaults so that
    the linearization law holds for non-string encoded values:
    ``linearize(to_content_blocks(v)) == json.dumps(v)``.
    """
    if isinstance(value, str):
        return [ChatCompletionTextObject(type="text", text=value)]

    buf: list[str] = []
    blocks: list[OpenAIMessageContentListBlock] = []

    def flush() -> None:
        if buf:
            blocks.append(ChatCompletionTextObject(type="text", text="".join(buf)))
            buf.clear()

    def walk(v: typing.Any) -> None:
        if isinstance(v, dict) and v.get("type") in CONTENT_BLOCK_TYPES:
            flush()
            blocks.append(typing.cast(OpenAIMessageContentListBlock, v))
        elif isinstance(v, dict):
            buf.append("{")
            for i, (k, val) in enumerate(v.items()):
                if i:
                    buf.append(", ")
                buf.append(json.dumps(k) + ": ")
                walk(val)
            buf.append("}")
        elif isinstance(v, list):
            buf.append("[")
            for i, item in enumerate(v):
                if i:
                    buf.append(", ")
                walk(item)
            buf.append("]")
        else:
            buf.append(json.dumps(v))

    walk(value)
    flush()
    return blocks


@dataclasses.dataclass(frozen=True, eq=True)
class DecodedToolCall[T]:
    """
    Structured representation of a tool call decoded from an LLM response.
    """

    tool: Tool[..., T]
    bound_args: inspect.BoundArguments
    id: ToolCallID
    name: str


@dataclasses.dataclass
class Encodable[T]:
    base: type[T]
    ctx: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @functools.cached_property
    def enc(self) -> pydantic.TypeAdapter[T]:
        return pydantic.TypeAdapter(TypeToPydanticType().evaluate(self.base))

    def encode(self, value: T) -> Mapping[str, Any]:
        return self.enc.dump_python(value, context=self.ctx, mode="json")

    def decode(self, encoded_value: Mapping[str, Any]) -> T:
        return self.enc.validate_python(encoded_value, context=self.ctx)

    def serialize(self, encoded_value: Mapping[str, Any]) -> OpenAIMessageContent:
        return to_content_blocks(encoded_value)

    @typing.final
    @classmethod
    def define(
        cls: type[typing.Self],
        t: type[T],
        ctx: Mapping[str, Any] | None = None,
    ) -> typing.Self:
        return cls(t, ctx or {})


class TypeToPydanticType(TypeEvaluator):
    """Substitute custom types with their Pydantic Annotated equivalents.

    Recursively walks a type annotation tree, replacing leaf types that have
    registered Pydantic annotations (e.g., Image.Image -> PydanticImage) and
    reconstructing the full generic type.

    The result can be passed to pydantic.TypeAdapter() for automatic
    validation and serialization of nested structures.
    """

    @staticmethod
    @functools.singledispatch
    def _registry(ty: type):
        raise RuntimeError("should not be here!")

    @classmethod
    def register(cls, *args, **kwargs):
        return cls._registry.register(*args, **kwargs)

    def evaluate(self, ty):
        app = super().evaluate(ty)
        if (
            isinstance(app, type | GenericAlias)
            and typing.get_origin(app) is not typing.Annotated
        ):
            return self._registry.dispatch(typing.get_origin(app) or app)(app)
        else:
            return app


@TypeToPydanticType.register(object)
@TypeToPydanticType.register(str)
@TypeToPydanticType.register(pydantic.BaseModel)
def _pydantic_type_base[T](ty: type[T]) -> type[T]:
    return ty


@TypeToPydanticType.register(Term)
def _pydantic_type_term(ty: type[Term]):
    raise TypeError("Terms cannot be converted to Pydantic types.")


@TypeToPydanticType.register(Operation)
def _pydantic_type_operation(ty: type[Operation]):
    raise TypeError("Operations cannot be converted to Pydantic types.")


@pydantic.validate_call(validate_return=False)
def _validate_image(value: ChatCompletionImageObject) -> Image.Image:
    value = pydantic.TypeAdapter(ChatCompletionImageObject).validate_python(value)
    image_url: litellm.ChatCompletionImageUrlObject | str = value["image_url"]
    url: str = image_url["url"] if isinstance(image_url, dict) else image_url
    prefix, data = url.split(",")
    if not prefix.startswith("data:image/"):
        raise ValueError(f"expected base64 encoded image as data uri, received {url}")
    return Image.open(fp=io.BytesIO(base64.b64decode(data)))


def _serialize_image(value: Image.Image) -> ChatCompletionImageObject:
    buf = io.BytesIO()
    value.save(buf, format="PNG")
    url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    return pydantic.TypeAdapter(ChatCompletionImageObject).validate_python(
        {"type": "image_url", "image_url": {"detail": "auto", "url": url}}
    )


@TypeToPydanticType.register(Image.Image)
def _pydantic_type_image(ty: type[Image.Image]):
    adapter = pydantic.TypeAdapter(ChatCompletionImageObject)
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate_image),
        pydantic.PlainSerializer(_serialize_image),
        pydantic.WithJsonSchema(adapter.json_schema()),
    ]


def _format_callable_type(callable_type: type[Callable]) -> str:
    """Format a Callable type annotation as a string for LLM instructions."""
    args = typing.get_args(callable_type)
    if not args:
        return "Callable"

    # Callable[[arg1, arg2, ...], return_type]
    if len(args) >= 2:
        param_types = args[0]
        return_type = args[-1]

        if param_types is ...:
            params_str = "..."
        elif isinstance(param_types, list | tuple):
            params_str = ", ".join(getattr(t, "__name__", str(t)) for t in param_types)
        else:
            params_str = str(param_types)

        return_str = getattr(return_type, "__name__", str(return_type))
        return f"Callable[[{params_str}], {return_str}]"

    return str(callable_type)


class SynthesizedFunction(pydantic.BaseModel):
    """Structured output for function synthesis.

    Pydantic model representing synthesized code with function name and module code.
    """

    module_code: str = pydantic.Field(
        ...,
        description="Complete Python module code (no imports needed)",
    )


def _create_typed_synthesized_function(
    callable_type: type[Callable],
) -> type[SynthesizedFunction]:
    """Create a SynthesizedFunction subclass with type signature in the model description.

    Uses pydantic.create_model to ensure the description is included in the JSON schema
    sent to the LLM, informing it of the expected function signature.
    """
    type_signature = _format_callable_type(callable_type)

    description = f"""Given the specification above, generate a Python function satisfying the following specification and type signature.

<signature>{type_signature}</signature>

<instructions>
1. Produce one block of Python code.
2. The function MUST have type annotations for all parameters and the return type.
3. The function definition must be the LAST statement - do not add any code after it.
4. Do not include usage examples or function calls.
</instructions>
"""

    # Use pydantic.create_model to create a proper model with the description
    # The __doc__ becomes the model's description in the JSON schema
    model = pydantic.create_model(
        "TypedSynthesizedFunction",
        __base__=SynthesizedFunction,
        __doc__=description,
    )
    return model


def _validate_signature_ast(
    func_ast: ast.FunctionDef | ast.AsyncFunctionDef,
    expected_params: list[type] | None,
) -> None:
    """Validate the function signature from AST before execution."""
    if expected_params is not None:
        ast_params = func_ast.args.args + func_ast.args.posonlyargs
        if len(ast_params) != len(expected_params):
            raise ValueError(
                f"decode() expected function with {len(expected_params)} parameters, "
                f"got {len(ast_params)}"
            )


def _validate_signature_callable(
    func: Callable,
    expected_params: list[type] | None,
    expected_return: type,
) -> None:
    """Validate the function signature from runtime callable after execution.

    The synthesized function must have type annotations for parameters and return type.
    """
    sig = inspect.signature(func)

    if expected_params is not None:
        actual_params = list(sig.parameters.values())
        if len(actual_params) != len(expected_params):
            raise ValueError(
                f"decode() expected function with {len(expected_params)} parameters, "
                f"got {len(actual_params)}"
            )

    actual_return = sig.return_annotation
    if actual_return is inspect.Parameter.empty:
        raise ValueError(
            "decode() requires synthesized function to have a return type annotation"
        )


@TypeToPydanticType.register(Callable)
def _pydantic_callable(callable_type: Any) -> Any:
    """Create a Pydantic-compatible Annotated type for a parameterized Callable.

    Usage: PydanticCallable(Callable[[int, str], bool])
    """
    type_args = typing.get_args(callable_type)

    if not type_args:
        typed_enc = _create_typed_synthesized_function(Callable[..., typing.Any])  # type: ignore[arg-type]
        expected_params = None
        expected_return = None
    else:
        if len(type_args) < 2:
            raise TypeError(
                f"Callable type signature incomplete: {callable_type}. "
                "Expected Callable[[ParamTypes...], ReturnType] or Callable[..., ReturnType]."
            )
        param_types, expected_return = type_args[0], type_args[-1]
        typed_enc = _create_typed_synthesized_function(callable_type)
        if param_types is not ... and isinstance(param_types, list | tuple):
            expected_params = list(param_types)
        else:
            expected_params = None

    def _validate(value: Any, info: pydantic.ValidationInfo) -> Callable:
        if callable(value) and not isinstance(value, dict):
            return value
        if isinstance(value, SynthesizedFunction):
            encoded = value
        elif isinstance(value, dict):
            encoded = typed_enc.model_validate(value)
        elif isinstance(value, str):
            encoded = typed_enc.model_validate_json(value)
        else:
            raise ValueError(
                f"Expected callable, SynthesizedFunction dict, or JSON string, "
                f"got {type(value)}"
            )

        if expected_return is None:
            raise TypeError(
                "Cannot decode/synthesize callable without a concrete type signature. "
                "Use Callable[[ParamTypes...], ReturnType] or Callable[..., ReturnType] "
                "with a concrete return type (not Any)."
            )

        ctx = info.context or {}
        filename = f"<synthesis:{id(encoded)}>"
        module: ast.AST = evaluation.parse(encoded.module_code, filename)

        if not isinstance(module, ast.Module) or not module.body:
            raise ValueError(
                "decode() requires module code with at least one statement."
            )

        last_stmt = module.body[-1]
        if not isinstance(last_stmt, ast.FunctionDef):
            raise ValueError(
                f"decode() requires the last statement to be a function definition, "
                f"got {type(last_stmt).__name__}"
            )

        _validate_signature_ast(last_stmt, expected_params)
        evaluation.type_check(module, ctx, expected_params, expected_return)

        g: MutableMapping[str, Any] = {}
        g.update(ctx)
        bytecode: CodeType = evaluation.compile(module, filename)
        evaluation.exec(bytecode, g)

        func_name = last_stmt.name
        if func_name not in g:
            raise ValueError(
                f"decode() expected function '{func_name}' to be defined in globals"
            )

        result = g[func_name]
        if not callable(result):
            raise ValueError(
                f"decode() expected '{func_name}' to be callable, got {type(result)}"
            )

        _validate_signature_callable(result, expected_params, expected_return)
        return result

    def _serialize(value: Callable) -> dict:
        if not callable(value):
            raise TypeError(f"Expected callable, got {type(value)}")

        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            source = None

        if source:
            return typed_enc(module_code=textwrap.dedent(source)).model_dump()

        name = getattr(value, "__name__", None)
        docstring = inspect.getdoc(value)
        if name is None or docstring is None:
            raise ValueError(
                f"Cannot encode callable {value}: no source code and no __name__ or docstring"
            )

        try:
            sig = inspect.signature(value)
            sig_str = str(sig)
        except (ValueError, TypeError):
            sig_str = "(...)"

        stub_code = f'''def {name}{sig_str}:
    """{docstring}"""
    ...
'''
        return typed_enc(module_code=stub_code).model_dump()

    return typing.Annotated[
        callable_type,
        pydantic.PlainValidator(_validate),
        pydantic.PlainSerializer(_serialize),
        pydantic.WithJsonSchema(pydantic.TypeAdapter(typed_enc).json_schema()),
    ]


def _validate_tool(
    value: ChatCompletionToolParam, info: pydantic.ValidationInfo
) -> Tool:
    assert isinstance(info.context, Mapping), "Tool decoding requires context"
    value = pydantic.TypeAdapter(ChatCompletionToolParam).validate_python(value)
    try:
        return info.context[value["function"]["name"]]
    except KeyError as e:
        raise NotImplementedError(f"Unknown tool: {value['function']['name']}") from e


def _serialize_tool(value: Tool) -> ChatCompletionToolParam:
    sig_model = pydantic.create_model(
        "Params",
        __config__={"extra": "forbid"},
        **{
            name: TypeToPydanticType().evaluate(param.annotation)
            for name, param in inspect.signature(value).parameters.items()
        },  # type: ignore
    )
    response_format = litellm.utils.type_to_response_format_param(sig_model)
    assert response_format is not None
    assert value.__default__.__doc__ is not None
    return pydantic.TypeAdapter(ChatCompletionToolParam).validate_python(
        {
            "type": "function",
            "function": {
                "name": value.__name__,
                "description": textwrap.dedent(value.__default__.__doc__),
                "parameters": response_format["json_schema"]["schema"],
                "strict": True,
            },
        }
    )


@TypeToPydanticType.register(Tool)
def _pydantic_type_tool(ty: type[Tool]):
    adapter = pydantic.TypeAdapter(ChatCompletionToolParam)
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate_tool),
        pydantic.PlainSerializer(_serialize_tool),
        pydantic.WithJsonSchema(adapter.json_schema()),
    ]


def _validate_tool_call(
    value: ChatCompletionMessageToolCall,
    info: pydantic.ValidationInfo,
) -> DecodedToolCall:
    if isinstance(value, dict):
        value = ChatCompletionMessageToolCall.model_validate(value)
    ctx = info.context or {}
    assert value.function.name is not None
    tool = ctx[value.function.name]
    assert isinstance(tool, Tool)
    sig = inspect.signature(tool)
    decoded_args = {}
    for name, raw_arg in json.loads(value.function.arguments).items():
        assert name in sig.parameters, (
            f"Unexpected argument {name} for tool {tool.__name__}"
        )
        param = sig.parameters[name]
        arg_enc = Encodable.define(param.annotation, ctx)
        decoded_args[name] = arg_enc.decode(raw_arg)
    return DecodedToolCall(
        tool=tool,
        bound_args=sig.bind(**decoded_args),
        id=value.id,
        name=value.function.name,
    )


def _serialize_tool_call(
    value: DecodedToolCall, info: pydantic.SerializationInfo
) -> dict:
    ctx = info.context or {}
    encoded_args = {}
    for k, v in value.bound_args.arguments.items():
        v_enc = Encodable.define(nested_type(v).value, ctx)
        encoded_args[k] = v_enc.encode(v)
    return ChatCompletionMessageToolCall.model_validate(
        {
            "type": "tool_call",
            "id": value.id,
            "function": {
                "name": value.tool.__name__,
                "arguments": encoded_args,
            },
        }
    ).model_dump(mode="json")


@TypeToPydanticType.register(DecodedToolCall)
def _pydantic_type_tool_call(ty: type[DecodedToolCall]):
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate_tool_call),
        pydantic.PlainSerializer(_serialize_tool_call),
        pydantic.WithJsonSchema(ChatCompletionMessageToolCall.model_json_schema()),
    ]
