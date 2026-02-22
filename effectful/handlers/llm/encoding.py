import ast
import base64
import functools
import inspect
import io
import json
import textwrap
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from dataclasses import dataclass
from types import CodeType
from typing import Any

import litellm
import pydantic
import typing_extensions
from litellm import (
    ChatCompletionImageUrlObject,
    ChatCompletionMessageToolCall,
    ChatCompletionTextObject,
    ChatCompletionToolParam,
    OpenAIMessageContentListBlock,
)
from PIL import Image

import effectful.handlers.llm.evaluation as evaluation
from effectful.handlers.llm.template import Tool
from effectful.internals.unification import nested_type
from effectful.ops.semantics import _simple_type
from effectful.ops.syntax import _CustomSingleDispatchCallable
from effectful.ops.types import Operation, Term

type ToolCallID = str


def _pil_image_to_base64_data(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_image_to_base64_data_uri(pil_image: Image.Image) -> str:
    return f"data:image/png;base64,{_pil_image_to_base64_data(pil_image)}"


class _BoxEncoding[T](pydantic.BaseModel):
    value: T


@functools.cache
def _boxed_model[T](ty: type[T]) -> type[_BoxEncoding[T]]:
    return pydantic.create_model(
        f"Response_{getattr(ty, '__name__', 'scalar')}",
        value=(ty, ...),
        __base__=_BoxEncoding,
        __config__={"extra": "forbid"},
    )


@functools.cache
def _typeddict_model(td: type[dict[str, Any]]) -> type[pydantic.BaseModel]:
    assert typing_extensions.is_typeddict(td), f"Expected a TypedDict type, got {td}"
    hints = typing.get_type_hints(td)
    required: frozenset[str] = getattr(td, "__required_keys__", frozenset())
    fields: dict[str, Any] = {}
    for k, v in hints.items():
        fields[k] = (v, ...) if k in required else (v, None)
    return pydantic.create_model(
        td.__name__,
        **fields,
    )


def _param_model(sig: inspect.Signature) -> type[pydantic.BaseModel]:
    return pydantic.create_model(
        "Params",
        __config__={"extra": "forbid"},
        **{
            name: param.annotation
            for name, param in sig.parameters.items()
        },  # type: ignore
    )


@dataclass(frozen=True, eq=True)
class DecodedToolCall[T]:
    """
    Structured representation of a tool call decoded from an LLM response.
    """

    tool: Tool[..., T]
    bound_args: inspect.BoundArguments
    id: ToolCallID
    name: str


class _RawResponseFormatSchema(typing.TypedDict):
    type: typing.Literal["json_schema"]
    schema: dict[str, Any]
    strict: bool


class Encodable[T, U](ABC):
    base: type[T]
    enc: type[pydantic.BaseModel] | pydantic.TypeAdapter | type[str]
    ctx: Mapping[str, Any]

    @typing.final
    @property
    def response_format(self) -> type[pydantic.BaseModel] | _RawResponseFormatSchema | None:
        """Return a response format expected by litellm"""
        if self.enc is str:
            return None  # No schema for plain strings, they are serialized as-is without JSON encoding
        elif isinstance(self.enc, pydantic.TypeAdapter):
            return _RawResponseFormatSchema(
                type="json_schema",
                schema=self.enc.json_schema(),
                strict=True,
            )
        elif issubclass(self.enc, pydantic.BaseModel):
            return self.enc
        else:
            raise TypeError(f"Unsupported enc type {self.enc} for response format schema")

    @abstractmethod
    def encode(self, value: T) -> U:
        raise NotImplementedError

    @abstractmethod
    def decode(self, encoded_value: U) -> T:
        raise NotImplementedError

    @abstractmethod
    def serialize(self, encoded_value: U) -> Sequence[OpenAIMessageContentListBlock]:
        raise NotImplementedError

    # serialize and deserialize have different types reflecting the LLM api chat.completions(list[content]) -> str
    @abstractmethod
    def deserialize(self, serialized_value: str) -> U:
        raise NotImplementedError

    @typing.final
    @staticmethod
    @_CustomSingleDispatchCallable
    def define(
        __dispatch: Callable[
            [type[T]], Callable[[type[T], Mapping[str, Any] | None], "Encodable[T, U]"]
        ],
        t: type[T],
        ctx: Mapping[str, Any] | None = None,
    ) -> "Encodable[T, U]":
        dispatch_ty = _simple_type(t)
        encodable: Encodable[T, U] = __dispatch(dispatch_ty)(t, ctx)
        return encodable


class AdapterEncodable[T](Encodable[T, Any]):
    enc: pydantic.TypeAdapter[T]

    def __init__(self, base: type[T], ctx: Mapping[str, Any]):
        self.base = base
        self.ctx = ctx
        self.enc = pydantic.TypeAdapter(base)

    def encode(self, value: T):
        return self.enc.dump_python(value)

    def decode(self, encoded_value: dict[str, Any]) -> T:
        return self.enc.validate_python(encoded_value)

    def serialize(
        self, encoded_value: dict[str, Any]
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": self.enc.dump_json(encoded_value).decode("utf-8")}]

    def deserialize(self, serialized_value: str) -> dict[str, Any]:
        return self.enc.dump_python(self.enc.validate_json(serialized_value))


class StrEncodable(Encodable[str, str]):

    def __init__(self, ctx: Mapping[str, Any]):
        self.ctx = ctx
        self.base = str
        self.enc = str

    def encode(self, value: str) -> str:
        return value

    def decode(self, encoded_value: str) -> str:
        return encoded_value

    def serialize(self, encoded_value: str) -> Sequence[ChatCompletionTextObject]:
        # Serialize strings without JSON encoding (no extra quotes)
        return [{"type": "text", "text": encoded_value}]

    def deserialize(self, serialized_value: str) -> str:
        return serialized_value


class PydanticBaseModelEncodable[T: pydantic.BaseModel](Encodable[T, T]):

    def __init__(self, base: type[T], ctx: Mapping[str, Any]):
        self.base = base
        self.enc = base
        self.ctx = ctx

    def decode(self, encoded_value: T) -> T:
        return self.base.model_validate(encoded_value)

    def encode(self, value: T) -> T:
        return self.base.model_validate(value).model_dump(mode="json")

    def serialize(self, encoded_value: T) -> Sequence[ChatCompletionTextObject]:
        return [{"type": "text", "text": self.decode(encoded_value).model_dump_json()}]

    def deserialize(self, serialized_value: str) -> T:
        return self.encode(self.base.model_validate_json(serialized_value))


def _validate_image(value: Image.Image | ChatCompletionImageUrlObject) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    value = pydantic.TypeAdapter(ChatCompletionImageUrlObject).validate_python(value)
    url = value.get("url")
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise ValueError(
            f"expected base64 encoded image as data uri, received {url}"
        )
    data = url.split(",")[1]
    return Image.open(fp=io.BytesIO(base64.b64decode(data)))


def _serialize_image(value: Image.Image) -> ChatCompletionImageUrlObject:
    adapter = pydantic.TypeAdapter(ChatCompletionImageUrlObject)
    return adapter.validate_python(
        {
            "detail": "auto",
            "url": _pil_image_to_base64_data_uri(value),
        }
    )


PydanticImage = typing.Annotated[
    Image.Image,
    pydantic.PlainValidator(_validate_image),
    pydantic.PlainSerializer(_serialize_image),
    pydantic.WithJsonSchema(pydantic.TypeAdapter(ChatCompletionImageUrlObject).json_schema()),
]


class ImageEncodable(Encodable[Image.Image, Any]):
    enc: pydantic.TypeAdapter[ChatCompletionImageUrlObject]

    def __init__(self, base: type[Image.Image], ctx: Mapping[str, Any]):
        self.base = base
        self.ctx = ctx
        self.enc = pydantic.TypeAdapter(ChatCompletionImageUrlObject)

    def encode(self, value: Image.Image) -> ChatCompletionImageUrlObject:
        return self.enc.validate_python(
            {
                "detail": "auto",
                "url": _pil_image_to_base64_data_uri(value),
            }
        )

    def decode(self, encoded_value: ChatCompletionImageUrlObject) -> Image.Image:
        image_url = self.enc.validate_python(encoded_value).get("url")
        assert isinstance(image_url, str)
        if not image_url.startswith("data:image/"):
            raise TypeError(
                f"expected base64 encoded image as data uri, received {image_url}"
            )
        data = image_url.split(",")[1]
        return Image.open(fp=io.BytesIO(base64.b64decode(data)))

    def serialize(
        self, encoded_value: ChatCompletionImageUrlObject
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [
            {
                "type": "image_url",
                "image_url": self.enc.dump_json(encoded_value).decode("utf-8"),
            }
        ]

    def deserialize(self, serialized_value: str) -> ChatCompletionImageUrlObject:
        # Images are serialized as image_url blocks, not text
        # This shouldn't be called in normal flow, but provide a fallback
        raise NotImplementedError("Image deserialization from string is not supported")


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


def PydanticCallable(callable_type: Any) -> Any:
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
        Callable,
        pydantic.PlainValidator(_validate),
        pydantic.PlainSerializer(_serialize),
        pydantic.WithJsonSchema(pydantic.TypeAdapter(typed_enc).json_schema()),
    ]


@dataclass
class CallableEncodable(Encodable[Callable, SynthesizedFunction]):
    base: type[Callable]
    enc: type[SynthesizedFunction]
    ctx: Mapping[str, Any]
    expected_params: list[type] | None = None
    expected_return: type | None = None  # None means decode is disabled

    def encode(self, value: Callable) -> SynthesizedFunction:
        # (https://github.com/python/mypy/issues/14928)
        if not isinstance(value, Callable):  # type: ignore
            raise TypeError(f"Expected callable, got {type(value)}")

        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            source = None

        if source:
            return self.enc(module_code=textwrap.dedent(source))

        # Source not available - create stub from name, signature, and docstring
        # This is useful for builtins and C extensions
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
            # Some builtins don't have inspectable signatures
            sig_str = "(...)"

        # Format as a stub function with docstring
        stub_code = f'''def {name}{sig_str}:
    """{docstring}"""
    ...
'''
        return self.enc(module_code=stub_code)

    def decode(self, encoded_value: SynthesizedFunction) -> Callable:
        # Decode requires a concrete return type for synthesis
        if self.expected_return is None:
            raise TypeError(
                "Cannot decode/synthesize callable without a concrete type signature. "
                "Use Callable[[ParamTypes...], ReturnType] or Callable[..., ReturnType] "
                "with a concrete return type (not Any)."
            )

        filename = f"<synthesis:{id(self)}>"

        module_code = encoded_value.module_code

        # Parse and validate AST before execution
        module: ast.AST = evaluation.parse(module_code, filename)

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

        # Validate signature from AST before execution
        _validate_signature_ast(last_stmt, self.expected_params)

        # Type-check with mypy; pass original module_code so mypy sees exact source
        evaluation.type_check(
            module, self.ctx, self.expected_params, self.expected_return
        )

        # Compile and execute
        # https://docs.python.org/3/library/functions.html#exec
        g: MutableMapping[str, Any] = {}
        g.update(self.ctx or {})

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

        # Validate signature from runtime callable after execution
        _validate_signature_callable(result, self.expected_params, self.expected_return)

        return result

    def serialize(
        self, encoded_value: SynthesizedFunction
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": encoded_value.model_dump_json()}]

    def deserialize(self, serialized_value: str) -> SynthesizedFunction:
        return SynthesizedFunction.model_validate_json(serialized_value)


_tool_adapter: pydantic.TypeAdapter[ChatCompletionToolParam] = pydantic.TypeAdapter(
    ChatCompletionToolParam
)


def _validate_tool(
    value: Tool | ChatCompletionToolParam,
    info: pydantic.ValidationInfo
) -> Tool:
    if isinstance(value, Tool):
        return value
    if isinstance(value, dict):
        name = value.get("function", {}).get("name")
        ctx = info.context or {}
        tool = ctx.get(name)
        if isinstance(tool, Tool):
            return tool
        raise ValueError(f"Unknown tool: {name}")
    raise ValueError(
        f"Expected Tool or ChatCompletionToolParam dict, got {type(value)}"
    )


def _serialize_tool(value: Tool) -> ChatCompletionToolParam:
    response_format = litellm.utils.type_to_response_format_param(
        _param_model(inspect.signature(value))
    )
    assert response_format is not None
    assert value.__default__.__doc__ is not None
    return _tool_adapter.validate_python(
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


PydanticTool = typing.Annotated[
    Tool,
    pydantic.PlainValidator(_validate_tool),
    pydantic.PlainSerializer(_serialize_tool),
    pydantic.WithJsonSchema(_tool_adapter.json_schema()),
]


class ToolEncodable[**P, T](Encodable[Tool[P, T], pydantic.BaseModel]):
    enc: pydantic.TypeAdapter[ChatCompletionToolParam]

    def __init__(self, base: type[Tool[P, T]], ctx: Mapping[str, Any]):
        self.base = base
        self.ctx = ctx
        self.enc = pydantic.TypeAdapter(ChatCompletionToolParam)

    def encode(self, value: Tool[P, T]) -> ChatCompletionToolParam:
        response_format = litellm.utils.type_to_response_format_param(
            _param_model(inspect.signature(value))
        )
        assert response_format is not None
        assert value.__default__.__doc__ is not None
        return self.enc.validate_python(
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

    def decode(self, encoded_value: ChatCompletionToolParam) -> Tool[P, T]:
        tool = self.ctx.get(encoded_value["function"]["name"], None)
        if isinstance(tool, Tool):
            return tool
        else:
            raise NotImplementedError("Tools cannot yet be decoded from LLM responses")

    def serialize(
        self, encoded_value: ChatCompletionToolParam
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [
            {
                "type": "text",
                "text": self.enc.dump_json(encoded_value).decode("utf-8"),
            }
        ]

    def deserialize(self, serialized_value: str) -> pydantic.BaseModel:
        return self.enc.dump_python(self.enc.validate_json(serialized_value))


def _validate_tool_call(
    value: DecodedToolCall | ChatCompletionMessageToolCall,
    info: pydantic.ValidationInfo
) -> DecodedToolCall:
    if isinstance(value, DecodedToolCall):
        return value
    if isinstance(value, ChatCompletionMessageToolCall):
        ctx = info.context or {}
        assert value.function.name is not None
        tool = ctx[value.function.name]
        assert isinstance(tool, Tool)
        sig = inspect.signature(tool)
        decoded_args = {}
        for name, raw_arg in json.loads(value.function.arguments).items():
            param = sig.parameters[name]
            arg_enc = Encodable.define(param.annotation, ctx)
            decoded_args[name] = arg_enc.decode(raw_arg)
        return DecodedToolCall(
            tool=tool,
            bound_args=sig.bind(**decoded_args),
            id=value.id,
            name=value.function.name,
        )
    raise ValueError(
        f"Expected DecodedToolCall or ChatCompletionMessageToolCall, got {type(value)}"
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
    ).model_dump()


PydanticToolCall = typing.Annotated[
    DecodedToolCall,
    pydantic.PlainValidator(_validate_tool_call),
    pydantic.PlainSerializer(_serialize_tool_call),
    pydantic.WithJsonSchema(ChatCompletionMessageToolCall.model_json_schema())
]


@_CustomSingleDispatchCallable
def pydantic_type(
    __dispatch: Callable[[type], Callable[[Any], Any]],
    ty: Any,
) -> Any:
    """Substitute custom types with their Pydantic Annotated equivalents.

    Recursively walks a type annotation tree, replacing leaf types that have
    registered Pydantic annotations (e.g., Image.Image -> PydanticImage) and
    reconstructing the full generic type.

    The result can be passed to pydantic.TypeAdapter() for automatic
    validation and serialization of nested structures.
    """
    if isinstance(ty, tuple | list | set | frozenset):
        return type(ty)(pydantic_type(t) for t in ty)
    elif isinstance(ty, typing.TypeVar | typing.ParamSpec | typing.ParamSpecArgs | typing.ParamSpecKwargs):
        return ty
    elif typing.get_origin(ty) is typing.Annotated:
        args = typing.get_args(ty)
        return typing.Annotated[pydantic_type(args[0]), *args[1:]]
    elif not typing.get_args(ty):
        return __dispatch(typing.get_origin(ty) or ty)(ty)
    else:
        origin = typing.get_origin(ty)
        return __dispatch(origin)(origin[pydantic_type(typing.get_args(ty))])


@pydantic_type.register(object)
@pydantic_type.register(str)
@pydantic_type.register(pydantic.BaseModel)
def _pydantic_type_base[T](ty: type[T]) -> type[T]:
    return ty


@pydantic_type.register(Term)
def _pydantic_type_term(ty: type[Term]):
    raise TypeError("Terms cannot be converted to Pydantic types.")


@pydantic_type.register(Operation)
def _pydantic_type_operation(ty: type[Operation]):
    raise TypeError("Operations cannot be converted to Pydantic types.")


@pydantic_type.register(Image.Image)
def _pydantic_type_image(ty: type[Image.Image]):
    adapter = pydantic.TypeAdapter(ChatCompletionImageUrlObject)
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate_image),
        pydantic.PlainSerializer(_serialize_image),
        pydantic.WithJsonSchema(adapter.json_schema()),
    ]


@pydantic_type.register(Tool)
def _pydantic_type_tool(ty: type[Tool]):
    adapter = pydantic.TypeAdapter(ChatCompletionToolParam)
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate_tool),
        pydantic.PlainSerializer(_serialize_tool),
        pydantic.WithJsonSchema(adapter.json_schema()),
    ]


@pydantic_type.register(DecodedToolCall)
def _pydantic_type_tool_call(ty: type[DecodedToolCall]):
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate_tool_call),
        pydantic.PlainSerializer(_serialize_tool_call),
        pydantic.WithJsonSchema(ChatCompletionMessageToolCall.model_json_schema()),
    ]


@pydantic_type.register(Callable)
def _pydantic_type_callable(ty: type[Callable]):
    return PydanticCallable(ty)


class ToolCallEncodable[T](
    Encodable[DecodedToolCall[T], ChatCompletionMessageToolCall]
):
    enc: type[ChatCompletionMessageToolCall]

    def __init__(self, base: type[DecodedToolCall[T]], ctx: Mapping[str, Any]):
        self.base = base
        self.ctx = ctx
        self.enc = ChatCompletionMessageToolCall

    def encode(self, value: DecodedToolCall[T]) -> ChatCompletionMessageToolCall:
        encoded_args = {}
        for k, v in value.bound_args.arguments.items():
            v_enc = Encodable.define(nested_type(v).value, self.ctx)
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
        )

    def decode(
        self, encoded_value: ChatCompletionMessageToolCall
    ) -> DecodedToolCall[T]:
        """Decode a tool call from the LLM response into a DecodedToolCall.

        Args:
            encoded_value: The tool call to decode.
        """
        assert encoded_value.function.name is not None
        tool: Tool[..., T] = self.ctx[encoded_value.function.name]
        assert isinstance(tool, Tool)

        json_str = encoded_value.function.arguments
        sig = inspect.signature(tool)

        decoded_args = {}
        for name, raw_arg in json.loads(json_str).items():
            param = sig.parameters[name]
            arg_enc = Encodable.define(param.annotation, self.ctx)
            decoded_args[name] = arg_enc.decode(raw_arg)

        return DecodedToolCall(
            tool=tool,
            bound_args=sig.bind(**decoded_args),
            id=encoded_value.id,
            name=encoded_value.function.name,
        )

    def serialize(
        self, encoded_value: ChatCompletionMessageToolCall
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": encoded_value.model_dump_json()}]

    def deserialize(self, serialized_value: str) -> ChatCompletionMessageToolCall:
        return self.enc.model_validate_json(serialized_value)


@Encodable.define.register(object)
def _encodable_object[T](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, _BoxEncoding[T]]:
    return AdapterEncodable(ty, ctx or {})


@Encodable.define.register(str)
def _encodable_str(ty: type[str], ctx: Mapping[str, Any] | None) -> Encodable[str, str]:
    """Handler for str type that serializes without JSON encoding."""
    return StrEncodable(ctx or {})


@Encodable.define.register(pydantic.BaseModel)
def _encodable_pydantic_base_model[T: pydantic.BaseModel](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, T]:
    return PydanticBaseModelEncodable(ty, ctx or {})


@Encodable.define.register(Term)
def _encodable_term[T: Term](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, Any]:
    raise TypeError("Terms cannot be encoded or decoded in general.")


@Encodable.define.register(Operation)
def _encodable_operation[T: Operation](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, Any]:
    raise TypeError("Operations cannot be encoded or decoded in general.")


@Encodable.define.register(Image.Image)
def _encodable_image(
    ty: type[Image.Image], ctx: Mapping[str, Any] | None
) -> Encodable[Image.Image, pydantic.BaseModel]:
    return ImageEncodable(ty, ctx or {})


@Encodable.define.register(Callable)
def _encodable_callable(
    ty: type[Callable], ctx: Mapping[str, Any] | None
) -> Encodable[Callable, SynthesizedFunction]:
    ctx = ctx or {}

    type_args = typing.get_args(ty)

    # Bare Callable without type args - allow encoding but disable decode
    # this occurs when decoding the result of Tools which return callable (need to Encodable.define(return_type) for return type)
    if not type_args:
        assert ty is types.FunctionType, f"Callable must have type signatures {ty}"
        typed_enc = _create_typed_synthesized_function(Callable[..., typing.Any])  # type: ignore[arg-type]
        return CallableEncodable(ty, typed_enc, ctx)

    if len(type_args) < 2:
        raise TypeError(
            f"Callable type signature incomplete: {ty}. "
            "Expected Callable[[ParamTypes...], ReturnType] or Callable[..., ReturnType]."
        )

    param_types, expected_return = type_args[0], type_args[-1]

    typed_enc = _create_typed_synthesized_function(ty)

    # Ellipsis means any params, skip param validation
    expected_params: list[type] | None = None
    if param_types is not ... and isinstance(param_types, list | tuple):
        expected_params = list(param_types)

    return CallableEncodable(ty, typed_enc, ctx, expected_params, expected_return)


@Encodable.define.register(Tool)
def _encodable_tool[**P, T](
    ty: type[Tool[P, T]], ctx: Mapping[str, Any] | None
) -> Encodable[Tool[P, T], pydantic.BaseModel]:
    return ToolEncodable(ty, ctx or {})


@Encodable.define.register(DecodedToolCall)
def _encodable_tool_call[T](
    ty: type[DecodedToolCall[T]], ctx: Mapping[str, Any] | None
) -> Encodable[DecodedToolCall[T], ChatCompletionMessageToolCall]:
    return ToolCallEncodable(ty, ctx or {})
