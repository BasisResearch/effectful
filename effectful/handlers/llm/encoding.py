import ast
import base64
import functools
import inspect
import io
import textwrap
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Hashable,
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


@dataclass(frozen=True, eq=True)
class DecodedToolCall[T]:
    """
    Structured representation of a tool call decoded from an LLM response.
    """

    tool: Tool[..., T]
    bound_args: inspect.BoundArguments
    id: ToolCallID
    name: str


class Encodable[T, U](ABC):
    base: type[T]
    enc: type[U]
    ctx: Mapping[str, Any]

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
        assert issubclass(
            pydantic.create_model("Model", v=(encodable.enc, ...)), pydantic.BaseModel
        ), f"enc type {encodable.enc} is not a valid pydantic field type for {t}"
        return encodable


class _BoxEncoding[T](pydantic.BaseModel):
    value: T


@dataclass
class BaseEncodable[T](Encodable[T, _BoxEncoding[T]]):
    base: type[T]
    enc: type[_BoxEncoding[T]]
    ctx: Mapping[str, Any]

    def encode(self, value: T) -> _BoxEncoding[T]:
        return self.enc(value=value)

    def decode(self, encoded_value: _BoxEncoding[T]) -> T:
        return typing.cast(T, encoded_value.value)

    def serialize(
        self, encoded_value: _BoxEncoding[T]
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": encoded_value.model_dump_json()}]

    def deserialize(self, serialized_value: str) -> _BoxEncoding[T]:
        return self.enc.model_validate_json(serialized_value)

    @staticmethod
    @functools.cache
    def wrapped_model(ty: Hashable) -> type[_BoxEncoding[Any]]:
        scalar_ty = typing.cast(type[Any], ty)
        return typing.cast(
            type[_BoxEncoding[Any]],
            pydantic.create_model(
                f"Response_{getattr(scalar_ty, '__name__', 'scalar')}",
                value=(scalar_ty, ...),
                __base__=_BoxEncoding,
                __config__={"extra": "forbid"},
            ),
        )


@dataclass
class StrEncodable(Encodable[str, str]):
    base: type[str]
    enc: type[str]
    ctx: Mapping[str, Any]

    def encode(self, value: str) -> str:
        return value

    def decode(self, encoded_value: str) -> str:
        return encoded_value

    def serialize(self, encoded_value: str) -> Sequence[ChatCompletionTextObject]:
        # Serialize strings without JSON encoding (no extra quotes)
        return [{"type": "text", "text": encoded_value}]

    def deserialize(self, serialized_value: str) -> str:
        return serialized_value


@dataclass
class PydanticBaseModelEncodable[T: pydantic.BaseModel](Encodable[T, T]):
    base: type[T]
    enc: type[T]
    ctx: Mapping[str, Any]

    def decode(self, encoded_value: T) -> T:
        return encoded_value

    def encode(self, value: T) -> T:
        return value

    def serialize(self, encoded_value: T) -> Sequence[ChatCompletionTextObject]:
        return [{"type": "text", "text": encoded_value.model_dump_json()}]

    def deserialize(self, serialized_value: str) -> T:
        return typing.cast(T, self.base.model_validate_json(serialized_value))


@dataclass
class ImageEncodable(Encodable[Image.Image, pydantic.BaseModel]):
    base: type[Image.Image]
    enc: type[pydantic.BaseModel]
    ctx: Mapping[str, Any]

    def encode(self, value: Image.Image) -> pydantic.BaseModel:
        return self.enc(
            detail="auto",
            url=_pil_image_to_base64_data_uri(value),
        )

    def decode(
        self, encoded_value: pydantic.BaseModel | Mapping[str, Any]
    ) -> Image.Image:
        normalized = self.enc.model_validate(encoded_value)
        image_url = typing.cast(str, getattr(normalized, "url"))
        if not image_url.startswith("data:image/"):
            raise TypeError(
                f"expected base64 encoded image as data uri, received {image_url}"
            )
        data = image_url.split(",")[1]
        return Image.open(fp=io.BytesIO(base64.b64decode(data)))

    def serialize(
        self, encoded_value: pydantic.BaseModel
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [
            {
                "type": "image_url",
                "image_url": typing.cast(
                    ChatCompletionImageUrlObject,
                    encoded_value.model_dump(exclude_none=True),
                ),
            }
        ]

    def deserialize(self, serialized_value: str) -> pydantic.BaseModel:
        # Images are serialized as image_url blocks, not text
        # This shouldn't be called in normal flow, but provide a fallback
        raise NotImplementedError("Image deserialization from string is not supported")


@dataclass
class TupleEncodable[T](Encodable[T, typing.Any]):
    base: type[T]
    enc: type[typing.Any]
    ctx: Mapping[str, Any]
    has_image: bool
    element_encoders: list[Encodable]

    def encode(self, value: T) -> typing.Any:
        if not isinstance(value, tuple):
            raise TypeError(f"Expected tuple, got {type(value)}")
        if len(value) != len(self.element_encoders):
            raise ValueError(
                f"Tuple length {len(value)} does not match expected length {len(self.element_encoders)}"
            )
        return tuple(
            [enc.encode(elem) for enc, elem in zip(self.element_encoders, value)]
        )

    def decode(self, encoded_value: typing.Any) -> T:
        if len(encoded_value) != len(self.element_encoders):
            raise ValueError(
                f"tuple length {len(encoded_value)} does not match expected length {len(self.element_encoders)}"
            )
        decoded_elements: list[typing.Any] = [
            enc.decode(elem) for enc, elem in zip(self.element_encoders, encoded_value)
        ]
        return typing.cast(T, tuple(decoded_elements))

    def serialize(
        self, encoded_value: typing.Any
    ) -> Sequence[OpenAIMessageContentListBlock]:
        if self.has_image:
            # If tuple contains images, serialize each element and flatten the results
            result: list[OpenAIMessageContentListBlock] = []
            if not isinstance(encoded_value, tuple):
                raise TypeError(f"Expected tuple, got {type(encoded_value)}")
            if len(encoded_value) != len(self.element_encoders):
                raise ValueError(
                    f"Tuple length {len(encoded_value)} does not match expected length {len(self.element_encoders)}"
                )
            for enc, elem in zip(self.element_encoders, encoded_value):
                result.extend(enc.serialize(elem))
            return result
        else:
            # Use base serialization for non-image tuples
            adapter: pydantic.TypeAdapter[tuple] = pydantic.TypeAdapter(self.enc)
            json_str = adapter.dump_json(encoded_value).decode("utf-8")
            return [{"type": "text", "text": json_str}]

    def deserialize(self, serialized_value: str) -> typing.Any:
        adapter: pydantic.TypeAdapter[tuple] = pydantic.TypeAdapter(self.enc)
        return adapter.validate_json(serialized_value)


@dataclass
class NamedTupleEncodable[T](TupleEncodable[T]):
    """Tuple encodable that reconstructs the original NamedTuple type on decode."""

    def decode(self, encoded_value: typing.Any) -> T:
        if len(encoded_value) != len(self.element_encoders):
            raise ValueError(
                f"tuple length {len(encoded_value)} does not match expected length {len(self.element_encoders)}"
            )
        decoded_elements: list[typing.Any] = [
            enc.decode(elem) for enc, elem in zip(self.element_encoders, encoded_value)
        ]
        return typing.cast(T, self.base(*decoded_elements))


@dataclass
class MutableSequenceEncodable[T](Encodable[MutableSequence[T], typing.Any]):
    base: type[MutableSequence[T]]
    enc: type[typing.Any]
    ctx: Mapping[str, Any]
    has_image: bool
    element_encoder: Encodable[T, typing.Any]

    def encode(self, value: MutableSequence[T]) -> typing.Any:
        if not isinstance(value, MutableSequence):
            raise TypeError(f"Expected MutableSequence, got {type(value)}")
        return [self.element_encoder.encode(elem) for elem in value]

    def decode(self, encoded_value: typing.Any) -> MutableSequence[T]:
        decoded_elements: list[T] = [
            self.element_encoder.decode(elem) for elem in encoded_value
        ]
        return typing.cast(MutableSequence[T], decoded_elements)

    def serialize(
        self, encoded_value: typing.Any
    ) -> Sequence[OpenAIMessageContentListBlock]:
        if self.has_image:
            # If list contains images, serialize each element and flatten the results
            result: list[OpenAIMessageContentListBlock] = []
            if not isinstance(encoded_value, MutableSequence):
                raise TypeError(f"Expected MutableSequence, got {type(encoded_value)}")
            for elem in encoded_value:
                result.extend(self.element_encoder.serialize(elem))
            return result
        else:
            # Use base serialization for non-image lists
            adapter = pydantic.TypeAdapter(self.enc)
            json_str = adapter.dump_json(encoded_value).decode("utf-8")
            return [{"type": "text", "text": json_str}]

    def deserialize(self, serialized_value: str) -> typing.Any:
        adapter = pydantic.TypeAdapter(self.enc)
        return adapter.validate_json(serialized_value)


@dataclass
class TypedDictEncodable[T](Encodable[T, pydantic.BaseModel]):
    base: type[T]
    enc: type[pydantic.BaseModel]
    ctx: Mapping[str, Any]

    def encode(self, value: T) -> pydantic.BaseModel:
        return self.enc.model_validate(value)

    def decode(self, encoded_value: pydantic.BaseModel) -> T:
        decoded_value: dict[str, Any] = encoded_value.model_dump()
        adapter = pydantic.TypeAdapter(self.base)
        return typing.cast(T, adapter.validate_python(decoded_value))

    def serialize(
        self, encoded_value: pydantic.BaseModel
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": encoded_value.model_dump_json()}]

    def deserialize(self, serialized_value: str) -> pydantic.BaseModel:
        return self.enc.model_validate_json(serialized_value)

    @staticmethod
    @functools.cache
    def _typeddict_model(td: type[Any]) -> type[pydantic.BaseModel]:
        hints = typing.get_type_hints(td)
        required = typing.cast(
            frozenset[str], getattr(td, "__required_keys__", frozenset())
        )
        fields: dict[str, Any] = {}
        for k, v in hints.items():
            fields[k] = (v, ...) if k in required else (v, None)
        return pydantic.create_model(
            td.__name__,
            **fields,
        )


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


def _param_model(sig: inspect.Signature) -> type[pydantic.BaseModel]:
    return pydantic.create_model(
        "Params",
        __config__={"extra": "forbid"},
        **{
            name: Encodable.define(param.annotation).enc
            for name, param in sig.parameters.items()
        },  # type: ignore
    )


@dataclass
class ToolEncodable[**P, T](Encodable[Tool[P, T], pydantic.BaseModel]):
    base: type[Tool]
    enc: type[pydantic.BaseModel]
    ctx: Mapping[str, Any]

    def encode(self, value: Tool[P, T]) -> pydantic.BaseModel:
        response_format = litellm.utils.type_to_response_format_param(
            _param_model(inspect.signature(value))
        )
        assert response_format is not None
        assert value.__default__.__doc__ is not None
        return self.enc.model_validate(
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

    def decode(self, encoded_value: pydantic.BaseModel) -> Tool[P, T]:
        raise NotImplementedError("Tools cannot yet be decoded from LLM responses")

    def serialize(
        self, encoded_value: pydantic.BaseModel
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [
            {
                "type": "text",
                "text": encoded_value.model_dump_json(exclude_none=True),
            }
        ]

    def deserialize(self, serialized_value: str) -> pydantic.BaseModel:
        return self.enc.model_validate_json(serialized_value)


@dataclass
class ToolCallEncodable[T](
    Encodable[DecodedToolCall[T], ChatCompletionMessageToolCall]
):
    base: type[DecodedToolCall[T]]
    enc: type[ChatCompletionMessageToolCall]
    ctx: Mapping[str, Any]

    def encode(self, value: DecodedToolCall[T]) -> ChatCompletionMessageToolCall:
        sig = inspect.signature(value.tool)
        encoded_args = _param_model(sig).model_validate(
            {
                k: Encodable.define(
                    typing.cast(type[Any], nested_type(v).value), self.ctx
                ).encode(v)
                for k, v in value.bound_args.arguments.items()
            }
        )
        return ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": value.id,
                "function": {
                    "name": value.tool.__name__,
                    "arguments": encoded_args.model_dump_json(),
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

        raw_args = _param_model(sig).model_validate_json(json_str)

        bound_args: inspect.BoundArguments = sig.bind(
            **{
                name: Encodable.define(
                    typing.cast(type[Any], sig.parameters[name].annotation), self.ctx
                ).decode(getattr(raw_args, name))
                for name in raw_args.model_fields_set
            }
        )
        return DecodedToolCall(
            tool=tool,
            bound_args=bound_args,
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
def _encodable_object[T, U](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, U]:
    ctx = {} if ctx is None else ctx
    wrapped = BaseEncodable.wrapped_model(typing.cast(Hashable, ty))
    return typing.cast(Encodable[T, U], BaseEncodable(ty, wrapped, ctx))


@Encodable.define.register(str)
def _encodable_str(ty: type[str], ctx: Mapping[str, Any] | None) -> Encodable[str, str]:
    """Handler for str type that serializes without JSON encoding."""
    return StrEncodable(ty, ty, ctx or {})


class _ComplexParts(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    real: float
    imag: float


@dataclass
class _ComplexEncodable(Encodable[complex, _ComplexParts]):
    base: type[complex]
    enc: type[_ComplexParts]
    ctx: Mapping[str, Any]

    def encode(self, value: complex) -> _ComplexParts:
        return _ComplexParts(real=value.real, imag=value.imag)

    def decode(self, encoded_value: _ComplexParts) -> complex:
        return complex(encoded_value.real, encoded_value.imag)

    def serialize(
        self, encoded_value: _ComplexParts
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": encoded_value.model_dump_json()}]

    def deserialize(self, serialized_value: str) -> _ComplexParts:
        return _ComplexParts.model_validate_json(serialized_value)


@Encodable.define.register(complex)
def _encodable_complex(
    ty: type[complex], ctx: Mapping[str, Any] | None
) -> Encodable[complex, _ComplexParts]:
    return _ComplexEncodable(ty, _ComplexParts, ctx or {})


@Encodable.define.register(Term)
def _encodable_term[T: Term, U](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, U]:
    raise TypeError("Terms cannot be encoded or decoded in general.")


@Encodable.define.register(Operation)
def _encodable_operation[T: Operation, U](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, U]:
    raise TypeError("Operations cannot be encoded or decoded in general.")


@Encodable.define.register(pydantic.BaseModel)
def _encodable_pydantic_base_model[T: pydantic.BaseModel](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, T]:
    return PydanticBaseModelEncodable(ty, ty, ctx or {})


@Encodable.define.register(Image.Image)
def _encodable_image(
    ty: type[Image.Image], ctx: Mapping[str, Any] | None
) -> Encodable[Image.Image, pydantic.BaseModel]:
    image_model = TypedDictEncodable._typeddict_model(ChatCompletionImageUrlObject)
    return ImageEncodable(ty, image_model, ctx or {})


@Encodable.define.register(tuple)
def _encodable_tuple[T, U](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, U]:
    def _is_namedtuple_type(ty: type[Any]) -> bool:
        return isinstance(ty, type) and issubclass(ty, tuple) and hasattr(ty, "_fields")

    args = typing.get_args(ty)
    ctx = {} if ctx is None else ctx

    # Handle plain tuple runtime type explicitly.
    if ty is tuple:
        return typing.cast(
            Encodable[T, U],
            TupleEncodable(ty, ty, ctx, False, []),
        )

    # NamedTuple handling is routed through tuple logic, but decoded back into
    # the concrete NamedTuple class.
    origin = typing.get_origin(ty)
    is_namedtuple = origin is None and _is_namedtuple_type(ty)
    if origin is None:
        if is_namedtuple:
            hints = typing.get_type_hints(ty)
            tuple_field_types: list[type[Any]] = list(hints.values())
            if not tuple_field_types:
                tuple_field_types = [typing.Any] * len(getattr(ty, "_fields", ()))
        else:
            tuple_field_types = []
    else:
        tuple_field_types = list(args)

    if not tuple_field_types:
        # Non-parameterized tuple subclasses still use object fallback.
        if not is_namedtuple:
            return _encodable_object(ty, ctx)
        # Empty namedtuple; keep tuple identity behavior.
        return typing.cast(Encodable[T, U], NamedTupleEncodable(ty, ty, ctx, False, []))

    # Handle empty tuple annotation (tuple[()]).
    if tuple_field_types == [()] or args == ((),):
        return TupleEncodable(ty, ty, ctx, False, [])

    element_encoders = [Encodable.define(arg, ctx) for arg in tuple_field_types]
    has_image = any(arg is Image.Image for arg in tuple_field_types)
    encoded_ty: type[typing.Any] = typing.cast(
        type[typing.Any],
        tuple[*(enc.enc for enc in element_encoders)],  # type: ignore
    )

    if is_namedtuple:
        return typing.cast(
            Encodable[T, U],
            NamedTupleEncodable(ty, encoded_ty, ctx, has_image, element_encoders),
        )

    if origin is None:
        return _encodable_object(ty, ctx)

    return typing.cast(
        Encodable[T, U],
        TupleEncodable(ty, encoded_ty, ctx, has_image, element_encoders),
    )


@Encodable.define.register(list)
@Encodable.define.register(MutableSequence)
def _encodable_mutable_sequence[T, U](
    ty: type[MutableSequence[T]], ctx: Mapping[str, Any] | None
) -> Encodable[T, U]:
    args = typing.get_args(ty)
    ctx = {} if ctx is None else ctx

    # Handle unparameterized list (list without type args)
    if not args:
        identity_encoder = typing.cast(
            Encodable[T, typing.Any],
            BaseEncodable(
                typing.cast(type[T], object),
                typing.cast(
                    type[_BoxEncoding[T]],
                    BaseEncodable.wrapped_model(typing.cast(Hashable, object)),
                ),
                ctx,
            ),
        )
        return typing.cast(
            Encodable[T, U],
            MutableSequenceEncodable(ty, list[Any], ctx, False, identity_encoder),
        )

    # Get the element type (first type argument)
    element_ty = args[0]
    element_encoder = Encodable.define(element_ty, ctx)

    # Check if element type is Image.Image
    has_image = element_ty is Image.Image

    # Use enc for Image (schema-valid), base otherwise
    encoded_ty: type[typing.Any] = typing.cast(
        type[typing.Any],
        list[element_encoder.enc],  # type: ignore
    )

    return typing.cast(
        Encodable[T, U],
        MutableSequenceEncodable(ty, encoded_ty, ctx, has_image, element_encoder),
    )


@Encodable.define.register(dict)
@Encodable.define.register(MutableMapping)
@Encodable.define.register(Mapping)
def _encodable_mapping[K, V, U](
    ty: type[Mapping[K, V]], ctx: Mapping[str, Any] | None
) -> Encodable[Mapping[K, V], U]:
    ctx = {} if ctx is None else ctx

    if typing.is_typeddict(ty):
        return typing.cast(
            Encodable[Mapping[K, V], U],
            TypedDictEncodable(ty, TypedDictEncodable._typeddict_model(ty), ctx),
        )

    return _encodable_object(ty, ctx)


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
    ctx = ctx or {}
    tool_model = TypedDictEncodable._typeddict_model(ChatCompletionToolParam)
    return ToolEncodable(ty, tool_model, ctx)


@Encodable.define.register(DecodedToolCall)
def _encodable_tool_call[T](
    ty: type[DecodedToolCall[T]], ctx: Mapping[str, Any] | None
) -> Encodable[DecodedToolCall[T], ChatCompletionMessageToolCall]:
    ctx = ctx or {}
    return ToolCallEncodable(ty, ChatCompletionMessageToolCall, ctx)
