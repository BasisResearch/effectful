import ast
import base64
import inspect
import io
import textwrap
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import CodeType
from typing import Any

import pydantic
from litellm import (
    ChatCompletionImageUrlObject,
    OpenAIMessageContentListBlock,
)
from PIL import Image

import effectful.handlers.llm.evaluation as evaluation
from effectful.handlers.llm.synthesis import SynthesizedFunction
from effectful.ops.semantics import _simple_type
from effectful.ops.syntax import _CustomSingleDispatchCallable
from effectful.ops.types import Operation, Term


def _pil_image_to_base64_data(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_image_to_base64_data_uri(pil_image: Image.Image) -> str:
    return f"data:image/png;base64,{_pil_image_to_base64_data(pil_image)}"


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
        return __dispatch(dispatch_ty)(t, ctx)


@dataclass
class BaseEncodable[T](Encodable[T, T]):
    base: type[T]
    enc: type[T]
    ctx: Mapping[str, Any]
    adapter: pydantic.TypeAdapter[T]

    def encode(self, value: T) -> T:
        return typing.cast(T, self.adapter.validate_python(value))

    def decode(self, encoded_value: T) -> T:
        return typing.cast(T, self.adapter.validate_python(encoded_value))

    def serialize(self, encoded_value: T) -> Sequence[OpenAIMessageContentListBlock]:
        json_str = self.adapter.dump_json(encoded_value).decode("utf-8")
        return [{"type": "text", "text": json_str}]

    def deserialize(self, serialized_value: str) -> T:
        # Parse JSON string into the encoded value, validated as `ty`.
        return typing.cast(T, self.adapter.validate_json(serialized_value))


@dataclass
class StrEncodable(Encodable[str, str]):
    base: type[str]
    enc: type[str]
    ctx: Mapping[str, Any]

    def encode(self, value: str) -> str:
        return value

    def decode(self, encoded_value: str) -> str:
        return encoded_value

    def serialize(self, encoded_value: str) -> Sequence[OpenAIMessageContentListBlock]:
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

    def serialize(self, encoded_value: T) -> Sequence[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": encoded_value.model_dump_json()}]

    def deserialize(self, serialized_value: str) -> T:
        return typing.cast(T, self.base.model_validate_json(serialized_value))


@dataclass
class ImageEncodable(Encodable[Image.Image, ChatCompletionImageUrlObject]):
    base: type[Image.Image]
    enc: type[ChatCompletionImageUrlObject]
    ctx: Mapping[str, Any]

    def encode(self, value: Image.Image) -> ChatCompletionImageUrlObject:
        return {
            "detail": "auto",
            "url": _pil_image_to_base64_data_uri(value),
        }

    def decode(self, encoded_value: ChatCompletionImageUrlObject) -> Image.Image:
        image_url = encoded_value["url"]
        if not image_url.startswith("data:image/"):
            raise RuntimeError(
                f"expected base64 encoded image as data uri, received {image_url}"
            )
        data = image_url.split(",")[1]
        return Image.open(fp=io.BytesIO(base64.b64decode(data)))

    def serialize(
        self, encoded_value: ChatCompletionImageUrlObject
    ) -> Sequence[OpenAIMessageContentListBlock]:
        return [{"type": "image_url", "image_url": encoded_value}]

    def deserialize(self, serialized_value: str) -> ChatCompletionImageUrlObject:
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
        return typing.cast(typing.Any, adapter.validate_json(serialized_value))


@dataclass
class ListEncodable[T](Encodable[list[T], typing.Any]):
    base: type[list[T]]
    enc: type[typing.Any]
    ctx: Mapping[str, Any]
    has_image: bool
    element_encoder: Encodable[T, typing.Any]

    def encode(self, value: list[T]) -> typing.Any:
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value)}")
        return [self.element_encoder.encode(elem) for elem in value]

    def decode(self, encoded_value: typing.Any) -> list[T]:
        decoded_elements: list[T] = [
            self.element_encoder.decode(elem) for elem in encoded_value
        ]
        return typing.cast(list[T], decoded_elements)

    def serialize(
        self, encoded_value: typing.Any
    ) -> Sequence[OpenAIMessageContentListBlock]:
        if self.has_image:
            # If list contains images, serialize each element and flatten the results
            result: list[OpenAIMessageContentListBlock] = []
            if not isinstance(encoded_value, list):
                raise TypeError(f"Expected list, got {type(encoded_value)}")
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
        return typing.cast(typing.Any, adapter.validate_json(serialized_value))


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
        elif isinstance(param_types, (list, tuple)):
            params_str = ", ".join(getattr(t, "__name__", str(t)) for t in param_types)
        else:
            params_str = str(param_types)

        return_str = getattr(return_type, "__name__", str(return_type))
        return f"Callable[[{params_str}], {return_str}]"

    return str(callable_type)


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
2. Do not include usage examples.
3. Your output function def must be the final statement in the code block.
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
    expected_return: type | None,
) -> None:
    """Validate the function signature from runtime callable after execution."""
    sig = inspect.signature(func)

    if expected_params is not None:
        actual_params = list(sig.parameters.values())
        if len(actual_params) != len(expected_params):
            raise ValueError(
                f"decode() expected function with {len(expected_params)} parameters, "
                f"got {len(actual_params)}"
            )

    if expected_return is not None:
        actual_return = sig.return_annotation
        if actual_return is not inspect.Parameter.empty:
            expected_name = getattr(expected_return, "__name__", str(expected_return))
            actual_name = getattr(actual_return, "__name__", str(actual_return))
            if expected_name != actual_name:
                raise ValueError(
                    f"decode() expected function with return type {expected_name}, "
                    f"got {actual_name}"
                )


@dataclass
class CallableEncodable(Encodable[Callable, SynthesizedFunction]):
    base: type[Callable]
    enc: type[SynthesizedFunction]
    ctx: Mapping[str, Any]
    expected_params: list[type] | None = None
    expected_return: type | None = None  # None means decode is disabled

    def encode(self, t: Callable) -> SynthesizedFunction:
        # (https://github.com/python/mypy/issues/14928)
        if not isinstance(t, Callable):  # type: ignore
            raise TypeError(f"Expected callable, got {type(t)}")
        try:
            source = inspect.getsource(t)
        except (OSError, TypeError):
            source = None

        if source:
            return self.enc(module_code=textwrap.dedent(source))

        # Source not available - create stub from name, signature, and docstring
        # This is useful for builtins and C extensions
        name = getattr(t, "__name__", None)
        if not name:
            raise RuntimeError(
                f"Cannot encode callable {t}: no source code and no __name__"
            )

        try:
            sig = inspect.signature(t)
            sig_str = str(sig)
        except (ValueError, TypeError):
            # Some builtins don't have inspectable signatures
            sig_str = "(...)"

        docstring = inspect.getdoc(t)
        if not docstring:
            raise RuntimeError(
                f"Cannot encode callable {t}: no source code and no docstring"
            )

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
        if not isinstance(last_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(
                f"decode() requires the last statement to be a function definition, "
                f"got {type(last_stmt).__name__}"
            )

        # Validate signature from AST before execution
        _validate_signature_ast(last_stmt, self.expected_params)

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

    @Operation.define
    @classmethod
    def encoding_instructions(cls) -> str | None:
        """Instructions to be prefixed onto synthesis prompts to tune the encoding of the result."""
        return None


@Encodable.define.register(object)
def _encodable_object[T, U](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, U]:
    adapter = pydantic.TypeAdapter(ty)
    ctx = {} if ctx is None else ctx
    return typing.cast(Encodable[T, U], BaseEncodable(ty, ty, ctx, adapter))


@Encodable.define.register(str)
def _encodable_str(ty: type[str], ctx: Mapping[str, Any] | None) -> Encodable[str, str]:
    """Handler for str type that serializes without JSON encoding."""
    return StrEncodable(ty, ty, ctx or {})


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
) -> Encodable[Image.Image, ChatCompletionImageUrlObject]:
    return ImageEncodable(ty, ChatCompletionImageUrlObject, ctx or {})


@Encodable.define.register(tuple)
def _encodable_tuple[T, U](
    ty: type[T], ctx: Mapping[str, Any] | None
) -> Encodable[T, U]:
    args = typing.get_args(ty)
    ctx = {} if ctx is None else ctx

    # handle namedtuples
    origin = typing.get_origin(ty)
    if origin is None:
        return _encodable_object(ty, ctx)
    # Handle empty tuple, or tuple with no args
    if not args or args == ((),):
        return _encodable_object(ty, ctx)

    # Create encoders for each element type
    element_encoders = [Encodable.define(arg, ctx) for arg in args]

    # Check if any element type is Image.Image
    has_image = any(arg is Image.Image for arg in args)

    encoded_ty: type[typing.Any] = typing.cast(
        type[typing.Any],
        tuple[*(enc.enc for enc in element_encoders)],  # type: ignore
    )

    return typing.cast(
        Encodable[T, U],
        TupleEncodable(ty, encoded_ty, ctx, has_image, element_encoders),
    )


@Encodable.define.register(list)
def _encodable_list[T, U](
    ty: type[list[T]], ctx: Mapping[str, Any] | None
) -> Encodable[T, U]:
    args = typing.get_args(ty)
    ctx = {} if ctx is None else ctx

    # Handle unparameterized list (list without type args)
    if not args:
        return _encodable_object(ty, ctx)

    # Get the element type (first type argument)
    element_ty = args[0]
    element_encoder = Encodable.define(element_ty, ctx)

    # Check if element type is Image.Image
    has_image = element_ty is Image.Image

    # Build the encoded type (list of encoded element type) - runtime-created, use Any
    encoded_ty: type[typing.Any] = typing.cast(
        type[typing.Any],
        list[element_encoder.enc],  # type: ignore
    )

    return typing.cast(
        Encodable[T, U], ListEncodable(ty, encoded_ty, ctx, has_image, element_encoder)
    )


@Encodable.define.register(Callable)
def _encodable_callable(
    ty: type[Callable], ctx: Mapping[str, Any] | None
) -> Encodable[Callable, SynthesizedFunction]:
    ctx = ctx or {}

    # Extract type args - Callable requires a type signature
    type_args = typing.get_args(ty)

    # Handle bare Callable without type args - allow encoding but disable decode
    # this occurs when encoding Tools which return callable (need to Encodable.define(return_type) for return type)
    if not type_args:
        typed_enc = _create_typed_synthesized_function(Callable[..., typing.Any])  # type: ignore[arg-type]
        return CallableEncodable(
            ty, typed_enc, ctx, expected_params=None, expected_return=None
        )

    if len(type_args) < 2:
        raise TypeError(
            f"Callable type signature incomplete: {ty}. "
            "Expected Callable[[ParamTypes...], ReturnType] or Callable[..., ReturnType]."
        )

    # Extract param and return types for validation
    param_types = type_args[0]
    expected_return: type | None = type_args[-1]

    # Handle Any as return type - allow encoding but disable decode
    # Any doesn't provide useful information for synthesis (expected_return=None)
    if expected_return is typing.Any:
        typed_enc = _create_typed_synthesized_function(ty)
        return CallableEncodable(
            ty, typed_enc, ctx, expected_params=None, expected_return=None
        )

    # Create a typed SynthesizedFunction model with the type signature in the description
    typed_enc = _create_typed_synthesized_function(ty)

    # Handle Callable[..., ReturnType] - ellipsis means any params, skip param validation
    expected_params: list[type] | None = None
    if param_types is not ...:
        if isinstance(param_types, (list, tuple)):
            expected_params = list(param_types)

    return CallableEncodable(
        ty,
        typed_enc,
        ctx,
        expected_params=expected_params,
        expected_return=expected_return,
    )
