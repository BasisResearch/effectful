import base64
import io
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pydantic
from litellm import (
    ChatCompletionImageUrlObject,
    OpenAIMessageContentListBlock,
)
from PIL import Image

from effectful.ops.semantics import _simple_type
from effectful.ops.syntax import _CustomSingleDispatchCallable
from effectful.ops.types import Operation, Term


def _pil_image_to_base64_data(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_image_to_base64_data_uri(pil_image: Image.Image) -> str:
    return f"data:image/png;base64,{_pil_image_to_base64_data(pil_image)}"


@dataclass
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
