import base64
import io
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import pydantic
from litellm import (
    ChatCompletionImageUrlObject,
    OpenAIMessageContentListBlock,
)
from PIL import Image

from effectful.ops.semantics import _simple_type
from effectful.ops.syntax import _CustomSingleDispatchCallable, defop
from effectful.ops.types import Operation, Term


def _pil_image_to_base64_data(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_image_to_base64_data_uri(pil_image: Image.Image) -> str:
    return f"data:image/png;base64,{_pil_image_to_base64_data(pil_image)}"


class EncodableAs[T, U](ABC):
    t: type[U]

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def encode(cls, vl: T, env: Mapping[str, Any] | None = None) -> U:
        pass

    @classmethod
    @abstractmethod
    def decode(cls, vl: U, env: Mapping[str, Any] | None = None) -> T:
        pass

    @classmethod
    def serialize(cls, value: U) -> list[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": str(value)}]


@defop
@_CustomSingleDispatchCallable
def encoding_instructions[T](
    __dispatch: Callable[[type[T]], Callable[..., list[str]]], ty: type[T]
) -> list[str]:
    """Additional instructions per type to be prefixed onto synthesis prompts to tune the encoding of the result."""
    dispatch_ty = _simple_type(ty)
    return __dispatch(dispatch_ty)(ty)


@encoding_instructions.register(object)  # type: ignore
def _(_ty: type[object]) -> list[str]:
    return []


@encoding_instructions.register(list)  # type: ignore
def _[T](ty: type[list[T]]):
    args = typing.get_args(ty)
    return encoding_instructions(args[0]) if args else []


@encoding_instructions.register(tuple)  # type: ignore
def _[T](ty: type[tuple[T, ...]]):
    args = typing.get_args(ty)
    return list({ins for arg in args for ins in (encoding_instructions(arg) or [])})


@encoding_instructions.register(Term)  # type: ignore
def _[T: Term](ty: type[T]) -> list[str]:
    raise TypeError("Terms cannot be encoded or decoded in general.")


@encoding_instructions.register(Operation)  # type: ignore
def _[T: Operation](ty: type[T]) -> list[str]:
    raise TypeError("Operations cannot be encoded or decoded in general.")


class Encodable[T](EncodableAs[T, type]):
    t = type


@_CustomSingleDispatchCallable
def type_to_encodable_type[T](
    __dispatch: Callable[[type[T]], Callable[..., Encodable[T]]], ty: type[T]
) -> Encodable[T]:
    dispatch_ty = _simple_type(ty)
    return __dispatch(dispatch_ty)(ty)


@type_to_encodable_type.register(object)
def _type_encodable_type_base[T](ty: type[T]) -> Encodable[T]:
    class BaseEncodable(EncodableAs[T, T]):
        t: type[T] = ty

        @classmethod
        def encode(cls, vl: T, env: Mapping[str, Any] | None = None) -> T:
            return vl

        @classmethod
        def decode(cls, vl: T, env: Mapping[str, Any] | None = None) -> T:
            return vl

    return typing.cast(Encodable[T], BaseEncodable())


@type_to_encodable_type.register(Term)
def _type_encodable_type_term[T: Term](ty: type[T]) -> Encodable[T]:
    raise TypeError("Terms cannot be encoded or decoded in general.")


@type_to_encodable_type.register(Operation)
def _type_encodable_type_operation[T: Operation](ty: type[T]) -> Encodable[T]:
    raise TypeError("Operations cannot be encoded or decoded in general.")


@type_to_encodable_type.register(pydantic.BaseModel)
def _type_encodable_type_pydantic_base_model[T: pydantic.BaseModel](
    ty: type[T],
) -> Encodable[T]:
    class EncodablePydanticBaseModel(EncodableAs[T, T]):
        t: type[T] = ty

        @classmethod
        def decode(cls, vl: T, env: Mapping[str, Any] | None = None) -> T:
            return vl

        @classmethod
        def encode(cls, vl: T, env: Mapping[str, Any] | None = None) -> T:
            return vl

        @classmethod
        def serialize(cls, value: T) -> list[OpenAIMessageContentListBlock]:
            return [{"type": "text", "text": value.model_dump_json()}]

    return typing.cast(Encodable[T], EncodablePydanticBaseModel())


@type_to_encodable_type.register(Image.Image)
class EncodableImage(EncodableAs[Image.Image, ChatCompletionImageUrlObject]):
    t = ChatCompletionImageUrlObject

    @classmethod
    def encode(
        cls, image: Image.Image, env: Mapping[str, Any] | None = None
    ) -> ChatCompletionImageUrlObject:
        return {
            "detail": "auto",
            "url": _pil_image_to_base64_data_uri(image),
        }

    @classmethod
    def decode(
        cls, image: ChatCompletionImageUrlObject, env: Mapping[str, Any] | None = None
    ) -> Image.Image:
        image_url = image["url"]
        if not image_url.startswith("data:image/"):
            raise RuntimeError(
                f"expected base64 encoded image as data uri, received {image_url}"
            )
        data = image_url.split(",")[1]
        return Image.open(fp=io.BytesIO(base64.b64decode(data)))

    @classmethod
    def serialize(
        cls, value: ChatCompletionImageUrlObject
    ) -> list[OpenAIMessageContentListBlock]:
        return [{"type": "image_url", "image_url": value}]


@type_to_encodable_type.register(tuple)
def _type_encodable_type_tuple[T](ty: type[T]) -> Encodable[T]:
    args = typing.get_args(ty)

    # Handle empty tuple, or tuple with no args
    if not args or args == ((),):
        return _type_encodable_type_base(ty)

    # Create encoders for each element type
    element_encoders = [type_to_encodable_type(arg) for arg in args]

    # Check if any element type is Image.Image
    has_image = any(arg is Image.Image for arg in args)

    encoded_ty: type[typing.Any] = typing.cast(
        type[typing.Any],
        tuple[*(enc.t for enc in element_encoders)],  # type: ignore
    )

    class TupleEncodable(EncodableAs[T, typing.Any]):
        t: type[typing.Any] = encoded_ty

        @classmethod
        def encode(
            cls, t: T, env: typing.Mapping[str, Any] | None = None
        ) -> typing.Any:
            if not isinstance(t, tuple):
                raise TypeError(f"Expected tuple, got {type(t)}")
            if len(t) != len(element_encoders):
                raise ValueError(
                    f"Tuple length {len(t)} does not match expected length {len(element_encoders)}"
                )
            return tuple(
                [enc.encode(elem, env) for enc, elem in zip(element_encoders, t)]
            )

        @classmethod
        def decode(cls, t: typing.Any, env: Mapping[str, Any] | None = None) -> T:
            if len(t) != len(element_encoders):
                raise ValueError(
                    f"tuple length {len(t)} does not match expected length {len(element_encoders)}"
                )
            decoded_elements: list[typing.Any] = [
                enc.decode(elem, env) for enc, elem in zip(element_encoders, t)
            ]
            return typing.cast(T, tuple(decoded_elements))

        @classmethod
        def serialize(cls, value: typing.Any) -> list[OpenAIMessageContentListBlock]:
            if has_image:
                # If tuple contains images, serialize each element and flatten the results
                result: list[OpenAIMessageContentListBlock] = []
                if not isinstance(value, tuple):
                    raise TypeError(f"Expected tuple, got {type(value)}")
                if len(value) != len(element_encoders):
                    raise ValueError(
                        f"Tuple length {len(value)} does not match expected length {len(element_encoders)}"
                    )
                for enc, elem in zip(element_encoders, value):
                    result.extend(enc.serialize(elem))
                return result
            else:
                return super().serialize(value)

    return typing.cast(Encodable[T], TupleEncodable())


@type_to_encodable_type.register(list)
def _type_encodable_type_list[T](ty: type[T]) -> Encodable[T]:
    args = typing.get_args(ty)

    # Handle unparameterized list (list without type args)
    if not args:
        return _type_encodable_type_base(ty)

    # Get the element type (first type argument)
    element_ty = args[0]
    element_encoder = type_to_encodable_type(element_ty)

    # Check if element type is Image.Image
    has_image = element_ty is Image.Image

    # Build the encoded type (list of encoded element type) - runtime-created, use Any
    encoded_ty: type[typing.Any] = typing.cast(
        type[typing.Any],
        list[element_encoder.t],  # type: ignore
    )

    class ListEncodable(EncodableAs[T, typing.Any]):
        t: type[typing.Any] = encoded_ty

        @classmethod
        def encode(cls, t: T, env: Mapping[str, Any] | None = None) -> typing.Any:
            if not isinstance(t, list):
                raise TypeError(f"Expected list, got {type(t)}")
            return [element_encoder.encode(elem, env) for elem in t]

        @classmethod
        def decode(cls, t: typing.Any, env: Mapping[str, Any] | None = None) -> T:
            decoded_elements: list[typing.Any] = [
                element_encoder.decode(elem, env) for elem in t
            ]
            return typing.cast(T, decoded_elements)

        @classmethod
        def serialize(cls, value: typing.Any) -> list[OpenAIMessageContentListBlock]:
            if has_image:
                # If list contains images, serialize each element and flatten the results
                result: list[OpenAIMessageContentListBlock] = []
                if not isinstance(value, list):
                    raise TypeError(f"Expected list, got {type(value)}")
                for elem in value:
                    result.extend(element_encoder.serialize(elem))
                return result
            else:
                return super().serialize(value)

    return typing.cast(Encodable[T], ListEncodable())
