import base64
import io
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable

import pydantic
from litellm import ChatCompletionImageUrlObject
from PIL import Image

from effectful.ops.syntax import _CustomSingleDispatchCallable


def _pil_image_to_base64_data(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_image_to_base64_data_uri(pil_image: Image.Image) -> str:
    return f"data:image/png;base64,{_pil_image_to_base64_data(pil_image)}"


class _Encodable[T, U](ABC):
    t: type[U]

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def encode(cls, t: T) -> U:
        pass

    @classmethod
    @abstractmethod
    def decode(cls, t: U) -> T:
        pass


class Encodable[T](_Encodable[T, type]):
    t = type


@_CustomSingleDispatchCallable
def type_to_encodable_type[T](
    __dispatch: Callable[[type[T]], Callable[..., Encodable[T]]], ty: type[T]
) -> Encodable[T]:
    origin_ty = typing.get_origin(ty) or ty
    return __dispatch(origin_ty)(ty)


@type_to_encodable_type.register(object)
@type_to_encodable_type.register(str)
@type_to_encodable_type.register(int)
@type_to_encodable_type.register(bool)
@type_to_encodable_type.register(float)
@type_to_encodable_type.register(complex)
def _type_encodable_type_base[T](ty: type[T]) -> Encodable[T]:
    class BaseEncodable(_Encodable[T, T]):
        t: type[T] = ty

        @classmethod
        def encode(cls, vl: T) -> T:
            return vl

        @classmethod
        def decode(cls, vl: T) -> T:
            return vl

    return typing.cast(Encodable[T], BaseEncodable())


@type_to_encodable_type.register(Image.Image)
class EncodableImage(_Encodable[Image.Image, ChatCompletionImageUrlObject]):
    t = ChatCompletionImageUrlObject

    @classmethod
    def encode(cls, image: Image.Image) -> ChatCompletionImageUrlObject:
        return {
            "detail": "auto",
            "url": _pil_image_to_base64_data_uri(image),
        }

    @classmethod
    def decode(cls, image: ChatCompletionImageUrlObject) -> Image.Image:
        image_url = image["url"]
        if not image_url.startswith("data:image/"):
            raise RuntimeError(
                f"expected base64 encoded image as data uri, received {image_url}"
            )
        data = image_url.split(",")[1]
        return Image.open(fp=io.BytesIO(base64.b64decode(data)))


U = typing.TypeVar("U", bound=pydantic.BaseModel)


@type_to_encodable_type.register(tuple)
def _type_encodable_type_tuple[T](ty: type[T]) -> Encodable[T]:
    args = typing.get_args(ty)

    # Handle empty tuple, or tuple with no args
    if not args or args == ((),):
        return _type_encodable_type_base(ty)

    # Create encoders for each element type
    element_encoders = [type_to_encodable_type(arg) for arg in args]

    # Build tuple type from element encoder types (runtime-created, use Any)
    encoded_ty: type[typing.Any] = typing.cast(type[typing.Any], tuple)

    class TupleEncodable(_Encodable[T, typing.Any]):
        t: type[typing.Any] = encoded_ty

        @classmethod
        def encode(cls, t: T) -> typing.Any:
            if not isinstance(t, tuple):
                raise TypeError(f"Expected tuple, got {type(t)}")
            if len(t) != len(element_encoders):
                raise ValueError(
                    f"Tuple length {len(t)} does not match expected length {len(element_encoders)}"
                )
            return tuple([enc.encode(elem) for enc, elem in zip(element_encoders, t)])

        @classmethod
        def decode(cls, t: typing.Any) -> T:
            if len(t) != len(element_encoders):
                raise ValueError(
                    f"tuple length {len(t)} does not match expected length {len(element_encoders)}"
                )
            decoded_elements: list[typing.Any] = [
                enc.decode(elem) for enc, elem in zip(element_encoders, t)
            ]
            return typing.cast(T, tuple(decoded_elements))

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

    # Build the encoded type (list of encoded element type) - runtime-created, use Any
    encoded_ty: type[typing.Any] = typing.cast(type[typing.Any], list)

    class ListEncodable(_Encodable[T, typing.Any]):
        t: type[typing.Any] = encoded_ty

        @classmethod
        def encode(cls, t: T) -> typing.Any:
            if not isinstance(t, list):
                raise TypeError(f"Expected list, got {type(t)}")
            return [element_encoder.encode(elem) for elem in t]

        @classmethod
        def decode(cls, t: typing.Any) -> T:
            decoded_elements: list[typing.Any] = [
                element_encoder.decode(elem) for elem in t
            ]
            return typing.cast(T, decoded_elements)

    return typing.cast(Encodable[T], ListEncodable())
