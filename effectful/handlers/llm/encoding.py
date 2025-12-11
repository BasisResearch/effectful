import base64
import io
import numbers
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable

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
@type_to_encodable_type.register(int)
@type_to_encodable_type.register(float)
@type_to_encodable_type.register(bool)
def _type_encodable_type_base[T](ty: type[T]) -> Encodable[T]:
    class BaseEncodable(_Encodable[T, T]):
        t = ty

        @classmethod
        def encode(cls, t: T) -> T:
            return t

        @classmethod
        def decode(cls, t: T) -> T:
            return t

    return typing.cast(Encodable[T], BaseEncodable())


@type_to_encodable_type.register(numbers.Number)
class EncodableNumber(_Encodable[numbers.Number, float]):
    t = float

    def __init__(self, _):
        pass

    @classmethod
    def encode(cls, n: numbers.Number) -> float:
        return float(n)  # type: ignore

    @classmethod
    def decode(cls, n: float) -> numbers.Number:
        return typing.cast(numbers.Number, n)


@type_to_encodable_type.register(Image.Image)
class EncodableImage(_Encodable[Image.Image, ChatCompletionImageUrlObject]):
    t = ChatCompletionImageUrlObject

    def __init__(self, _):
        pass

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


@type_to_encodable_type.register(tuple)
def _type_encodable_type_tuple[T](ty: type[T]) -> Encodable[T]:
    args = typing.get_args(ty)

    # Handle empty tuple, or tuple with no args
    if not args or args == ((),):
        return _type_encodable_type_base(ty)

    # Create encoders for each element type
    element_encoders = [type_to_encodable_type(arg) for arg in args]

    encoded_ty: type = tuple[*(encoder.t for encoder in element_encoders)]  # type: ignore

    class TupleEncodable(_Encodable[T, encoded_ty]):  # type: ignore
        t = encoded_ty

        @classmethod
        def encode(cls, t: T) -> encoded_ty:  # type: ignore
            if not isinstance(t, tuple):
                raise TypeError(f"Expected tuple, got {type(t)}")
            if len(t) != len(element_encoders):
                raise ValueError(
                    f"Tuple length {len(t)} does not match expected length {len(element_encoders)}"
                )
            return tuple([enc.encode(elem) for enc, elem in zip(element_encoders, t)])

        @classmethod
        def decode(cls, t: encoded_ty) -> T:  # type: ignore
            if len(t) != len(element_encoders):
                raise ValueError(
                    f"tuple length {len(t)} does not match expected length {len(element_encoders)}"
                )
            decoded_elements = [  # type: ignore
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
    element_encoder: Encodable[T] = type_to_encodable_type(element_ty)

    # Build the encoded type (list of encoded element type)
    encoded_ty: type = list[element_encoder.t]  # type: ignore

    class ListEncodable(_Encodable[T, encoded_ty]):  # type: ignore
        t = encoded_ty

        @classmethod
        def encode(cls, t: T) -> encoded_ty:  # type: ignore
            if not isinstance(t, list):
                raise TypeError(f"Expected list, got {type(t)}")
            return [element_encoder.encode(elem) for elem in t]

        @classmethod
        def decode(cls, t: encoded_ty) -> T:  # type: ignore
            decoded_elements = [element_encoder.decode(elem) for elem in t]  # type: ignore
            return typing.cast(T, decoded_elements)

    return typing.cast(Encodable[T], ListEncodable())
