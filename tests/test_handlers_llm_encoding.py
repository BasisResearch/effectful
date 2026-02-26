<<<<<<< HEAD
import inspect
from collections import ChainMap
from dataclasses import asdict, dataclass
from typing import NamedTuple, TypedDict

import pydantic
import pytest
from PIL import Image

from effectful.handlers.llm.encodable_type import (
    EncodableSynthesizedType,
    SynthesizedType,
)
from effectful.handlers.llm.encoding import type_to_encodable_type
from effectful.handlers.llm.synthesis import SynthesisError
from effectful.ops.types import Operation, Term


def test_type_to_encodable_type_term():
    with pytest.raises(TypeError):
        type_to_encodable_type(Term)


def test_type_to_encodable_type_operation():
    with pytest.raises(TypeError):
        type_to_encodable_type(Operation)


def test_type_to_encodable_type_str():
    encodable = type_to_encodable_type(str)
    encoded = encodable.encode("hello")
    decoded = encodable.decode(encoded)
    assert decoded == "hello"
    Model = pydantic.create_model("Model", value=encodable.t)
    decoded = Model.model_validate({"value": "hello"})
    assert decoded.value == "hello"


def test_type_to_encodable_type_int():
    encodable = type_to_encodable_type(int)
    encoded = encodable.encode(42)
    decoded = encodable.decode(encoded)
    assert decoded == 42
    assert isinstance(decoded, int)
    Model = pydantic.create_model("Model", value=encodable.t)
    decoded = Model.model_validate({"value": 42})
    assert decoded.value == 42
    assert isinstance(decoded.value, int)


def test_type_to_encodable_type_bool():
    encodable = type_to_encodable_type(bool)
    encoded = encodable.encode(True)
    decoded = encodable.decode(encoded)
    assert decoded is True
    assert isinstance(decoded, bool)
    encoded_false = encodable.encode(False)
    decoded_false = encodable.decode(encoded_false)
    assert decoded_false is False
    Model = pydantic.create_model("Model", value=encodable.t)
    decoded = Model.model_validate({"value": True})
    assert decoded.value is True
    assert isinstance(decoded.value, bool)


def test_type_to_encodable_type_float():
    encodable = type_to_encodable_type(float)
    encoded = encodable.encode(3.14)
    decoded = encodable.decode(encoded)
    assert decoded == 3.14
    assert isinstance(decoded, float)
    Model = pydantic.create_model("Model", value=encodable.t)
    decoded = Model.model_validate({"value": 3.14})
    assert decoded.value == 3.14
    assert isinstance(decoded.value, float)


def test_type_to_encodable_type_image():
    encodable = type_to_encodable_type(Image.Image)
    image = Image.new("RGB", (10, 10), color="red")
    encoded = encodable.encode(image)
    assert isinstance(encoded, dict)
    assert "url" in encoded
    assert "detail" in encoded
    assert encoded["detail"] == "auto"
    assert encoded["url"].startswith("data:image/png;base64,")
    decoded = encodable.decode(encoded)
    assert isinstance(decoded, Image.Image)
    assert decoded.size == (10, 10)
    Model = pydantic.create_model("Model", value=encodable.t)
    decoded = Model.model_validate({"value": encoded})
    assert decoded.value["url"] == encoded["url"]
    assert decoded.value["detail"] == "auto"


def test_type_to_encodable_type_image_roundtrip():
    encodable = type_to_encodable_type(Image.Image)
    original = Image.new("RGB", (20, 20), color="green")
    encoded = encodable.encode(original)
    decoded = encodable.decode(encoded)
    assert isinstance(decoded, Image.Image)
    assert decoded.size == original.size
    assert decoded.mode == original.mode


def test_type_to_encodable_type_image_decode_invalid_url():
    encodable = type_to_encodable_type(Image.Image)
    encoded = {"url": "http://example.com/image.png", "detail": "auto"}
    with pytest.raises(RuntimeError, match="expected base64 encoded image as data uri"):
        encodable.decode(encoded)


def test_type_to_encodable_type_tuple():
    encodable = type_to_encodable_type(tuple[int, str])
    value = (1, "test")
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, tuple)
    assert decoded[0] == 1
    assert decoded[1] == "test"
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded})
    assert model_instance.value == encoded
    assert isinstance(model_instance.value, tuple)
    assert model_instance.value[0] == 1
    assert model_instance.value[1] == "test"
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == value
    assert isinstance(decoded_from_model, tuple)


def test_type_to_encodable_type_tuple_empty():
    encodable = type_to_encodable_type(tuple[()])
    value = ()
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, tuple)
    assert len(decoded) == 0
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded})
    assert model_instance.value == encoded
    assert isinstance(model_instance.value, tuple)
    assert len(model_instance.value) == 0
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == value
    assert isinstance(decoded_from_model, tuple)


def test_type_to_encodable_type_tuple_three_elements():
    encodable = type_to_encodable_type(tuple[int, str, bool])
    value = (42, "hello", True)
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, tuple)
    assert decoded[0] == 42
    assert decoded[1] == "hello"
    assert decoded[2] is True
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded})
    assert model_instance.value == encoded
    assert isinstance(model_instance.value, tuple)
    assert model_instance.value[0] == 42
    assert model_instance.value[1] == "hello"
    assert model_instance.value[2] is True
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == value
    assert isinstance(decoded_from_model, tuple)


def test_type_to_encodable_type_list():
    encodable = type_to_encodable_type(list[int])
    value = [1, 2, 3, 4, 5]
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, list)
    assert all(isinstance(elem, int) for elem in decoded)
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded})
    assert model_instance.value == encoded
    assert isinstance(model_instance.value, list)
    assert model_instance.value == [1, 2, 3, 4, 5]
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == value
    assert isinstance(decoded_from_model, list)
    assert all(isinstance(elem, int) for elem in decoded_from_model)


def test_type_to_encodable_type_list_str():
    encodable = type_to_encodable_type(list[str])
    value = ["hello", "world", "test"]
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, list)
    assert all(isinstance(elem, str) for elem in decoded)
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded})
    assert model_instance.value == encoded
    assert isinstance(model_instance.value, list)
    assert model_instance.value == ["hello", "world", "test"]
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == value
    assert isinstance(decoded_from_model, list)
    assert all(isinstance(elem, str) for elem in decoded_from_model)


def test_type_to_encodable_type_namedtuple():
    class Point(NamedTuple):
        x: int
        y: int

    encodable = type_to_encodable_type(Point)
    point = Point(10, 20)
    encoded = encodable.encode(point)
    decoded = encodable.decode(encoded)
    assert decoded == point
    assert isinstance(decoded, Point)
    assert decoded.x == 10
    assert decoded.y == 20
    Model = pydantic.create_model("Model", value=encodable.t)
    decoded = Model.model_validate({"value": {"x": 10, "y": 20}})
    assert decoded.value == point
    assert isinstance(decoded.value, Point)


def test_type_to_encodable_type_namedtuple_with_str():
    class Person(NamedTuple):
        name: str
        age: int

    encodable = type_to_encodable_type(Person)
    person = Person("Alice", 30)
    encoded = encodable.encode(person)
    decoded = encodable.decode(encoded)
    assert decoded == person
    assert isinstance(decoded, Person)
    assert decoded.name == "Alice"
    assert decoded.age == 30
    Model = pydantic.create_model("Model", value=encodable.t)
    decoded = Model.model_validate({"value": {"name": "Alice", "age": 30}})
    assert decoded.value == person
    assert isinstance(decoded.value, Person)


def test_type_to_encodable_type_typeddict():
    class User(TypedDict):
        name: str
        age: int

    encodable = type_to_encodable_type(User)
    user = User(name="Bob", age=25)
    encoded = encodable.encode(user)
    decoded = encodable.decode(encoded)
    assert decoded == user
    assert isinstance(decoded, dict)
    assert decoded["name"] == "Bob"
    assert decoded["age"] == 25
    Model = pydantic.create_model("Model", value=encodable.t)
    decoded = Model.model_validate({"value": {"name": "Bob", "age": 25}})
    assert decoded.value == user
    assert isinstance(decoded.value, dict)


def test_type_to_encodable_type_typeddict_optional():
    class Config(TypedDict, total=False):
        host: str
        port: int

    encodable = type_to_encodable_type(Config)
    config = Config(host="localhost", port=8080)
    encoded = encodable.encode(config)
    decoded = encodable.decode(encoded)
    assert decoded == config
    assert decoded["host"] == "localhost"
    assert decoded["port"] == 8080
    Model = pydantic.create_model("Model", value=encodable.t)
    decoded = Model.model_validate({"value": {"host": "localhost", "port": 8080}})
    assert decoded.value == config
    assert isinstance(decoded.value, dict)


def test_type_to_encodable_type_complex():
    encodable = type_to_encodable_type(complex)
    value = 3 + 4j
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, complex)
    assert decoded.real == 3.0
    assert decoded.imag == 4.0
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded})
    assert model_instance.value == encoded
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == value
    assert isinstance(decoded_from_model, complex)


def test_type_to_encodable_type_tuple_of_images():
    encodable = type_to_encodable_type(tuple[Image.Image, Image.Image])
    image1 = Image.new("RGB", (10, 10), color="red")
    image2 = Image.new("RGB", (20, 20), color="blue")
    value = (image1, image2)

    encoded = encodable.encode(value)
    assert isinstance(encoded, tuple)
    assert len(encoded) == 2
    assert isinstance(encoded[0], dict)
    assert isinstance(encoded[1], dict)
    assert "url" in encoded[0]
    assert "url" in encoded[1]
    assert encoded[0]["url"].startswith("data:image/png;base64,")
    assert encoded[1]["url"].startswith("data:image/png;base64,")

    decoded = encodable.decode(encoded)
    assert isinstance(decoded, tuple)
    assert len(decoded) == 2
    assert isinstance(decoded[0], Image.Image)
    assert isinstance(decoded[1], Image.Image)
    assert decoded[0].size == (10, 10)
    assert decoded[1].size == (20, 20)

    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded})
    assert model_instance.value == encoded
    assert isinstance(model_instance.value, tuple)
    assert len(model_instance.value) == 2
    assert isinstance(model_instance.value[0], dict)
    assert isinstance(model_instance.value[1], dict)
    assert model_instance.value[0]["url"] == encoded[0]["url"]
    assert model_instance.value[1]["url"] == encoded[1]["url"]
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert isinstance(decoded_from_model, tuple)
    assert len(decoded_from_model) == 2
    assert isinstance(decoded_from_model[0], Image.Image)
    assert isinstance(decoded_from_model[1], Image.Image)
    assert decoded_from_model[0].size == (10, 10)
    assert decoded_from_model[1].size == (20, 20)

    # Roundtrip test
    original = (
        Image.new("RGB", (15, 15), color="green"),
        Image.new("RGB", (25, 25), color="yellow"),
    )
    encoded_roundtrip = encodable.encode(original)
    decoded_roundtrip = encodable.decode(encoded_roundtrip)
    assert isinstance(decoded_roundtrip, tuple)
    assert len(decoded_roundtrip) == 2
    assert decoded_roundtrip[0].size == original[0].size
    assert decoded_roundtrip[1].size == original[1].size
    assert decoded_roundtrip[0].mode == original[0].mode
    assert decoded_roundtrip[1].mode == original[1].mode


def test_type_to_encodable_type_list_of_images():
    encodable = type_to_encodable_type(list[Image.Image])
    images = [
        Image.new("RGB", (10, 10), color="red"),
        Image.new("RGB", (20, 20), color="blue"),
        Image.new("RGB", (30, 30), color="green"),
    ]

    encoded = encodable.encode(images)
    assert isinstance(encoded, list)
    assert len(encoded) == 3
    assert all(isinstance(elem, dict) for elem in encoded)
    assert all("url" in elem for elem in encoded)
    assert all(elem["url"].startswith("data:image/png;base64,") for elem in encoded)

    decoded = encodable.decode(encoded)
    assert isinstance(decoded, list)
    assert len(decoded) == 3
    assert all(isinstance(elem, Image.Image) for elem in decoded)
    assert decoded[0].size == (10, 10)
    assert decoded[1].size == (20, 20)
    assert decoded[2].size == (30, 30)

    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded})
    assert model_instance.value == encoded
    assert isinstance(model_instance.value, list)
    assert len(model_instance.value) == 3
    assert all(isinstance(elem, dict) for elem in model_instance.value)
    assert all("url" in elem for elem in model_instance.value)
    assert model_instance.value[0]["url"] == encoded[0]["url"]
    assert model_instance.value[1]["url"] == encoded[1]["url"]
    assert model_instance.value[2]["url"] == encoded[2]["url"]
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert isinstance(decoded_from_model, list)
    assert len(decoded_from_model) == 3
    assert all(isinstance(elem, Image.Image) for elem in decoded_from_model)
    assert decoded_from_model[0].size == (10, 10)
    assert decoded_from_model[1].size == (20, 20)
    assert decoded_from_model[2].size == (30, 30)

    # Roundtrip test
    original = [
        Image.new("RGB", (15, 15), color="yellow"),
        Image.new("RGB", (25, 25), color="purple"),
    ]
    encoded_roundtrip = encodable.encode(original)
    decoded_roundtrip = encodable.decode(encoded_roundtrip)
    assert isinstance(decoded_roundtrip, list)
    assert len(decoded_roundtrip) == 2
    assert decoded_roundtrip[0].size == original[0].size
    assert decoded_roundtrip[1].size == original[1].size
    assert decoded_roundtrip[0].mode == original[0].mode
    assert decoded_roundtrip[1].mode == original[1].mode


def test_type_to_encodable_type_dataclass():
    @dataclass
    class Point:
        x: int
        y: int

    encodable = type_to_encodable_type(Point)
    point = Point(10, 20)
    encoded = encodable.encode(point)
    decoded = encodable.decode(encoded)
    assert decoded == point
    assert isinstance(decoded, Point)
    assert decoded.x == 10
    assert decoded.y == 20
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": asdict(encoded)})
    assert model_instance.value.x == 10
    assert model_instance.value.y == 20
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == point
    assert isinstance(decoded_from_model, Point)


def test_type_to_encodable_type_dataclass_with_str():
    @dataclass
    class Person:
        name: str
        age: int

    encodable = type_to_encodable_type(Person)
    person = Person("Alice", 30)
    encoded = encodable.encode(person)
    decoded = encodable.decode(encoded)
    assert decoded == person
    assert isinstance(decoded, Person)
    assert decoded.name == "Alice"
    assert decoded.age == 30
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": asdict(encoded)})
    assert model_instance.value.name == "Alice"
    assert model_instance.value.age == 30
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == person
    assert isinstance(decoded_from_model, Person)


def test_type_to_encodable_type_dataclass_with_list():
    @dataclass
    class Container:
        items: list[int]
        name: str

    encodable = type_to_encodable_type(Container)
    container = Container(items=[1, 2, 3], name="test")
    encoded = encodable.encode(container)
    decoded = encodable.decode(encoded)
    assert decoded == container
    assert isinstance(decoded, Container)
    assert decoded.items == [1, 2, 3]
    assert decoded.name == "test"
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": asdict(encoded)})
    assert model_instance.value.items == [1, 2, 3]
    assert model_instance.value.name == "test"
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == container
    assert isinstance(decoded_from_model, Container)


def test_type_to_encodable_type_dataclass_with_tuple():
    @dataclass
    class Pair:
        values: tuple[int, str]
        count: int

    encodable = type_to_encodable_type(Pair)
    pair = Pair(values=(42, "hello"), count=2)
    encoded = encodable.encode(pair)
    decoded = encodable.decode(encoded)
    assert decoded == pair
    assert isinstance(decoded, Pair)
    assert decoded.values == (42, "hello")
    assert decoded.count == 2
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": asdict(encoded)})
    assert model_instance.value.values == (42, "hello")
    assert model_instance.value.count == 2
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == pair
    assert isinstance(decoded_from_model, Pair)


def test_type_to_encodable_type_dataclass_with_optional():
    @dataclass
    class Config:
        host: str
        port: int
        timeout: float | None = None

    encodable = type_to_encodable_type(Config)
    config = Config(host="localhost", port=8080, timeout=5.0)
    encoded = encodable.encode(config)
    decoded = encodable.decode(encoded)
    assert decoded == config
    assert isinstance(decoded, Config)
    assert decoded.host == "localhost"
    assert decoded.port == 8080
    assert decoded.timeout == 5.0

    # Test with None value
    config_none = Config(host="localhost", port=8080, timeout=None)
    encoded_none = encodable.encode(config_none)
    decoded_none = encodable.decode(encoded_none)
    assert decoded_none == config_none
    assert decoded_none.timeout is None

    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": asdict(encoded)})
    assert model_instance.value.host == "localhost"
    assert model_instance.value.port == 8080
    assert model_instance.value.timeout == 5.0
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == config


def test_type_to_encodable_type_nested_dataclass():
    @dataclass
    class Address:
        street: str
        city: str

    @dataclass
    class Person:
        name: str
        age: int
        address: Address

    encodable = type_to_encodable_type(Person)
    address = Address(street="123 Main St", city="New York")
    person = Person(name="Bob", age=25, address=address)

    encoded = encodable.encode(person)
    assert isinstance(encoded, Person)
    assert hasattr(encoded, "name")
    assert hasattr(encoded, "age")
    assert hasattr(encoded, "address")
    assert isinstance(encoded.address, Address)
    assert encoded.address.street == "123 Main St"
    assert encoded.address.city == "New York"

    decoded = encodable.decode(encoded)
    assert isinstance(decoded, Person)
    assert isinstance(decoded.address, Address)
    assert decoded.name == "Bob"
    assert decoded.age == 25
    assert decoded.address.street == "123 Main St"
    assert decoded.address.city == "New York"

    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": asdict(encoded)})
    assert model_instance.value.name == "Bob"
    assert model_instance.value.age == 25
    assert model_instance.value.address.street == "123 Main St"
    assert model_instance.value.address.city == "New York"
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == person
    assert isinstance(decoded_from_model, Person)
    assert isinstance(decoded_from_model.address, Address)


def test_type_to_encodable_type_pydantic_model():
    class Point(pydantic.BaseModel):
        x: int
        y: int

    encodable = type_to_encodable_type(Point)
    point = Point(x=10, y=20)
    encoded = encodable.encode(point)
    decoded = encodable.decode(encoded)
    assert decoded == point
    assert isinstance(decoded, Point)
    assert decoded.x == 10
    assert decoded.y == 20
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded.model_dump()})
    assert model_instance.value.x == 10
    assert model_instance.value.y == 20
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == point
    assert isinstance(decoded_from_model, Point)


def test_type_to_encodable_type_pydantic_model_with_str():
    class Person(pydantic.BaseModel):
        name: str
        age: int

    encodable = type_to_encodable_type(Person)
    person = Person(name="Alice", age=30)
    encoded = encodable.encode(person)
    decoded = encodable.decode(encoded)
    assert decoded == person
    assert isinstance(decoded, Person)
    assert decoded.name == "Alice"
    assert decoded.age == 30
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded.model_dump()})
    assert model_instance.value.name == "Alice"
    assert model_instance.value.age == 30
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == person
    assert isinstance(decoded_from_model, Person)


def test_type_to_encodable_type_pydantic_model_with_list():
    class Container(pydantic.BaseModel):
        items: list[int]
        name: str

    encodable = type_to_encodable_type(Container)
    container = Container(items=[1, 2, 3], name="test")
    encoded = encodable.encode(container)
    decoded = encodable.decode(encoded)
    assert decoded == container
    assert isinstance(decoded, Container)
    assert decoded.items == [1, 2, 3]
    assert decoded.name == "test"
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded.model_dump()})
    assert model_instance.value.items == [1, 2, 3]
    assert model_instance.value.name == "test"
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == container
    assert isinstance(decoded_from_model, Container)


def test_type_to_encodable_type_nested_pydantic_model():
    class Address(pydantic.BaseModel):
        street: str
        city: str

    class Person(pydantic.BaseModel):
        name: str
        age: int
        address: Address

    encodable = type_to_encodable_type(Person)
    address = Address(street="123 Main St", city="New York")
    person = Person(name="Bob", age=25, address=address)

    encoded = encodable.encode(person)
    assert isinstance(encoded, pydantic.BaseModel)
    assert hasattr(encoded, "name")
    assert hasattr(encoded, "age")
    assert hasattr(encoded, "address")
    assert isinstance(encoded.address, pydantic.BaseModel)
    assert encoded.address.street == "123 Main St"
    assert encoded.address.city == "New York"

    decoded = encodable.decode(encoded)
    assert isinstance(decoded, Person)
    assert isinstance(decoded.address, Address)
    assert decoded.name == "Bob"
    assert decoded.age == 25
    assert decoded.address.street == "123 Main St"
    assert decoded.address.city == "New York"

    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.t)
    model_instance = Model.model_validate({"value": encoded.model_dump()})
    assert model_instance.value.name == "Bob"
    assert model_instance.value.age == 25
    assert model_instance.value.address.street == "123 Main St"
    assert model_instance.value.address.city == "New York"
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == person
    assert isinstance(decoded_from_model, Person)
    assert isinstance(decoded_from_model.address, Address)


class TestEncodableSynthesizedType:
    """Tests for EncodableSynthesizedType encode/decode functionality."""

    def test_decode_simple_class(self):
        """Test decoding a simple class from SynthesizedType."""
        synth = SynthesizedType(
            type_name="Greeter",
            module_code="""\
class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"
""",
        )

        result = EncodableSynthesizedType.decode(synth)

        assert isinstance(result, type)
        assert result.__name__ == "Greeter"

        # Test instantiation and method call
        instance = result()
        assert instance.greet("World") == "Hello, World!"

    def test_decode_with_inheritance(self):
        """Test decoding a class that inherits from a base class in context."""

        class Animal:
            def speak(self) -> str:
                raise NotImplementedError

        synth = SynthesizedType(
            type_name="Dog",
            module_code="""\
class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"
""",
        )

        # Attach context with base class
        object.__setattr__(synth, "_decode_context", ChainMap({"Animal": Animal}))

        result = EncodableSynthesizedType.decode(synth)

        assert isinstance(result, type)
        assert issubclass(result, Animal)
        assert result.__name__ == "Dog"

        instance = result()
        assert instance.speak() == "Woof!"

    def test_decode_attaches_source_attribute(self):
        """Test that decoded types have __source__ attribute."""
        synth = SynthesizedType(
            type_name="Simple",
            module_code="class Simple:\n    pass",
        )

        result = EncodableSynthesizedType.decode(synth)

        assert hasattr(result, "__source__")
        assert "class Simple" in result.__source__

    def test_decode_attaches_synthesized_attribute(self):
        """Test that decoded types have __synthesized__ attribute."""
        synth = SynthesizedType(
            type_name="Simple",
            module_code="class Simple:\n    pass",
        )

        result = EncodableSynthesizedType.decode(synth)

        assert hasattr(result, "__synthesized__")
        assert result.__synthesized__ is synth

    def test_decode_inspect_getsource_works(self):
        """Test that inspect.getsource() works on synthesized types."""
        synth = SynthesizedType(
            type_name="Documented",
            module_code='''\
class Documented:
    """A documented class."""

    def method(self) -> int:
        return 42
''',
        )

        result = EncodableSynthesizedType.decode(synth)
        source = inspect.getsource(result)

        assert "class Documented" in source
        assert "A documented class" in source
        assert "def method" in source
        assert source == result.__source__

    def test_decode_with_helper_in_class(self):
        """Test decoding a class that uses a helper method."""
        synth = SynthesizedType(
            type_name="Counter",
            module_code="""\
class Counter:
    def __init__(self):
        self.value = 0

    def _increment(self, x):
        return x + 1

    def increment(self):
        self.value = self._increment(self.value)
        return self.value
""",
        )

        result = EncodableSynthesizedType.decode(synth)
        instance = result()

        assert instance.increment() == 1
        assert instance.increment() == 2
        assert instance.increment() == 3

    def test_decode_syntax_error_raises_synthesis_error(self):
        """Test that syntax errors raise SynthesisError."""
        synth = SynthesizedType(
            type_name="Broken",
            module_code="class Broken\n    pass  # missing colon",
        )

        with pytest.raises(SynthesisError, match="Syntax error"):
            EncodableSynthesizedType.decode(synth)

    def test_decode_missing_type_raises_synthesis_error(self):
        """Test that missing type name raises SynthesisError."""
        synth = SynthesizedType(
            type_name="Missing",
            module_code="class WrongName:\n    pass",
        )

        with pytest.raises(SynthesisError, match="not found after execution"):
            EncodableSynthesizedType.decode(synth)

    def test_decode_non_type_raises_synthesis_error(self):
        """Test that non-type result raises SynthesisError."""
        synth = SynthesizedType(
            type_name="NotAType",
            module_code="NotAType = 42",
        )

        with pytest.raises(SynthesisError, match="is not a type"):
            EncodableSynthesizedType.decode(synth)

    def test_encode_simple_class(self):
        """Test encoding a simple class to SynthesizedType."""

        class MyClass:
            def method(self) -> str:
                return "hello"

        result = EncodableSynthesizedType.encode(MyClass)

        assert isinstance(result, SynthesizedType)
        assert result.type_name == "MyClass"
        assert "class MyClass" in result.module_code
        assert "def method" in result.module_code

    def test_encode_builtin_class_fallback(self):
        """Test encoding a builtin class (source unavailable) uses fallback."""
        # int is a builtin, so inspect.getsource() will fail
        result = EncodableSynthesizedType.encode(int)

        assert isinstance(result, SynthesizedType)
        assert result.type_name == "int"
        assert "class int" in result.module_code
        assert "Source unavailable" in result.module_code

    def test_serialize_produces_json(self):
        """Test that serialize produces valid JSON content blocks."""
        synth = SynthesizedType(
            type_name="TestType",
            module_code="class TestType:\n    pass",
        )

        result = EncodableSynthesizedType.serialize(synth)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        # Verify it's valid JSON
        import json

        parsed = json.loads(result[0]["text"])
        assert parsed["type_name"] == "TestType"
        assert "class TestType" in parsed["module_code"]

    def test_decode_unique_module_names(self):
        """Test that each decoded type gets a unique module name."""
        synth1 = SynthesizedType(
            type_name="Unique",
            module_code="class Unique:\n    value = 1",
        )
        synth2 = SynthesizedType(
            type_name="Unique",
            module_code="class Unique:\n    value = 2",
        )

        result1 = EncodableSynthesizedType.decode(synth1)
        result2 = EncodableSynthesizedType.decode(synth2)

        # Both should be different types with different module names
        assert result1 is not result2
        assert result1.__module__ != result2.__module__
        assert result1.value == 1
        assert result2.value == 2

    def test_decode_context_with_multiple_items(self):
        """Test decoding with context containing multiple items."""

        class BaseA:
            pass

        class BaseB:
            pass

        def helper() -> int:
            return 100

        synth = SynthesizedType(
            type_name="Combined",
            module_code="""\
class Combined(BaseA, BaseB):
    def get_value(self) -> int:
        return helper()
""",
        )

        context = ChainMap({"BaseA": BaseA, "BaseB": BaseB, "helper": helper})
        object.__setattr__(synth, "_decode_context", context)

        result = EncodableSynthesizedType.decode(synth)

        assert issubclass(result, BaseA)
        assert issubclass(result, BaseB)
        instance = result()
        assert instance.get_value() == 100
=======
"""
Law-based test suite for effectful.handlers.llm.encoding.

Each test function verifies a single equational law of the Encodable[T, U]
interface, parametrized over many types and values.
"""

import inspect
import io
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, NamedTuple, TypedDict

import litellm
import pydantic
import pytest
from litellm import ChatCompletionMessageToolCall
from PIL import Image

from effectful.handlers.llm.encoding import (
    DecodedToolCall,
    Encodable,
    SynthesizedFunction,
)
from effectful.handlers.llm.evaluation import RestrictedEvalProvider, UnsafeEvalProvider
from effectful.handlers.llm.template import Tool
from effectful.internals.unification import nested_type
from effectful.ops.semantics import handler
from effectful.ops.types import Operation, Term
from tests.test_handlers_llm_tool_calling_book import requires_openai

CHEAP_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Module-level type definitions
# ---------------------------------------------------------------------------


@dataclass
class _Point:
    x: int
    y: int


@dataclass
class _Person:
    name: str
    age: int


@dataclass
class _Address:
    street: str
    city: str


@dataclass
class _PersonWithAddress:
    name: str
    address: _Address


@dataclass
class _Config:
    host: str
    port: int
    timeout: float | None = None


@dataclass
class _Container:
    items: list[int]
    label: str


class _Coord(NamedTuple):
    x: int
    y: int


class _PersonNT(NamedTuple):
    name: str
    age: int


class _UserTD(TypedDict):
    name: str
    age: int


class _ConfigTD(TypedDict, total=False):
    host: str
    port: int


@dataclass
class _Pair:
    values: tuple[int, str]
    count: int


class _PointModel(pydantic.BaseModel):
    x: int
    y: int


class _PersonModel(pydantic.BaseModel):
    name: str
    age: int


class _ContainerModel(pydantic.BaseModel):
    items: list[int]
    name: str


class _AddressModel(pydantic.BaseModel):
    street: str
    city: str


class _PersonWithAddressModel(pydantic.BaseModel):
    name: str
    address: _AddressModel


# ---------------------------------------------------------------------------
# Module-level tool definitions
# ---------------------------------------------------------------------------


@Tool.define
def _tool_add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@Tool.define
def _tool_greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@Tool.define
def _tool_process(items: list[int], label: str) -> str:
    """Process a list of items."""
    return f"{label}: {sum(items)}"


@Tool.define
def _tool_get_value() -> int:
    """Return a constant value."""
    return 42


@Tool.define
def _tool_distance(p: _PointModel) -> float:
    """Compute distance from origin."""
    return (p.x**2 + p.y**2) ** 0.5


# ---------------------------------------------------------------------------
# Module-level callable definitions
# ---------------------------------------------------------------------------


def fn_add(a: int, b: int) -> int:
    return a + b


def fn_greet(name: str) -> str:
    return f"Hello, {name}!"


def fn_is_positive(x: int) -> bool:
    return x > 0


def fn_identity(x: int) -> int:
    return x


def fn_constant() -> int:
    return 42


fn_multiply_factor = 3


def fn_multiply(x: int) -> int:
    return x * fn_multiply_factor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_image(mode, size, color):
    """Create a loaded PngImageFile from the given spec.

    Image.new() returns a plain Image.Image, but encode/decode roundtrips
    through PNG, producing a PngImageFile.  PIL's __eq__ uses strict class
    identity, so Image.Image != PngImageFile.  By constructing test values
    as PngImageFile from the start, we can use plain == in assertions.
    """
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, "PNG")
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    return img


def _make_dtc(tool, kwargs, call_id):
    """Construct a DecodedToolCall from a tool, kwargs, and call id."""
    sig = inspect.signature(tool)
    bound = sig.bind(**kwargs)
    return DecodedToolCall(tool=tool, bound_args=bound, id=call_id, name=tool.__name__)


# ---------------------------------------------------------------------------
# Test case lists
# ---------------------------------------------------------------------------

# (type_annotation, value, ctx) triples — reused across law tests.
# ctx=None means Encodable.define(ty), otherwise Encodable.define(ty, ctx).
ROUNDTRIP_CASES = [
    # --- str ---
    pytest.param(str, "hello", None, id="str-hello"),
    pytest.param(str, "", None, id="str-empty"),
    pytest.param(str, "with spaces and\ttabs", None, id="str-whitespace"),
    pytest.param(str, "line1\nline2", None, id="str-multiline"),
    pytest.param(str, '{"key": "value"}', None, id="str-json-like"),
    # --- int ---
    pytest.param(int, 42, None, id="int-positive"),
    pytest.param(int, -7, None, id="int-negative"),
    pytest.param(int, 0, None, id="int-zero"),
    pytest.param(int, 999999, None, id="int-large"),
    # --- bool ---
    pytest.param(bool, True, None, id="bool-true"),
    pytest.param(bool, False, None, id="bool-false"),
    # --- float ---
    pytest.param(float, 3.14, None, id="float-positive"),
    pytest.param(float, -2.5, None, id="float-negative"),
    pytest.param(float, 0.0, None, id="float-zero"),
    # --- complex ---
    pytest.param(complex, 3 + 4j, None, id="complex-positive"),
    pytest.param(complex, -1 + 0j, None, id="complex-real"),
    # --- dataclass ---
    pytest.param(_Point, _Point(10, 20), None, id="dc-point"),
    pytest.param(_Person, _Person("Alice", 30), None, id="dc-person"),
    pytest.param(
        _Config, _Config("localhost", 8080, 5.0), None, id="dc-config-timeout"
    ),
    pytest.param(_Config, _Config("localhost", 8080), None, id="dc-config-none"),
    pytest.param(
        _PersonWithAddress,
        _PersonWithAddress("Bob", _Address("123 Main", "NYC")),
        None,
        id="dc-nested",
    ),
    pytest.param(_Container, _Container([1, 2, 3], "test"), None, id="dc-with-list"),
    pytest.param(_Pair, _Pair(values=(42, "hello"), count=2), None, id="dc-with-tuple"),
    # --- NamedTuple ---
    pytest.param(_Coord, _Coord(3, 4), None, id="nt-coord"),
    pytest.param(_PersonNT, _PersonNT("Alice", 30), None, id="nt-person"),
    # --- TypedDict ---
    pytest.param(_UserTD, _UserTD(name="Bob", age=25), None, id="td-user"),
    pytest.param(
        _ConfigTD, _ConfigTD(host="localhost", port=8080), None, id="td-config"
    ),
    # --- pydantic BaseModel ---
    pytest.param(_PointModel, _PointModel(x=10, y=20), None, id="pm-point"),
    pytest.param(
        _PersonModel, _PersonModel(name="Alice", age=30), None, id="pm-person"
    ),
    pytest.param(
        _ContainerModel,
        _ContainerModel(items=[1, 2, 3], name="test"),
        None,
        id="pm-with-list",
    ),
    pytest.param(
        _PersonWithAddressModel,
        _PersonWithAddressModel(
            name="Bob", address=_AddressModel(street="123 Main", city="NYC")
        ),
        None,
        id="pm-nested",
    ),
    # --- tuple ---
    pytest.param(tuple[int, str], (1, "hello"), None, id="tuple-int-str"),
    pytest.param(tuple[int, str, bool], (42, "hello", True), None, id="tuple-three"),
    pytest.param(tuple[()], (), None, id="tuple-empty"),
    # --- list ---
    pytest.param(list[int], [1, 2, 3, 4, 5], None, id="list-int"),
    pytest.param(list[str], ["hello", "world"], None, id="list-str"),
    pytest.param(list[int], [], None, id="list-empty"),
    # --- Image ---
    pytest.param(
        Image.Image, _make_png_image("RGB", (10, 10), "red"), None, id="img-red"
    ),
    pytest.param(
        Image.Image,
        _make_png_image("RGBA", (20, 20), (0, 0, 255, 128)),
        None,
        id="img-blue-alpha",
    ),
    # --- composite with Image ---
    pytest.param(
        tuple[Image.Image, str],
        (_make_png_image("RGB", (5, 5), "green"), "label"),
        None,
        id="tuple-img-str",
    ),
    pytest.param(
        list[Image.Image],
        [
            _make_png_image("RGB", (10, 10), "red"),
            _make_png_image("RGB", (15, 15), "blue"),
        ],
        None,
        id="list-img",
    ),
    # --- Tool ---
    pytest.param(type(_tool_add), _tool_add, None, id="tool-add"),
    pytest.param(type(_tool_greet), _tool_greet, None, id="tool-greet"),
    pytest.param(type(_tool_process), _tool_process, None, id="tool-process"),
    pytest.param(type(_tool_get_value), _tool_get_value, None, id="tool-no-params"),
    pytest.param(type(_tool_distance), _tool_distance, None, id="tool-pydantic-param"),
    # --- DecodedToolCall ---
    pytest.param(
        DecodedToolCall,
        _make_dtc(_tool_add, {"a": 3, "b": 5}, "call_1"),
        {"_tool_add": _tool_add},
        id="dtc-add-3-5",
    ),
    pytest.param(
        DecodedToolCall,
        _make_dtc(_tool_add, {"a": 0, "b": -1}, "call_2"),
        {"_tool_add": _tool_add},
        id="dtc-add-0-neg",
    ),
    pytest.param(
        DecodedToolCall,
        _make_dtc(_tool_greet, {"name": "Alice"}, "call_3"),
        {"_tool_greet": _tool_greet},
        id="dtc-greet-alice",
    ),
    pytest.param(
        DecodedToolCall,
        _make_dtc(_tool_process, {"items": [1, 2, 3], "label": "total"}, "call_4"),
        {"_tool_process": _tool_process},
        id="dtc-process-items",
    ),
    pytest.param(
        DecodedToolCall,
        _make_dtc(_tool_distance, {"p": _PointModel(x=3, y=4)}, "call_5"),
        {"_tool_distance": _tool_distance},
        id="dtc-pydantic-param",
    ),
]

# Filter ID sets
_IMAGE_IDS = frozenset({"img-red", "img-blue-alpha", "tuple-img-str", "list-img"})
_TOOL_IDS = frozenset(
    {"tool-add", "tool-greet", "tool-process", "tool-no-params", "tool-pydantic-param"}
)

_tool_decode_xfail = pytest.mark.xfail(
    raises=NotImplementedError, reason="Tool.decode not yet implemented"
)


def _xfail_tools(cases):
    """Add xfail mark to Tool cases (whose decode raises NotImplementedError)."""
    return [
        pytest.param(*c.values, marks=[*c.marks, _tool_decode_xfail], id=c.id)
        if c.id in _TOOL_IDS
        else c
        for c in cases
    ]


# Derived case lists
# decode: Tool cases are xfail (decode raises NotImplementedError)
DECODE_CASES = _xfail_tools(ROUNDTRIP_CASES)

# Text-serializable: everything except Image-containing types
TEXT_CASES = [c for c in ROUNDTRIP_CASES if c.id not in _IMAGE_IDS]

# Full pipeline (encode→serialize→deserialize→decode): needs both text and decode
FULL_PIPELINE_CASES = _xfail_tools(
    [c for c in ROUNDTRIP_CASES if c.id not in _IMAGE_IDS]
)


# ============================================================================
# Law 1: decode(encode(v)) == v
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", DECODE_CASES)
def test_encode_decode_roundtrip(ty, value, ctx):
    enc = Encodable.define(ty, ctx)
    assert enc.decode(enc.encode(value)) == value


# ============================================================================
# Law 2: deserialize(serialize(encode(v))[0]["text"]) == encode(v)
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", TEXT_CASES)
def test_serialize_deserialize_roundtrip(ty, value, ctx):
    enc = Encodable.define(ty, ctx)
    encoded = enc.encode(value)
    blocks = enc.serialize(encoded)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert enc.deserialize(blocks[0]["text"]) == encoded


# ============================================================================
# Law 3: decode(deserialize(serialize(encode(v))[0]["text"])) == v
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", FULL_PIPELINE_CASES)
def test_full_pipeline_roundtrip(ty, value, ctx):
    enc = Encodable.define(ty, ctx)
    encoded = enc.encode(value)
    text = enc.serialize(encoded)[0]["text"]
    assert enc.decode(enc.deserialize(text)) == value


# ============================================================================
# Law 4: serialize(encode(v)) succeeds
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", ROUNDTRIP_CASES)
def test_serialize_succeeds(ty, value, ctx):
    enc = Encodable.define(ty, ctx)
    enc.serialize(enc.encode(value))


# ============================================================================
# Law 5: encode(encode(v)) == encode(v) (idempotency)
# ============================================================================


@pytest.mark.parametrize(
    "ty,value,ctx",
    ROUNDTRIP_CASES,
)
def test_encode_idempotent(ty, value, ctx):
    enc = Encodable.define(ty, ctx)
    once = enc.encode(value)
    twice = Encodable.define(nested_type(once).value, ctx).encode(once)
    assert once == twice


# ============================================================================
# Term-specific: Encodable.define raises TypeError for Term and Operation
# ============================================================================


@pytest.mark.parametrize("ty", [Term, Operation])
def test_define_raises_for_invalid_types(ty):
    with pytest.raises(TypeError):
        Encodable.define(ty)


# ============================================================================
# Image-specific: deserialize raises, decode rejects invalid URLs
# ============================================================================


def test_image_deserialize_raises():
    enc = Encodable.define(Image.Image)
    with pytest.raises(NotImplementedError):
        enc.deserialize("anything")


def test_image_decode_rejects_non_data_uri():
    enc = Encodable.define(Image.Image)
    with pytest.raises(TypeError):
        enc.decode({"url": "http://example.com/image.png", "detail": "auto"})


# ============================================================================
# DecodedToolCall-specific: error cases
# ============================================================================

TOOL_CALL_ERROR_CASES = [
    pytest.param(
        "nonexistent", "{}", {}, (KeyError, AssertionError), id="unknown-tool"
    ),
    pytest.param(
        "_tool_add",
        '{"a": "not_an_int", "b": 2}',
        {"_tool_add": _tool_add},
        pydantic.ValidationError,
        id="wrong-arg-type",
    ),
    pytest.param(
        "_tool_add",
        '{"a": 1}',
        {"_tool_add": _tool_add},
        (pydantic.ValidationError, TypeError),
        id="missing-required-arg",
    ),
    pytest.param(
        "_tool_add",
        '{"a": 1, "b": 2, "c": 3}',
        {"_tool_add": _tool_add},
        pydantic.ValidationError,
        id="extra-arg",
    ),
    pytest.param(
        "_tool_add",
        "{not valid json}",
        {"_tool_add": _tool_add},
        pydantic.ValidationError,
        id="invalid-json",
    ),
    pytest.param(
        "_tool_process",
        '{"items": ["a", "b"], "label": "total"}',
        {"_tool_process": _tool_process},
        pydantic.ValidationError,
        id="wrong-list-element-type",
    ),
]


@pytest.mark.parametrize("tool_name,args_json,ctx,exc_type", TOOL_CALL_ERROR_CASES)
def test_toolcall_decode_rejects_invalid(tool_name, args_json, ctx, exc_type):
    tool_call = ChatCompletionMessageToolCall.model_validate(
        {
            "type": "tool_call",
            "id": "call_err",
            "function": {"name": tool_name, "arguments": args_json},
        }
    )
    enc = Encodable.define(DecodedToolCall, ctx)
    with pytest.raises(exc_type):
        enc.decode(tool_call)


# ============================================================================
# Callable: behavioral roundtrip, serialize/deserialize, error cases
# ============================================================================

EVAL_PROVIDERS = [
    pytest.param(UnsafeEvalProvider(), id="unsafe"),
    pytest.param(RestrictedEvalProvider(), id="restricted"),
]

# (callable_type, function, ctx, test_args, expected_result)
CALLABLE_CASES = [
    pytest.param(Callable[[int, int], int], fn_add, {}, (2, 3), 5, id="add"),
    pytest.param(
        Callable[[str], str], fn_greet, {}, ("Alice",), "Hello, Alice!", id="greet"
    ),
    pytest.param(Callable[[int], bool], fn_is_positive, {}, (5,), True, id="pos-true"),
    pytest.param(
        Callable[[int], bool], fn_is_positive, {}, (-1,), False, id="pos-false"
    ),
    pytest.param(Callable[[int], int], fn_identity, {}, (42,), 42, id="identity"),
    pytest.param(Callable[[], int], fn_constant, {}, (), 42, id="zero-params"),
    pytest.param(
        Callable[[int], int],
        fn_multiply,
        {"fn_multiply_factor": fn_multiply_factor},
        (4,),
        12,
        id="env-factor",
    ),
    pytest.param(
        Callable[[Annotated[int, "value"]], Annotated[int, "result"]],
        fn_identity,
        {},
        (7,),
        7,
        id="annotated-expected-type",
    ),
]


@pytest.mark.parametrize("ty,func,ctx,args,expected", CALLABLE_CASES)
@pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
def test_callable_encode_decode_behavioral(
    ty, func, ctx, args, expected, eval_provider
):
    """Decoded callable is behaviorally equivalent to the original."""
    enc = Encodable.define(ty, ctx)
    with handler(eval_provider):
        decoded = enc.decode(enc.encode(func))
        assert decoded(*args) == expected


@pytest.mark.parametrize("ty,func,ctx,args,expected", CALLABLE_CASES)
@pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
def test_callable_full_pipeline_behavioral(
    ty, func, ctx, args, expected, eval_provider
):
    """Full encode->serialize->deserialize->decode pipeline is behaviorally equivalent."""
    enc = Encodable.define(ty, ctx)
    text = enc.serialize(enc.encode(func))[0]["text"]
    with handler(eval_provider):
        decoded = enc.decode(enc.deserialize(text))
    assert decoded(*args) == expected


# Callable error cases: (type, ctx, source, exc_type, match)
CALLABLE_ERROR_CASES = [
    pytest.param(
        Callable[..., int],
        {},
        SynthesizedFunction(module_code="x = 42"),
        ValueError,
        id="non-function-last-stmt",
    ),
    pytest.param(
        Callable[[int, int], int],
        {},
        SynthesizedFunction(module_code="def add(a: int) -> int:\n    return a"),
        ValueError,
        id="wrong-param-count",
    ),
    pytest.param(
        Callable[[int, int], int],
        {},
        SynthesizedFunction(
            module_code="def add(a: int, b: int) -> str:\n    return str(a + b)"
        ),
        TypeError,
        id="wrong-return-type",
    ),
    pytest.param(
        Callable[[int, int], int],
        {},
        SynthesizedFunction(module_code="def add(a: int, b: int):\n    return a + b"),
        ValueError,
        id="missing-return-annotation",
    ),
]


@pytest.mark.parametrize("ty,ctx,source,exc_type", CALLABLE_ERROR_CASES)
@pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
def test_callable_decode_rejects_invalid(ty, ctx, source, exc_type, eval_provider):
    enc = Encodable.define(ty, ctx)
    with pytest.raises(exc_type):
        with handler(eval_provider):
            enc.decode(source)


def test_callable_encode_non_callable():
    enc = Encodable.define(Callable[..., int], {})
    with pytest.raises(TypeError):
        enc.encode("not a callable")


def test_callable_encode_no_source_no_docstring():

    class _NoDocCallable:
        __name__ = "nodoc"
        __doc__ = None

        def __call__(self):
            pass

    enc = Encodable.define(Callable[..., int], {})
    with pytest.raises(ValueError):
        enc.encode(_NoDocCallable())


# ---------------------------------------------------------------------------
# Provider integration tests
# ---------------------------------------------------------------------------

_tuple_schema_bug_xfail = pytest.mark.xfail(
    reason="Known tuple schema bug; expected to fail until fixed."
)
_provider_response_format_xfail = pytest.mark.xfail(
    reason="Known OpenAI/LiteLLM response_format limitation for this type."
)


def _provider_case_marks(case_id: str) -> list[pytest.MarkDecorator]:
    marks: list[pytest.MarkDecorator] = []
    if case_id.startswith("tuple-") or case_id in {
        "dc-with-tuple",
        "nt-coord",
        "nt-person",
    }:
        marks.append(_tuple_schema_bug_xfail)
    if case_id.startswith(("list-", "img-", "tool-", "dtc-")):
        marks.append(_provider_response_format_xfail)
    return marks


def _cases_with_provider_xfails(cases: list[Any]) -> list[Any]:
    out: list[Any] = []
    for c in cases:
        case_id = c.id if isinstance(c.id, str) else None
        if case_id is None:
            out.append(c)
            continue
        marks = [*c.marks, *_provider_case_marks(case_id)]
        if marks == list(c.marks):
            out.append(c)
            continue
        out.append(pytest.param(*c.values, id=case_id, marks=marks))
    return out


PROVIDER_CASES = _cases_with_provider_xfails(ROUNDTRIP_CASES)


def _encode_tool_spec(tool: Tool[..., Any]) -> dict[str, Any]:
    tool_ty: type[Any] = type(tool)
    tool_enc: Encodable[Any, Any] = Encodable.define(tool_ty)
    tool_spec_obj = tool_enc.encode(tool)
    if isinstance(tool_spec_obj, Mapping):
        return dict(tool_spec_obj)
    elif hasattr(tool_spec_obj, "model_dump"):
        return dict(tool_spec_obj.model_dump())
    raise TypeError(f"Unexpected encoded tool spec type: {type(tool_spec_obj)}")


@requires_openai
@pytest.mark.parametrize("ty,_value,ctx", PROVIDER_CASES)
def test_litellm_completion_accepts_encodable_response_model_for_supported_types(
    ty: Any, _value: Any, ctx: Mapping[str, Any] | None
) -> None:
    enc = Encodable.define(ty, ctx)
    kwargs: dict[str, Any] = {
        "model": CHEAP_MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"Return an instance of {getattr(ty, '__name__', repr(ty))}.",
            }
        ],
        "max_tokens": 200,
    }
    if enc.enc is not str:
        kwargs["response_format"] = enc.enc
    response = litellm.completion(**kwargs)
    assert response is not None

    content = response.choices[0].message.content
    assert content is not None, (
        f"Expected content in response for {getattr(ty, '__name__', repr(ty))}"
    )

    deserialized = enc.deserialize(content)
    pydantic.TypeAdapter(enc.enc).validate_python(deserialized)

    decoded = enc.decode(deserialized)
    pydantic.TypeAdapter(enc.base).validate_python(decoded)


@requires_openai
@pytest.mark.parametrize("ty,_value,ctx", PROVIDER_CASES)
def test_litellm_completion_accepts_tool_with_type_as_param(
    ty: Any, _value: Any, ctx: Mapping[str, Any] | None
) -> None:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", getattr(ty, "__name__", repr(ty)))

    def _fn(value):
        raise RuntimeError("should not be called")

    _fn.__name__ = f"accept_{name}"
    _fn.__doc__ = f"Accept a value of type {name}."
    _fn.__annotations__ = {"value": ty, "return": None}

    tool: Tool[..., Any] = Tool.define(_fn)
    response = litellm.completion(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": "Return hello, do NOT call any tools."}],
        tools=[_encode_tool_spec(tool)],
        tool_choice="none",
        max_tokens=200,
    )
    assert response is not None


@requires_openai
@pytest.mark.parametrize("ty,_value,ctx", PROVIDER_CASES)
def test_litellm_completion_accepts_tool_with_type_as_return(
    ty: Any, _value: Any, ctx: Mapping[str, Any] | None
) -> None:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", getattr(ty, "__name__", repr(ty)))

    def _fn():
        raise RuntimeError("should not be called")

    _fn.__name__ = f"return_{name}"
    _fn.__doc__ = f"Return a value of type {name}."
    _fn.__annotations__ = {"return": ty}

    tool: Tool[..., Any] = Tool.define(_fn)
    response = litellm.completion(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": "Return hello, do NOT call any tools."}],
        tools=[_encode_tool_spec(tool)],
        tool_choice="none",
        max_tokens=200,
    )
    assert response is not None
>>>>>>> 68d7645f081b17247fde3494e548fd16f92694e8
