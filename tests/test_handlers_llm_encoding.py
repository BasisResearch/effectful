import builtins
import inspect
import json
import typing
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Annotated, Any, NamedTuple, TypedDict

import pydantic
import pytest
from PIL import Image
from RestrictedPython import RestrictingNodeTransformer

from effectful.handlers.llm.encoding import (
    DecodedToolCall,
    Encodable,
    SynthesizedFunction,
    ToolCallEncodable,
    ToolEncodable,
    _param_model,
)
from effectful.handlers.llm.evaluation import RestrictedEvalProvider, UnsafeEvalProvider
from effectful.handlers.llm.template import Tool
from effectful.internals.unification import nested_type
from effectful.ops.semantics import handler
from effectful.ops.types import Operation, Term

# Eval providers for parameterized tests
EVAL_PROVIDERS = [
    pytest.param(UnsafeEvalProvider(), id="unsafe"),
    pytest.param(RestrictedEvalProvider(), id="restricted"),
]


def test_type_to_encodable_type_term():
    with pytest.raises(TypeError):
        Encodable.define(Term)


def test_type_to_encodable_type_operation():
    with pytest.raises(TypeError):
        Encodable.define(Operation)


def test_type_to_encodable_type_str():
    encodable = Encodable.define(str)
    encoded = encodable.encode("hello")
    decoded = encodable.decode(encoded)
    assert decoded == "hello"
    Model = pydantic.create_model("Model", value=encodable.enc)
    decoded = Model.model_validate({"value": "hello"})
    assert decoded.value == "hello"


def test_type_to_encodable_type_int():
    encodable = Encodable.define(int)
    encoded = encodable.encode(42)
    decoded = encodable.decode(encoded)
    assert decoded == 42
    assert isinstance(decoded, int)
    Model = pydantic.create_model("Model", value=encodable.enc)
    decoded = Model.model_validate({"value": 42})
    assert decoded.value == 42
    assert isinstance(decoded.value, int)


def test_type_to_encodable_type_bool():
    encodable = Encodable.define(bool)
    encoded = encodable.encode(True)
    decoded = encodable.decode(encoded)
    assert decoded is True
    assert isinstance(decoded, bool)
    encoded_false = encodable.encode(False)
    decoded_false = encodable.decode(encoded_false)
    assert decoded_false is False
    Model = pydantic.create_model("Model", value=encodable.enc)
    decoded = Model.model_validate({"value": True})
    assert decoded.value is True
    assert isinstance(decoded.value, bool)


def test_type_to_encodable_type_float():
    encodable = Encodable.define(float)
    encoded = encodable.encode(3.14)
    decoded = encodable.decode(encoded)
    assert decoded == 3.14
    assert isinstance(decoded, float)
    Model = pydantic.create_model("Model", value=encodable.enc)
    decoded = Model.model_validate({"value": 3.14})
    assert decoded.value == 3.14
    assert isinstance(decoded.value, float)


def test_type_to_encodable_type_image():
    encodable = Encodable.define(Image.Image)
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
    Model = pydantic.create_model("Model", value=encodable.enc)
    decoded = Model.model_validate({"value": encoded})
    assert decoded.value["url"] == encoded["url"]
    assert decoded.value["detail"] == "auto"


def test_type_to_encodable_type_image_roundtrip():
    encodable = Encodable.define(Image.Image)
    original = Image.new("RGB", (20, 20), color="green")
    encoded = encodable.encode(original)
    decoded = encodable.decode(encoded)
    assert isinstance(decoded, Image.Image)
    assert decoded.size == original.size
    assert decoded.mode == original.mode


def test_type_to_encodable_type_image_decode_invalid_url():
    encodable = Encodable.define(Image.Image)
    encoded = {"url": "http://example.com/image.png", "detail": "auto"}
    with pytest.raises(RuntimeError, match="expected base64 encoded image as data uri"):
        encodable.decode(encoded)


def test_type_to_encodable_type_tuple():
    encodable = Encodable.define(tuple[int, str])
    value = (1, "test")
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, tuple)
    assert decoded[0] == 1
    assert decoded[1] == "test"
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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
    encodable = Encodable.define(tuple[()])
    value = ()
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, tuple)
    assert len(decoded) == 0
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
    model_instance = Model.model_validate({"value": encoded})
    assert model_instance.value == encoded
    assert isinstance(model_instance.value, tuple)
    assert len(model_instance.value) == 0
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == value
    assert isinstance(decoded_from_model, tuple)


def test_type_to_encodable_type_tuple_three_elements():
    encodable = Encodable.define(tuple[int, str, bool])
    value = (42, "hello", True)
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, tuple)
    assert decoded[0] == 42
    assert decoded[1] == "hello"
    assert decoded[2] is True
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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
    encodable = Encodable.define(list[int])
    value = [1, 2, 3, 4, 5]
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, list)
    assert all(isinstance(elem, int) for elem in decoded)
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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
    encodable = Encodable.define(list[str])
    value = ["hello", "world", "test"]
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, list)
    assert all(isinstance(elem, str) for elem in decoded)
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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

    encodable = Encodable.define(Point)
    point = Point(10, 20)
    encoded = encodable.encode(point)
    decoded = encodable.decode(encoded)
    assert decoded == point
    assert isinstance(decoded, Point)
    assert decoded.x == 10
    assert decoded.y == 20
    Model = pydantic.create_model("Model", value=encodable.enc)
    decoded = Model.model_validate({"value": {"x": 10, "y": 20}})
    assert decoded.value == point
    assert isinstance(decoded.value, Point)


def test_type_to_encodable_type_namedtuple_with_str():
    class Person(NamedTuple):
        name: str
        age: int

    encodable = Encodable.define(Person)
    person = Person("Alice", 30)
    encoded = encodable.encode(person)
    decoded = encodable.decode(encoded)
    assert decoded == person
    assert isinstance(decoded, Person)
    assert decoded.name == "Alice"
    assert decoded.age == 30
    Model = pydantic.create_model("Model", value=encodable.enc)
    decoded = Model.model_validate({"value": {"name": "Alice", "age": 30}})
    assert decoded.value == person
    assert isinstance(decoded.value, Person)


def test_type_to_encodable_type_typeddict():
    class User(TypedDict):
        name: str
        age: int

    encodable = Encodable.define(User)
    user = User(name="Bob", age=25)
    encoded = encodable.encode(user)
    decoded = encodable.decode(encoded)
    assert decoded == user
    assert isinstance(decoded, dict)
    assert decoded["name"] == "Bob"
    assert decoded["age"] == 25
    Model = pydantic.create_model("Model", value=encodable.enc)
    decoded = Model.model_validate({"value": {"name": "Bob", "age": 25}})
    assert decoded.value == user
    assert isinstance(decoded.value, dict)


def test_type_to_encodable_type_typeddict_optional():
    class Config(TypedDict, total=False):
        host: str
        port: int

    encodable = Encodable.define(Config)
    config = Config(host="localhost", port=8080)
    encoded = encodable.encode(config)
    decoded = encodable.decode(encoded)
    assert decoded == config
    assert decoded["host"] == "localhost"
    assert decoded["port"] == 8080
    Model = pydantic.create_model("Model", value=encodable.enc)
    decoded = Model.model_validate({"value": {"host": "localhost", "port": 8080}})
    assert decoded.value == config
    assert isinstance(decoded.value, dict)


def test_type_to_encodable_type_complex():
    encodable = Encodable.define(complex)
    value = 3 + 4j
    encoded = encodable.encode(value)
    decoded = encodable.decode(encoded)
    assert decoded == value
    assert isinstance(decoded, complex)
    assert decoded.real == 3.0
    assert decoded.imag == 4.0
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
    model_instance = Model.model_validate({"value": encoded})
    assert model_instance.value == encoded
    # Decode from model
    decoded_from_model = encodable.decode(model_instance.value)
    assert decoded_from_model == value
    assert isinstance(decoded_from_model, complex)


def test_type_to_encodable_type_tuple_of_images():
    encodable = Encodable.define(tuple[Image.Image, Image.Image])
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
    Model = pydantic.create_model("Model", value=encodable.enc)
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
    encodable = Encodable.define(list[Image.Image])
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
    Model = pydantic.create_model("Model", value=encodable.enc)
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


def test_type_to_encodable_nested_type_mutable_sequence_images():
    """Encodable.define(nested_type(images).value) preserves image list behavior."""
    images = [
        Image.new("RGB", (10, 10), color="red"),
        Image.new("RGB", (20, 20), color="blue"),
        Image.new("RGB", (30, 30), color="green"),
    ]

    inferred_type = nested_type(images).value
    encodable = Encodable.define(inferred_type)

    encoded = encodable.encode(images)
    serialized = encodable.serialize(encoded)

    assert len(serialized) == 3
    for i, block in enumerate(serialized):
        assert block["type"] == "image_url", (
            f"Expected image_url block at index {i}, got {block['type']}"
        )
        assert "image_url" in block
        assert block["image_url"]["url"].startswith("data:image/png;base64,")
        assert block["image_url"]["detail"] == "auto"

    # Round-trip through decode
    decoded = encodable.decode(encoded)
    assert len(decoded) == 3
    assert decoded[0].size == (10, 10)
    assert decoded[1].size == (20, 20)
    assert decoded[2].size == (30, 30)

    # Empty list
    empty_encoded = encodable.encode([])
    empty_serialized = encodable.serialize(empty_encoded)
    assert empty_serialized == []


def test_type_to_encodable_type_dataclass():
    @dataclass
    class Point:
        x: int
        y: int

    encodable = Encodable.define(Point)
    point = Point(10, 20)
    encoded = encodable.encode(point)
    decoded = encodable.decode(encoded)
    assert decoded == point
    assert isinstance(decoded, Point)
    assert decoded.x == 10
    assert decoded.y == 20
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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

    encodable = Encodable.define(Person)
    person = Person("Alice", 30)
    encoded = encodable.encode(person)
    decoded = encodable.decode(encoded)
    assert decoded == person
    assert isinstance(decoded, Person)
    assert decoded.name == "Alice"
    assert decoded.age == 30
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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

    encodable = Encodable.define(Container)
    container = Container(items=[1, 2, 3], name="test")
    encoded = encodable.encode(container)
    decoded = encodable.decode(encoded)
    assert decoded == container
    assert isinstance(decoded, Container)
    assert decoded.items == [1, 2, 3]
    assert decoded.name == "test"
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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

    encodable = Encodable.define(Pair)
    pair = Pair(values=(42, "hello"), count=2)
    encoded = encodable.encode(pair)
    decoded = encodable.decode(encoded)
    assert decoded == pair
    assert isinstance(decoded, Pair)
    assert decoded.values == (42, "hello")
    assert decoded.count == 2
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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

    encodable = Encodable.define(Config)
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
    Model = pydantic.create_model("Model", value=encodable.enc)
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

    encodable = Encodable.define(Person)
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
    Model = pydantic.create_model("Model", value=encodable.enc)
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

    encodable = Encodable.define(Point)
    point = Point(x=10, y=20)
    encoded = encodable.encode(point)
    decoded = encodable.decode(encoded)
    assert decoded == point
    assert isinstance(decoded, Point)
    assert decoded.x == 10
    assert decoded.y == 20
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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

    encodable = Encodable.define(Person)
    person = Person(name="Alice", age=30)
    encoded = encodable.encode(person)
    decoded = encodable.decode(encoded)
    assert decoded == person
    assert isinstance(decoded, Person)
    assert decoded.name == "Alice"
    assert decoded.age == 30
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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

    encodable = Encodable.define(Container)
    container = Container(items=[1, 2, 3], name="test")
    encoded = encodable.encode(container)
    decoded = encodable.decode(encoded)
    assert decoded == container
    assert isinstance(decoded, Container)
    assert decoded.items == [1, 2, 3]
    assert decoded.name == "test"
    # Test with pydantic model validation
    Model = pydantic.create_model("Model", value=encodable.enc)
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

    encodable = Encodable.define(Person)
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
    Model = pydantic.create_model("Model", value=encodable.enc)
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


# ============================================================================
# Tests for serialize/deserialize contract
# ============================================================================


class TestEncodableSerializeDeserialize:
    """Systematic tests for the serialize/deserialize contract.

    The serialize/deserialize path is the LLM communication interface:
    serialize(encode(value)) produces content blocks sent to the LLM,
    and deserialize(response_string) parses the LLM's response string.
    The full roundtrip encode → serialize → deserialize → decode must
    recover the original value for every Encodable type.
    """

    # -- str (StrEncodable): serializes without JSON encoding --

    def test_str_serialize_plain_text(self):
        """str serializes as plain text, NOT JSON-encoded with quotes."""
        enc = Encodable.define(str)
        blocks = enc.serialize(enc.encode("hello"))
        assert blocks == [{"type": "text", "text": "hello"}]

    def test_str_serialize_preserves_special_chars(self):
        """str serialization preserves quotes, newlines, etc."""
        enc = Encodable.define(str)
        value = 'line1\nline2\t"quoted"'
        blocks = enc.serialize(enc.encode(value))
        assert blocks[0]["text"] == value

    def test_str_deserialize_identity(self):
        """str deserialization is identity — no JSON parsing."""
        enc = Encodable.define(str)
        assert enc.deserialize("hello") == "hello"

    def test_str_deserialize_json_like_not_parsed(self):
        """str containing JSON-like content is NOT parsed as JSON."""
        enc = Encodable.define(str)
        value = '{"key": "value"}'
        assert enc.deserialize(value) == value

    def test_str_roundtrip(self):
        enc = Encodable.define(str)
        for value in ["hello", "", "with spaces", '{"json": true}', "line1\nline2"]:
            text = enc.serialize(enc.encode(value))[0]["text"]
            assert enc.decode(enc.deserialize(text)) == value

    # -- int (BaseEncodable) --

    def test_int_serialize(self):
        enc = Encodable.define(int)
        assert enc.serialize(enc.encode(42)) == [{"type": "text", "text": "42"}]

    def test_int_serialize_negative(self):
        enc = Encodable.define(int)
        assert enc.serialize(enc.encode(-7)) == [{"type": "text", "text": "-7"}]

    def test_int_deserialize(self):
        enc = Encodable.define(int)
        assert enc.deserialize("42") == 42
        assert enc.deserialize("-7") == -7
        assert enc.deserialize("0") == 0

    def test_int_deserialize_rejects_string(self):
        enc = Encodable.define(int)
        with pytest.raises(pydantic.ValidationError):
            enc.deserialize('"hello"')

    def test_int_roundtrip(self):
        enc = Encodable.define(int)
        for value in [42, -7, 0, 999999]:
            text = enc.serialize(enc.encode(value))[0]["text"]
            assert enc.decode(enc.deserialize(text)) == value

    # -- bool (BaseEncodable) --

    def test_bool_serialize(self):
        enc = Encodable.define(bool)
        assert enc.serialize(enc.encode(True)) == [{"type": "text", "text": "true"}]
        assert enc.serialize(enc.encode(False)) == [{"type": "text", "text": "false"}]

    def test_bool_deserialize(self):
        enc = Encodable.define(bool)
        assert enc.deserialize("true") is True
        assert enc.deserialize("false") is False

    def test_bool_roundtrip(self):
        enc = Encodable.define(bool)
        for value in [True, False]:
            text = enc.serialize(enc.encode(value))[0]["text"]
            assert enc.decode(enc.deserialize(text)) is value

    # -- float (BaseEncodable) --

    def test_float_serialize(self):
        enc = Encodable.define(float)
        assert enc.serialize(enc.encode(3.14)) == [{"type": "text", "text": "3.14"}]

    def test_float_deserialize(self):
        enc = Encodable.define(float)
        assert enc.deserialize("3.14") == 3.14

    def test_float_roundtrip(self):
        enc = Encodable.define(float)
        for value in [3.14, -2.5, 0.0]:
            text = enc.serialize(enc.encode(value))[0]["text"]
            assert enc.decode(enc.deserialize(text)) == value

    # -- dataclass (BaseEncodable) --

    def test_dataclass_serialize(self):
        @dataclass
        class Point:
            x: int
            y: int

        enc = Encodable.define(Point)
        text = enc.serialize(enc.encode(Point(1, 2)))[0]["text"]
        assert json.loads(text) == {"x": 1, "y": 2}

    def test_dataclass_deserialize(self):
        @dataclass
        class Point:
            x: int
            y: int

        enc = Encodable.define(Point)
        assert enc.deserialize('{"x": 1, "y": 2}') == Point(1, 2)

    def test_dataclass_roundtrip(self):
        @dataclass
        class Point:
            x: int
            y: int

        enc = Encodable.define(Point)
        original = Point(10, 20)
        text = enc.serialize(enc.encode(original))[0]["text"]
        assert enc.decode(enc.deserialize(text)) == original

    def test_dataclass_with_str_fields_roundtrip(self):
        @dataclass
        class Person:
            name: str
            age: int

        enc = Encodable.define(Person)
        original = Person("Alice", 30)
        text = enc.serialize(enc.encode(original))[0]["text"]
        assert enc.decode(enc.deserialize(text)) == original

    def test_dataclass_with_optional_roundtrip(self):
        @dataclass
        class Config:
            host: str
            port: int
            timeout: float | None = None

        enc = Encodable.define(Config)
        for original in [Config("localhost", 8080, 5.0), Config("localhost", 8080)]:
            text = enc.serialize(enc.encode(original))[0]["text"]
            assert enc.decode(enc.deserialize(text)) == original

    def test_nested_dataclass_roundtrip(self):
        @dataclass
        class Address:
            street: str
            city: str

        @dataclass
        class Person:
            name: str
            address: Address

        enc = Encodable.define(Person)
        original = Person("Bob", Address("123 Main", "NYC"))
        text = enc.serialize(enc.encode(original))[0]["text"]
        assert enc.decode(enc.deserialize(text)) == original

    def test_dataclass_with_list_field_roundtrip(self):
        @dataclass
        class Container:
            items: list[int]
            label: str

        enc = Encodable.define(Container)
        original = Container([1, 2, 3], "test")
        text = enc.serialize(enc.encode(original))[0]["text"]
        assert enc.decode(enc.deserialize(text)) == original

    # -- NamedTuple (BaseEncodable) --

    def test_namedtuple_serialize(self):
        class Coord(NamedTuple):
            x: int
            y: int

        enc = Encodable.define(Coord)
        text = enc.serialize(enc.encode(Coord(3, 4)))[0]["text"]
        # pydantic serializes NamedTuples as JSON arrays
        assert json.loads(text) == [3, 4]

    def test_namedtuple_roundtrip(self):
        class Coord(NamedTuple):
            x: int
            y: int

        enc = Encodable.define(Coord)
        original = Coord(3, 4)
        text = enc.serialize(enc.encode(original))[0]["text"]
        assert enc.decode(enc.deserialize(text)) == original

    # -- TypedDict --

    def test_typeddict_serialize(self):
        class Config(TypedDict):
            host: str
            port: int

        enc = Encodable.define(Config)
        text = enc.serialize(enc.encode(Config(host="localhost", port=8080)))[0]["text"]
        assert json.loads(text) == {"host": "localhost", "port": 8080}

    def test_typeddict_roundtrip(self):
        class Config(TypedDict):
            host: str
            port: int

        enc = Encodable.define(Config)
        original = Config(host="localhost", port=8080)
        text = enc.serialize(enc.encode(original))[0]["text"]
        assert enc.decode(enc.deserialize(text)) == original

    # -- pydantic BaseModel (PydanticBaseModelEncodable) --

    def test_pydantic_model_serialize(self):
        class User(pydantic.BaseModel):
            name: str
            age: int

        enc = Encodable.define(User)
        text = enc.serialize(enc.encode(User(name="Alice", age=30)))[0]["text"]
        assert json.loads(text) == {"name": "Alice", "age": 30}

    def test_pydantic_model_deserialize(self):
        class User(pydantic.BaseModel):
            name: str
            age: int

        enc = Encodable.define(User)
        result = enc.deserialize('{"name": "Alice", "age": 30}')
        assert result == User(name="Alice", age=30)

    def test_pydantic_model_roundtrip(self):
        class User(pydantic.BaseModel):
            name: str
            age: int

        enc = Encodable.define(User)
        original = User(name="Alice", age=30)
        text = enc.serialize(enc.encode(original))[0]["text"]
        assert enc.decode(enc.deserialize(text)) == original

    # -- tuple (TupleEncodable) --

    def test_tuple_serialize(self):
        enc = Encodable.define(tuple[int, str])
        text = enc.serialize(enc.encode((1, "hello")))[0]["text"]
        assert json.loads(text) == [1, "hello"]

    def test_tuple_roundtrip(self):
        enc = Encodable.define(tuple[int, str])
        original = (1, "hello")
        text = enc.serialize(enc.encode(original))[0]["text"]
        assert enc.decode(enc.deserialize(text)) == original

    # -- list (MutableSequenceEncodable) --

    def test_list_serialize(self):
        enc = Encodable.define(list[int])
        text = enc.serialize(enc.encode([1, 2, 3]))[0]["text"]
        assert json.loads(text) == [1, 2, 3]

    def test_list_roundtrip(self):
        enc = Encodable.define(list[int])
        original = [1, 2, 3]
        text = enc.serialize(enc.encode(original))[0]["text"]
        assert enc.decode(enc.deserialize(text)) == original

    def test_list_of_str_roundtrip(self):
        enc = Encodable.define(list[str])
        original = ["hello", "world"]
        text = enc.serialize(enc.encode(original))[0]["text"]
        assert enc.decode(enc.deserialize(text)) == original


class TestCallableEncodable:
    """Tests for CallableEncodable - encoding/decoding callables as SynthesizedFunction."""

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_encode_decode_function(self, eval_provider):
        def add(a: int, b: int) -> int:
            return a + b

        # Use typed Callable with matching signature
        encodable = Encodable.define(Callable[[int, int], int], {})
        encoded = encodable.encode(add)
        assert isinstance(encoded, SynthesizedFunction)
        assert "def add" in encoded.module_code
        assert "return a + b" in encoded.module_code

        with handler(eval_provider):
            decoded = encodable.decode(encoded)
        assert callable(decoded)
        assert decoded(2, 3) == 5
        assert decoded.__name__ == "add"

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_decode_with_ellipsis_params(self, eval_provider):
        # Callable[..., int] allows any params but validates return type
        encodable = Encodable.define(Callable[..., int], {})

        # Test decoding a function - must end with function def with return annotation
        func_source = SynthesizedFunction(
            module_code="def double(x) -> int:\n    return x * 2"
        )
        with handler(eval_provider):
            decoded = encodable.decode(func_source)
        assert callable(decoded)
        assert decoded(5) == 10

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_decode_with_env(self, eval_provider):
        # Test decoding a function that uses env variables
        encodable = Encodable.define(Callable[..., int], {"factor": 3})
        source = SynthesizedFunction(
            module_code="""def multiply(x) -> int:
    return x * factor"""
        )

        with handler(eval_provider):
            decoded = encodable.decode(source)
        assert callable(decoded)
        assert decoded(4) == 12

    def test_encode_non_callable_raises(self):
        encodable = Encodable.define(Callable[..., int], {})
        with pytest.raises(TypeError, match="Expected callable"):
            encodable.encode("not a callable")

    def test_encode_builtin_creates_stub(self):
        encodable = Encodable.define(Callable[..., int], {})
        # Built-in functions don't have source code but have docstrings
        encoded = encodable.encode(len)
        assert isinstance(encoded, SynthesizedFunction)
        assert "def len" in encoded.module_code
        assert '"""' in encoded.module_code  # docstring present
        assert "..." in encoded.module_code  # stub body

    def test_encode_builtin_no_docstring_raises(self):
        # Create a callable without source and without docstring
        class NoDocCallable:
            __name__ = "nodoc"
            __doc__ = None

            def __call__(self):
                pass

        encodable = Encodable.define(Callable[..., int], {})
        with pytest.raises(RuntimeError, match="no source code and no docstring"):
            encodable.encode(NoDocCallable())

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_decode_no_function_at_end_raises(self, eval_provider):
        encodable = Encodable.define(Callable[..., int], {})
        # Source code where last statement is not a function definition
        source = SynthesizedFunction(module_code="x = 42")
        with pytest.raises(
            ValueError, match="last statement to be a function definition"
        ):
            with handler(eval_provider):
                encodable.decode(source)

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_decode_multiple_functions_uses_last(self, eval_provider):
        encodable = Encodable.define(Callable[..., int], {})
        # Source code that defines multiple functions - should use the last one
        source = SynthesizedFunction(
            module_code="""def foo() -> int:
    return 1

def bar() -> int:
    return 2"""
        )
        with handler(eval_provider):
            decoded = encodable.decode(source)
        assert callable(decoded)
        assert decoded.__name__ == "bar"
        assert decoded() == 2

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_decode_class_raises(self, eval_provider):
        encodable = Encodable.define(Callable[..., int], {})
        # Classes are callable but the last statement must be a function definition
        source = SynthesizedFunction(
            module_code="""class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!\""""
        )

        with pytest.raises(
            ValueError, match="last statement to be a function definition"
        ):
            with handler(eval_provider):
                encodable.decode(source)

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_roundtrip(self, eval_provider):
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        encodable = Encodable.define(Callable[[str], str], {})
        with handler(eval_provider):
            encoded = encodable.encode(greet)
            decoded = encodable.decode(encoded)

        assert callable(decoded)
        assert decoded("Alice") == "Hello, Alice!"
        assert decoded.__name__ == "greet"

    def test_serialize_deserialize(self):
        def add(a: int, b: int) -> int:
            return a + b

        encodable = Encodable.define(Callable[[int, int], int], {})
        encoded = encodable.encode(add)

        # Test serialization
        serialized = encodable.serialize(encoded)
        assert len(serialized) == 1
        assert serialized[0]["type"] == "text"
        assert "module_code" in serialized[0]["text"]

        # Test deserialization
        deserialized = encodable.deserialize(serialized[0]["text"])
        assert isinstance(deserialized, SynthesizedFunction)
        assert "def add" in deserialized.module_code

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_decode_validates_last_statement(self, eval_provider):
        encodable = Encodable.define(Callable[..., int], {})

        # Helper function followed by assignment - should fail
        source = SynthesizedFunction(
            module_code="""def helper():
    return 42

result = helper()"""
        )
        with pytest.raises(
            ValueError, match="last statement to be a function definition"
        ):
            with handler(eval_provider):
                encodable.decode(source)

    def test_typed_callable_includes_signature_in_docstring(self):
        # Test that the enc type has the signature in its docstring
        encodable = Encodable.define(Callable[[int, int], int], {})
        assert encodable.enc.__doc__ is not None
        assert "Callable[[int, int], int]" in encodable.enc.__doc__
        assert "<signature>" in encodable.enc.__doc__

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_typed_callable_validates_param_count(self, eval_provider):
        encodable = Encodable.define(Callable[[int, int], int], {})

        # Function with wrong number of parameters
        source = SynthesizedFunction(
            module_code="""def add(a: int) -> int:
    return a"""
        )
        with pytest.raises(ValueError, match="expected function with 2 parameters"):
            with handler(eval_provider):
                encodable.decode(source)

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_typed_callable_validates_return_type(self, eval_provider):
        encodable = Encodable.define(Callable[[int, int], int], {})

        # Function with wrong return type
        source = SynthesizedFunction(
            module_code="""def add(a: int, b: int) -> str:
    return str(a + b)"""
        )
        with pytest.raises(TypeError, match="Incompatible types in assignment"):
            with handler(eval_provider):
                encodable.decode(source)

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_typed_callable_requires_return_annotation(self, eval_provider):
        encodable = Encodable.define(Callable[[int, int], int], {})

        # Function missing return type annotation
        source = SynthesizedFunction(
            module_code="""def add(a: int, b: int):
    return a + b"""
        )
        with pytest.raises(
            ValueError,
            match="requires synthesized function to have a return type annotation",
        ):
            with handler(eval_provider):
                encodable.decode(source)

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_typed_callable_accepts_correct_signature(self, eval_provider):
        encodable = Encodable.define(Callable[[int, int], int], {})

        # Function with correct signature
        source = SynthesizedFunction(
            module_code="""def add(a: int, b: int) -> int:
    return a + b"""
        )
        with handler(eval_provider):
            result = encodable.decode(source)
        assert callable(result)
        assert result(2, 3) == 5

    @pytest.mark.parametrize(
        "eval_provider", [pytest.param(UnsafeEvalProvider(), id="unsafe")]
    )
    def test_typed_callable_decode_when_source_uses_annotated(self, eval_provider):
        """Decoding works when synthesized module code uses typing.Annotated in the function."""
        encodable = Encodable.define(Callable[[int], int], {"typing": typing})
        source = SynthesizedFunction(
            module_code='def f(x: typing.Annotated[int, "positive"]) -> int:\n    return x'
        )
        with handler(eval_provider):
            result = encodable.decode(source)
        assert callable(result)
        assert result(42) == 42

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_typed_callable_decode_with_expected_annotated(self, eval_provider):
        """Decoding works when expected signature uses Annotated (stripped for typecheck stub)."""
        encodable = Encodable.define(
            Callable[[Annotated[int, "value"]], Annotated[int, "result"]], {}
        )
        source = SynthesizedFunction(module_code="def g(x: int) -> int:\n    return x")
        with handler(eval_provider):
            result = encodable.decode(source)
        assert callable(result)
        assert result(7) == 7

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_ellipsis_callable_skips_param_validation(self, eval_provider):
        # Callable[..., int] should skip param validation but still validate return
        encodable = Encodable.define(Callable[..., int], {})

        source = SynthesizedFunction(
            module_code="""def anything(a, b, c, d, e) -> int:
    return 42"""
        )
        with handler(eval_provider):
            result = encodable.decode(source)
        assert callable(result)
        assert result(1, 2, 3, 4, 5) == 42

    def test_typed_callable_json_schema_includes_signature(self):
        # Test that the JSON schema includes the type signature for the LLM
        encodable = Encodable.define(Callable[[int, int], int], {})

        # Get the JSON schema from the enc model
        schema = encodable.enc.model_json_schema()

        # The description should contain the type signature
        assert "description" in schema
        assert "Callable[[int, int], int]" in schema["description"]
        assert "<signature>" in schema["description"]
        assert "<instructions>" in schema["description"]

    def test_typed_callable_json_schema_different_signatures(self):
        # Test that different type signatures produce different schemas
        enc1 = Encodable.define(Callable[[str], str], {})
        enc2 = Encodable.define(Callable[[int, int, int], bool], {})

        schema1 = enc1.enc.model_json_schema()
        schema2 = enc2.enc.model_json_schema()

        assert "Callable[[str], str]" in schema1["description"]
        assert "Callable[[int, int, int], bool]" in schema2["description"]

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_validates_param_count_via_ast(self, eval_provider):
        # Test that param validation happens via AST analysis
        encodable = Encodable.define(Callable[[int, int], int], {})

        # Function with 3 params when 2 expected
        source = SynthesizedFunction(
            module_code="""def add(a: int, b: int, c: int) -> int:
    return a + b + c"""
        )
        with pytest.raises(ValueError, match="expected function with 2 parameters"):
            with handler(eval_provider):
                encodable.decode(source)

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_validates_param_count_zero_params(self, eval_provider):
        # Test callable with no params
        encodable = Encodable.define(Callable[[], int], {})

        # Function with params when 0 expected
        source = SynthesizedFunction(
            module_code="""def get_value(x: int) -> int:
    return x"""
        )
        with pytest.raises(ValueError, match="expected function with 0 parameters"):
            with handler(eval_provider):
                encodable.decode(source)

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_validates_accepts_zero_params(self, eval_provider):
        # Test callable with no params - correct signature
        encodable = Encodable.define(Callable[[], int], {})

        source = SynthesizedFunction(
            module_code="""def get_value() -> int:
    return 42"""
        )
        with handler(eval_provider):
            result = encodable.decode(source)
        assert callable(result)
        assert result() == 42

    def test_ellipsis_callable_json_schema_includes_signature(self):
        # Test that Callable[..., int] has signature in schema
        encodable = Encodable.define(Callable[..., int], {})

        schema = encodable.enc.model_json_schema()
        assert "description" in schema
        assert "Callable[[...], int]" in schema["description"]
        assert "<signature>" in schema["description"]

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_ellipsis_callable_validates_return_type(self, eval_provider):
        # Callable[..., int] should still validate return type
        encodable = Encodable.define(Callable[..., int], {})

        source = SynthesizedFunction(
            module_code="""def get_value() -> str:
    return "wrong type\""""
        )
        with pytest.raises(TypeError, match="Incompatible types in assignment"):
            with handler(eval_provider):
                encodable.decode(source)

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_callable_with_single_param(self, eval_provider):
        encodable = Encodable.define(Callable[[str], int], {})

        source = SynthesizedFunction(
            module_code="""def count_chars(s: str) -> int:
    return len(s)"""
        )
        with handler(eval_provider):
            result = encodable.decode(source)
        assert callable(result)
        assert result("hello") == 5

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_callable_with_many_params(self, eval_provider):
        encodable = Encodable.define(Callable[[int, int, int, int], int], {})

        source = SynthesizedFunction(
            module_code="""def sum_four(a: int, b: int, c: int, d: int) -> int:
    return a + b + c + d"""
        )
        with handler(eval_provider):
            result = encodable.decode(source)
        assert callable(result)
        assert result(1, 2, 3, 4) == 10

    @pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
    def test_callable_with_bool_return(self, eval_provider):
        encodable = Encodable.define(Callable[[int], bool], {})

        source = SynthesizedFunction(
            module_code="""def is_positive(x: int) -> bool:
    return x > 0"""
        )
        with handler(eval_provider):
            result = encodable.decode(source)
        assert callable(result)
        assert result(5) is True
        assert result(-1) is False

    def test_callable_type_variations_schema(self):
        # Test various callable type variations have correct schemas
        test_cases = [
            (Callable[[], int], "Callable[[], int]"),
            (Callable[[str], str], "Callable[[str], str]"),
            (Callable[[int, str], bool], "Callable[[int, str], bool]"),
            (Callable[..., int], "Callable[[...], int]"),
            (Callable[..., Any], "Callable[[...], Any]"),
        ]

        for callable_type, expected_sig in test_cases:
            encodable = Encodable.define(callable_type, {})
            schema = encodable.enc.model_json_schema()
            assert "description" in schema, f"No description for {callable_type}"
            assert expected_sig in schema["description"], (
                f"Expected {expected_sig} in schema for {callable_type}, "
                f"got: {schema['description'][:100]}..."
            )


class TestRestrictedEvalProviderConfig:
    """Tests for RestrictedEvalProvider configuration options."""

    def test_restricted_blocks_private_attribute_access(self):
        """RestrictedPython blocks access to underscore-prefixed attributes by default."""
        encodable = Encodable.define(Callable[[str], int], {})
        source = SynthesizedFunction(
            module_code="""def get_private(s: str) -> int:
    return s.__class__.__name__"""
        )
        # Should raise due to restricted attribute access
        with pytest.raises(Exception):  # Could be NameError or AttributeError
            with handler(RestrictedEvalProvider()):
                fn = encodable.decode(source)
                fn("test")

    def test_restricted_with_custom_policy(self):
        """Can pass custom policy via kwargs."""

        # Create a custom policy that's the same as default (just to test the plumbing)
        class CustomPolicy(RestrictingNodeTransformer):
            pass

        encodable = Encodable.define(Callable[[int, int], int], {})
        source = SynthesizedFunction(
            module_code="""def add(a: int, b: int) -> int:
    return a + b"""
        )
        with handler(RestrictedEvalProvider(policy=CustomPolicy)):
            fn = encodable.decode(source)
        assert fn(2, 3) == 5

    def test_builtins_in_env_does_not_bypass_security(self):
        """Including __builtins__ in env should not bypass RestrictedEvalProvider security.

        RestrictedEvalProvider explicitly filters out __builtins__ from the env
        to prevent callers from replacing the restricted builtins with full Python builtins.
        This test verifies that even if __builtins__ is passed in the context,
        dangerous operations remain blocked.
        """

        # Attempt to pass full builtins in the context, which should be filtered out
        dangerous_ctx = {"__builtins__": builtins.__dict__}

        # Test 1: open() should not be usable even with __builtins__ in context
        # The function may fail at compile/exec time or at call time, but either way
        # it should not be able to actually open files
        encodable_open = Encodable.define(Callable[[str], str], dangerous_ctx)
        source_open = SynthesizedFunction(
            module_code="""def read_file(path: str) -> str:
    return open(path).read()"""
        )
        with pytest.raises(Exception):  # Could be NameError, ValueError, or other
            with handler(RestrictedEvalProvider()):
                fn = encodable_open.decode(source_open)
                # If decode succeeded (shouldn't), calling should still fail
                fn("/etc/passwd")

        # Test 2: __import__ should not be usable
        encodable_import = Encodable.define(Callable[[], str], dangerous_ctx)
        source_import = SynthesizedFunction(
            module_code="""def get_os_name() -> str:
    os = __import__('os')
    return os.name"""
        )
        with pytest.raises(Exception):
            with handler(RestrictedEvalProvider()):
                fn = encodable_import.decode(source_import)
                fn()

        # Test 3: Verify safe code still works with dangerous context
        # This confirms we're not just breaking everything
        encodable_safe = Encodable.define(Callable[[int, int], int], dangerous_ctx)
        source_safe = SynthesizedFunction(
            module_code="""def add(a: int, b: int) -> int:
    return a + b"""
        )
        with handler(RestrictedEvalProvider()):
            fn = encodable_safe.decode(source_safe)
            assert fn(2, 3) == 5, "Safe code should still work"

        # Test 4: Private attribute access should still be blocked
        encodable_private = Encodable.define(Callable[[str], str], dangerous_ctx)
        source_private = SynthesizedFunction(
            module_code="""def get_class(s: str) -> str:
    return s.__class__.__name__"""
        )
        with pytest.raises(Exception):
            with handler(RestrictedEvalProvider()):
                fn = encodable_private.decode(source_private)
                fn("test")


# ============================================================================
# Tests for ToolEncodable
# ============================================================================


class TestToolEncodable:
    """Tests for ToolEncodable - encoding/decoding Tools as ChatCompletionToolParam."""

    def test_define_returns_tool_encodable(self):
        """Encodable.define(Tool) returns a ToolEncodable instance."""
        encodable = Encodable.define(Tool)
        assert isinstance(encodable, ToolEncodable)

    def test_encode_simple_tool(self):
        """Encoding a simple Tool produces correct ChatCompletionToolParam structure."""

        @Tool.define
        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        encodable = Encodable.define(type(add))
        encoded = encodable.encode(add)

        assert encoded["type"] == "function"
        assert encoded["function"]["name"] == "add"
        assert "Add two numbers together." in encoded["function"]["description"]
        assert encoded["function"]["strict"] is True
        assert "parameters" in encoded["function"]
        params = encoded["function"]["parameters"]
        assert "a" in params.get("properties", {})
        assert "b" in params.get("properties", {})

    def test_encode_tool_with_str_param(self):
        """Tool with string parameter encodes correctly."""

        @Tool.define
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"

        encodable = Encodable.define(type(greet))
        encoded = encodable.encode(greet)

        assert encoded["type"] == "function"
        assert encoded["function"]["name"] == "greet"
        params = encoded["function"]["parameters"]
        assert "name" in params.get("properties", {})

    def test_encode_tool_no_params(self):
        """Tool with no parameters encodes correctly."""

        @Tool.define
        def get_value() -> int:
            """Return a constant value."""
            return 42

        encodable = Encodable.define(type(get_value))
        encoded = encodable.encode(get_value)

        assert encoded["type"] == "function"
        assert encoded["function"]["name"] == "get_value"

    def test_decode_raises_not_implemented(self):
        """Decoding a Tool from ChatCompletionToolParam is not supported."""
        encodable = Encodable.define(Tool)
        with pytest.raises(NotImplementedError, match="Tools cannot yet be decoded"):
            encodable.decode({})

    def test_serialize_deserialize_roundtrip(self):
        """Serialization and deserialization of encoded Tool roundtrips."""

        @Tool.define
        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        encodable = Encodable.define(type(multiply))
        encoded = encodable.encode(multiply)
        serialized = encodable.serialize(encoded)

        assert len(serialized) == 1
        assert serialized[0]["type"] == "text"

        deserialized = encodable.deserialize(serialized[0]["text"])
        assert deserialized["type"] == "function"
        assert deserialized["function"]["name"] == "multiply"


# ============================================================================
# Tests for ToolCallEncodable
# ============================================================================


class TestToolCallEncodable:
    """Tests for ToolCallEncodable - encoding/decoding DecodedToolCall."""

    def test_define_returns_tool_call_encodable(self):
        """Encodable.define(DecodedToolCall) returns a ToolCallEncodable instance."""
        encodable = Encodable.define(DecodedToolCall, {})
        assert isinstance(encodable, ToolCallEncodable)

    def test_encode_decoded_tool_call(self):
        """Encoding a DecodedToolCall produces a ChatCompletionMessageToolCall."""

        @Tool.define
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        sig = inspect.signature(add)
        bound = sig.bind(3, 5)
        dtc = DecodedToolCall(tool=add, bound_args=bound, id="call_123", name="add")

        encodable = Encodable.define(DecodedToolCall, {"add": add})
        encoded = encodable.encode(dtc)

        assert encoded.id == "call_123"
        assert encoded.function.name == "add"
        args = json.loads(encoded.function.arguments)
        assert args["a"] == 3
        assert args["b"] == 5

    def test_decode_tool_call(self):
        """Decoding a ChatCompletionMessageToolCall produces a DecodedToolCall."""
        from litellm import ChatCompletionMessageToolCall

        @Tool.define
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_456",
                "function": {
                    "name": "add",
                    "arguments": '{"a": 10, "b": 20}',
                },
            }
        )

        encodable = Encodable.define(DecodedToolCall, {"add": add})
        decoded = encodable.decode(tool_call)

        assert isinstance(decoded, DecodedToolCall)
        assert decoded.tool is add
        assert decoded.id == "call_456"
        assert decoded.bound_args.arguments == {"a": 10, "b": 20}

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding a DecodedToolCall roundtrips."""

        @Tool.define
        def greet(name: str) -> str:
            """Greet by name."""
            return f"Hello, {name}!"

        sig = inspect.signature(greet)
        bound = sig.bind("Alice")
        original = DecodedToolCall(
            tool=greet, bound_args=bound, id="call_rt", name="greet"
        )

        ctx = {"greet": greet}
        encodable = Encodable.define(DecodedToolCall, ctx)
        encoded = encodable.encode(original)
        decoded = encodable.decode(encoded)

        assert decoded.tool is greet
        assert decoded.id == "call_rt"
        assert decoded.bound_args.arguments == {"name": "Alice"}

    def test_decode_unknown_tool_raises(self):
        """Decoding a tool call with a name not in context raises."""
        from litellm import ChatCompletionMessageToolCall

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_bad",
                "function": {
                    "name": "nonexistent",
                    "arguments": "{}",
                },
            }
        )

        encodable = Encodable.define(DecodedToolCall, {})
        with pytest.raises((KeyError, AssertionError)):
            encodable.decode(tool_call)

    def test_serialize_deserialize(self):
        """Serialization and deserialization of an encoded tool call."""
        from litellm import ChatCompletionMessageToolCall

        @Tool.define
        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        sig = inspect.signature(double)
        bound = sig.bind(7)
        dtc = DecodedToolCall(
            tool=double, bound_args=bound, id="call_ser", name="double"
        )

        encodable = Encodable.define(DecodedToolCall, {"double": double})
        encoded = encodable.encode(dtc)
        serialized = encodable.serialize(encoded)

        assert len(serialized) == 1
        assert serialized[0]["type"] == "text"

        deserialized = encodable.deserialize(serialized[0]["text"])
        assert isinstance(deserialized, ChatCompletionMessageToolCall)
        assert deserialized.function.name == "double"
        assert deserialized.id == "call_ser"

    def test_decode_tool_with_str_args(self):
        """Decoding a tool call with string arguments works correctly."""
        from litellm import ChatCompletionMessageToolCall

        @Tool.define
        def concat(a: str, b: str) -> str:
            """Concatenate two strings."""
            return a + b

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_str",
                "function": {
                    "name": "concat",
                    "arguments": '{"a": "hello", "b": " world"}',
                },
            }
        )

        encodable = Encodable.define(DecodedToolCall, {"concat": concat})
        decoded = encodable.decode(tool_call)

        assert decoded.bound_args.arguments == {"a": "hello", "b": " world"}


# ============================================================================
# Tests for _param_model
# ============================================================================


class TestParamModel:
    """Tests for _param_model helper that builds pydantic models from signatures."""

    def test_creates_model_from_signature(self):
        @Tool.define
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        sig = inspect.signature(add)
        model = _param_model(sig)

        assert issubclass(model, pydantic.BaseModel)
        instance = model.model_validate({"a": 1, "b": 2})
        assert instance.a == 1
        assert instance.b == 2

    def test_forbids_extra_fields(self):
        @Tool.define
        def f(x: int) -> int:
            """Identity."""
            return x

        sig = inspect.signature(f)
        model = _param_model(sig)

        with pytest.raises(pydantic.ValidationError):
            model.model_validate({"x": 1, "extra": "bad"})

    def test_model_with_str_param(self):
        @Tool.define
        def echo(msg: str) -> str:
            """Echo."""
            return msg

        sig = inspect.signature(echo)
        model = _param_model(sig)
        instance = model.model_validate({"msg": "hello"})
        assert instance.msg == "hello"


# ============================================================================
# Tests for ToolCallEncodable argument validation
# ============================================================================


class TestToolCallEncodableArgValidation:
    """Tests that ToolCallEncodable.decode validates arguments against the tool signature."""

    def test_decode_wrong_arg_type_raises(self):
        """Passing a string where int is expected should fail pydantic validation."""
        from litellm import ChatCompletionMessageToolCall

        @Tool.define
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_type",
                "function": {
                    "name": "add",
                    "arguments": '{"a": "not_an_int", "b": 2}',
                },
            }
        )

        encodable = Encodable.define(DecodedToolCall, {"add": add})
        with pytest.raises(pydantic.ValidationError):
            encodable.decode(tool_call)

    def test_decode_missing_required_arg_raises(self):
        """Omitting a required argument should fail validation."""
        from litellm import ChatCompletionMessageToolCall

        @Tool.define
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_missing",
                "function": {
                    "name": "add",
                    "arguments": '{"a": 1}',
                },
            }
        )

        encodable = Encodable.define(DecodedToolCall, {"add": add})
        # Either pydantic rejects the missing field or sig.bind fails
        with pytest.raises((pydantic.ValidationError, TypeError)):
            encodable.decode(tool_call)

    def test_decode_extra_arg_raises(self):
        """Extra arguments not in the signature should fail (extra='forbid')."""
        from litellm import ChatCompletionMessageToolCall

        @Tool.define
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_extra",
                "function": {
                    "name": "add",
                    "arguments": '{"a": 1, "b": 2, "c": 3}',
                },
            }
        )

        encodable = Encodable.define(DecodedToolCall, {"add": add})
        with pytest.raises(pydantic.ValidationError):
            encodable.decode(tool_call)

    def test_decode_bool_not_coerced_to_int(self):
        """Bool arg where int is expected: pydantic may coerce, verify it doesn't crash."""
        from litellm import ChatCompletionMessageToolCall

        @Tool.define
        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_bool",
                "function": {
                    "name": "double",
                    "arguments": '{"x": true}',
                },
            }
        )

        encodable = Encodable.define(DecodedToolCall, {"double": double})
        decoded = encodable.decode(tool_call)
        assert decoded.bound_args.arguments["x"] in (True, 1)

    def test_decode_complex_param_types(self):
        """Tool with list and nested types decodes correctly."""
        from litellm import ChatCompletionMessageToolCall

        @Tool.define
        def process(items: list[int], label: str) -> str:
            """Process items."""
            return f"{label}: {sum(items)}"

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_complex",
                "function": {
                    "name": "process",
                    "arguments": '{"items": [1, 2, 3], "label": "total"}',
                },
            }
        )

        encodable = Encodable.define(DecodedToolCall, {"process": process})
        decoded = encodable.decode(tool_call)
        assert decoded.bound_args.arguments["items"] == [1, 2, 3]
        assert decoded.bound_args.arguments["label"] == "total"

    def test_decode_list_with_wrong_element_type_raises(self):
        """list[int] parameter receiving list of strings should fail."""
        from litellm import ChatCompletionMessageToolCall

        @Tool.define
        def sum_items(items: list[int]) -> int:
            """Sum items."""
            return sum(items)

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_bad_list",
                "function": {
                    "name": "sum_items",
                    "arguments": '{"items": ["a", "b", "c"]}',
                },
            }
        )

        encodable = Encodable.define(DecodedToolCall, {"sum_items": sum_items})
        with pytest.raises(pydantic.ValidationError):
            encodable.decode(tool_call)

    def test_decode_invalid_json_raises(self):
        """Malformed JSON in arguments should fail."""
        from litellm import ChatCompletionMessageToolCall

        @Tool.define
        def f(x: int) -> int:
            """Identity."""
            return x

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_bad_json",
                "function": {
                    "name": "f",
                    "arguments": "{not valid json}",
                },
            }
        )

        encodable = Encodable.define(DecodedToolCall, {"f": f})
        with pytest.raises(pydantic.ValidationError):
            encodable.decode(tool_call)

    def test_decode_pydantic_model_param(self):
        """Tool with a pydantic BaseModel parameter decodes through the Encodable chain."""
        from litellm import ChatCompletionMessageToolCall

        class Point(pydantic.BaseModel):
            x: int
            y: int

        @Tool.define
        def distance_from_origin(p: Point) -> float:
            """Compute distance from origin."""
            return (p.x**2 + p.y**2) ** 0.5

        tool_call = ChatCompletionMessageToolCall.model_validate(
            {
                "type": "tool_call",
                "id": "call_pydantic",
                "function": {
                    "name": "distance_from_origin",
                    "arguments": '{"p": {"x": 3, "y": 4}}',
                },
            }
        )

        encodable = Encodable.define(
            DecodedToolCall, {"distance_from_origin": distance_from_origin}
        )
        decoded = encodable.decode(tool_call)
        p = decoded.bound_args.arguments["p"]
        assert isinstance(p, Point)
        assert p.x == 3
        assert p.y == 4
