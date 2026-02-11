import builtins
import typing
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Annotated, Any, NamedTuple, TypedDict

import pydantic
import pytest
from PIL import Image
from RestrictedPython import RestrictingNodeTransformer

from effectful.handlers.llm.encoding import Encodable, SynthesizedFunction
from effectful.handlers.llm.evaluation import RestrictedEvalProvider, UnsafeEvalProvider
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


def test_type_to_encodable_bare_list_of_images():
    """Encodable.define(type(images)) handles list[Image.Image] via BareListEncodable.

    Regression test for GitHub issue #552.  ``type(obj)`` on a plain list
    erases generic args.  BareListEncodable infers element types at encode
    time so images still produce ``image_url`` content blocks.
    """
    images = [
        Image.new("RGB", (10, 10), color="red"),
        Image.new("RGB", (20, 20), color="blue"),
        Image.new("RGB", (30, 30), color="green"),
    ]

    # type(images) is bare `list` — no type args
    encodable = Encodable.define(type(images))

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


def test_type_to_encodable_nested_type_mutable_sequence_images():
    """Encodable.define(nested_type(images).value) preserves image list behavior."""
    images = [
        Image.new("RGB", (10, 10), color="red"),
        Image.new("RGB", (20, 20), color="blue"),
    ]

    inferred_type = nested_type(images).value
    encodable = Encodable.define(inferred_type)

    encoded = encodable.encode(images)
    serialized = encodable.serialize(encoded)
    assert len(serialized) == 2
    assert all(block["type"] == "image_url" for block in serialized)
    assert all(
        block["image_url"]["url"].startswith("data:image/png;base64,")
        for block in serialized
    )

    decoded = encodable.decode(encoded)
    assert isinstance(decoded, list)
    assert len(decoded) == 2
    assert all(isinstance(elem, Image.Image) for elem in decoded)
    assert decoded[0].size == (10, 10)
    assert decoded[1].size == (20, 20)


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
