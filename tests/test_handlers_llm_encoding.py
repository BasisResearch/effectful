from dataclasses import asdict, dataclass
from typing import NamedTuple, TypedDict

import pydantic
import pytest
from PIL import Image

from effectful.handlers.llm.encoding import Encodable
from effectful.handlers.llm.evaluation import UnsafeEvalProvider
from effectful.ops.semantics import handler
from effectful.ops.types import Operation, Term


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
    """Tests for CallableEncodable - encoding/decoding callables as source code."""

    def test_encode_decode_function(self):
        from collections.abc import Callable

        def add(a: int, b: int) -> int:
            return a + b

        encodable = Encodable.define(Callable, {})
        encoded = encodable.encode(add)
        assert isinstance(encoded, str)
        assert "def add" in encoded
        assert "return a + b" in encoded

        with handler(UnsafeEvalProvider()):
            decoded = encodable.decode(encoded)
        assert callable(decoded)
        assert decoded(2, 3) == 5
        assert decoded.__name__ == "add"

    def test_decode_lambda(self):
        from collections.abc import Callable

        # Lambdas should work if defined in a way that inspect.getsource can find them
        # Note: lambdas defined inline may not always have retrievable source
        encodable = Encodable.define(Callable, {})

        # Test decoding a lambda from source string
        lambda_source = "f = lambda x: x * 2"
        with handler(UnsafeEvalProvider()):
            decoded = encodable.decode(lambda_source)
        assert callable(decoded)
        assert decoded(5) == 10

    def test_decode_with_env(self):
        from collections.abc import Callable

        # Test decoding a function that uses env variables
        encodable = Encodable.define(Callable, {"factor": 3})
        source = """def multiply(x):
    return x * factor"""

        with handler(UnsafeEvalProvider()):
            decoded = encodable.decode(source)
        assert callable(decoded)
        assert decoded(4) == 12

    def test_encode_non_callable_raises(self):
        from collections.abc import Callable

        encodable = Encodable.define(Callable, {})
        with pytest.raises(TypeError, match="Expected callable"):
            encodable.encode("not a callable", {})

    def test_encode_builtin_raises(self):
        from collections.abc import Callable

        encodable = Encodable.define(Callable, {})
        # Built-in functions don't have source code
        with pytest.raises(RuntimeError, match="Source code of callable .* not found"):
            with handler(UnsafeEvalProvider()):
                encodable.encode(len)

    def test_decode_no_callable_raises(self):
        from collections.abc import Callable

        encodable = Encodable.define(Callable, {})
        # Source code that defines no callable
        source = "x = 42"
        with pytest.raises(ValueError, match="exactly one callable"):
            with handler(UnsafeEvalProvider()):
                encodable.decode(source)

    def test_decode_multiple_callables_raises(self):
        from collections.abc import Callable

        encodable = Encodable.define(Callable, {})
        # Source code that defines multiple callables
        source = """def foo():
    return 1

def bar():
    return 2"""
        with pytest.raises(ValueError, match="exactly one callable"):
            with handler(UnsafeEvalProvider()):
                encodable.decode(source)

    def test_decode_class(self):
        from collections.abc import Callable

        encodable = Encodable.define(Callable, {})
        # Classes are callable, decode should work with class definitions
        source = """class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!\""""

        with handler(UnsafeEvalProvider()):
            decoded = encodable.decode(source)
        assert callable(decoded)
        instance = decoded("World")
        assert instance.greet() == "Hello, World!"

    def test_roundtrip(self):
        from collections.abc import Callable

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        encodable = Encodable.define(Callable, {})
        with handler(UnsafeEvalProvider()):
            encoded = encodable.encode(greet)
            decoded = encodable.decode(encoded)

        assert callable(decoded)
        assert decoded("Alice") == "Hello, Alice!"
        assert decoded.__name__ == "greet"
