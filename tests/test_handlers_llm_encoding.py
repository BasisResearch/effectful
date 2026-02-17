import builtins
import dataclasses
import random
import typing
from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set,
)
from dataclasses import asdict, dataclass
from typing import Annotated, Any, NamedTuple, TypedDict

import pydantic
import pytest
from PIL import Image, ImageChops
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _images_equal(a: Image.Image, b: Image.Image) -> bool:
    """Pixel-level image comparison.

    Decoded images are ``PngImageFile`` (not ``Image.Image``), so ``==``
    returns ``False`` even when pixels are identical.  Use
    ``ImageChops.difference`` instead.
    """
    if not isinstance(a, Image.Image) or not isinstance(b, Image.Image):
        return False
    if a.size != b.size:
        return False
    return ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox() is None


def _values_equal(typ, decoded, value):
    """Deep equality that handles PIL Image comparisons in nested structures."""
    if typ is Image.Image:
        return _images_equal(decoded, value)
    origin = typing.get_origin(typ)
    args = typing.get_args(typ)
    if origin is tuple:
        if not isinstance(decoded, tuple) or len(decoded) != len(value):
            return False
        if not args or args == ((),):
            return decoded == value
        return all(_values_equal(t, d, v) for t, d, v in zip(args, decoded, value))
    if origin is not None and origin is not tuple and issubclass(origin, Sequence):
        if not isinstance(decoded, list) or len(decoded) != len(value):
            return False
        if not args:
            return decoded == value
        return all(_values_equal(args[0], d, v) for d, v in zip(decoded, value))
    return decoded == value


def _pydantic_validation_input(encodable, encoded, value):
    """Prepare input for ``Model.model_validate({'value': ...})``.

    Different base types need different dict conversion for pydantic
    model_validate to accept them:

    * **pydantic models** -- ``model_dump()``
    * **dataclasses** -- ``asdict()``
    * **NamedTuples** -- ``_asdict()``
    * everything else -- pass-through (already a valid pydantic input)
    """
    base = encodable.base
    if isinstance(base, type) and issubclass(base, pydantic.BaseModel):
        return encoded.model_dump()
    if isinstance(base, type) and dataclasses.is_dataclass(base):
        return asdict(encoded)
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return value._asdict()
    return encoded


def _contains_image(typ) -> bool:
    """Check if a type contains ``Image.Image`` anywhere (including nested)."""
    if typ is Image.Image:
        return True
    return any(_contains_image(arg) for arg in typing.get_args(typ))


# ---------------------------------------------------------------------------
# Type definitions for parameterized tests
# ---------------------------------------------------------------------------


class _NTPoint(NamedTuple):
    x: int
    y: int


class _TDUser(TypedDict):
    name: str
    age: int


class _TDConfig(TypedDict, total=False):
    host: str
    port: int


@dataclass
class _DCPoint:
    x: int
    y: int


@dataclass
class _DCAddress:
    street: str
    city: str


@dataclass
class _DCPerson:
    name: str
    age: int
    address: _DCAddress


@dataclass
class _DCOptConfig:
    host: str
    port: int
    timeout: float | None = None


class _PMPoint(pydantic.BaseModel):
    x: int
    y: int


class _PMAddress(pydantic.BaseModel):
    street: str
    city: str


class _PMPerson(pydantic.BaseModel):
    name: str
    age: int
    address: _PMAddress


# ---------------------------------------------------------------------------
# Parameterized test cases
#
# Each entry is (type, value).  The parameterized tests below exercise
# encode/decode roundtrip, pydantic model_validate roundtrip, and
# serialize/deserialize roundtrip for every case.
# ---------------------------------------------------------------------------

ENCODABLE_CASES = [
    # --- Primitives ---
    pytest.param(str, "hello", id="str"),
    pytest.param(int, 42, id="int"),
    pytest.param(bool, True, id="bool"),
    pytest.param(float, 3.14, id="float"),
    pytest.param(complex, 3 + 4j, id="complex"),
    # --- Image ---
    # Edge case: decoded images are PngImageFile, not Image.Image;
    # equality requires pixel-level comparison (_images_equal).
    pytest.param(Image.Image, Image.new("RGB", (10, 10), "red"), id="image"),
    # --- Tuples ---
    pytest.param(tuple[int, str], (1, "test"), id="tuple[int,str]"),
    pytest.param(tuple[()], (), id="tuple_empty"),  # special case: empty tuple
    pytest.param(tuple[int, str, bool], (42, "hello", True), id="tuple_3elem"),
    # --- Lists ---
    pytest.param(list[int], [1, 2, 3], id="list[int]"),
    pytest.param(list[str], ["hello", "world"], id="list[str]"),
    # --- Composite image types ---
    # These serialize to multiple image_url blocks (not text),
    # so serialize/deserialize roundtrip is unsupported.
    pytest.param(
        tuple[Image.Image, Image.Image],
        (Image.new("RGB", (10, 10), "red"), Image.new("RGB", (20, 20), "blue")),
        id="tuple[Image,Image]",
    ),
    pytest.param(
        list[Image.Image],
        [Image.new("RGB", (10, 10), "red"), Image.new("RGB", (20, 20), "blue")],
        id="list[Image]",
    ),
    # --- NamedTuple ---
    # Pydantic validation uses _asdict() for the dict representation.
    pytest.param(_NTPoint, _NTPoint(10, 20), id="namedtuple"),
    # --- TypedDict ---
    pytest.param(_TDUser, _TDUser(name="Bob", age=25), id="typeddict"),
    pytest.param(
        _TDConfig,
        _TDConfig(host="localhost", port=8080),
        id="typeddict_partial",
    ),
    # --- Dataclasses ---
    # Pydantic validation uses asdict() for the dict representation.
    pytest.param(_DCPoint, _DCPoint(10, 20), id="dataclass"),
    pytest.param(
        _DCPerson,
        _DCPerson("Bob", 25, _DCAddress("123 Main St", "NYC")),
        id="dataclass_nested",
    ),
    pytest.param(
        _DCOptConfig,
        _DCOptConfig("localhost", 8080, 5.0),
        id="dataclass_optional",
    ),
    pytest.param(
        _DCOptConfig,
        _DCOptConfig("localhost", 8080, None),
        id="dataclass_optional_none",
    ),
    # --- Pydantic models ---
    # Pydantic validation uses model_dump() for the dict representation.
    pytest.param(_PMPoint, _PMPoint(x=10, y=20), id="pydantic"),
    pytest.param(
        _PMPerson,
        _PMPerson(
            name="Bob", age=25, address=_PMAddress(street="123 Main St", city="NYC")
        ),
        id="pydantic_nested",
    ),
    # --- Collection types ---
    # MutableSequence[T] is supported (uses the same handler as list[T]).
    pytest.param(MutableSequence[int], [1, 2, 3], id="MutableSequence[int]"),
    # The following collection types are NOT yet supported by Encodable.define().
    # Each is marked xfail so failures are expected; issues will be filed.
    pytest.param(
        tuple[int, ...],
        (1, 2, 3),
        id="tuple[int,...]",
        marks=pytest.mark.xfail(
            reason="variable-length homogeneous tuple not supported"
        ),
    ),
    # These collection types are supported via the fallback/object handler.
    pytest.param(Sequence[int], [1, 2, 3], id="Sequence[int]"),
    pytest.param(Mapping[str, int], {"a": 1, "b": 2}, id="Mapping[str,int]"),
    pytest.param(
        MutableMapping[str, int], {"a": 1, "b": 2}, id="MutableMapping[str,int]"
    ),
    pytest.param(Set[int], {1, 2, 3}, id="Set[int]"),
    pytest.param(frozenset[int], frozenset({1, 2, 3}), id="frozenset[int]"),
]

# Types that support serialize -> deserialize roundtrip (single text block).
# Image types serialize to image_url blocks (not text), so Image.deserialize
# raises NotImplementedError and composite types produce multiple blocks.
SERIALIZABLE_CASES = [p for p in ENCODABLE_CASES if not _contains_image(p.values[0])]


# ---------------------------------------------------------------------------
# Error case tests
# ---------------------------------------------------------------------------


def test_define_rejects_term():
    with pytest.raises(TypeError):
        Encodable.define(Term)


def test_define_rejects_operation():
    with pytest.raises(TypeError):
        Encodable.define(Operation)


def test_image_decode_rejects_non_data_uri():
    """Only base64 data URIs are accepted; HTTP URLs are rejected."""
    encodable = Encodable.define(Image.Image)
    with pytest.raises(RuntimeError, match="expected base64 encoded image as data uri"):
        encodable.decode({"url": "http://example.com/image.png", "detail": "auto"})


# ---------------------------------------------------------------------------
# Parameterized roundtrip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("typ,value", ENCODABLE_CASES)
def test_encode_decode_roundtrip(typ, value):
    """encode -> decode preserves the original value."""
    encodable = Encodable.define(typ)
    decoded = encodable.decode(encodable.encode(value))
    assert _values_equal(typ, decoded, value)


@pytest.mark.parametrize("typ,value", ENCODABLE_CASES)
def test_pydantic_roundtrip(typ, value):
    """encode -> pydantic model_validate -> decode preserves values."""
    encodable = Encodable.define(typ)
    encoded = encodable.encode(value)

    Model = pydantic.create_model("Model", value=encodable.enc)
    validation_input = _pydantic_validation_input(encodable, encoded, value)
    model_instance = Model.model_validate({"value": validation_input})
    decoded = encodable.decode(model_instance.value)

    assert _values_equal(typ, decoded, value)


@pytest.mark.parametrize("typ,value", SERIALIZABLE_CASES)
def test_serialize_deserialize_roundtrip(typ, value):
    """encode -> serialize -> deserialize -> decode preserves values.

    Image types are excluded: ``Image.serialize`` produces ``image_url``
    blocks (not text), so ``deserialize`` is unsupported.
    """
    encodable = Encodable.define(typ)
    encoded = encodable.encode(value)
    serialized = encodable.serialize(encoded)

    assert len(serialized) == 1 and serialized[0]["type"] == "text"
    decoded = encodable.decode(encodable.deserialize(serialized[0]["text"]))

    assert _values_equal(typ, decoded, value)


# ---------------------------------------------------------------------------
# Image-specific edge cases
# ---------------------------------------------------------------------------


def test_image_encoding_format():
    """Image encodes to a dict with base64 data URI and ``detail='auto'``."""
    encodable = Encodable.define(Image.Image)
    encoded = encodable.encode(Image.new("RGB", (10, 10), color="red"))

    assert isinstance(encoded, dict)
    assert encoded["url"].startswith("data:image/png;base64,")
    assert encoded["detail"] == "auto"


def test_image_list_serializes_to_image_url_blocks():
    """``nested_type``-inferred image lists serialize to ``image_url`` content blocks."""
    images = [
        Image.new("RGB", (10, 10), color="red"),
        Image.new("RGB", (20, 20), color="blue"),
    ]

    inferred_type = nested_type(images).value
    encodable = Encodable.define(inferred_type)
    serialized = encodable.serialize(encodable.encode(images))

    assert len(serialized) == 2
    for block in serialized:
        assert block["type"] == "image_url"
        assert block["image_url"]["url"].startswith("data:image/png;base64,")
        assert block["image_url"]["detail"] == "auto"

    # Empty list produces no blocks
    assert encodable.serialize(encodable.encode([])) == []


# ---------------------------------------------------------------------------
# Callable encoding tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Random fuzz tests
#
# These use seeded random generators to produce random type/value
# combinations (primitives, nested tuples/lists, callables) and verify
# that all roundtrip pipelines hold.
# ---------------------------------------------------------------------------

_PRIMITIVES = [
    # (type, annotation_str, sample_values)
    (int, "int", [0, -1, 42]),
    (str, "str", ["", "hello", "café"]),
    (bool, "bool", [True, False]),
    (float, "float", [0.0, -3.14, 1e10]),
    (complex, "complex", [0j, 3 + 4j]),
    (
        Image.Image,
        "Image",
        [Image.new("RGB", (5, 5), c) for c in ("red", "blue", "green")],
    ),
]


def _random_type_and_value(rng, depth=0):
    """Generate a random ``(type, value)`` pair, with nesting up to depth 2."""
    if depth >= 2:
        typ, _ann, vals = rng.choice(_PRIMITIVES)
        return typ, rng.choice(vals)

    kind = rng.choice(["primitive", "tuple", "list", "mutablesequence"])
    if kind == "primitive":
        typ, _ann, vals = rng.choice(_PRIMITIVES)
        return typ, rng.choice(vals)
    if kind == "tuple":
        n = rng.randint(1, 3)
        pairs = [_random_type_and_value(rng, depth + 1) for _ in range(n)]
        types, values = zip(*pairs)
        return tuple[*types], values  # values is already a tuple from zip
    if kind == "mutablesequence":
        elem_typ, elem_val = _random_type_and_value(rng, depth + 1)
        n = rng.randint(0, 3)
        return MutableSequence[elem_typ], [elem_val] * n
    # list
    elem_typ, elem_val = _random_type_and_value(rng, depth + 1)
    n = rng.randint(0, 3)
    return list[elem_typ], [elem_val] * n


# --- Callable random generation ---
# Types are randomly generated: primitives (including Image.Image), lists,
# tuples, and dynamically-created NamedTuples / pydantic models whose fields
# are themselves random primitives.  Dynamic types are created with a
# ``<locals>``-containing ``__qualname__`` so the mypy stub generator treats
# them as runtime-only and produces class stubs.


def _dyn_name(rng, prefix):
    """Generate a short unique-ish name like ``NT_a3f``."""
    suffix = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=4))
    return f"{prefix}_{suffix}"


def _make_dyn_namedtuple(rng):
    """Create a NamedTuple with random primitive fields (including Image).

    Uses ``class`` syntax inside an exec'd function so the resulting type
    naturally gets a ``<locals>``-containing ``__qualname__``.
    """
    name = _dyn_name(rng, "NT")
    n = rng.randint(1, 3)
    specs = [rng.choice(_PRIMITIVES) for _ in range(n)]
    ns: dict[str, Any] = {"NamedTuple": NamedTuple}
    for i, s in enumerate(specs):
        ns[f"_T{i}"] = s[0]
    fields = "\n        ".join(f"f{i}: _T{i}" for i in range(n))
    exec(
        f"def _f():\n    class {name}(NamedTuple):\n        {fields}\n    return {name}",
        ns,
    )
    NT = ns["_f"]()
    val = NT(*(rng.choice(s[2]) for s in specs))
    return NT, name, val


def _make_dyn_pydantic(rng):
    """Create a pydantic BaseModel with random primitive fields (including Image).

    Uses ``class`` syntax inside an exec'd function so the resulting type
    naturally gets a ``<locals>``-containing ``__qualname__``.
    """
    name = _dyn_name(rng, "PM")
    n = rng.randint(1, 3)
    specs = [rng.choice(_PRIMITIVES) for _ in range(n)]
    ns: dict[str, Any] = {"BaseModel": pydantic.BaseModel}
    for i, s in enumerate(specs):
        ns[f"_T{i}"] = s[0]
    fields = "\n        ".join(f"f{i}: _T{i}" for i in range(n))
    exec(
        f"def _f():\n"
        f"    class {name}(BaseModel):\n"
        f"        model_config = dict(arbitrary_types_allowed=True)\n"
        f"        {fields}\n"
        f"    return {name}",
        ns,
    )
    PM = ns["_f"]()
    kwargs = {f"f{i}": rng.choice(s[2]) for i, s in enumerate(specs)}
    val = PM(**kwargs)
    return PM, name, val


def _random_callable_type_value_ann(rng, depth=0):
    """Randomly generate ``(type, value, annotation_str, ctx, skip_reasons)`` for a callable param.

    Recursively builds nested lists, tuples, dynamically-created NamedTuples,
    and pydantic models up to *depth* 2, then falls back to primitives.
    *skip_reasons* is a ``set[str]`` of reasons the enclosing test should skip
    (e.g. ``"complex"`` or ``"pydantic"``).
    """

    def _primitive(rng):
        typ, ann, vals = rng.choice(_PRIMITIVES)
        return typ, rng.choice(vals), ann, {}, set()

    if depth >= 2:
        return _primitive(rng)

    kind = rng.choice(
        ["primitive", "list", "mutablesequence", "tuple", "namedtuple", "pydantic"]
    )
    if kind == "primitive":
        return _primitive(rng)
    if kind == "list":
        et, ev, ea, ec, es = _random_callable_type_value_ann(rng, depth + 1)
        n = rng.randint(0, 3)
        return list[et], [ev] * n, f"list[{ea}]", ec, es
    if kind == "mutablesequence":
        et, ev, ea, ec, es = _random_callable_type_value_ann(rng, depth + 1)
        n = rng.randint(0, 3)
        ec = {**ec, "MutableSequence": MutableSequence}
        return MutableSequence[et], [ev] * n, f"MutableSequence[{ea}]", ec, es
    if kind == "tuple":
        n = rng.randint(1, 3)
        parts = [_random_callable_type_value_ann(rng, depth + 1) for _ in range(n)]
        types = tuple(p[0] for p in parts)
        vals = tuple(p[1] for p in parts)
        anns = [p[2] for p in parts]
        ctx: dict[str, Any] = {}
        skip: set[str] = set()
        for p in parts:
            ctx.update(p[3])
            skip |= p[4]
        return tuple[*types], vals, f"tuple[{', '.join(anns)}]", ctx, skip
    if kind == "namedtuple":
        NT, name, val = _make_dyn_namedtuple(rng)
        return NT, val, name, {name: NT}, set()
    # pydantic
    PM, name, val = _make_dyn_pydantic(rng)
    return (
        PM,
        val,
        name,
        {name: PM},
        {
            "stub generator produces incompatible method signatures for BaseModel subclasses"
        },
    )


def _random_callable_case(rng):
    """Generate a callable with a randomly-generated type signature.

    Single param produces an identity function; multiple params pack into a
    tuple.  ``Image.Image`` is always available in ctx as ``Image``.
    Returns ``(callable_type, source, ctx, call_args, expected, skip_reasons)``.
    """
    n_params = rng.randint(1, 3)
    parts = [_random_callable_type_value_ann(rng) for _ in range(n_params)]
    param_types = [p[0] for p in parts]
    param_vals = tuple(p[1] for p in parts)
    param_anns = [p[2] for p in parts]
    skip_reasons: set[str] = set()
    for p in parts:
        skip_reasons |= p[4]
    ctx: dict[str, Any] = {"Image": Image.Image}
    for p in parts:
        ctx.update(p[3])

    names = [chr(ord("a") + i) for i in range(n_params)]
    params_str = ", ".join(f"{nm}: {a}" for nm, a in zip(names, param_anns))

    if n_params == 1:
        # Identity: f(x: T) -> T
        code = f"def f({params_str}) -> {param_anns[0]}:\n    return a"
        callable_typ = Callable[[param_types[0]], param_types[0]]
        expected = param_vals[0]
    else:
        # Pack: f(a: A, b: B) -> tuple[A, B]
        ret_ann = f"tuple[{', '.join(param_anns)}]"
        body = f"({', '.join(names)},)"
        code = f"def f({params_str}) -> {ret_ann}:\n    return {body}"
        callable_typ = Callable[param_types, tuple[*param_types]]
        expected = param_vals

    return (
        callable_typ,
        SynthesizedFunction(module_code=code),
        ctx,
        param_vals,
        expected,
        skip_reasons,
    )


def _build_callable_cases(n=10):
    cases = []
    for seed in range(n):
        case = _random_callable_case(random.Random(seed))
        sig = case[1].module_code.split("\n")[0]
        skip_reasons = case[-1]
        marks = (
            [pytest.mark.xfail(reason="; ".join(skip_reasons))] if skip_reasons else []
        )
        cases.append(pytest.param(*case, id=sig, marks=marks))
    return cases


_RANDOM_CALLABLE_CASES = _build_callable_cases()


def _build_roundtrip_cases(n=15):
    cases = []
    for seed in range(n):
        rng = random.Random(seed)
        typ, value = _random_type_and_value(rng)
        cases.append(pytest.param(typ, value, id=str(typ)))
    return cases


_RANDOM_ROUNDTRIP_CASES = _build_roundtrip_cases()


class TestRandomFuzz:
    """Fuzz tests with seeded random type/value generation."""

    @pytest.mark.parametrize("typ,value", _RANDOM_ROUNDTRIP_CASES)
    def test_roundtrips(self, typ, value):
        """Random types through encode/decode, pydantic, and serialize roundtrips."""
        encodable = Encodable.define(typ)
        encoded = encodable.encode(value)
        decoded = encodable.decode(encoded)
        assert _values_equal(typ, decoded, value), f"{typ}: {decoded!r} != {value!r}"

        # encode -> pydantic model_validate -> decode
        Model = pydantic.create_model("M", value=encodable.enc)
        validated = Model.model_validate(
            {"value": _pydantic_validation_input(encodable, encoded, value)}
        )
        decoded = encodable.decode(validated.value)
        assert _values_equal(typ, decoded, value), (
            f"{typ}: pydantic {decoded!r} != {value!r}"
        )

        # encode -> serialize -> deserialize -> decode (skip image types)
        if not _contains_image(typ):
            serialized = encodable.serialize(encoded)
            if len(serialized) == 1 and serialized[0]["type"] == "text":
                decoded = encodable.decode(encodable.deserialize(serialized[0]["text"]))
                assert _values_equal(typ, decoded, value), (
                    f"{typ}: serde {decoded!r} != {value!r}"
                )

    @pytest.mark.parametrize(
        "typ,source,ctx,args,expected,skip_reasons", _RANDOM_CALLABLE_CASES
    )
    def test_callable_roundtrips(self, typ, source, ctx, args, expected, skip_reasons):
        """Callable with random signature through decode, pydantic, and serialize.

        Uses ``UnsafeEvalProvider`` only -- ``RestrictedPython`` rewrites generic
        subscripts like ``list[int]`` into ``_getitem_`` calls, breaking type
        annotations in synthesized source code.
        """
        info = (
            f"\ntype={typ}\nctx_keys={list(ctx.keys())}"
            f"\nsource:\n{source.module_code}\nargs={args!r}"
        )
        encodable = Encodable.define(typ, ctx)
        provider = UnsafeEvalProvider()

        try:
            # decode
            with handler(provider):
                decoded = encodable.decode(source)
            assert callable(decoded) and decoded(*args) == expected

            # pydantic roundtrip
            Model = pydantic.create_model("M", value=encodable.enc)
            validated = Model.model_validate({"value": source.model_dump()})
            with handler(provider):
                decoded = encodable.decode(validated.value)
            assert callable(decoded) and decoded(*args) == expected

            # serialize roundtrip
            serialized = encodable.serialize(source)
            deserialized = encodable.deserialize(serialized[0]["text"])
            with handler(provider):
                decoded = encodable.decode(deserialized)
            assert callable(decoded) and decoded(*args) == expected
        except Exception as exc:
            raise type(exc)(f"{exc}{info}") from exc


# ---------------------------------------------------------------------------
# RestrictedEvalProvider security tests
# ---------------------------------------------------------------------------


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
