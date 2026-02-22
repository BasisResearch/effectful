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
    PydanticImage,
    PydanticTool,
    PydanticToolCall,
    SynthesizedFunction,
    pydantic_type,
)
from effectful.handlers.llm.evaluation import RestrictedEvalProvider, UnsafeEvalProvider
from effectful.handlers.llm.template import Tool
from effectful.internals.unification import nested_type
from effectful.ops.semantics import handler
from effectful.ops.types import Operation, Term

CHEAP_MODEL = "lm_studio/openai/gpt-oss-120b"

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


# @requires_openai
@pytest.mark.parametrize("ty,_value,ctx", PROVIDER_CASES)
def test_litellm_completion_accepts_encodable_response_model_for_supported_types(
    ty: Any, _value: Any, ctx: Mapping[str, Any] | None
) -> None:
    enc = Encodable.define(ty, ctx)
    response = litellm.completion(
        model=CHEAP_MODEL,
        response_format=enc.response_format,
        messages=[
            {
                "role": "user",
                "content": f"Return an instance of {getattr(ty, '__name__', repr(ty))}.",
            }
        ],
        max_tokens=200,
    )
    assert isinstance(response, litellm.ModelResponse)

    content = response.choices[0].message.content
    assert content is not None, (
        f"Expected content in response for {getattr(ty, '__name__', repr(ty))}"
    )

    deserialized = enc.deserialize(content)
    decoded = enc.decode(deserialized)
    pydantic.TypeAdapter(enc.base).validate_python(decoded)


# @requires_openai
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
    enc = Encodable.define(type(tool), ctx)
    response = litellm.completion(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": "Return hello, do NOT call any tools."}],
        tools=[enc.encode(tool)],
        tool_choice="none",
        max_tokens=200,
    )
    assert isinstance(response, litellm.ModelResponse)


# @requires_openai
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
    enc = Encodable.define(type(tool), ctx)
    response = litellm.completion(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": "Return hello, do NOT call any tools."}],
        tools=[enc.encode(tool)],
        tool_choice="none",
        max_tokens=200,
    )
    assert isinstance(response, litellm.ModelResponse)


# ---------------------------------------------------------------------------
# pydantic_type substitution tests
# ---------------------------------------------------------------------------


class TestPydanticTypeLeafSubstitutions:
    """Leaf types are replaced with their Pydantic Annotated equivalents."""

    def test_image(self):
        assert pydantic_type(Image.Image) is PydanticImage

    def test_tool(self):
        assert pydantic_type(Tool) is PydanticTool

    def test_tool_call(self):
        assert pydantic_type(DecodedToolCall) is PydanticToolCall

    def test_callable(self):
        result = pydantic_type(Callable[[int], str])
        import typing

        assert typing.get_origin(result) is Annotated


class TestPydanticTypePassthrough:
    """Types without registrations pass through unchanged."""

    def test_int(self):
        assert pydantic_type(int) is int

    def test_str(self):
        assert pydantic_type(str) is str

    def test_float(self):
        assert pydantic_type(float) is float

    def test_list_int(self):
        ty = list[int]
        assert pydantic_type(ty) is ty

    def test_dict_str_int(self):
        ty = dict[str, int]
        assert pydantic_type(ty) is ty

    def test_pydantic_model(self):
        assert pydantic_type(_PointModel) is _PointModel


class TestPydanticTypeNestedGenerics:
    """Generic types have their type args recursively substituted."""

    def test_list_image(self):
        result = pydantic_type(list[Image.Image])
        assert result == list[PydanticImage]

    def test_dict_str_tool(self):
        result = pydantic_type(dict[str, Tool])
        assert result == dict[str, PydanticTool]

    def test_tuple_image_str(self):
        result = pydantic_type(tuple[Image.Image, str])
        assert result == tuple[PydanticImage, str]

    def test_list_list_image(self):
        result = pydantic_type(list[list[Image.Image]])
        assert result == list[list[PydanticImage]]

    def test_mixed_no_substitution_args(self):
        """Only args that need substitution are changed."""
        result = pydantic_type(dict[str, Image.Image])
        assert result == dict[str, PydanticImage]


class TestPydanticTypeErrors:
    """Invalid types raise TypeError."""

    def test_term(self):
        with pytest.raises(TypeError):
            pydantic_type(Term)

    def test_operation(self):
        with pytest.raises(TypeError):
            pydantic_type(Operation)


class TestPydanticTypeRoundtrip:
    """Substituted types work with pydantic.TypeAdapter for validation and serialization."""

    def test_list_image_roundtrip(self):
        img = Image.new("RGB", (2, 2), color="red")
        adapter = pydantic.TypeAdapter(pydantic_type(list[Image.Image]))

        serialized = adapter.dump_python([img], mode="json")
        assert isinstance(serialized, list)
        assert len(serialized) == 1
        assert isinstance(serialized[0], dict)
        assert "url" in serialized[0]

        roundtripped = adapter.validate_python(serialized)
        assert isinstance(roundtripped, list)
        assert len(roundtripped) == 1
        assert isinstance(roundtripped[0], Image.Image)
