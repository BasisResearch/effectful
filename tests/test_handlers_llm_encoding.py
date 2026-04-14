"""
Law-based test suite for effectful.handlers.llm.encoding.

Each test function verifies a single equational law of the Encodable[T]
type-level encoding, parametrized over many types and values.
"""

import inspect
import io
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, NamedTuple, TypedDict

import litellm
import pydantic
import pytest
from litellm import ChatCompletionMessageToolCall, OpenAIMessageContentListBlock
from PIL import Image

from effectful.handlers.llm.encoding import (
    CONTENT_BLOCK_TYPES,
    DecodedToolCall,
    Encodable,
    SynthesizedFunction,
    to_content_blocks,
)
from effectful.handlers.llm.evaluation import RestrictedEvalProvider, UnsafeEvalProvider
from effectful.handlers.llm.template import Tool
from effectful.internals.unification import nested_type
from effectful.ops.semantics import handler
from effectful.ops.types import Operation, Term
from tests.conftest import EFFECTFUL_LLM_MODEL, requires_llm

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
class _PairManual:
    values: Encodable[tuple[int, str]]
    count: int


@dataclass
class _WithCallableManual:
    name: str
    fn: Encodable[Callable[[int], int]]


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
# ctx=None means no context, otherwise passed as context to dump_python/validate_python.
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
    pytest.param(
        _PairManual,
        _PairManual(values=(42, "hello"), count=2),
        None,
        id="dc-manual-tuple-field",
    ),
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
    pytest.param(tuple, (1, "hello", True), None, id="tuple-bare"),
    pytest.param(tuple[int, ...], (1, 2, 3), None, id="tuple-variadic"),
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
        tuple[str, Image.Image, str],
        ("before", _make_png_image("RGB", (5, 5), "green"), "after"),
        None,
        id="tuple-str-img-str",
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
    # --- deeper generic composition with Image ---
    pytest.param(
        list[tuple[str, Image.Image]],
        [
            ("first", _make_png_image("RGB", (4, 4), "red")),
            ("second", _make_png_image("RGB", (4, 4), "blue")),
        ],
        None,
        id="list-tuple-str-img",
    ),
    # --- Tool ---
    pytest.param(type(_tool_add), _tool_add, {"_tool_add": _tool_add}, id="tool-add"),
    pytest.param(
        type(_tool_greet), _tool_greet, {"_tool_greet": _tool_greet}, id="tool-greet"
    ),
    pytest.param(
        type(_tool_process),
        _tool_process,
        {"_tool_process": _tool_process},
        id="tool-process",
    ),
    pytest.param(
        type(_tool_get_value),
        _tool_get_value,
        {"_tool_get_value": _tool_get_value},
        id="tool-no-params",
    ),
    pytest.param(
        type(_tool_distance),
        _tool_distance,
        {"_tool_distance": _tool_distance},
        id="tool-pydantic-param",
    ),
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

# ============================================================================
# Law 1: decode(encode(v)) == v
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", ROUNDTRIP_CASES)
def test_encode_decode_roundtrip(ty, value, ctx):
    enc = pydantic.TypeAdapter(Encodable[ty])
    encoded = enc.dump_python(value, mode="json", context=ctx or {})
    assert enc.validate_python(encoded, context=ctx or {}) == value


# ============================================================================
# Law 2: json.loads(json.dumps(encode(v))) == encode(v)
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", ROUNDTRIP_CASES)
def test_serialize_deserialize_roundtrip(ty, value, ctx):
    enc = pydantic.TypeAdapter(Encodable[ty])
    encoded = enc.dump_python(value, mode="json", context=ctx or {})
    assert json.loads(json.dumps(encoded)) == encoded


# ============================================================================
# Law 3: decode(json.loads(json.dumps(encode(v)))) == v
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", ROUNDTRIP_CASES)
def test_full_pipeline_roundtrip(ty, value, ctx):
    enc = pydantic.TypeAdapter(Encodable[ty])
    encoded = enc.dump_python(value, mode="json", context=ctx or {})
    assert (
        enc.validate_python(json.loads(json.dumps(encoded)), context=ctx or {}) == value
    )


# ============================================================================
# Law 5: encode(encode(v)) == encode(v) (idempotency)
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", ROUNDTRIP_CASES)
def test_encode_idempotent(ty, value, ctx):
    once = pydantic.TypeAdapter(Encodable[ty]).dump_python(
        value, mode="json", context=ctx or {}
    )
    twice = pydantic.TypeAdapter(Encodable[nested_type(once).value]).dump_python(
        once, mode="json", context=ctx or {}
    )
    assert once == twice


# ============================================================================
# Term-specific: Encodable raises TypeError for Term and Operation
# ============================================================================


@pytest.mark.parametrize("ty", [Term, Operation])
def test_define_raises_for_invalid_types(ty):
    with pytest.raises(TypeError):
        Encodable[ty]


# ============================================================================
# to_content_blocks helpers
# ============================================================================


def _linearize(blocks: list[OpenAIMessageContentListBlock]) -> str:
    """Concatenate content blocks back into a JSON string."""
    return "".join(b["text"] if b["type"] == "text" else json.dumps(b) for b in blocks)


def _has_content_block(v):
    """Recursively check whether v contains any content-block-shaped dicts."""
    if isinstance(v, dict) and v.get("type") in CONTENT_BLOCK_TYPES:
        return True
    if isinstance(v, dict):
        return any(_has_content_block(val) for val in v.values())
    if isinstance(v, list):
        return any(_has_content_block(item) for item in v)
    return False


# ============================================================================
# Law 6: linearize(to_content_blocks(encode(v))) == json.dumps(encode(v))
#         (for non-string encoded values; bare strings are emitted unquoted)
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", ROUNDTRIP_CASES)
def test_to_content_blocks_linearization(ty, value, ctx):
    encoded = pydantic.TypeAdapter(Encodable[ty]).dump_python(
        value, mode="json", context=ctx or {}
    )
    if isinstance(encoded, str):
        # Bare strings are emitted without JSON quoting for natural template rendering
        assert _linearize(to_content_blocks(encoded)) == encoded
    else:
        assert _linearize(to_content_blocks(encoded)) == json.dumps(encoded)


# ============================================================================
# Law 7: decode(json.loads(linearize(to_content_blocks(encode(v))))) == v
#         (for non-string encoded values; bare strings roundtrip directly)
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", ROUNDTRIP_CASES)
def test_to_content_blocks_full_pipeline(ty, value, ctx):
    enc = pydantic.TypeAdapter(Encodable[ty])
    encoded = enc.dump_python(value, mode="json", context=ctx or {})
    linearized = _linearize(to_content_blocks(encoded))
    if isinstance(encoded, str):
        assert enc.validate_python(linearized, context=ctx or {}) == value
    else:
        assert enc.validate_python(json.loads(linearized), context=ctx or {}) == value


# ============================================================================
# Law 8: no content blocks hidden in text (maximal extraction)
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", ROUNDTRIP_CASES)
def test_to_content_blocks_maximal_extraction(ty, value, ctx):
    encoded = pydantic.TypeAdapter(Encodable[ty]).dump_python(
        value, mode="json", context=ctx or {}
    )
    if isinstance(encoded, str):
        # Bare strings are emitted unquoted; they can't contain content blocks
        return
    blocks = to_content_blocks(encoded)
    skeleton = json.loads(
        "".join(b["text"] if b["type"] == "text" else "null" for b in blocks)
    )
    assert not _has_content_block(skeleton)


# ============================================================================
# Tuple-specific: schema validation
# ============================================================================

TUPLE_SCHEMA_CASES = [
    pytest.param(tuple[int, str], id="tuple-int-str"),
    pytest.param(tuple[int, str, bool], id="tuple-three"),
    pytest.param(tuple[()], id="tuple-empty"),
]


@pytest.mark.parametrize("ty", TUPLE_SCHEMA_CASES)
def test_tuple_schema_no_prefix_items(ty):
    """Finitary tuple schemas use properties/required, not prefixItems."""
    schema = pydantic.TypeAdapter(Encodable[ty]).json_schema()
    assert "prefixItems" not in str(schema), (
        f"Schema for {ty} should not contain prefixItems: {schema}"
    )


# ============================================================================
# Regression tests: manual Encodable annotation on fields (#626, #631)
# ============================================================================


def test_encodable_tuple_produces_object_schema_626():
    """Encodable[tuple[int, str]] produces object-based schema, not prefixItems."""
    adapter = pydantic.TypeAdapter(Encodable[tuple[int, str]])
    schema = adapter.json_schema()
    assert schema["type"] == "object"
    assert "prefixItems" not in json.dumps(schema)


def test_encodable_callable_produces_valid_schema_631():
    """Encodable[Callable[[int], int]] produces valid JSON schema."""
    adapter = pydantic.TypeAdapter(Encodable[Callable[[int], int]])
    schema = adapter.json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema


def test_dataclass_with_encodable_tuple_field_626():
    """Users annotate tuple fields with Encodable for OpenAI compatibility (#626)."""

    @dataclass
    class Pair:
        values: Encodable[tuple[int, str]]
        count: int

    adapter = pydantic.TypeAdapter(Pair)
    val = Pair(values=(42, "hello"), count=2)
    encoded = adapter.dump_python(val, mode="json")
    decoded = adapter.validate_python(encoded)
    assert decoded == val
    assert "prefixItems" not in json.dumps(adapter.json_schema())


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
    with pytest.raises(exc_type):
        pydantic.TypeAdapter(Encodable[DecodedToolCall]).validate_python(
            tool_call, context=ctx
        )


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
    enc = pydantic.TypeAdapter(Encodable[ty])
    with handler(eval_provider):
        decoded = enc.validate_python(
            enc.dump_python(func, mode="json", context=ctx), context=ctx
        )
        assert decoded(*args) == expected


@pytest.mark.parametrize("ty,func,ctx,args,expected", CALLABLE_CASES)
@pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
def test_callable_full_pipeline_behavioral(
    ty, func, ctx, args, expected, eval_provider
):
    """Full encode->serialize->deserialize->decode pipeline is behaviorally equivalent."""
    enc = pydantic.TypeAdapter(Encodable[ty])
    text = json.dumps(enc.dump_python(func, mode="json", context=ctx))
    with handler(eval_provider):
        decoded = enc.validate_python(json.loads(text), context=ctx)
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
    with pytest.raises(exc_type):
        with handler(eval_provider):
            pydantic.TypeAdapter(Encodable[ty]).validate_python(source, context=ctx)


def test_callable_encode_non_callable():
    with pytest.raises(Exception):
        pydantic.TypeAdapter(Encodable[Callable[..., int]]).dump_python(
            "not a callable", mode="json", context={}
        )


def test_callable_encode_no_source_no_docstring():

    class _NoDocCallable:
        __name__ = "nodoc"
        __doc__ = None

        def __call__(self):
            pass

    with pytest.raises(ValueError):
        pydantic.TypeAdapter(Encodable[Callable[..., int]]).dump_python(
            _NoDocCallable(), mode="json", context={}
        )


# ---------------------------------------------------------------------------
# Provider integration tests
# ---------------------------------------------------------------------------

_provider_response_format_xfail = pytest.mark.xfail(
    reason="Known OpenAI/LiteLLM response_format limitation for this type."
)


def _apply_xfails(
    cases: list[Any],
    should_xfail: Callable[[str], bool],
) -> list[Any]:
    out: list[Any] = []
    for c in cases:
        case_id = c.id if isinstance(c.id, str) else None
        if case_id is not None and should_xfail(case_id):
            out.append(
                pytest.param(
                    *c.values,
                    id=case_id,
                    marks=[*c.marks, _provider_response_format_xfail],
                )
            )
        else:
            out.append(c)
    return out


# response_model: image types can't roundtrip (LLM returns URLs, not data URIs);
# bare tuple has no type information so its schema is rejected by OpenAI.
RESPONSE_MODEL_CASES = _apply_xfails(
    ROUNDTRIP_CASES,
    lambda cid: cid.startswith("img-") or "-img" in cid or cid == "tuple-bare",
)

# tool-as-param: bare tuple has no type information.
TOOL_PARAM_CASES = _apply_xfails(
    ROUNDTRIP_CASES,
    lambda cid: cid == "tuple-bare",
)


@requires_llm
@pytest.mark.parametrize("ty,_value,ctx", RESPONSE_MODEL_CASES)
def test_litellm_completion_accepts_encodable_response_model_for_supported_types(
    ty: Any, _value: Any, ctx: Mapping[str, Any] | None
) -> None:
    enc: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(Encodable[ty])
    # Use pydantic.create_model so litellm handles strictification
    response_model: type[pydantic.BaseModel] = pydantic.create_model(
        "Response", value=(Encodable[ty], ...)
    )
    response_format = litellm.utils.type_to_response_format_param(response_model)
    response = litellm.completion(
        model=EFFECTFUL_LLM_MODEL,
        response_format=response_format,
        messages=[
            {
                "role": "user",
                "content": f"Return an instance of {getattr(ty, '__name__', repr(ty))}.",
            }
        ],
        max_tokens=400,
    )
    assert isinstance(response, litellm.ModelResponse)

    choice = response.choices[0]
    assert isinstance(choice, litellm.Choices)
    content = choice.message.content
    assert content is not None, (
        f"Expected content in response for {getattr(ty, '__name__', repr(ty))}"
    )

    deserialized = json.loads(content)["value"]
    decoded = enc.validate_python(deserialized, context=ctx or {})
    pydantic.TypeAdapter(ty).validate_python(decoded)


@requires_llm
@pytest.mark.parametrize("ty,_value,ctx", TOOL_PARAM_CASES)
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
    enc: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(
        Encodable[type(tool)]  # type: ignore[misc]
    )
    tool_spec = enc.dump_python(tool, mode="json", context=ctx or {})
    response = litellm.completion(
        model=EFFECTFUL_LLM_MODEL,
        messages=[{"role": "user", "content": "Return hello, do NOT call any tools."}],
        tools=[tool_spec],
        tool_choice="none",
        max_tokens=400,
    )
    assert isinstance(response, litellm.ModelResponse)


@requires_llm
@pytest.mark.parametrize("ty,_value,ctx", ROUNDTRIP_CASES)
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
    enc: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(
        Encodable[type(tool)]  # type: ignore[misc]
    )
    tool_spec = enc.dump_python(tool, mode="json", context=ctx or {})
    response = litellm.completion(
        model=EFFECTFUL_LLM_MODEL,
        messages=[{"role": "user", "content": "Return hello, do NOT call any tools."}],
        tools=[tool_spec],
        tool_choice="none",
        max_tokens=400,
    )
    assert isinstance(response, litellm.ModelResponse)
