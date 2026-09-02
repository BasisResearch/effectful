"""
Law-based test suite for effectful.handlers.llm.encoding.

Each test function verifies a single equational law of the Encodable[T]
type-level encoding, parametrized over many types and values.
"""

import inspect
import io
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import CodeType
from typing import (
    Annotated,
    Any,
    Literal,
    NamedTuple,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
)

import litellm
import pydantic
import pytest
from litellm import ChatCompletionMessageToolCall, OpenAIMessageContentListBlock
from PIL import Image

from effectful.handlers.llm.harness.execution.builtin import BuiltinExecutor
from effectful.handlers.llm.harness.execution.restricted import (
    RestrictedPythonExecutor,
)
from effectful.handlers.llm.harness.serialization import (
    _NAME2TOOL_KEY,
    _TYPE_CHECK_ANCHOR_KEY,
    CONTENT_BLOCK_TYPES,
    DecodedToolCall,
    _BoxedResponse,
    _is_decodable,
    _NameAndTool,
    _UndecodableReturn,
    to_content_blocks,
)
from effectful.handlers.llm.harness.validation.ty import TyTypeChecker
from effectful.handlers.llm.types import Encodable, Skill, Tool
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
# Module-level type aliases
#
# An alias is meant to be *transparent*: writing one in a signature must encode
# exactly as writing the type it stands for.  Covered over the shapes with a
# registered encoding of their own (tuple, complex, Image), since those are the
# ones an alias could silently bypass -- Pydantic would still handle the type
# natively, so the failure is a wrong wire format rather than an error.
# ---------------------------------------------------------------------------

type _PairAlias = tuple[int, str]
type _ComplexAlias = complex
type _ImageAlias = Image.Image
type _PointAlias = _Point
type _PairsAlias = list[_PairAlias]

# The pre-PEP-695 spelling, which is a plain assignment: indistinguishable from
# the type itself, and here to keep it that way. (`UP040` wants the `type`
# keyword, which is the spelling *above* -- the old one is the point here.)
_LegacyPairAlias: TypeAlias = tuple[int, str]  # noqa: UP040

# A *generic* alias, whose encoding is reached only once it is subscripted.
type _GenPairAlias[T] = tuple[T, T]
type _GenImageAlias[T] = list[tuple[T, Image.Image]]

# Recursive aliases, which cannot be expanded to a finite type expression. The
# generic one is the reason expansion needs a guard rather than a special case:
# Pydantic resolves it natively, so it works until something expands it eagerly.
type _RecursiveAlias = int | list[_RecursiveAlias]
type _RecursiveGenAlias[T] = T | list[_RecursiveGenAlias[T]]

# Stands in for `Kernel` in `docs/source/llm_examples/optimization/kernels.py`:
# an alias that is in a skill's lexical *scope*, so the alias object itself
# reaches the encoding as a value.
type _KernelAlias = Callable[[list[float]], list[float]]


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


@Tool.define
def _tool_style(style: Literal["moral", "funny"]) -> str:
    """Return the requested style."""
    return style


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
    # --- Literal / special forms (regression for #644) ---
    pytest.param(Literal["moral", "funny"], "moral", None, id="literal-two-strings"),
    pytest.param(Literal[1, 2, 3], 2, None, id="literal-ints"),
    pytest.param(Annotated[int, "meta"], 42, None, id="annotated-int"),
    # typing.Union (vs PEP 604 `|`) uses typing.Union as origin — a _SpecialForm
    # that would re-trigger the #644 dispatch bug if the guard regressed.
    pytest.param(Union[int, str], 42, None, id="union-old-int"),  # noqa: UP007
    pytest.param(int | str, 42, None, id="union-new-int"),  # noqa: UP007
    pytest.param(Union[int, str], "hello", None, id="union-old-str"),  # noqa: UP007
    pytest.param(list[Literal["a", "b"]], ["a", "b", "a"], None, id="list-literal"),
    pytest.param(tuple[Literal["x", "y"], int], ("x", 5), None, id="tuple-literal-int"),
    pytest.param(Literal["a", "b"] | None, "a", None, id="literal-or-none-some"),
    pytest.param(Literal["a", "b"] | None, None, None, id="literal-or-none-none"),
    # --- type aliases ---
    # Each pairs with an entry above for the type it aliases; the schemas are
    # asserted equal in `test_type_alias_is_transparent`.
    pytest.param(_PairAlias, (1, "hello"), None, id="alias-tuple"),
    pytest.param(_ComplexAlias, 3 + 4j, None, id="alias-complex"),
    pytest.param(_PointAlias, _Point(10, 20), None, id="alias-dataclass"),
    pytest.param(
        _PairsAlias, [(1, "a"), (2, "b")], None, id="alias-list-of-alias"
    ),  # an alias reached through a generic, not at the top level
    pytest.param(_LegacyPairAlias, (1, "hello"), None, id="alias-legacy-tuple"),
    pytest.param(_GenPairAlias[int], (1, 2), None, id="alias-generic-subscripted"),
    pytest.param(
        Annotated[_PairAlias, "meta"], (1, "hello"), None, id="alias-annotated"
    ),
    # id carries "-img" so the response_format xfails above pick it up, as they
    # do for every other image case.
    pytest.param(
        _ImageAlias, _make_png_image("RGB", (10, 10), "red"), None, id="alias-img"
    ),
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
    # --- _NameAndTool (the tool advertisement `call_assistant` sends) ---
    pytest.param(
        _NameAndTool,
        _NameAndTool("_tool_add", _tool_add),
        {_NAME2TOOL_KEY: {"_tool_add": _tool_add}},
        id="tool-add",
    ),
    pytest.param(
        _NameAndTool,
        _NameAndTool("_tool_greet", _tool_greet),
        {_NAME2TOOL_KEY: {"_tool_greet": _tool_greet}},
        id="tool-greet",
    ),
    pytest.param(
        _NameAndTool,
        _NameAndTool("_tool_process", _tool_process),
        {_NAME2TOOL_KEY: {"_tool_process": _tool_process}},
        id="tool-process",
    ),
    pytest.param(
        _NameAndTool,
        _NameAndTool("_tool_get_value", _tool_get_value),
        {_NAME2TOOL_KEY: {"_tool_get_value": _tool_get_value}},
        id="tool-no-params",
    ),
    pytest.param(
        _NameAndTool,
        _NameAndTool("_tool_distance", _tool_distance),
        {_NAME2TOOL_KEY: {"_tool_distance": _tool_distance}},
        id="tool-pydantic-param",
    ),
    pytest.param(
        _NameAndTool,
        _NameAndTool("_tool_style", _tool_style),
        {_NAME2TOOL_KEY: {"_tool_style": _tool_style}},
        id="tool-literal-param",
    ),
    # A tool advertised under a name that is not its `__name__` -- what
    # `call_assistant` does when two tools in scope share one.
    pytest.param(
        _NameAndTool,
        _NameAndTool("_tool_add_2", _tool_add),
        {_NAME2TOOL_KEY: {"_tool_add_2": _tool_add}},
        id="tool-renamed",
    ),
    # --- DecodedToolCall ---
    pytest.param(
        DecodedToolCall,
        _make_dtc(_tool_add, {"a": 3, "b": 5}, "call_1"),
        {_NAME2TOOL_KEY: {"_tool_add": _tool_add}},
        id="dtc-add-3-5",
    ),
    pytest.param(
        DecodedToolCall,
        _make_dtc(_tool_add, {"a": 0, "b": -1}, "call_2"),
        {_NAME2TOOL_KEY: {"_tool_add": _tool_add}},
        id="dtc-add-0-neg",
    ),
    pytest.param(
        DecodedToolCall,
        _make_dtc(_tool_greet, {"name": "Alice"}, "call_3"),
        {_NAME2TOOL_KEY: {"_tool_greet": _tool_greet}},
        id="dtc-greet-alice",
    ),
    pytest.param(
        DecodedToolCall,
        _make_dtc(_tool_process, {"items": [1, 2, 3], "label": "total"}, "call_4"),
        {_NAME2TOOL_KEY: {"_tool_process": _tool_process}},
        id="dtc-process-items",
    ),
    pytest.param(
        DecodedToolCall,
        _make_dtc(_tool_distance, {"p": _PointModel(x=3, y=4)}, "call_5"),
        {_NAME2TOOL_KEY: {"_tool_distance": _tool_distance}},
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
# Law 6: decode(v) == v for an already-decoded v (decoding is idempotent)
# ============================================================================


@pytest.mark.parametrize("ty,value,ctx", ROUNDTRIP_CASES)
def test_decode_idempotent(ty, value, ctx):
    """`Encodable[T]` validates a real `T`, not only the wire form of one.

    A value that never crossed the model boundary -- the result of applying a
    synthesized function, say -- can be checked against its declared type, which
    is the whole point of validating one at all.
    """
    dec = pydantic.TypeAdapter(Encodable[ty])
    assert dec.validate_python(value, context=ctx or {}) == value


# ============================================================================
# The encoding leaves room for validators it did not supply
# ============================================================================

_CALLER_VALIDATOR_CASES = [
    pytest.param(complex, 3 + 4j, -3 + 4j, lambda z: z.real >= 0, id="complex"),
    pytest.param(tuple[int, str], (1, "a"), (-1, "a"), lambda t: t[0] >= 0, id="tuple"),
    pytest.param(
        Image.Image,
        _make_png_image("RGB", (10, 10), "red"),
        _make_png_image("RGB", (2, 2), "red"),
        lambda im: im.width >= 10,
        id="image",
    ),
    pytest.param(_Coord, _Coord(3, 4), _Coord(-3, 4), lambda c: c.x >= 0, id="nt"),
]


@pytest.mark.parametrize("ty,ok,bad,predicate", _CALLER_VALIDATOR_CASES)
def test_caller_validators_run_on_the_decoded_value(ty, ok, bad, predicate):
    """An `AfterValidator` a caller annotates a type with runs on the decoded
    value, in both directions.

    The encoding wraps validation rather than replacing it, so it composes with
    whatever else the annotation carries instead of shadowing it.
    """

    def check(value):
        assert predicate(value), "rejected"
        return value

    dec = pydantic.TypeAdapter(Encodable[Annotated[ty, pydantic.AfterValidator(check)]])
    enc = pydantic.TypeAdapter(Encodable[ty])

    assert dec.validate_python(ok) == ok
    assert dec.validate_python(enc.dump_python(ok, mode="json")) == ok
    for rejected in (bad, enc.dump_python(bad, mode="json")):
        with pytest.raises(pydantic.ValidationError, match="rejected"):
            dec.validate_python(rejected)


def test_annotated_metadata_reaches_pydantic():
    """`Encodable[Annotated[T, ...]]` keeps each metadata item its own item.

    Packed into a single tuple, metadata is still *present* but invisible to
    everything that looks for it by type -- Pydantic finding a validator, say.
    """

    def check(x: int) -> int:
        assert x > 0, "rejected"
        return x

    marker = pydantic.AfterValidator(check)
    assert marker in Encodable[Annotated[int, marker]].__metadata__
    with pytest.raises(pydantic.ValidationError, match="rejected"):
        pydantic.TypeAdapter(Encodable[Annotated[int, marker]]).validate_python(-1)


_T = TypeVar("_T")

_ANNOTATABLE_TYPES = [
    pytest.param(complex, id="complex"),
    pytest.param(tuple[int, str], id="tuple"),
    pytest.param(list[Image.Image], id="list-of-image"),
    pytest.param(_Coord, id="namedtuple"),
    pytest.param(Callable[[int], int], id="callable"),
    pytest.param(Tool, id="tool"),
    # An alias is replaced wholesale by the type it names, so a caller's
    # metadata has to survive that substitution too.
    pytest.param(_PairAlias, id="type-alias"),
    # A free type variable: a value here, but mypy reads the subscript as a
    # type expression and wants it bound.
    pytest.param(Sequence[_T], id="generic"),  # type: ignore[valid-type]
]


@pytest.mark.parametrize("ty", _ANNOTATABLE_TYPES)
def test_metadata_survives_every_encoded_shape(ty):
    """The type walk rebuilds a generic type from its arguments, so a caller's
    metadata has to be carried through rather than rebuilt away.

    Checked across the shapes with a *registered* encoding of their own, since
    those are the ones whose annotation is replaced wholesale: a validator the
    caller attached must still be there afterwards, or a contract written on
    such a type would silently do nothing.
    """
    marker = pydantic.AfterValidator(lambda v: v)
    assert marker in Encodable[Annotated[ty, marker]].__metadata__


@pytest.mark.parametrize("ty", _ANNOTATABLE_TYPES)
def test_metadata_does_not_change_the_encoding_itself(ty):
    """Annotating a type neither adds nor removes what the model is shown."""
    marker = pydantic.AfterValidator(lambda v: v)
    plain = pydantic.TypeAdapter(Encodable[ty])
    annotated = pydantic.TypeAdapter(Encodable[Annotated[ty, marker]])
    for mode in ("validation", "serialization"):
        try:
            expected = plain.json_schema(mode=mode)
        except Exception:
            continue  # a shape with no schema in this direction; nothing to compare
        assert annotated.json_schema(mode=mode) == expected


def test_metadata_does_not_make_an_unencodable_type_encodable():
    """Orthogonality in the other direction: metadata is not a way in.

    A type the registry cannot encode is equally undecodable annotated, and
    refuses the same way -- so attaching a contract never turns a refusal into
    a schema that promises the type.

    Stated on the *validation* schema rather than on building the adapter,
    since an unencodable type is still serializable (see
    `test_unencodable_type_serializes_but_does_not_decode`); it is being asked
    for as model *output* that has nowhere to go.
    """

    class Widget:
        pass

    marker = pydantic.AfterValidator(lambda v: v)
    for ty in (Widget, Annotated[Widget, marker]):
        schema = pydantic.TypeAdapter(Encodable[ty]).json_schema()
        assert schema["type"] == "string"
        assert "No decoding exists" in schema["description"]


# ============================================================================
# Term-specific: Encodable raises TypeError for Term and Operation
# ============================================================================


@pytest.mark.parametrize("ty", [Term, Operation])
def test_define_raises_for_invalid_types(ty):
    with pytest.raises(pydantic.errors.PydanticSchemaGenerationError):
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
# Type aliases
#
# An alias names a type; it does not make a new one. So `Encodable` must see
# through it -- to the *same* schema, not merely a workable one. Anything less
# is a trap that only springs on the types with a registered encoding: those
# are exactly the ones Pydantic would otherwise handle natively and wrongly
# (the `prefixItems` tuple schema OpenAI's strict mode rejects, #626), so an
# alias that stops the substitution produces a bad advertisement rather than
# an error, and nothing complains until a provider does.
# ============================================================================

_ALIAS_TRANSPARENCY_CASES = [
    pytest.param(_PairAlias, tuple[int, str], id="tuple"),
    pytest.param(_ComplexAlias, complex, id="complex"),
    pytest.param(_ImageAlias, Image.Image, id="image"),
    pytest.param(_PointAlias, _Point, id="dataclass"),
    pytest.param(_PairsAlias, list[tuple[int, str]], id="list-of-alias"),
    pytest.param(_LegacyPairAlias, tuple[int, str], id="legacy"),
    pytest.param(_KernelAlias, Callable[[list[float]], list[float]], id="callable"),
    pytest.param(
        Annotated[_PairAlias, "meta"],
        Annotated[tuple[int, str], "meta"],
        id="annotated",
    ),
    # The alias under a generic it did not itself introduce.
    pytest.param(list[_PairAlias], list[tuple[int, str]], id="under-generic"),
    # A generic alias, at the point it is applied.
    pytest.param(_GenPairAlias[int], tuple[int, int], id="generic-subscripted"),
    pytest.param(_GenImageAlias[str], list[tuple[str, Image.Image]], id="generic-leaf"),
]


@pytest.mark.parametrize("alias,target", _ALIAS_TRANSPARENCY_CASES)
@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_type_alias_is_transparent(alias, target, mode):
    """`Encodable[Alias]` and `Encodable[<what it aliases>]` are the same type.

    Stated on the schema rather than on a roundtrip because a roundtrip passes
    either way: Pydantic can validate a tuple whether or not the encoding was
    applied, and it is the shape shown to the model that differs.
    """
    assert pydantic.TypeAdapter(Encodable[alias]).json_schema(
        mode=mode
    ) == pydantic.TypeAdapter(Encodable[target]).json_schema(mode=mode)


def test_subscripted_generic_alias_keeps_the_tuple_encoding():
    """`type GenPair[T] = tuple[T, T]` subscripted still encodes as a tuple.

    The unsubscripted alias does, so this is not about generics being
    unsupported: the same alias loses its encoding by being *applied*.
    """
    schema = pydantic.TypeAdapter(Encodable[_GenPairAlias[int]]).json_schema()
    assert "prefixItems" not in json.dumps(schema)


def test_subscripted_generic_alias_reaches_a_registered_leaf():
    """A registered leaf encoding survives being reached through a subscripted
    generic alias -- `Image.Image` has no schema at all without it."""
    pydantic.TypeAdapter(Encodable[_GenImageAlias[str]]).json_schema()


@pytest.mark.parametrize(
    "ty", [_RecursiveAlias, _RecursiveGenAlias[int]], ids=["plain", "generic"]
)
def test_recursive_alias_does_not_diverge(ty):
    """A self-referential alias is left for Pydantic, which resolves it.

    The generic case is the one that constrains the fix: it *already* worked,
    by being passed through unexpanded, so expanding subscripted aliases had to
    come with a guard rather than replace that pass-through.
    """
    schema = json.dumps(pydantic.TypeAdapter(Encodable[ty]).json_schema())
    assert "$ref" in schema, "recursion should resolve to a reference, not inline"


# ---------------------------------------------------------------------------
# An alias as a *value*
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="The alias now reaches the type-expression encoding and gets a real "
    "schema, but `Kernel` aliases a *callable*, whose schema is `EncodedFunction` "
    "-- correct, and signature-free, as every callable schema here is. So the "
    "encoding says 'a function' without saying `list[float]`. An alias to any "
    "other shape (see the tuple case) comes out fully described.",
)
def test_type_alias_value_has_an_encoding():
    """A type alias in a skill's lexical scope can be shown to the model.

    This is the failure in the ``optimization`` example, whose ``Kernel`` is a
    PEP 695 alias *and* the declared return type of the skill. Where such an
    alias reaches the model as a *value* rather than as an annotation, its
    encoding is reconstructed from the value itself -- the two lines below,
    which are what runs. That path is unguarded, so an alias with no encoding
    takes down the whole request rather than degrading.

    A `TypeAliasType` is not a type, so it arrives as a value; the value that
    answers "what does this alias stand for" is the type it names.
    """
    typ = nested_type(_KernelAlias).value
    schema = pydantic.TypeAdapter(Encodable[typ]).json_schema()
    assert schema

    encoded = pydantic.TypeAdapter(Encodable[typ]).dump_python(
        _KernelAlias, mode="json", context={}
    )
    # Whatever shape it takes, it has to answer the question that made the
    # agent read the name: what does this alias stand for?
    assert "list[float]" in json.dumps(encoded)


# ---------------------------------------------------------------------------
# Types as values, generally
# ---------------------------------------------------------------------------


class _PlainClass:
    """A class whose ``__init__`` is unannotated, so `nested_type` reports it as
    `type` rather than reconstructing a `Callable` signature from it -- see
    `test_dataclass_value_routes_to_the_callable_encoding` for the other side."""

    def __init__(self, n):
        self.n = n


# (type expression as a value, the schema its encoding should produce)
_TYPE_VALUE_CASES = [
    pytest.param(int, {"type": "integer"}, id="builtin-class"),
    pytest.param(
        list[int], {"type": "array", "items": {"type": "integer"}}, id="generic-alias"
    ),
    pytest.param(
        int | str, {"anyOf": [{"type": "integer"}, {"type": "string"}]}, id="union"
    ),
    pytest.param(Sequence, {"type": "array", "items": {}}, id="abc"),
    pytest.param(Any, {}, id="any"),
]


@pytest.mark.parametrize("value,expected", _TYPE_VALUE_CASES)
def test_type_value_encodes_as_the_schema_of_what_it_denotes(value, expected):
    """A type expression handed over as a value encodes as its `Encodable` schema.

    The schema is the *object*, not a rendering of one, so it nests into the
    surrounding JSON instead of arriving escaped inside a string.

    Each spelling reaches the encoding as a different class -- a class and an
    ABC through `type`, a subscripted generic through `types.GenericAlias`, a
    union through `types.UnionType` -- so this is one law over the family
    rather than a property of any one of them.
    """
    typ = nested_type(value).value
    encoded = pydantic.TypeAdapter(Encodable[typ]).dump_python(
        value, mode="json", context={}
    )
    assert isinstance(encoded, dict), encoded
    assert encoded == expected


def test_type_value_schema_nests_rather_than_escaping():
    """The reason the serialized value is an object: `to_content_blocks` renders
    a nested object as readable JSON, where a string of JSON arrives escaped
    inside another string."""
    typ = nested_type(int).value
    encoded = pydantic.TypeAdapter(Encodable[typ]).dump_python(
        int, mode="json", context={}
    )
    text = "".join(b["text"] for b in to_content_blocks(encoded))
    assert text == '{"type": "integer"}'
    assert "\\" not in text


def test_class_value_is_not_encoded_as_a_function():
    """Regression: a class is callable, so `type` is a virtual subclass of
    `Callable` and singledispatch routed a class-as-value to the callable
    encoding -- which recovered `int`'s *docstring* and emitted a fabricated
    ``def int(...)``. Nothing raised; the model was simply told something
    false about every class in its lexical scope.
    """
    typ = nested_type(int).value
    encoded = pydantic.TypeAdapter(Encodable[typ]).dump_python(
        int, mode="json", context={}
    )
    assert "code" not in encoded
    assert "def int" not in json.dumps(encoded)


def test_dataclass_value_routes_to_the_callable_encoding():
    """A known asymmetry, recorded rather than claimed as desirable.

    `nested_type` reports a class as `type` *unless* it can reconstruct a full
    `Callable` signature from ``__init__``, which it can for a dataclass. So a
    dataclass handed over as a value encodes as its source -- like `_LexicalEnum`
    above, and informative in its own right -- while `int` encodes as a type.
    The two are not the same encoding, and evening them out means changing
    `nested_type` rather than anything here.
    """
    typ = nested_type(_Point).value
    encoded = pydantic.TypeAdapter(Encodable[typ]).dump_python(
        _Point, mode="json", context={}
    )
    assert "class _Point" in encoded["code"]


@pytest.mark.parametrize("value", [int, list[int], int | str], ids=str)
def test_type_value_decodes_a_real_type_but_not_its_schema(value):
    """Validation accepts the type itself and refuses the rendering of it.

    The advertised schema is an object, because a reader for one of these has
    to be advertisable at all -- but the schema is what the model *reads*, not
    something it can hand back and have reconstructed.
    """
    dec = pydantic.TypeAdapter(Encodable[nested_type(value).value])
    assert dec.validate_python(value) is value
    with pytest.raises(pydantic.ValidationError):
        dec.validate_python(dec.dump_python(value, mode="json", context={}))


def test_alias_value_does_not_satisfy_its_own_inferred_type():
    """The cost of resolving aliases in `nested_type`, recorded.

    `nested_type` reports an alias as the type of what it *aliases*, which is
    what puts a type-expression encoding in reach of the value. The alias
    object is not an instance of that, though, so `Encodable[nested_type(v)]`
    no longer validates ``v`` -- the one place the decoding-is-idempotent law
    does not hold.

    It costs nothing on the paths that matter: every `nested_type` call site is
    encoding-only, and serializing is unaffected. It would matter to a caller
    that round-trips an alias through validation, and there is none.
    """
    dec = pydantic.TypeAdapter(Encodable[nested_type(_KernelAlias).value])
    assert dec.dump_python(_KernelAlias, mode="json", context={})
    with pytest.raises(pydantic.ValidationError):
        dec.validate_python(_KernelAlias)


# ---------------------------------------------------------------------------
# The base case: a type with no encoding at all
# ---------------------------------------------------------------------------


class _Widget:
    def __init__(self, n):
        self.n = n


class _WidgetWithRepr(_Widget):
    def __repr__(self):
        return f"_WidgetWithRepr(n={self.n})"


def test_unencodable_type_serializes_but_does_not_decode():
    """A value the registry cannot encode is still *showable*.

    The two directions carry different obligations. Serializing is a
    degradation -- worst case the model reads a `repr` -- and it happens on
    paths that never asked the model for anything: a tool result, a value
    spliced into a prompt, a trace. Decoding is not, because nothing rebuilds
    an arbitrary object from that text, so the validation schema asks for a
    string that nothing satisfies rather than one that would decode.
    """
    adapter = pydantic.TypeAdapter(Encodable[_WidgetWithRepr])
    assert (
        adapter.dump_python(_WidgetWithRepr(1), mode="json") == "_WidgetWithRepr(n=1)"
    )
    assert adapter.json_schema(mode="serialization")["type"] == "string"

    validation = adapter.json_schema()
    assert validation["type"] == "string"
    assert "_WidgetWithRepr" in validation["description"]
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python("_WidgetWithRepr(n=1)")


@pytest.mark.parametrize(
    "ty",
    [_WidgetWithRepr, list[_WidgetWithRepr], tuple[int, _WidgetWithRepr]],
    ids=["bare", "nested", "in-tuple"],
)
def test_unencodable_response_format_reaches_the_provider(ty):
    """Refusing with a schema rather than an exception is what keeps a response
    format buildable, which is the obligation `Encodable` carries: a `Skill`
    returning such a type answers by calling a final-answer tool, and never gets
    to if assembling the request cannot be done at all.

    However deep the unencodable type sits, the refusal is emitted at that leaf
    and names it, leaving every other part its real schema.
    """
    box = pydantic.create_model(
        "BoxedResponse", value=Encodable[ty], __base__=_BoxedResponse
    )
    schema = litellm.utils.type_to_response_format_param(box)
    assert "_WidgetWithRepr" in json.dumps(schema)


@pytest.mark.parametrize("ty", [_UndecodableReturn, _WidgetWithRepr], ids=str)
def test_refusals_announce_themselves_on_the_wire(ty):
    """Both refusing schemas carry the title `_is_decodable` recognizes them by
    -- the one for a return type never instantiated, and the one for a type with
    no encoding -- and strict-mode post-processing leaves it alone."""
    box = pydantic.create_model(
        "BoxedResponse", value=Encodable[ty], __base__=_BoxedResponse
    )
    schema = litellm.utils.type_to_response_format_param(box)
    assert _UndecodableReturn.__schema_title__ in json.dumps(schema)


@pytest.mark.parametrize(
    "ty,expected",
    [
        (int, True),
        (Image.Image, True),
        (dict[str, int], True),
        (_WidgetWithRepr, False),
        (list[_WidgetWithRepr], False),
        (tuple[int, _WidgetWithRepr], False),
        (Operation, False),
    ],
    ids=str,
)
def test_is_decodable(ty, expected):
    """Whether the model can be *asked* for a value, as opposed to shown one.

    False covers both ways a type can fail to name something the model could
    send: a schema that refuses however deeply it sits, and no schema at all.
    """
    assert _is_decodable(ty) is expected


def test_is_decodable_looks_past_an_encoding_that_replaces_its_arguments():
    """A refusal only counts where the model would actually meet it.

    `SkillBody` is asked for as source and decoded by compiling it, so its own
    schema stands in for its arguments' -- which is what lets a Skill returning
    an undecodable type still be answered by synthesizing one, the whole point
    of refusing a direct reply. Reading the type rather than the schema it
    generates gets this backwards and withdraws the tool that was the way out.
    """
    from effectful.handlers.llm.harness.synthesis.body import SkillBody

    assert not _is_decodable(_WidgetWithRepr)
    assert _is_decodable(SkillBody[[int], _WidgetWithRepr])


def test_unencodable_element_does_not_block_encoding_its_container():
    """A container is sendable when its parts are. One part having no decoding
    does not change that: the refusal is that element's, in one direction."""
    adapter = pydantic.TypeAdapter(Encodable[tuple[int, _WidgetWithRepr]])
    assert adapter.dump_python((7, _WidgetWithRepr(1)), mode="json", context={}) == {
        "item_0": 7,
        "item_1": "_WidgetWithRepr(n=1)",
    }


def test_unencodable_value_rendering_is_stable_across_runs():
    """The default `object.__repr__` embeds an address, and this text goes into
    a message history that is persisted and replayed -- so it is not used."""
    encoded = pydantic.TypeAdapter(Encodable[_Widget]).dump_python(
        _Widget(1), mode="json"
    )
    assert "0x" not in encoded
    assert encoded == pydantic.TypeAdapter(Encodable[_Widget]).dump_python(
        _Widget(2), mode="json"
    )


def test_base_case_leaves_types_pydantic_already_handles_alone():
    """The fallback is for what Pydantic *cannot* do, not a replacement for it:
    a dataclass, a model, a builtin all keep their own schema."""
    for ty in (_Point, _PointModel, int, list[int]):
        schema = pydantic.TypeAdapter(Encodable[ty]).json_schema()
        assert schema.get("type") != "string" or ty is str, ty


# ============================================================================
# DecodedToolCall-specific: error cases
# ============================================================================

TOOL_CALL_ERROR_CASES = [
    pytest.param(
        "nonexistent",
        "{}",
        {_NAME2TOOL_KEY: {}},
        (KeyError, AssertionError),
        id="unknown-tool",
    ),
    pytest.param(
        "_tool_add",
        '{"a": "not_an_int", "b": 2}',
        {_NAME2TOOL_KEY: {"_tool_add": _tool_add}},
        pydantic.ValidationError,
        id="wrong-arg-type",
    ),
    pytest.param(
        "_tool_add",
        '{"a": 1}',
        {_NAME2TOOL_KEY: {"_tool_add": _tool_add}},
        (pydantic.ValidationError, TypeError),
        id="missing-required-arg",
    ),
    pytest.param(
        "_tool_add",
        '{"a": 1, "b": 2, "c": 3}',
        {_NAME2TOOL_KEY: {"_tool_add": _tool_add}},
        pydantic.ValidationError,
        id="extra-arg",
    ),
    pytest.param(
        "_tool_add",
        "{not valid json}",
        {_NAME2TOOL_KEY: {"_tool_add": _tool_add}},
        pydantic.ValidationError,
        id="invalid-json",
    ),
    pytest.param(
        "_tool_process",
        '{"items": ["a", "b"], "label": "total"}',
        {_NAME2TOOL_KEY: {"_tool_process": _tool_process}},
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
    pytest.param(BuiltinExecutor(), id="unsafe"),
    pytest.param(RestrictedPythonExecutor(), id="restricted"),
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
    with handler(TyTypeChecker()), handler(eval_provider):
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
    with handler(TyTypeChecker()), handler(eval_provider):
        decoded = enc.validate_python(json.loads(text), context=ctx)
    assert decoded(*args) == expected


# A Skill-style anchor whose return type is the Callable being decoded.
# Decoding only runs the (source-anchored) type check when an anchor is in scope
# (bound by call_agent); the result path has one, the argument path does
# not. The return-type case needs the body checked against the expected signature,
# so it provides an anchor; the structural cases (param count, missing/last-stmt)
# are caught without one.
def _int_pair_anchor() -> Callable[[int, int], int]:
    raise NotImplementedError


# Callable error cases: (type, ctx, source, exc_type, anchor)
#
# Sources are passed as raw ``{"code": ...}`` dicts, not pre-built
# ``SynthesizedFunction`` instances: structurally-invalid code (e.g. a non-function
# last statement) is rejected by ``SynthesizedFunction``'s own field validator, so
# building it eagerly here would raise at collection. A dict defers that validation
# to the decoder (``model_validate``), which is the real path an LLM's JSON takes.
CALLABLE_ERROR_CASES = [
    pytest.param(
        Callable[..., int],
        {},
        {"code": "x = 42"},
        ValueError,
        None,
        id="non-function-last-stmt",
    ),
    pytest.param(
        Callable[[int, int], int],
        {},
        {"code": "def add(a: int) -> int:\n    return a"},
        ValueError,
        None,
        id="wrong-param-count",
    ),
    pytest.param(
        Callable[[int, int], int],
        {},
        {"code": "def add(a: int, b: int) -> str:\n    return str(a + b)"},
        TypeError,
        _int_pair_anchor,
        id="wrong-return-type",
    ),
    pytest.param(
        Callable[[int, int], int],
        {},
        {"code": "def add(a: int, b: int):\n    return a + b"},
        ValueError,
        None,
        id="missing-return-annotation",
    ),
]


@pytest.mark.parametrize("ty,ctx,source,exc_type,anchor", CALLABLE_ERROR_CASES)
@pytest.mark.parametrize("eval_provider", EVAL_PROVIDERS)
def test_callable_decode_rejects_invalid(
    ty, ctx, source, exc_type, anchor, eval_provider
):
    with pytest.raises(exc_type):
        with handler(TyTypeChecker()), handler(eval_provider):
            pydantic.TypeAdapter(Encodable[ty]).validate_python(
                source, context={**ctx, _TYPE_CHECK_ANCHOR_KEY: anchor}
            )


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


# response_model xfails:
#   - image: LLM returns URLs, not data URIs
#   - tool/dtc: ChatCompletionToolParam schema has optional fields and bare
#     "type": "object" without properties — incompatible with OpenAI strict mode
#   - tuple-bare: no type information for structured output
RESPONSE_MODEL_CASES = _apply_xfails(
    ROUNDTRIP_CASES,
    lambda cid: (
        cid.startswith("img-")
        or "-img" in cid
        or cid.startswith("tool-")
        or cid.startswith("dtc-")
        or cid == "tuple-bare"
    ),
)

# tool-as-param xfails: same as response_model for tool/dtc, plus bare tuple.
TOOL_PARAM_CASES = _apply_xfails(
    ROUNDTRIP_CASES,
    lambda cid: (
        cid == "tuple-bare" or cid.startswith("tool-") or cid.startswith("dtc-")
    ),
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
    name = getattr(ty, "__name__", repr(ty))
    # What's under test is that the type survives as a response_model -- the schema is
    # accepted and what comes back decodes. A model that answers with nothing says
    # nothing about that either way, so retry an empty completion rather than read it
    # as a failure of the type. (Seen occasionally on degenerate schemas, e.g. the
    # empty tuple, whose schema admits exactly one instance: `{"value": {}}`.)
    for _attempt in range(3):
        response = litellm.completion(
            model=EFFECTFUL_LLM_MODEL,
            response_format=response_format,
            messages=[{"role": "user", "content": f"Return an instance of {name}."}],
            max_tokens=400,
        )
        assert isinstance(response, litellm.ModelResponse)

        choice = response.choices[0]
        assert isinstance(choice, litellm.Choices)
        content = choice.message.content
        if content:
            break
    assert content, f"Expected content in response for {name}"

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
    enc: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(Encodable[_NameAndTool])
    tool_spec = enc.dump_python(
        _NameAndTool(tool.__name__, tool), mode="json", context=ctx or {}
    )
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
    enc: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(Encodable[_NameAndTool])
    tool_spec = enc.dump_python(
        _NameAndTool(tool.__name__, tool), mode="json", context=ctx or {}
    )
    response = litellm.completion(
        model=EFFECTFUL_LLM_MODEL,
        messages=[{"role": "user", "content": "Return hello, do NOT call any tools."}],
        tools=[tool_spec],
        tool_choice="none",
        max_tokens=400,
    )
    assert isinstance(response, litellm.ModelResponse)


# ============================================================================
# Encodable[CodeType] -- syntax checking at the Encodable boundary
# ============================================================================


def test_encodable_code_compiles_source_to_a_code_object():
    """Decoding `Encodable[CodeType]` compiles the source through the eval
    provider, yielding a ready-to-run code object."""
    src = "x = 1\nprint(x)\n"
    adapter = pydantic.TypeAdapter(Encodable[CodeType])
    with handler(TyTypeChecker()), handler(BuiltinExecutor()):
        decoded = adapter.validate_python(src)
    assert isinstance(decoded, CodeType)


def test_encodable_code_round_trips_to_source():
    """Re-encoding a decoded code object recovers its source string (from
    `linecache`)."""
    src = "a = 2\n"
    adapter = pydantic.TypeAdapter(Encodable[CodeType])
    with handler(TyTypeChecker()), handler(BuiltinExecutor()):
        decoded = adapter.validate_python(src)
        assert adapter.dump_python(decoded) == src


def test_encodable_code_rejects_syntax_error():
    """Source that does not parse is rejected at decode."""
    with handler(TyTypeChecker()), handler(BuiltinExecutor()):
        with pytest.raises(SyntaxError):
            pydantic.TypeAdapter(Encodable[CodeType]).validate_python("def f(:")


def test_encodable_code_rejects_compile_only_error():
    """`return` outside a function parses but does not compile -- still rejected,
    so the check is `compile`, not merely `ast.parse`."""
    with handler(TyTypeChecker()), handler(BuiltinExecutor()):
        with pytest.raises(SyntaxError):
            pydantic.TypeAdapter(Encodable[CodeType]).validate_python("return 5")


def test_encodable_code_schema_is_a_string():
    """The LLM sees a `CodeType` parameter as a plain string."""
    schema = pydantic.TypeAdapter(Encodable[CodeType]).json_schema()
    assert schema["type"] == "string"


# ============================================================================
# Serializing a callable: `Encodable[Callable]`'s two directions carry different
# obligations
#
# Validation decodes code the model *wrote*, and holds it to `SynthesizedFunction`'s
# constraints. Serialization encodes a value that already exists, which was never
# under those constraints -- a class read out of the lexical scope, or an inner
# function a Skill *body* returned. Conflating the two aborted the enclosing call
# with a `ValidationError` raised from inside pydantic's serializer.
# ============================================================================


def _outer_returning_unannotated():
    """Stands in for a synthesized Skill body: `SkillBody` waives annotations,
    so the function it returns need not have any."""

    def step(state, action):
        return state + action

    return step


class _LexicalEnum(int):
    """Stands in for a class reaching the model from lexical scope -- a *callable*,
    so it routes to the `Callable` serializer."""


@pytest.mark.parametrize(
    "label,value",
    [
        ("inner function with unannotated parameters", _outer_returning_unannotated()),
        ("a class, whose source is a `class` statement", _LexicalEnum),
    ],
)
def test_serialize_callable_does_not_reapply_synthesis_constraints(label, value):
    """Serializing recovers the value's source and encodes it, rather than
    re-validating it as if the model had just submitted it for synthesis."""
    encoded = pydantic.TypeAdapter(Encodable[Callable[[int, int], int]]).dump_python(
        value, mode="json", context={}
    )
    assert "code" in encoded, label
    assert encoded["code"].strip(), label


def test_serialize_callable_matches_its_declared_schema():
    """What serialization emits is what its JSON schema promises: the plain
    `EncodedFunction` shape, with none of the synthesis constraints attached."""
    adapter = pydantic.TypeAdapter(Encodable[Callable[[int, int], int]])
    schema = adapter.json_schema(mode="serialization")
    encoded = adapter.dump_python(
        _outer_returning_unannotated(), mode="json", context={}
    )
    assert set(encoded) <= set(schema["properties"])


@pytest.mark.parametrize(
    "ty", [Tool, Skill], ids=["tool", "skill"]
)  # `Skill` reaches the same encoding through its base
def test_serialize_tool_value_encodes_the_callable_it_is(ty):
    """A `Tool` arriving as a *value* -- returned by another tool, spliced into a
    prompt -- encodes as its source, like any other callable.

    The `ChatCompletionToolParam` advertisement is the encoding of `_NameAndTool`
    (above), not of `Tool`: it needs a name, which a bare `Tool` does not carry.
    """
    encoded = pydantic.TypeAdapter(Encodable[ty]).dump_python(
        _tool_add, mode="json", context={}
    )
    assert "def _tool_add" in encoded["code"]
