"""Conversion between Python values and the model's wire format.

Python values are converted to the content blocks, tool schemas and JSON
payloads exchanged with the model, and the model's output is converted back.
"""

import abc
import base64
import collections.abc
import contextvars
import dataclasses
import functools
import inspect
import io
import json
import re
import string
import textwrap
import typing

import litellm
import pydantic
import typing_extensions
from litellm import (
    ChatCompletionImageObject,
    ChatCompletionMessageToolCall,
    ChatCompletionTextObject,
    ChatCompletionToolParam,
    OpenAIMessageContentListBlock,
)
from openai.types.chat import (
    ChatCompletionMessageToolCall as OpenAIChatCompletionMessageToolCall,
)
from PIL import Image

from effectful.handlers.llm.types import Encodable, Tool
from effectful.internals.unification import (
    GenericAlias,
    TypeEvaluator,
    UnionType,
    canonicalize,
    nested_type,
)
from effectful.ops.types import Operation, Term

type ToolCallID = str

# Keys under which special metadata are stashed in the Pydantic decoding context.
# Deliberately not identifiers so they cannot be confused with lexical variables.
_NAME2TOOL_KEY: typing.Literal["$NAME2TOOL"] = "$NAME2TOOL"
_IS_FINAL_KEY: typing.Literal["$IS_FINAL"] = "$IS_FINAL"
_TYPE_CHECK_ANCHOR_KEY: typing.Literal["$TYPE_CHECK_ANCHOR"] = "$TYPE_CHECK_ANCHOR"


CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    literal
    for member in typing.get_args(OpenAIMessageContentListBlock)
    for literal in typing.get_args(typing.get_type_hints(member).get("type", str))
    if isinstance(literal, str)
)


def _is_empty_text_block(block: typing.Any) -> bool:
    """Whether `block` is a text block with no text."""
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and not block.get("text")
    )


def _text_blocks(text: str) -> list[OpenAIMessageContentListBlock]:
    """`text` as one content block, or no block at all when `_is_empty_text_block`
    would reject it."""
    block = ChatCompletionTextObject(type="text", text=text)
    return [] if _is_empty_text_block(block) else [block]


@pydantic.validate_call(validate_return=True)
def to_content_blocks(
    value: typing.Any,
) -> collections.abc.Sequence[OpenAIMessageContentListBlock]:
    """Convert an encoded JSON-compatible value into a flat list of content blocks.

    Walks the value tree, extracting content-block-shaped dicts (identified by
    their ``type`` discriminator) and emitting JSON syntax as text around them.

    Top-level strings are emitted bare (for natural template rendering).
    Inside JSON structures, separators match ``json.dumps`` defaults so that
    the linearization law holds for non-string encoded values:
    ``linearize(to_content_blocks(v)) == json.dumps(v)``.

    Every text block goes through `_text_blocks`, so none of them is empty;
    Anthropic rejects a request containing one.
    """
    if isinstance(value, str):
        return _text_blocks(value)

    buf: list[str] = []
    blocks: list[OpenAIMessageContentListBlock] = []

    def flush() -> None:
        blocks.extend(_text_blocks("".join(buf)))
        buf.clear()

    def walk(v: typing.Any) -> None:
        if isinstance(v, dict) and v.get("type") in CONTENT_BLOCK_TYPES:
            flush()
            blocks.append(typing.cast(OpenAIMessageContentListBlock, v))
        elif isinstance(v, dict):
            buf.append("{")
            for i, (k, val) in enumerate(v.items()):
                if i:
                    buf.append(", ")
                buf.append(json.dumps(k) + ": ")
                walk(val)
            buf.append("}")
        elif isinstance(v, list):
            buf.append("[")
            for i, item in enumerate(v):
                if i:
                    buf.append(", ")
                walk(item)
            buf.append("]")
        else:
            buf.append(json.dumps(v))

    walk(value)
    flush()
    return blocks


def format_as_content_blocks(
    template: str,
    env: collections.abc.Mapping[str, typing.Any],
) -> list[OpenAIMessageContentListBlock]:
    """
    Format a template applied to arguments into a list of content blocks.
    This is similar to str.format() but produces a list of content blocks
    instead of a single string, so that non-text content is preserved.

    A conversion or format spec runs even on a value that encodes to ``""``, so
    ``{x!r}`` renders ``''``. Text that formats to nothing produces no block.
    """
    formatter = string.Formatter()
    parts: list[OpenAIMessageContentListBlock] = []

    buf: list[str] = []

    def flush_text() -> None:
        parts.extend(_text_blocks("".join(buf)))
        buf.clear()

    for literal, field_name, format_spec, conversion in formatter.parse(
        textwrap.dedent(template)
    ):
        if literal:
            buf.append(literal)

        if field_name is None:
            continue

        def formatted(text: str, conversion=conversion, spec=format_spec) -> str:
            if conversion:
                text = formatter.convert_field(text, conversion)
            return formatter.format_field(text, spec or "")

        obj, _ = formatter.get_field(field_name, (), env)
        encoder: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
            Encodable[nested_type(obj).value]  # type: ignore[misc]
        )
        encoded_obj = encoder.dump_python(obj, mode="json", context=env)
        if isinstance(encoded_obj, str):
            # Formatted here rather than through `to_content_blocks`, which
            # drops an empty string before a conversion or format spec could
            # turn it into something.
            buf.append(formatted(encoded_obj))
            continue
        for part in to_content_blocks(encoded_obj):
            if part["type"] == "text":
                buf.append(formatted(part["text"]))
            else:
                flush_text()
                parts.append(part)

    flush_text()

    return parts


class PromptSection(typing.TypedDict):
    type: typing.Literal["prompt_section"]
    title: str
    content: collections.abc.Sequence[
        typing.Union[OpenAIMessageContentListBlock, "PromptSection"]
    ]


# Matches an ATX heading's leading ``#``s (1-6, followed by whitespace) at the
# start of a line, e.g. ``## Foo``. The lookahead avoids matching ``#!`` or a
# ``#tag`` that is not a heading.
_ATX_HEADING = re.compile(r"^(#{1,6})(?=\s)")


def _shift_headings(md: str, by: int) -> str:
    """Shift every ATX heading in `md` by `by` levels (clamped to 1..6).

    Fenced code blocks (``` ``` ``` / ``` ~~~ ```) are skipped so ``#`` inside code --
    Python comments, shell shebangs -- is left untouched.
    """
    if by == 0 or not md:
        return md
    out: list[str] = []
    fence: str | None = None
    for line in md.splitlines():
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
        elif fence is not None and stripped.startswith(fence):
            fence = None
        elif fence is None:
            m = _ATX_HEADING.match(line)
            if m:
                level = max(1, min(6, len(m.group(1)) + by))
                line = "#" * level + line[m.end(1) :]
        out.append(line)
    return "\n".join(out)


def _rebase_headings(md: str, top: int) -> str:
    """Renumber the headings in `md` so its shallowest one sits at level `top`,
    preserving relative nesting; text with no headings is returned unchanged.

    Applied to every text block as a prompt is rendered, so a docstring written
    with its own heading hierarchy nests beneath the section that carries it and
    the assembled document has a single coherent outline.
    """
    if not md:
        return md
    fence: str | None = None
    levels: list[int] = []
    for line in md.splitlines():
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
        elif fence is not None and stripped.startswith(fence):
            fence = None
        elif fence is None:
            m = _ATX_HEADING.match(line)
            if m:
                levels.append(len(m.group(1)))
    if not levels:
        return md
    return _shift_headings(md, top - min(levels))


def _render_prompt_section(
    prompt: PromptSection, level: int = 0
) -> list[OpenAIMessageContentListBlock]:
    """Flatten an assembled prompt into content blocks.

    Top-level sections are rendered as ``#`` headings and nest from there, with
    each block of text rebased below the section carrying it, so a docstring
    written with its own heading hierarchy joins one coherent outline.  Runs of
    text are coalesced, separated by blank lines: the result is a single Markdown
    document, interrupted only by whatever non-text blocks it carries -- an image
    in a section reaches the model as an image.

    Recurs on each `PromptSection` at `level`, whose `title` becomes its heading.
    A section that renders to nothing -- an unfilled harness section, say --
    leaves no stray heading behind.  At `level` 0 sits the whole document, the
    section `call_system` assembles from its two arguments; its `title` names the
    document for a `SystemPromptDumper` or a trace and is not rendered, so the
    arguments themselves are the document's ``#`` headings.
    """
    blocks: list[OpenAIMessageContentListBlock] = []

    def emit(block: OpenAIMessageContentListBlock) -> None:
        """Add `block`, running text into the text block before it, if any."""
        if block["type"] == "text" and blocks and blocks[-1]["type"] == "text":
            joined = f"{blocks[-1]['text']}\n\n{block['text']}"
            blocks[-1] = {**blocks[-1], "text": joined}
        else:
            blocks.append(block)

    for item in prompt["content"]:
        if item.get("type") == "prompt_section":
            for block in _render_prompt_section(
                typing.cast(PromptSection, item), level + 1
            ):
                emit(block)
        elif item.get("type") == "text":
            text = _rebase_headings(
                typing.cast(str, item.get("text") or ""), level + 1
            ).strip()
            if text:
                emit(ChatCompletionTextObject(type="text", text=text))
        else:
            emit(typing.cast(OpenAIMessageContentListBlock, item))

    if not level:
        return blocks
    if not blocks:
        return []
    heading = f"{'#' * min(level, 6)} {prompt['title']}"
    if blocks[0]["type"] == "text":
        joined = f"{heading}\n\n{blocks[0]['text']}"
        return [{**blocks[0], "text": joined}, *blocks[1:]]
    return [ChatCompletionTextObject(type="text", text=heading), *blocks]


@dataclasses.dataclass(frozen=True, eq=True)
class DecodedToolCall[T]:
    """
    Structured representation of a tool call decoded from an LLM response.
    """

    tool: Tool[..., T]
    bound_args: inspect.BoundArguments
    id: ToolCallID
    name: str

    # The Python source this call was decoded from, when the model produced a
    # call *expression* rather than JSON arguments (see
    # `~effectful.handlers.llm.harness.synthesis.toolcall.CallExpression`, which
    # narrows this to required).  `None` means the ordinary JSON pathway.  This
    # is what `_serialize_tool_call` must emit to round-trip the call: the
    # advertised schema for an expression call is ``{"call": <source>}``, and
    # its evaluated `bound_args` may hold values with no JSON encoding at all.
    source: str | None = None

    @property
    def result_type(self) -> type[T]:
        return inspect.signature(self.tool).return_annotation


class _NameAndTool(typing.NamedTuple):
    """A `Tool` together with the name it is advertised to the model under.

    A name is a property of the advertisement, not of the tool: the same `Tool`
    can be offered to two different requests under two different names, and two
    tools sharing a ``__name__`` (an `Agent` method bound to two instances, say)
    must still be told apart.  `call_assistant` assigns the names and pairs each
    one with its tool here, immediately before encoding; nothing else constructs
    or consumes a `_NameAndTool`.

    This exists so that the `ChatCompletionToolParam` encoding below can hang off
    a type that actually *is* a tool advertisement, leaving `Tool` itself free to
    encode as what it is -- a callable.
    """

    name: str
    tool: Tool


# TODO move upstream to unification.py
@nested_type.register
def _nested_type_alias(ty: typing.TypeAliasType):
    return nested_type(ty.__value__)


# TODO move upstream to unification.py
def _expand_alias(
    evaluator: TypeEvaluator, typ: typing.TypeAliasType, value: typing.Any
):
    """Evaluate what ``typ`` aliases, unless ``typ`` is already being expanded."""
    if not hasattr(evaluator, "_expanding_aliases"):
        setattr(evaluator, "_expanding_aliases", set())
    seen: set[typing.Any] = getattr(evaluator, "_expanding_aliases")
    if typ in seen:
        return typ
    seen.add(typ)
    try:
        return evaluator.evaluate(value)
    finally:
        seen.discard(typ)


# TODO move upstream to unification.py
@TypeEvaluator.evaluate.register  # type: ignore[attr-defined]
def _evaluate_type_alias(self, typ: typing.TypeAliasType):
    return _expand_alias(self, typ, typ.__value__)


# TODO move upstream to unification.py
@TypeEvaluator.evaluate.register  # type: ignore[attr-defined]
def _evaluate_generic_alias(self, typ: GenericAlias):
    origin, args = typing.get_origin(typ), typing.get_args(typ)
    if isinstance(origin, typing.TypeAliasType):
        return _expand_alias(self, typ, origin.__value__[args])
    return origin[self.evaluate(args)]  # type: ignore[index]


# TODO move upstream to unification.py
_CANONICALIZING: contextvars.ContextVar[frozenset] = contextvars.ContextVar(
    "_CANONICALIZING", default=frozenset()
)


# TODO move upstream to unification.py
@dataclasses.dataclass(frozen=True)
class _SelfReferentialAlias(Exception):
    typ: typing.TypeAliasType


# TODO move upstream to unification.py
@canonicalize.register
def _canonicalize_type_alias(typ: typing.TypeAliasType) -> typing.Any:
    """Canonicalize what the alias names, unless it names itself."""
    seen = _CANONICALIZING.get()
    if typ in seen:
        raise _SelfReferentialAlias(typ)
    token = _CANONICALIZING.set(seen | {typ})
    try:
        return canonicalize(typ.__value__)
    except _SelfReferentialAlias as e:
        if e.typ is not typ:
            raise  # another alias's recursion; the frame expanding it will catch
        return typ
    finally:
        _CANONICALIZING.reset(token)


class TypeToPydanticType(TypeEvaluator):
    """Substitute custom types with their Pydantic Annotated equivalents.

    Recursively walks a type annotation tree, replacing leaf types that have
    registered Pydantic annotations (e.g., Image.Image -> PydanticImage) and
    reconstructing the full generic type.

    The result can be passed to pydantic.TypeAdapter() for automatic
    validation and serialization of nested structures.
    """

    @staticmethod
    @functools.singledispatch
    def _registry(ty: type):
        raise RuntimeError("should not be here!")

    @classmethod
    def register(cls, *args, **kwargs):
        return cls._registry.register(*args, **kwargs)

    def evaluate(self, ty):
        app = super().evaluate(ty)
        origin = typing.get_origin(app)
        # Only dispatch on regular types. Special forms (Literal, Annotated,
        # Union) have non-type origins that singledispatch can't resolve; pass
        # them through for Pydantic to handle natively.
        if isinstance(app, type | GenericAlias) and (
            origin is None or isinstance(origin, type)
        ):
            return self._registry.dispatch(origin or app)(app)
        else:
            return app


@TypeToPydanticType.register(str)
def _pydantic_type_str[T](ty: type[T]) -> type[T]:
    return ty


def _serialize_unencodable(value: typing.Any) -> str:
    """Render a value whose type has no encoding, as text.

    Whichever of `repr` and `str` the type actually defines, preferring `repr`
    as the more precise of the two.  Falls back to the type's name only when it
    defines neither, because `object.__repr__` prints an address, which differs
    between two runs of the same conversation -- and this text goes into a
    message history that is persisted and replayed.
    """
    cls = type(value)
    if cls.__repr__ is not object.__repr__:
        return repr(value)
    if cls.__str__ is not object.__str__:
        return str(value)
    return f"<{inspect.formatannotation(cls)}>"


@TypeToPydanticType.register(object)
def _pydantic_type_base(ty: typing.Any) -> typing.Any:
    """Pydantic's own handling, or a serialize-only encoding if it has none."""
    try:
        pydantic.TypeAdapter(ty)
        return ty
    except pydantic.errors.PydanticSchemaGenerationError:
        name = inspect.formatannotation(ty)
        return typing.Annotated[
            ty,
            pydantic.InstanceOf,
            pydantic.PlainSerializer(
                _serialize_unencodable,
                return_type=typing.Annotated[str, pydantic.Field(description=name)],
            ),
            pydantic.BeforeValidator(
                lambda value: value,
                json_schema_input_type=typing.Annotated[
                    str,
                    pydantic.Field(
                        title=_UndecodableReturn.__schema_title__,
                        description=(
                            f"No decoding exists for `{name}`, so a direct reply of "
                            f"it cannot be decoded. Do not answer directly: call a "
                            f"tool that produces a final answer instead."
                        ),
                    ),
                ],
            ),
        ]


def _best_effort_schema(
    annotation: typing_extensions.TypeForm,
    mode: typing.Literal["validation", "serialization"] = "serialization",
) -> dict[str, typing.Any]:
    """The `Encodable` JSON schema of ``annotation``, or a fallback"""
    try:
        return pydantic.TypeAdapter(Encodable[annotation]).json_schema(mode=mode)  # type: ignore
    except (pydantic.errors.PydanticUserError, TypeError):
        return {"description": inspect.formatannotation(annotation)}


def _is_decodable(annotation: typing_extensions.TypeForm) -> bool:
    """Whether the model can be asked to produce a value of ``annotation``."""

    def refuses(node: typing.Any) -> bool:
        if isinstance(node, dict):
            return node.get("title") == _UndecodableReturn.__schema_title__ or any(
                refuses(v) for v in node.values()
            )
        return isinstance(node, list) and any(refuses(v) for v in node)

    try:
        return not refuses(pydantic.TypeAdapter(Encodable[annotation]).json_schema())  # type: ignore
    except Exception:
        return False


@TypeToPydanticType.register(type)
@TypeToPydanticType.register(abc.ABCMeta)
@TypeToPydanticType.register(GenericAlias)
@TypeToPydanticType.register(UnionType)
def _pydantic_type_type(ty: typing.Any) -> typing.Any:
    """Encode a type *value* as the JSON schema of what it denotes."""
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.PlainSerializer(
            _best_effort_schema,
            return_type=typing.Annotated[
                dict[str, typing.Any],
                pydantic.Field(
                    description=(
                        "A Python type, as the JSON schema of its encoding. Shown "
                        "for reference; a type cannot be reconstructed from its "
                        "schema."
                    )
                ),
            ],
        ),
    ]


class _UndecodableReturn:
    """A concrete type could not be instantiated for this value, so a direct
    reply cannot be decoded soundly. Do not answer directly: call a tool that
    produces a final answer instead."""

    __schema_title__: typing.ClassVar[typing.Literal["$UNDECODABLE"]] = "$UNDECODABLE"


def _fail_validation(value: typing.Any) -> typing.Any:
    raise ValueError(inspect.getdoc(_UndecodableReturn))


@TypeToPydanticType.register(_UndecodableReturn)
def _pydantic_type_undecodable_return(ty: type[_UndecodableReturn]) -> typing.Any:
    return typing.Annotated[
        str,
        pydantic.PlainValidator(_fail_validation, json_schema_input_type=str),
        pydantic.Field(
            title=_UndecodableReturn.__schema_title__,
            description=inspect.getdoc(_UndecodableReturn),
        ),
    ]


class _ComplexModel(typing.TypedDict):
    real: float
    imag: float


@pydantic.validate_call(validate_return=True)
def _validate_complex(value: _ComplexModel | complex) -> complex:
    return (
        value if isinstance(value, complex) else complex(value["real"], value["imag"])
    )


@pydantic.validate_call(validate_return=True)
def _serialize_complex(value: complex) -> _ComplexModel:
    return {"real": value.real, "imag": value.imag}


@TypeToPydanticType.register(complex)
def _pydantic_type_complex(ty):
    """Encode ``complex`` as ``{"real": float, "imag": float}``."""

    return typing.Annotated[
        ty,
        pydantic.BeforeValidator(
            _validate_complex, json_schema_input_type=_ComplexModel
        ),
        pydantic.PlainSerializer(_serialize_complex, return_type=_ComplexModel),
    ]


@TypeToPydanticType.register(tuple)
def _pydantic_type_tuple(ty):
    """Convert finitary tuples to object-based schemas (``properties/required``).

    OpenAI's strict mode rejects the ``prefixItems`` array schema that Pydantic
    emits for fixed-length tuples.  We convert them to a Pydantic model with
    positional ``item_0``, ``item_1``, … fields instead.

    NamedTuples are handled similarly using their field names.
    Bare ``tuple`` and variadic ``tuple[T, ...]`` are passed through unchanged.
    """
    # NamedTuple subclasses dispatch here via MRO; use field names.
    if isinstance(ty, type) and hasattr(ty, "_fields"):
        hints = typing.get_type_hints(ty)
        nt_fields: list[str] = list(ty._fields)
        nt_types = [hints.get(f, typing.Any) for f in nt_fields]
        nt_model = pydantic.create_model(
            ty.__name__,
            __config__={"extra": "forbid"},
            __doc__=ty.__doc__,
            **{f: (t, ...) for f, t in zip(nt_fields, nt_types)},
        )

        def _nt_serialize(value):
            return nt_model.model_construct(**{f: getattr(value, f) for f in nt_fields})

        def _nt_decode(value):
            """Reshape the named-field object form back into the positional tuple
            Pydantic validates a `NamedTuple` from."""
            if isinstance(value, collections.abc.Mapping):
                return tuple(value[f] for f in nt_fields)
            return value

        return typing.Annotated[
            ty,
            pydantic.BeforeValidator(_nt_decode, json_schema_input_type=nt_model),
            pydantic.PlainSerializer(_nt_serialize, return_type=nt_model),
        ]

    args = typing.get_args(ty)

    # Bare tuple or tuple[T, ...] — Pydantic's native handling is fine.
    # Note: tuple[()] also has get_args() == (), but has origin=tuple.
    if (not args and typing.get_origin(ty) is None) or (
        len(args) == 2 and args[1] is Ellipsis
    ):
        return ty

    # tuple[()] (empty args with origin) maps to zero fields; otherwise use args.
    effective: list[typing.Any] = list(args)

    model = pydantic.create_model(
        "TupleItems",
        __config__={"extra": "forbid"},
        **{f"item_{i}": (a, ...) for i, a in enumerate(effective)},
    )

    def _decode(value):
        """Reshape the ``item_0``/``item_1``/... object form back into a positional
        tuple, leaving the tuple itself for Pydantic to validate elementwise."""
        if isinstance(value, collections.abc.Mapping):
            return tuple(value[f"item_{i}"] for i in range(len(effective)))
        return value

    def _serialize(value):
        return model.model_construct(**{f"item_{i}": v for i, v in enumerate(value)})

    return typing.Annotated[
        ty,
        pydantic.BeforeValidator(_decode, json_schema_input_type=model),
        pydantic.PlainSerializer(_serialize, return_type=model),
    ]


@TypeToPydanticType.register(Term)
def _pydantic_type_term(ty: type[Term]):
    raise pydantic.errors.PydanticSchemaGenerationError(
        "Terms cannot be converted to Pydantic types."
    )


@TypeToPydanticType.register(Operation)
def _pydantic_type_operation(ty: type[Operation]):
    raise pydantic.errors.PydanticSchemaGenerationError(
        "Operations cannot be converted to Pydantic types."
    )


def _validate_image(value: ChatCompletionImageObject | Image.Image) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    value = pydantic.TypeAdapter(ChatCompletionImageObject).validate_python(value)
    image_url: litellm.ChatCompletionImageUrlObject | str = value["image_url"]
    url: str = image_url["url"] if isinstance(image_url, dict) else image_url
    prefix, data = url.split(",")
    if not prefix.startswith("data:image/"):
        raise ValueError(f"expected base64 encoded image as data uri, received {url}")
    return Image.open(fp=io.BytesIO(base64.b64decode(data)))


def _serialize_image(value: Image.Image) -> ChatCompletionImageObject:
    buf = io.BytesIO()
    value.save(buf, format="PNG")
    url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    return pydantic.TypeAdapter(ChatCompletionImageObject).validate_python(
        {"type": "image_url", "image_url": {"detail": "auto", "url": url}}
    )


@TypeToPydanticType.register(Image.Image)
def _pydantic_type_image(ty: type[Image.Image]):
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.BeforeValidator(
            _validate_image, json_schema_input_type=ChatCompletionImageObject
        ),
        pydantic.PlainSerializer(
            _serialize_image, return_type=ChatCompletionImageObject
        ),
    ]


# The *serialization* view of a synthesized callable: the shape the model reads when
# a function is handed to it as a value (e.g. a tool's return) -- the bare source,
# with none of the synthesis instructions the validation direction carries.
EncodedFunction = typing.Annotated[
    str,
    pydantic.Field(
        description="A function, as a string of its complete Python source."
    ),
]


def _serialize_callable(value: collections.abc.Callable) -> str:
    """Encode a callable as its source, or as a stub when there is none.

    The synthesis constraints ("the last statement must be a function definition",
    "every parameter must be annotated") are demands on code a model is *writing*;
    a value being serialized is under no such obligation -- it may be a class from a
    lexical-scope read, or an inner function a Skill body returned -- so they are
    deliberately not re-applied here.
    """
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        source = None

    if source:
        return textwrap.dedent(source)

    name = getattr(value, "__name__", None)
    docstring = inspect.getdoc(value)
    if name is None or docstring is None:
        raise ValueError(
            f"Cannot encode callable {value}: no source code and no __name__ or docstring"
        )

    try:
        sig_str = str(inspect.signature(value))
    except (ValueError, TypeError):
        sig_str = "(...)"

    return f'''def {name}{sig_str}:
    """{docstring}"""
    ...
'''


@TypeToPydanticType.register(collections.abc.Callable)
def _pydantic_callable_serialize_only(ty: typing.Any) -> typing.Any:
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.PlainSerializer(_serialize_callable, return_type=EncodedFunction),
    ]


@TypeToPydanticType.register(Tool)
def _pydantic_type_tool(ty: type[Tool]) -> typing.Any:
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.PlainSerializer(_serialize_callable, return_type=EncodedFunction),
    ]


def _validate_name_and_tool(
    value: typing.Any, info: pydantic.ValidationInfo
) -> _NameAndTool:
    if isinstance(value, _NameAndTool):
        return value
    assert isinstance(info.context, collections.abc.Mapping), (
        "Tool decoding requires context"
    )
    value = pydantic.TypeAdapter(ChatCompletionToolParam).validate_python(value)
    name = value["function"]["name"]
    try:
        return _NameAndTool(name, info.context[_NAME2TOOL_KEY][name])
    except KeyError as e:
        raise NotImplementedError(f"Unknown tool: {name}") from e


def _tool_description(tool: Tool, *, param_schemas: bool = False) -> str:
    """The model-facing prose describing ``tool``: ``qualname : signature``, its
    docstring, and the `Encodable` schema of its return type.

    This is the description half of a tool advertisement, shared between the
    JSON pathway (`_serialize_name_and_tool`, where the parameter schemas ride
    separately as the machine-enforced ``parameters``) and the code-generation
    pathway (`~effectful.handlers.llm.harness.synthesis.toolcall`, which sets
    ``param_schemas=True`` so the parameter type structure the JSON ``parameters``
    would have carried is preserved as prose instead).
    """
    description = (
        f"{getattr(tool, '__qualname__', tool.__name__)} : {tool.__signature__}"
    )
    description += f"\n\n{textwrap.dedent(tool.__doc__ or '')}"
    if param_schemas and tool.__signature__.parameters:
        rows = "\n".join(
            f"- `{param_name}`: "
            f"{json.dumps(_best_effort_schema(param.annotation, 'validation'))}"
            for param_name, param in tool.__signature__.parameters.items()
        )
        description += f"\n\nAnnotated JSON schema of each parameter type:\n{rows}"
    description += (
        f"\n\nAnnotated JSON schema of return type: "
        f"{json.dumps(_best_effort_schema(tool.__signature__.return_annotation))}"
    )
    return description


def _serialize_name_and_tool(value: _NameAndTool) -> ChatCompletionToolParam:
    name, tool = value
    params = inspect.signature(tool).parameters
    for param_name, param in params.items():
        if not _is_decodable(param.annotation):
            raise pydantic.errors.PydanticSchemaGenerationError(
                f"`{name}` cannot be advertised as JSON: no value of parameter "
                f"`{param_name}` could be decoded from the model's output"
            )
    fields: dict[str, typing.Any] = {
        param_name: TypeToPydanticType().evaluate(param.annotation)
        for param_name, param in params.items()
    }
    sig_model = pydantic.create_model(
        "Params",
        __config__={"extra": "forbid"},
        **fields,
    )
    response_format = litellm.utils.type_to_response_format_param(sig_model)
    assert response_format is not None
    description = _tool_description(tool)
    return pydantic.TypeAdapter(ChatCompletionToolParam).validate_python(
        {
            "type": "function",
            # Advertise under the assigned name, which is the name decoding
            # (`_validate_tool_call`) resolves the call back by.
            "function": {
                "name": name,
                "description": description,
                "parameters": response_format["json_schema"]["schema"],
                "strict": True,
            },
        }
    )


@TypeToPydanticType.register(_NameAndTool)
def _pydantic_type_name_and_tool(ty: type[_NameAndTool]):
    """Encode a named tool as the `ChatCompletionToolParam` advertising it.

    Registered on the exact `_NameAndTool` class, which singledispatch prefers
    over any base: a `NamedTuple` has `tuple` in its MRO, so without this it
    would route to `_pydantic_type_tuple`'s NamedTuple branch, which would try
    to build a `TypeAdapter` for the `Tool` field and fail.
    """
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.BeforeValidator(
            _validate_name_and_tool, json_schema_input_type=ChatCompletionToolParam
        ),
        pydantic.PlainSerializer(
            _serialize_name_and_tool, return_type=ChatCompletionToolParam
        ),
    ]


def _validate_tool_call(
    value: ChatCompletionMessageToolCall | DecodedToolCall,
    info: pydantic.ValidationInfo,
) -> DecodedToolCall:
    if isinstance(value, DecodedToolCall):
        return value
    call = (
        OpenAIChatCompletionMessageToolCall.model_validate(value)
        if isinstance(value, dict)
        else value
    )
    ctx = info.context or {}
    assert call.function.name is not None
    tool = ctx[_NAME2TOOL_KEY][call.function.name]
    assert isinstance(tool, Tool)
    sig = inspect.signature(tool)
    decoded_args = {}
    for name, raw_arg in json.loads(call.function.arguments).items():
        assert name in sig.parameters, (
            f"Unexpected argument {name} for tool {tool.__name__}"
        )
        param = sig.parameters[name]
        arg_enc: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
            Encodable[param.annotation]  # type: ignore[name-defined]
        )
        decoded_args[name] = arg_enc.validate_python(raw_arg, context=ctx)
    return DecodedToolCall(
        tool=tool,
        bound_args=sig.bind(**decoded_args),
        id=call.id,
        name=call.function.name,
    )


def _serialize_tool_call(
    value: DecodedToolCall, info: pydantic.SerializationInfo
) -> OpenAIChatCompletionMessageToolCall:
    ctx = info.context or {}
    encoded_args: dict[str, typing.Any] = {}
    if value.source is not None:
        # Decoded from a call *expression*: what the model sent -- and what the
        # advertised ``{"call": string}`` schema describes -- is the source, so
        # that is what round-trips. The evaluated `bound_args` are not
        # re-encoded; they may hold values with no JSON encoding at all.
        encoded_args["call"] = value.source
    else:
        for k, v in value.bound_args.arguments.items():
            v_enc: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
                Encodable[nested_type(v).value]  # type: ignore[misc]
            )
            encoded_args[k] = v_enc.dump_python(v, mode="json", context=ctx)
    return OpenAIChatCompletionMessageToolCall.model_validate(
        {
            "type": "function",
            "id": value.id,
            "function": {
                # Use the name the tool was called by (possibly disambiguated by
                # `call_assistant`), not the tool's `__name__`, so the call
                # round-trips to the same identity the model and decoder share.
                "name": value.name,
                "arguments": json.dumps(encoded_args),
            },
        }
    )


@TypeToPydanticType.register(DecodedToolCall)
def _pydantic_type_tool_call(ty: type[DecodedToolCall]):
    # Use OpenAI's ChatCompletionMessageToolCall (has actual fields: id, function,
    # type) rather than litellm's (empty dict with extra="allow").
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.BeforeValidator(
            _validate_tool_call,
            json_schema_input_type=OpenAIChatCompletionMessageToolCall,
        ),
        pydantic.PlainSerializer(
            _serialize_tool_call, return_type=OpenAIChatCompletionMessageToolCall
        ),
    ]


def _advertised_names(
    tools: collections.abc.Set[Tool],
    _TOOL_NAME_MAX: int = 64,
    _NOT_IN_TOOL_NAME: re.Pattern = re.compile(r"[^0-9a-zA-Z_-]+"),
) -> dict[str, Tool]:
    """Assign each tool the name the model calls it by.

    A tool's `__name__` is the obvious name and the one almost every tool gets,
    but it guarantees nothing: two `Agent` instances contribute the same bound
    method under the same `__name__`, and nothing stops two modules from naming
    a tool alike.  So the name is *assigned* here -- unique by construction, and
    provider-legal -- rather than assumed to be unique and asserted.  The result
    is the single naming scheme the request and its response agree on: tools are
    advertised under these keys and `DecodedToolCall` resolves a call back
    through the same mapping (`_NAME2TOOL_KEY`).

    Ties are broken by a numeric suffix, so a collision costs the *later*
    claimants a ``_2``/``_3`` and leaves the first with the name it wanted.
    Which one is first is decided by a sort, so the assignment is stable across
    the turns of a conversation -- a name keeps pointing at the tool the earlier
    turns in the history called by it.  Genuinely indistinguishable tools (that
    same bound method, twice) fall through to `id`, which orders them stably
    within the process but not across runs; nothing outlives the process.
    """

    def base(tool: Tool) -> str:
        name = _NOT_IN_TOOL_NAME.sub("_", tool.__name__)
        # Leave room for a suffix, so disambiguation cannot push a long name
        # over the limit (or, worse, truncate two names into a fresh collision).
        return name[: _TOOL_NAME_MAX - 4] or "tool"

    result: dict[str, Tool] = {}
    for tool in sorted(
        tools,
        # A synthetic tool need not carry a `__qualname__`; it is only a
        # tiebreaker, so fall back rather than insist on one.
        key=lambda t: (base(t), getattr(t, "__qualname__", ""), t.__module__, id(t)),
    ):
        name = base(tool)
        suffix = 1
        while name in result:
            suffix += 1
            name = f"{base(tool)}_{suffix}"
        result[name] = tool
    return result


class _BoxedResponse[T](pydantic.BaseModel):
    value: T
