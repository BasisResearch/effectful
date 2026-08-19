import base64
import collections.abc
import dataclasses
import functools
import inspect
import io
import json
import re
import string
import textwrap
import typing
from collections.abc import (
    Callable,
    Mapping,
)

import litellm
import pydantic
from litellm import (
    ChatCompletionImageObject,
    ChatCompletionMessageToolCall,
    ChatCompletionTextObject,
    ChatCompletionToolParam,
    OpenAIMessageContentListBlock,
)
from openai.lib._pydantic import _ensure_strict_json_schema
from openai.types.chat import (
    ChatCompletionMessageToolCall as OpenAIChatCompletionMessageToolCall,
)
from PIL import Image

from effectful.handlers.llm.types import Encodable, Tool
from effectful.internals.unification import GenericAlias, TypeEvaluator, nested_type
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
    """
    if isinstance(value, str):
        return [ChatCompletionTextObject(type="text", text=value)]

    buf: list[str] = []
    blocks: list[OpenAIMessageContentListBlock] = []

    def flush() -> None:
        if buf:
            blocks.append(ChatCompletionTextObject(type="text", text="".join(buf)))
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
    """
    formatter = string.Formatter()
    parts: list[OpenAIMessageContentListBlock] = []

    buf: list[str] = []

    def flush_text() -> None:
        if buf:
            parts.append(ChatCompletionTextObject(type="text", text="".join(buf)))
            buf.clear()

    for literal, field_name, format_spec, conversion in formatter.parse(
        textwrap.dedent(template)
    ):
        if literal:
            buf.append(literal)

        if field_name is None:
            continue

        obj, _ = formatter.get_field(field_name, (), env)
        encoder: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
            Encodable[nested_type(obj).value]  # type: ignore[misc]
        )
        encoded_obj = encoder.dump_python(obj, mode="json", context=env)
        for part in to_content_blocks(encoded_obj):
            if part["type"] == "text":
                text = (
                    formatter.convert_field(part["text"], conversion)
                    if conversion
                    else part["text"]
                )
                buf.append(formatter.format_field(text, format_spec or ""))
            else:
                flush_text()
                parts.append(part)

    flush_text()

    return parts


class PromptSection(typing.TypedDict):
    type: typing.Literal["prompt_section"]
    title: str
    content: collections.abc.Sequence[
        litellm.OpenAIMessageContentListBlock | "PromptSection"
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


def _render_system_prompt(
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
            for block in _render_system_prompt(
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


def _inline_refs(schema: dict) -> dict:
    """Inline ``$ref`` pointers so ``WithJsonSchema`` never emits orphan refs.

    Workaround for https://github.com/pydantic/pydantic/issues/12145 —
    Pydantic's ``GenerateJsonSchema`` does not merge user-provided ``$defs``
    into its internal ref map, so any ``$ref`` in a ``WithJsonSchema`` value
    causes a ``KeyError`` when the annotated type is composed into a model.
    """
    defs = schema.get("$defs", {})

    def _resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    return _resolve(defs[ref_name])
            return {k: _resolve(v) for k, v in obj.items() if k != "$defs"}
        if isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    return _resolve(schema)


@dataclasses.dataclass(frozen=True, eq=True)
class DecodedToolCall[T]:
    """
    Structured representation of a tool call decoded from an LLM response.
    """

    tool: Tool[..., T]
    bound_args: inspect.BoundArguments
    id: ToolCallID
    name: str

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


@TypeToPydanticType.register(object)
def _pydantic_type_base(ty: type) -> typing.Any:
    return ty


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

    schema = pydantic.TypeAdapter(_ComplexModel).json_schema()
    schema = _ensure_strict_json_schema(_inline_refs(schema), path=(), root={})

    return typing.Annotated[
        ty,
        pydantic.BeforeValidator(_validate_complex),
        pydantic.PlainSerializer(_serialize_complex),
        pydantic.WithJsonSchema(schema),
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
        nt_adapters = [pydantic.TypeAdapter(t) for t in nt_types]
        nt_model = pydantic.create_model(
            ty.__name__,
            __config__={"extra": "forbid"},
            __doc__=ty.__doc__,
            **{f: (t, ...) for f, t in zip(nt_fields, nt_types)},
        )

        def _nt_serialize(value, info: pydantic.SerializationInfo):
            return {
                f: nt_adapters[i].dump_python(
                    getattr(value, f), mode="json", context=info.context
                )
                for i, f in enumerate(nt_fields)
            }

        return typing.Annotated[
            ty,
            pydantic.PlainSerializer(_nt_serialize),
            pydantic.WithJsonSchema(_inline_refs(nt_model.model_json_schema())),
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

    adapters = [pydantic.TypeAdapter(a) for a in effective]

    model = pydantic.create_model(
        "TupleItems",
        __config__={"extra": "forbid"},
        **{f"item_{i}": (a, ...) for i, a in enumerate(effective)},
    )

    def _decode(value):
        """Reshape the ``item_0``/``item_1``/... object form back into a positional
        tuple, leaving the tuple itself for Pydantic to validate elementwise."""
        if isinstance(value, Mapping):
            return tuple(value[f"item_{i}"] for i in range(len(effective)))
        return value

    def _serialize(value, info: pydantic.SerializationInfo):
        return {
            f"item_{i}": adapters[i].dump_python(v, mode="json", context=info.context)
            for i, v in enumerate(value)
        }

    return typing.Annotated[
        ty,
        pydantic.BeforeValidator(_decode),
        pydantic.PlainSerializer(_serialize),
        pydantic.WithJsonSchema(_inline_refs(model.model_json_schema())),
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
    adapter = pydantic.TypeAdapter(ChatCompletionImageObject)
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.BeforeValidator(_validate_image),
        pydantic.PlainSerializer(_serialize_image),
        pydantic.WithJsonSchema(_inline_refs(adapter.json_schema())),
    ]


# The *serialization* view of a synthesized callable: the shape the model reads
# when a function is handed to it as a value (e.g. a tool's return) -- just the
# source, with none of the synthesis instructions the `SynthesizedFunction` subtype
# carries for the generation direction. Its JSON schema (docstring included, since
# pydantic renders it as the schema `description`) is the ``mode="serialization"``
# schema of every synthesized-callable encoding, so keep the docstring model-facing.
class EncodedFunction(pydantic.BaseModel):
    """A function, encoded as a string of its complete Python source."""

    module_code: str = pydantic.Field(
        ..., description="Python source defining the function."
    )


def _serialize_callable(value: Callable) -> dict:
    """Encode a callable back to its ``module_code`` form (source, or a stub).

    Emits a plain `EncodedFunction` -- which is exactly what the serialization JSON
    schema declares -- rather than the `SynthesizedFunction` subclass that governs
    the *other* direction. The two directions carry different obligations, and
    conflating them was a real bug: `SynthesizedFunction`'s constraints ("the last
    statement must be a function definition", "every parameter must be annotated")
    are demands on code a model is *writing*, and a value being serialized is under
    no such obligation. It is any callable that reached this point -- a class handed
    back by a lexical-scope read, or an inner function a Skill *body* returned,
    which was never required to annotate anything -- and re-validating its recovered
    source rejected those with a `ValidationError` raised from inside pydantic's
    serializer, aborting the whole call rather than encoding the value.
    """
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        source = None

    if source:
        return EncodedFunction(module_code=textwrap.dedent(source)).model_dump()

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

    stub_code = f'''def {name}{sig_str}:
    """{docstring}"""
    ...
'''
    return EncodedFunction(module_code=stub_code).model_dump()


@TypeToPydanticType.register(Callable)
def _pydantic_callable_serialize_only(ty: typing.Any) -> typing.Any:
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.PlainSerializer(_serialize_callable),
        pydantic.WithJsonSchema(
            EncodedFunction.model_json_schema(), mode="serialization"
        ),
    ]


@TypeToPydanticType.register(Tool)
def _pydantic_type_tool(ty: type[Tool]) -> typing.Any:
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.PlainSerializer(_serialize_callable),
        pydantic.WithJsonSchema(
            EncodedFunction.model_json_schema(), mode="serialization"
        ),
    ]


def _validate_name_and_tool(
    value: typing.Any, info: pydantic.ValidationInfo
) -> _NameAndTool:
    if isinstance(value, _NameAndTool):
        return value
    assert isinstance(info.context, Mapping), "Tool decoding requires context"
    value = pydantic.TypeAdapter(ChatCompletionToolParam).validate_python(value)
    name = value["function"]["name"]
    try:
        return _NameAndTool(name, info.context[_NAME2TOOL_KEY][name])
    except KeyError as e:
        raise NotImplementedError(f"Unknown tool: {name}") from e


def _serialize_name_and_tool(value: _NameAndTool) -> ChatCompletionToolParam:
    name, tool = value
    fields: dict[str, typing.Any] = {
        param_name: TypeToPydanticType().evaluate(param.annotation)
        for param_name, param in inspect.signature(tool).parameters.items()
    }
    sig_model = pydantic.create_model(
        "Params",
        __config__={"extra": "forbid"},
        **fields,
    )
    response_format = litellm.utils.type_to_response_format_param(sig_model)
    assert response_format is not None
    ret_schema = pydantic.TypeAdapter(
        Encodable[tool.__signature__.return_annotation]  # type: ignore[name-defined]
    ).json_schema(mode="serialization")
    description = (
        f"{getattr(tool, '__qualname__', tool.__name__)} : {tool.__signature__}"
    )
    description += f"\n\n{textwrap.dedent(tool.__doc__ or '')}"
    description += f"\n\nAnnotated JSON schema of return type: {json.dumps(ret_schema)}"
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
    schema = pydantic.TypeAdapter(ChatCompletionToolParam).json_schema()
    schema = _ensure_strict_json_schema(_inline_refs(schema), path=(), root={})
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.BeforeValidator(_validate_name_and_tool),
        pydantic.PlainSerializer(_serialize_name_and_tool),
        pydantic.WithJsonSchema(schema),
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
) -> dict:
    ctx = info.context or {}
    encoded_args = {}
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
    ).model_dump(mode="json")


@TypeToPydanticType.register(DecodedToolCall)
def _pydantic_type_tool_call(ty: type[DecodedToolCall]):
    # Use OpenAI's ChatCompletionMessageToolCall (has actual fields: id, function,
    # type) rather than litellm's (empty dict with extra="allow").
    schema = OpenAIChatCompletionMessageToolCall.model_json_schema()
    schema = _ensure_strict_json_schema(_inline_refs(schema), path=(), root={})
    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.BeforeValidator(_validate_tool_call),
        pydantic.PlainSerializer(_serialize_tool_call),
        pydantic.WithJsonSchema(schema),
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
