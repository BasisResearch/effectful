import ast
import base64
import collections.abc
import dataclasses
import functools
import inspect
import io
import json
import linecache
import string
import textwrap
import types
import typing
import uuid
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

import effectful.handlers.llm.harness.execution as execution
from effectful.handlers.llm.types import Encodable, Template, Tool
from effectful.internals.unification import GenericAlias, TypeEvaluator, nested_type
from effectful.ops.semantics import fwd, handler
from effectful.ops.types import Operation, Term

type ToolCallID = str

# Key under which the name->Tool mapping is stashed in the decoding context.
# Deliberately not a valid Python identifier, so it can never collide with a
# lexical variable name sharing the context (e.g. a reader named after its var).
_TOOLS_KEY: typing.Literal["$TOOLS"] = "$TOOLS"
# Reserved key under which the type-check anchor -- the enclosing `Template`
# itself -- rides in the Pydantic decoding context, alongside the lexical
# environment. `decode` reads it to type-check a synthesized function against the
# Template's source (recovered from the Template via `inspect.unwrap`); absent
# (tool-argument decoding) means skip. Deliberately not a valid identifier so
# `LexicalReaders` skips it (no tool leak) and it can never collide with a lexical
# name.
TYPE_CHECK_ANCHOR_KEY = "<type_check_anchor>"

# Anchor for REPL `exec_code` snippets and synthesized tool arguments (including a
# `TemplateBody`), separate from the structured-output-result synthesis anchor
# (TYPE_CHECK_ANCHOR_KEY): the two decoders check against different contracts -- a
# REPL snippet or a `TemplateBody` against the Template body, a synthesized general
# `Callable` tool argument against its own parameter type. Both keys carry the
# enclosing `Template`.
REPL_ANCHOR_KEY = "<repl_anchor>"

CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    literal
    for member in typing.get_args(OpenAIMessageContentListBlock)
    for literal in typing.get_args(typing.get_type_hints(member).get("type", str))
    if isinstance(literal, str)
)


@pydantic.validate_call(validate_return=True)
def to_content_blocks(value: typing.Any) -> list[OpenAIMessageContentListBlock]:
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
def _validate_complex(value: _ComplexModel) -> complex:
    return complex(value["real"], value["imag"])


@pydantic.validate_call(validate_return=True)
def _serialize_complex(value: complex) -> _ComplexModel:
    return {"real": value.real, "imag": value.imag}


@TypeToPydanticType.register(complex)
def _pydantic_type_complex(ty):
    """Encode ``complex`` as ``{"real": float, "imag": float}``."""

    adapted_schema = pydantic.TypeAdapter(_ComplexModel).json_schema()

    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate_complex),
        pydantic.PlainSerializer(_serialize_complex),
        pydantic.WithJsonSchema({**adapted_schema, "additionalProperties": False}),
    ]


_CODE_FILENAME_PREFIX = "<exec_code-"


@TypeToPydanticType.register(types.CodeType)
def _pydantic_type_code(ty):
    """Encode a `types.CodeType` as a JSON string of Python source.

    This is the internal `Encodable` implementation for code objects -- the
    public type is `types.CodeType`, with no separate model (analogous to
    `_ComplexModel`).  Decoding compiles the source through the `parse`/`compile`
    effect operations under a unique per-snippet filename, so invalid source is
    rejected here rather than at run time and the snippet's source lands in
    `linecache` (keeping each snippet's tracebacks resolvable).  A decoded value
    is therefore a ready-to-run code object; re-encoding recovers its source from
    `linecache`, which carries everything the source string did.
    """

    def validate(
        value: types.CodeType | str, info: pydantic.ValidationInfo
    ) -> types.CodeType:
        if isinstance(value, types.CodeType):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"expected Python source as a string, got {type(value).__name__}"
            )
        filename = f"{_CODE_FILENAME_PREFIX}{uuid.uuid4()}>"
        try:
            module = execution.parse(value, filename)
            # Reject `__future__`/star imports: both are `SyntaxError` once nested in a
            # function body, so such a snippet can't be spliced into the Template for
            # type checking.
            execution.scan_non_nestable(module)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"source is not valid REPL code: {exc}") from exc

        # Type-check the snippet in its execution context, exactly as a synthesized
        # `Callable` is (see `_pydantic_callable`): when the enclosing Template is the
        # type-check anchor in the decode context, splice the accumulated REPL session
        # (`PythonRepl.repl_history` returns the prior snippets of the session in scope)
        # plus this snippet into the Template body and check it. A type error raises here
        # -> the tool-call decode fails -> `RetryLLMHandler` retries, so ill-typed code
        # never reaches `runcode`.
        ctx = info.context or {}
        anchor = ctx.get(REPL_ANCHOR_KEY)
        if anchor is not None:
            # Imported lazily (not at module load) to avoid an import cycle: `completions`
            # imports this module. `repl_history` returns the managed session's prior
            # snippets, or `[]` when no REPL is in scope.
            from effectful.handlers.llm.harness.completion import PythonRepl

            # Prepend the already-run (type-clean) session snippets so their bindings
            # resolve; `value` is the current snippet. The whole cumulative body is
            # spliced and checked.
            prior = PythonRepl.repl_history()
            prior_src = "".join(s if s.endswith("\n") else s + "\n" for s in prior)
            session = ast.parse(prior_src + value)
            checked = execution.splice_repl_code_into_body(session, anchor)
            if checked is not None:
                execution.type_check(*checked, lenient=True)
        try:
            return execution.compile(module, filename)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"source does not compile: {exc}") from exc

    return typing.Annotated[
        ty,
        pydantic.PlainValidator(validate),
        pydantic.PlainSerializer(
            lambda value: "".join(linecache.getlines(value.co_filename))
        ),
        pydantic.WithJsonSchema({"type": "string"}),
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

        def _nt_validate(value, info: pydantic.ValidationInfo):
            if isinstance(value, tuple | list):
                value = dict(zip(nt_fields, value))
            return ty(
                **{
                    f: nt_adapters[i].validate_python(value[f], context=info.context)
                    for i, f in enumerate(nt_fields)
                }
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
            pydantic.PlainValidator(_nt_validate),
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

    def _validate(value, info: pydantic.ValidationInfo):
        if isinstance(value, tuple | list):
            value = {f"item_{i}": v for i, v in enumerate(value)}
        return tuple(
            adapters[i].validate_python(value[f"item_{i}"], context=info.context)
            for i in range(len(effective))
        )

    def _serialize(value, info: pydantic.SerializationInfo):
        return {
            f"item_{i}": adapters[i].dump_python(v, mode="json", context=info.context)
            for i, v in enumerate(value)
        }

    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate),
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


@pydantic.validate_call(validate_return=False)
def _validate_image(value: ChatCompletionImageObject) -> Image.Image:
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
        pydantic.PlainValidator(_validate_image),
        pydantic.PlainSerializer(_serialize_image),
        pydantic.WithJsonSchema(_inline_refs(adapter.json_schema())),
    ]


def _callable_type_from_signature(
    signature: inspect.Signature,
) -> type[types.FunctionType]:
    """Construct a `Callable` type from a signature.

    Raises if the signature is recursive (e.g. a Template that returns itself)
    or contains variadic parameters (which cannot be expressed in a `Callable`
    type).
    """
    param_types = []
    for pname, param in signature.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise NotImplementedError(
                f"Cannot synthesize a function for parameter "
                f"'{pname}' of kind {param.kind.description}: variadic parameters "
                "cannot be expressed as a Callable type signature."
            )
        param_types.append(
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else typing.Any
        )
    return_type = signature.return_annotation
    return collections.abc.Callable[param_types, return_type]  # type: ignore


class TemplateBody:
    """The synthesized *body* of a `Template`, as opposed to a general `Callable`.

    Used only as the type of `submit_solution`'s ``implementation`` parameter (see
    `effectful.handlers.llm.completions.SynthesizeAndCall`).  A `TemplateBody[[P],
    R]` carries the Template's parameter and return types exactly like a
    `Callable`, but gets its own `TypeToPydanticType` case (`_pydantic_template_body`)
    so the synthesized function is type-checked against the enclosing Template's
    source and its doctests run with self/recursive calls routed to the synthesized
    implementation.  The enclosing `Template` is recovered from the decode context
    (the ``anchor``), so no state rides on the type itself.
    """

    def __class_getitem__(cls, item):
        return types.GenericAlias(cls, item)


class MethodTemplateBody(TemplateBody):
    """A `TemplateBody` for an *instance-method* Template.

    Carries the method/free distinction on the type's origin (context-free schema
    generation reads it) so `submit_solution`'s description names the leading
    receiver ``self`` and the receiver is exempt from the annotation requirement --
    the model no longer has to reverse-engineer that the first parameter is ``self``.
    The Template's real signature (which includes the receiver) remains the
    type-check contract; see `splice_template_body`.
    """


def _class_template_of(op: typing.Any) -> typing.Any | None:
    """The class-level `Template` underlying an Agent-method Template ``op``.

    Returns ``None`` for a free-function template (whose ``__default__`` is a plain
    function rather than a bound method).
    """
    default = getattr(op, "__default__", None)
    if isinstance(default, types.MethodType):
        return default.__func__.__wrapped__  # type: ignore[attr-defined]
    return None


def _method_instance(op: typing.Any, class_template: typing.Any) -> typing.Any | None:
    """The instance ``op`` is bound to, if ``op`` is ``class_template`` on *some*
    instance; otherwise ``None``.
    """
    if class_template is not None and _class_template_of(op) is class_template:
        return op.__default__.__self__
    return None


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


class SynthesizedFunction(EncodedFunction):
    """
    Structured output for function synthesis.
    """

    module_code: str = pydantic.Field(
        ...,
        description=textwrap.dedent("""
        A string containing the complete Python source code for the function.
        The code MUST satisfy the following constraints, or it will fail validation:

        <constraints>
        1. The code MUST be one complete syntactically valid Python module.
        2. The code MUST NOT use star imports or ``__future__`` imports.
        3. The function definition MUST be the LAST statement - do not add any code after it.
        4. The function MUST have type annotations for all parameters and the return type.
        5. You may include doctest examples (lines starting with >>>) inside the function's
        docstring to demonstrate and verify its behavior; these examples are run as tests.
        </constraints>
        """),
    )

    # A general `Callable` is type-checked against the requested signature, so it must
    # be fully annotated. A Template *body* is instead checked against the enclosing
    # Template's own signature (`splice_template_body`), which already carries the
    # annotations -- so its subclasses waive this and may omit the `self` receiver.
    _require_annotations: typing.ClassVar[bool] = True

    @pydantic.field_validator("module_code")
    @classmethod
    def _validate_module_code(cls, value: str) -> str:
        module: ast.AST = ast.parse(value)

        if not isinstance(module, ast.Module) or not module.body:
            raise ValueError(
                "decode() requires module code with at least one statement."
            )

        last_stmt = module.body[-1]
        if not isinstance(last_stmt, ast.FunctionDef):
            raise ValueError(
                f"decode() requires the last statement to be a function definition, "
                f"got {type(last_stmt).__name__}"
            )

        if cls._require_annotations:
            for arg in last_stmt.args.args:
                if arg.annotation is None:
                    raise ValueError(
                        f"decode() requires all parameters to have type annotations, "
                        f"parameter '{arg.arg}' is missing an annotation"
                    )
            if last_stmt.returns is None:
                raise ValueError(
                    "decode() requires the function to have a return type annotation"
                )

        for stmt in module.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                raise ValueError(
                    "decode() does not allow __future__ imports in the module code"
                )

        for stmt in module.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.names:
                for alias in stmt.names:
                    if alias.name == "*":
                        raise ValueError(
                            "decode() does not allow star imports in the module code"
                        )

        return value

    @classmethod
    def _create_model_from_callable_type(cls, typ: type[Callable]) -> type[typing.Self]:
        """Create a SynthesizedFunction subclass carrying the requested signature in
        the model-facing description.

        Uses ``pydantic.create_model`` so the rendered signature (and any
        subclass-specific instructions) ride in the JSON schema ``description`` sent
        to the model. Subclasses customize the receiver rendering via `_param_names`
        and add guidance via `_extra_instructions`.
        """
        doc = (
            f"Python function with signature "
            f"<signature>{cls._signature_str(typ)}</signature>"
            f"{cls._extra_instructions()}"
        )
        return pydantic.create_model(
            "TypedSynthesizedFunction",
            __base__=cls,
            __doc__=doc,
        )

    @classmethod
    def _signature_str(cls, typ: type[Callable]) -> str:
        """Render a ``Callable[[...], ...]`` signature by type *name* (not its
        fully-qualified ``repr``), so the model sees ``Callable[[State], int]`` rather
        than ``collections.abc.Callable[[pkg.mod.State], builtins.int]``."""
        args = typing.get_args(typ)
        if not args:
            return "Callable"
        param_types, return_type = args
        params_str = (
            "..." if param_types is ... else ", ".join(cls._param_names(param_types))
        )
        return_str = getattr(return_type, "__name__", str(return_type))
        return f"Callable[[{params_str}], {return_str}]"

    @classmethod
    def _param_names(cls, param_types: typing.Iterable[typing.Any]) -> list[str]:
        return [getattr(t, "__name__", str(t)) for t in param_types]

    @classmethod
    def _extra_instructions(cls) -> str:
        return ""


class SynthesizedTemplateBody(SynthesizedFunction):
    """Structured output for synthesizing a `Template`'s body (`submit_solution`).

    Decoded through `_pydantic_template_body`: the function is type-checked against
    the enclosing Template's source and its doctests are run with self/recursive
    calls routed to the synthesized implementation.

    Unlike `SynthesizedFunction`, the parameter and return *annotations* are not
    required: a Template body is type-checked against the Template's own signature
    (see `splice_template_body`), so the model may omit or vary them -- in
    particular it need not annotate the ``self`` receiver of an instance-method
    Template.
    """

    module_code: str = pydantic.Field(
        ...,
        description=textwrap.dedent("""
        The complete Python source implementing the Template shown in its spec.
        The code MUST satisfy the following constraints, or it will fail validation:

        <constraints>
        1. The code MUST be one complete syntactically valid Python module.
        2. The code MUST NOT use star imports or ``__future__`` imports.
        3. The function definition MUST be the LAST statement - do not add any code after it.
        4. Write the function with the Template's signature; parameter and return
        annotations are optional.
        5. Do not include a docstring or doctests; the Template's are supplied automatically.
        </constraints>
        """),
    )

    # A Template body is checked against the Template's own (already-annotated)
    # signature, so the synthesized body's annotations are optional.
    _require_annotations: typing.ClassVar[bool] = False


class SynthesizedMethodTemplateBody(SynthesizedTemplateBody):
    """Structured output for synthesizing an *instance-method* `Template`'s body.

    Decoded through `_pydantic_template_body`: the function is type-checked against
    the enclosing Template's source and its doctests are run with self/recursive
    calls routed to the synthesized implementation.

    Unlike `SynthesizedFunction`, the parameter and return *annotations* are not
    required: a Template body is type-checked against the Template's own signature
    (see `splice_template_body`), so the model may omit or vary them -- in
    particular it need not annotate the ``self`` receiver of an instance-method
    Template.
    """

    module_code: str = pydantic.Field(
        ...,
        description=textwrap.dedent("""
        The complete Python source implementing the instance-method Template shown in
        its spec. The code MUST satisfy the following constraints, or it will fail
        validation:

        <constraints>
        1. The code MUST be one complete syntactically valid Python module.
        2. The code MUST NOT use star imports or ``__future__`` imports.
        3. The function definition MUST be the LAST statement - do not add any code after it.
        4. Write the function with the Template's signature: its FIRST parameter is the
        instance receiver ``self`` (which you may leave unannotated); all other parameter
        and return annotations are optional too.
        5. Do not include a docstring or doctests; the Template's are supplied automatically.
        </constraints>
        """),
    )

    @classmethod
    def _param_names(cls, param_types: typing.Iterable[typing.Any]) -> list[str]:
        # The method's callable type already carries the receiver as its first
        # parameter (with an uninformative Agent-class type); relabel it ``self`` so
        # the model reproduces it rather than inventing one -- do NOT prepend a receiver.
        names = super()._param_names(param_types)
        if names:
            names[0] = "self"
        return names

    @classmethod
    def _extra_instructions(cls) -> str:
        return (
            "\n\nThis implements an instance method: the first parameter is the "
            "instance receiver `self`. Include it as the first parameter; you may "
            "leave it unannotated."
        )


def _serialize_synthesized(value: Callable) -> dict:
    """Encode a callable back to its ``module_code`` form (source, or a stub).

    Emits a plain `EncodedFunction` -- which is exactly what the serialization JSON
    schema declares -- rather than the `SynthesizedFunction` subclass that governs
    the *other* direction. The two directions carry different obligations, and
    conflating them was a real bug: `SynthesizedFunction`'s constraints ("the last
    statement must be a function definition", "every parameter must be annotated")
    are demands on code a model is *writing*, and a value being serialized is under
    no such obligation. It is any callable that reached this point -- a class handed
    back by a lexical-scope read, or an inner function a Template *body* returned,
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


def _synthesize_callable(
    module_code: str,
    ctx: Mapping,
    *,
    template_body: bool,
) -> tuple[Callable, dict[str, typing.Any]]:
    """Parse, type-check, compile and exec a synthesized module, returning the
    function it defines and the exec namespace.

    The code is type-checked against the enclosing Template's source when an
    ``anchor`` is present in ``ctx``.  ``template_body`` selects the splice: a
    `TemplateBody` (submit_solution) is spliced as the Template's own body; a
    general `Callable` uses the strict result splice (`splice_into_source`) when it
    is a structured-output result, else the lenient REPL splice.
    """
    filename = f"<synthesis:{id(module_code)}>"
    module: ast.Module = execution.parse(module_code, filename)

    if template_body:
        anchor = ctx.get(TYPE_CHECK_ANCHOR_KEY) or ctx.get(REPL_ANCHOR_KEY)
        if anchor is not None:
            # Check the synthesized function *as the Template's body*, strictly: it
            # is the final answer, so -- unlike incrementally-built REPL code -- it
            # must honor the Template's declared types and gets no redefinition slack
            # (no name reuse with a new type, no duplicate definitions).
            spliced = execution.splice_template_body(module, anchor)
            if spliced is not None:
                execution.type_check(*spliced)
    elif ctx.get(TYPE_CHECK_ANCHOR_KEY) is not None:
        spliced = execution.splice_into_source(module, ctx[TYPE_CHECK_ANCHOR_KEY])
        if spliced is not None:
            execution.type_check(*spliced)
    elif ctx.get(REPL_ANCHOR_KEY) is not None:
        spliced = execution.splice_repl_code_into_body(module, ctx[REPL_ANCHOR_KEY])
        if spliced is not None:
            execution.type_check(*spliced, lenient=True)

    bytecode: types.CodeType = execution.compile(module, filename)
    g: dict[str, typing.Any] = {k: v for k, v in ctx.items() if k.isidentifier()}
    execution.exec(bytecode, g)
    result = g[module.body[-1].name]  # type: ignore
    return result, g


def _reject_param_count_mismatch(fn: Callable, ty: typing.Any) -> None:
    """Raise ``ValueError`` if the synthesized ``fn``'s positional arity does not
    match the expected ``Callable[[...], ret]`` type.

    The mypy signature check only runs when a type-check anchor is in scope; this
    structural check runs unconditionally, so a wrong parameter count is still
    rejected on the anchorless argument-decoding path.
    """
    args = typing.get_args(ty)
    if not args or args[0] is ...:
        return  # bare ``Callable`` or ``Callable[..., R]``: any arity is acceptable
    expected = len(args[0])
    params = list(inspect.signature(fn).parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return  # ``*args`` accepts any number of positional arguments
    positional = sum(
        p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )
    if positional != expected:
        raise ValueError(
            f"synthesized function takes {positional} positional parameter(s), "
            f"but the expected signature has {expected}"
        )


@TypeToPydanticType.register(Callable)
def _pydantic_callable(ty: typing.Any) -> typing.Any:
    """Pydantic-compatible Annotated type for a parameterized `Callable` value.

    The model *produces* a function (as ``module_code``); it is synthesized,
    type-checked in the enclosing Template's scope, and its own doctests are run.
    Template-body synthesis (`submit_solution`) has its own encoding,
    `_pydantic_template_body`.
    """
    typed_enc = SynthesizedFunction._create_model_from_callable_type(
        Callable[..., typing.Any] if not typing.get_args(ty) else ty  # type: ignore[arg-type]
    )

    def _validate(
        value: SynthesizedFunction | dict | str, info: pydantic.ValidationInfo
    ) -> Callable:
        if isinstance(value, str):
            value = typed_enc.model_validate_json(value)
        if isinstance(value, dict):
            value = typed_enc.model_validate(value)
        result, g = _synthesize_callable(
            value.module_code, info.context or {}, template_body=False
        )
        _reject_param_count_mismatch(result, ty)
        execution.run_doctests(result, g)
        return result

    # Distinct schemas per direction: validation (the model *produces* a function)
    # carries the synthesis instructions; serialization (the model *reads* an
    # encoded function) shows only the `module_code` shape `_serialize_synthesized`
    # emits, with no synthesis prose.
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate),
        pydantic.PlainSerializer(lambda value: _serialize_synthesized(value)),
        pydantic.WithJsonSchema(
            _inline_refs(pydantic.TypeAdapter(typed_enc).json_schema()),
            mode="validation",
        ),
        pydantic.WithJsonSchema(
            EncodedFunction.model_json_schema(), mode="serialization"
        ),
    ]


@TypeToPydanticType.register(TemplateBody)
def _pydantic_template_body(ty: typing.Any) -> typing.Any:
    """`TypeToPydanticType` case for a free-function `Template` body.

    Like `_pydantic_callable`, but the synthesized function is checked against the
    enclosing Template's source (the ``anchor`` in the decode context) and its
    doctests are run with the Template's own name/op routed back to the synthesized
    implementation, so a doctest that calls the Template (including for recursion)
    exercises the freshly synthesized code rather than re-invoking the model.
    """
    typed_enc = SynthesizedTemplateBody._create_model_from_callable_type(
        ty if typing.get_args(ty) else Callable[..., typing.Any],  # type: ignore[arg-type]
    )

    def _validate(
        value: SynthesizedTemplateBody | dict | str, info: pydantic.ValidationInfo
    ) -> Callable:
        if isinstance(value, str):
            value = typed_enc.model_validate_json(value)
        if isinstance(value, dict):
            value = typed_enc.model_validate(value)
        ctx = info.context or {}
        result, g = _synthesize_callable(value.module_code, ctx, template_body=True)
        anchor = ctx.get(TYPE_CHECK_ANCHOR_KEY) or ctx.get(REPL_ANCHOR_KEY)
        if anchor is None:
            execution.run_doctests(result, g)
            return result
        # Shadow the global name the doctests call and route the Template op back
        # into the synthesized function.
        result = functools.wraps(anchor)(result)
        g.update({anchor.__name__: result})
        with handler({anchor: result}):
            execution.run_doctests(result, g)
        return result

    # Distinct schemas per direction: validation (the model *produces* a function)
    # carries the synthesis instructions; serialization (the model *reads* an
    # encoded function) shows only the `module_code` shape `_serialize_synthesized`
    # emits, with no synthesis prose.
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate),
        pydantic.PlainSerializer(lambda value: _serialize_synthesized(value)),
        pydantic.WithJsonSchema(
            _inline_refs(pydantic.TypeAdapter(typed_enc).json_schema()),
            mode="validation",
        ),
        pydantic.WithJsonSchema(
            EncodedFunction.model_json_schema(), mode="serialization"
        ),
    ]


@TypeToPydanticType.register(MethodTemplateBody)
def _pydantic_method_template_body(ty: typing.Any) -> typing.Any:
    """`TypeToPydanticType` case for an instance-method `Template` body.

    Registered separately from `TemplateBody` (rather than reached via subclass
    MRO) so the method/free distinction is an explicit dispatch: it surfaces the
    leading ``self`` receiver in the signature hint, and its doctests -- which build
    their own instances -- route ``agent.method(...)`` on *any* instance to the
    synthesized implementation.
    """
    typed_enc = SynthesizedMethodTemplateBody._create_model_from_callable_type(
        ty if typing.get_args(ty) else Callable[..., typing.Any],  # type: ignore[arg-type]
    )

    def _validate(
        value: SynthesizedMethodTemplateBody | dict | str, info: pydantic.ValidationInfo
    ) -> Callable:
        if isinstance(value, str):
            value = typed_enc.model_validate_json(value)
        if isinstance(value, dict):
            value = typed_enc.model_validate(value)
        ctx = info.context or {}
        result, g = _synthesize_callable(value.module_code, ctx, template_body=True)
        anchor = ctx.get(TYPE_CHECK_ANCHOR_KEY) or ctx.get(REPL_ANCHOR_KEY)
        class_template = _class_template_of(anchor) if anchor is not None else None
        if class_template is None:
            execution.run_doctests(result, g)
            return result
        # A fresh instance's `agent.method(...)` dispatches through
        # `Template.__apply__`, which we intercept and redirect to the synthesized
        # implementation.
        result = functools.wraps(class_template)(result)

        def _doctest_apply(op, *args, **kwargs):
            instance = _method_instance(op, class_template)
            if instance is None:
                return fwd()
            return class_template(instance, *args, **kwargs)

        with handler({Template.__apply__: _doctest_apply, class_template: result}):
            execution.run_doctests(result, g)
        return result

    # Distinct schemas per direction: validation (the model *produces* a function)
    # carries the synthesis instructions; serialization (the model *reads* an
    # encoded function) shows only the `module_code` shape `_serialize_synthesized`
    # emits, with no synthesis prose.
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate),
        pydantic.PlainSerializer(lambda value: _serialize_synthesized(value)),
        pydantic.WithJsonSchema(
            _inline_refs(pydantic.TypeAdapter(typed_enc).json_schema()),
            mode="validation",
        ),
        pydantic.WithJsonSchema(
            EncodedFunction.model_json_schema(), mode="serialization"
        ),
    ]


def _validate_tool(
    value: ChatCompletionToolParam, info: pydantic.ValidationInfo
) -> Tool:
    assert isinstance(info.context, Mapping), "Tool decoding requires context"
    value = pydantic.TypeAdapter(ChatCompletionToolParam).validate_python(value)
    try:
        return info.context[_TOOLS_KEY][value["function"]["name"]]
    except KeyError as e:
        raise NotImplementedError(f"Unknown tool: {value['function']['name']}") from e


def _serialize_tool(
    value: Tool, info: pydantic.SerializationInfo
) -> ChatCompletionToolParam:
    fields: dict[str, typing.Any] = {
        name: TypeToPydanticType().evaluate(param.annotation)
        for name, param in inspect.signature(value).parameters.items()
    }
    sig_model = pydantic.create_model(
        "Params",
        __config__={"extra": "forbid"},
        **fields,
    )
    response_format = litellm.utils.type_to_response_format_param(sig_model)
    assert response_format is not None
    # Advertise under the context key, since decode (`_validate_tool`) resolves the call by that name.
    tool_name = value.__name__
    context = info.context
    if isinstance(context, Mapping):
        for key, tool in context.items():
            if tool is value:
                tool_name = key
                break
    ret_schema = pydantic.TypeAdapter(
        Encodable[value.__signature__.return_annotation]  # type: ignore[name-defined]
    ).json_schema(mode="serialization")
    description = (
        f"{getattr(value, '__qualname__', value.__name__)} : {value.__signature__}"
    )
    description += f"\n\n{textwrap.dedent(value.__doc__ or '')}"
    description += f"\n\nAnnotated JSON schema of return type: {json.dumps(ret_schema)}"
    return pydantic.TypeAdapter(ChatCompletionToolParam).validate_python(
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": response_format["json_schema"]["schema"],
                "strict": True,
            },
        }
    )


@TypeToPydanticType.register(Tool)
def _pydantic_type_tool(ty: type[Tool]):
    schema = _inline_refs(pydantic.TypeAdapter(ChatCompletionToolParam).json_schema())
    schema = _ensure_strict_json_schema(schema, path=(), root={})
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate_tool),
        pydantic.PlainSerializer(_serialize_tool),
        pydantic.WithJsonSchema(schema),
    ]


def _validate_tool_call(
    value: ChatCompletionMessageToolCall,
    info: pydantic.ValidationInfo,
) -> DecodedToolCall:
    if isinstance(value, dict):
        value = OpenAIChatCompletionMessageToolCall.model_validate(value)
    ctx = info.context or {}
    assert value.function.name is not None
    tool = ctx[_TOOLS_KEY][value.function.name]
    assert isinstance(tool, Tool)
    sig = inspect.signature(tool)
    decoded_args = {}
    for name, raw_arg in json.loads(value.function.arguments).items():
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
        id=value.id,
        name=value.function.name,
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
    schema = _inline_refs(OpenAIChatCompletionMessageToolCall.model_json_schema())
    schema = _ensure_strict_json_schema(schema, path=(), root={})
    return typing.Annotated[
        ty,
        pydantic.PlainValidator(_validate_tool_call),
        pydantic.PlainSerializer(_serialize_tool_call),
        pydantic.WithJsonSchema(schema),
    ]
