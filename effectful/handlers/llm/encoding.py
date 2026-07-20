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
    MutableMapping,
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

import effectful.handlers.llm.evaluation as evaluation
from effectful.handlers.llm.template import Template, Tool
from effectful.internals.unification import GenericAlias, TypeEvaluator, nested_type
from effectful.ops.semantics import fwd, handler
from effectful.ops.types import Operation, Term

type ToolCallID = str

# Key under which the name->Tool mapping is stashed in the decoding context.
# Deliberately not a valid Python identifier, so it can never collide with a
# lexical variable name sharing the context (e.g. a reader named after its var).
_TOOLS_KEY: typing.Literal["$TOOLS"] = "$TOOLS"
# Reserved key under which the type-check anchor (the enclosing Template's
# underlying function) rides in the Pydantic decoding context, alongside the
# lexical environment. `decode` reads it to type-check a synthesized function
# against the Template's source; absent (tool-argument decoding) means skip.
# Deliberately not a valid identifier so `LexicalReaders` skips it (no tool leak)
# and it can never collide with a lexical name.
TYPE_CHECK_ANCHOR_KEY = "<type_check_anchor>"

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


if typing.TYPE_CHECKING:
    type Encodable[T] = typing.Annotated[T, "encoded"]
else:

    class Encodable:
        """The type-driven JSON bridge between Python values and the LLM.

        `Encodable[T]` maps a Python type `T` to a Pydantic-compatible type
        whose JSON schema and (de)serialization the harness uses to move
        values across the model boundary in both directions:

        - **Encoding (Python -> model):** argument and tool-result *values*
          spliced into prompts are serialized to JSON via `Encodable[type]`,
          so the model sees a faithful, schema-shaped rendering of each value
          (including non-text values such as images, emitted as content
          blocks).
        - **Decoding (model -> Python):** a `Template`'s structured return
          value and the arguments of every tool call are validated and
          decoded from the model's JSON back into real Python objects through
          the same `Encodable[type]` schema, so the value handed to your code
          already has the declared type.

        Custom types register their JSON representation with
        `TypeToPydanticType`; see
        `effectful.handlers.llm.encoding.type_to_encodable_type`. Because the
        encoding is derived from the *type*, it is the single source of truth
        for both the schema shown to the model and the validation applied to
        its output.
        """

        def __class_getitem__(cls, item):
            return TypeToPydanticType().evaluate(item)


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
        if typing.get_origin(ty) is typing.Annotated and any(
            isinstance(m, _SynthesisSpec) for m in ty.__metadata__
        ):
            inner, *meta = typing.get_args(ty)
            return self._registry.dispatch(typing.get_origin(inner) or inner)(
                inner, *meta
            )

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

    def validate(value: object) -> types.CodeType:
        if isinstance(value, types.CodeType):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"expected Python source as a string, got {type(value).__name__}"
            )
        filename = f"{_CODE_FILENAME_PREFIX}{uuid.uuid4()}>"
        try:
            return evaluation.compile(evaluation.parse(value, filename), filename)
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


@dataclasses.dataclass(frozen=True)
class _SynthesisSpec[T]:
    template: Template[..., T]

    @property
    def _class_template(self) -> Template[..., T] | None:
        if isinstance(self.template.__default__, types.MethodType):
            return self.template.__default__.__func__.__wrapped__  # type: ignore[attr-defined]
        else:
            return None

    def _method_instance(self, other: Template) -> typing.Any | None:
        """The instance ``op`` is bound to, if ``op`` is this synthesized
        Agent-method on *some* instance; otherwise ``None``.
        """
        if (
            self._class_template is not None
            and _SynthesisSpec(other)._class_template is self._class_template
        ):
            return other.__default__.__self__  # type: ignore[attr-defined]
        else:
            return None


class SynthesizedFunction(pydantic.BaseModel):
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

        # Check that the function has type annotations for all parameters
        for arg in last_stmt.args.args:
            if arg.annotation is None:
                raise ValueError(
                    f"decode() requires all parameters to have type annotations, "
                    f"parameter '{arg.arg}' is missing an annotation"
                )

        # Check that the function has a return type annotation
        if last_stmt.returns is None:
            raise ValueError(
                "decode() requires the function to have a return type annotation"
            )

        # no __future__ imports are allowed
        for stmt in module.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                raise ValueError(
                    "decode() does not allow __future__ imports in the module code"
                )

        # no star imports are allowed
        for stmt in module.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.names:
                for alias in stmt.names:
                    if alias.name == "*":
                        raise ValueError(
                            "decode() does not allow star imports in the module code"
                        )

        return value


def _create_typed_synthesized_function(
    callable_type: type[Callable],
) -> type[SynthesizedFunction]:
    """Create a SynthesizedFunction subclass with type signature in the model description.

    Uses pydantic.create_model to ensure the description is included in the JSON schema
    sent to the LLM, informing it of the expected function signature.
    """
    if not typing.get_args(callable_type):
        type_signature = "Callable"
    # Callable[[arg1, arg2, ...], return_type]
    elif len(typing.get_args(callable_type)) >= 2:
        param_types = typing.get_args(callable_type)[0]
        return_type = typing.get_args(callable_type)[-1]

        if param_types is ...:
            params_str = "..."
        elif isinstance(param_types, list | tuple):
            params_str = ", ".join(getattr(t, "__name__", str(t)) for t in param_types)
        else:
            params_str = str(param_types)

        return_str = getattr(return_type, "__name__", str(return_type))
        type_signature = f"Callable[[{params_str}], {return_str}]"
    else:
        type_signature = str(callable_type)

    return pydantic.create_model(
        "TypedSynthesizedFunction",
        __base__=SynthesizedFunction,
        __doc__=f"""Python function with signature <signature>{type_signature}</signature>""",
    )


def _validate_signature_callable(
    func: Callable,
    expected_params: list[type] | None,
    expected_return: type,
) -> None:
    """Validate the function signature from runtime callable after execution.

    The synthesized function must have type annotations for parameters and return type.
    """
    sig = inspect.signature(func)

    if expected_params is not None:
        actual_params = list(sig.parameters.values())
        if len(actual_params) != len(expected_params):
            raise ValueError(
                f"decode() expected function with {len(expected_params)} parameters, "
                f"got {len(actual_params)}"
            )

    actual_return = sig.return_annotation
    if actual_return is inspect.Parameter.empty:
        raise ValueError(
            "decode() requires synthesized function to have a return type annotation"
        )


@TypeToPydanticType.register(Callable)
def _pydantic_callable(
    callable_type: typing.Any, metadata: _SynthesisSpec | None = None
) -> typing.Any:
    """Create a Pydantic-compatible Annotated type for a parameterized Callable.

    Usage: PydanticCallable(Callable[[int, str], bool])
    """
    type_args = typing.get_args(callable_type)

    if not type_args:
        typed_enc = _create_typed_synthesized_function(Callable[..., typing.Any])  # type: ignore[arg-type]
        expected_params = None
        expected_return = None
    else:
        if len(type_args) < 2:
            raise pydantic.errors.PydanticSchemaGenerationError(
                f"Callable type signature incomplete: {callable_type}. "
                "Expected Callable[[ParamTypes...], ReturnType] or Callable[..., ReturnType]."
            )
        if type_args[1] is None:
            raise pydantic.errors.PydanticSchemaGenerationError(
                "Cannot decode/synthesize callable without a concrete type signature. "
                "Use Callable[[ParamTypes...], ReturnType] or Callable[..., ReturnType] "
                "with a concrete return type (not Any)."
            )
        param_types, expected_return = type_args[0], type_args[1]
        typed_enc = _create_typed_synthesized_function(callable_type)
        if param_types is not ... and isinstance(param_types, list | tuple):
            expected_params = list(param_types)
        else:
            expected_params = None

    def _validate(value: typing.Any, info: pydantic.ValidationInfo) -> Callable:
        if callable(value) and not isinstance(value, dict):
            return value
        if isinstance(value, SynthesizedFunction):
            encoded = value
        elif isinstance(value, dict):
            encoded = typed_enc.model_validate(value)
        elif isinstance(value, str):
            encoded = typed_enc.model_validate_json(value)
        else:
            raise ValueError(
                f"Expected callable, SynthesizedFunction dict, or JSON string, "
                f"got {type(value)}"
            )

        ctx = info.context or {}
        filename = f"<synthesis:{id(encoded)}>"
        module: ast.AST = evaluation.parse(encoded.module_code, filename)

        # The anchor (Template's underlying function) rides in the decoding context
        # under TYPE_CHECK_ANCHOR_KEY; absent for tool-argument decoding, whose
        # synthesized Callables are contracted by the tool param's type, not the
        # Template's return type, so the Template anchor doesn't apply. When
        # present, the code is spliced into the Template body, so first reject
        # constructs illegal once nested (star / `__future__` imports), then check.
        anchor = ctx.get(TYPE_CHECK_ANCHOR_KEY)
        if anchor is not None:
            evaluation.scan_non_nestable(module)
            spliced = evaluation.splice_into_source(module, anchor)
            if spliced is not None:
                evaluation.type_check(*spliced)

        g: MutableMapping[str, typing.Any] = {}
        g.update({k: v for k, v in ctx.items() if k.isidentifier()})

        bytecode: types.CodeType = evaluation.compile(module, filename)
        evaluation.exec(bytecode, g)

        result = g[module.body[-1].name]  # type: ignore
        _validate_signature_callable(result, expected_params, expected_return)

        if metadata is not None:
            if metadata._class_template is not None:
                # Agent-method template: doctests build their own instances, so the
                # method must route to `synth` on *any* instance (not just the one
                # that triggered synthesis).  A fresh instance's call dispatches
                # through `Template.__apply__`, which we intercept here.
                result = functools.wraps(metadata._class_template)(result)

                def _doctest_apply(op, *args, **kwargs):
                    instance = metadata._method_instance(op)
                    if instance is None:
                        return fwd()
                    return metadata._class_template(instance, *args, **kwargs)

                with handler(
                    {
                        Template.__apply__: _doctest_apply,
                        metadata._class_template: result,
                    }
                ):
                    evaluation.run_doctests(result, g)
                return result
            else:
                # Free-function template: shadow the global name the doctest calls,
                # and route the template op back into `synth` for recursion.
                result = functools.wraps(metadata.template)(result)
                g.update({metadata.template.__name__: result})
                with handler({metadata.template: result}):
                    evaluation.run_doctests(result, g)
                return result
        else:
            evaluation.run_doctests(result, g)
            return result

    def _serialize(value: Callable) -> dict:
        if not callable(value):
            raise TypeError(f"Expected callable, got {type(value)}")

        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            source = None

        if source:
            return typed_enc(module_code=textwrap.dedent(source)).model_dump()

        name = getattr(value, "__name__", None)
        docstring = inspect.getdoc(value)
        if name is None or docstring is None:
            raise ValueError(
                f"Cannot encode callable {value}: no source code and no __name__ or docstring"
            )

        try:
            sig = inspect.signature(value)
            sig_str = str(sig)
        except (ValueError, TypeError):
            sig_str = "(...)"

        stub_code = f'''def {name}{sig_str}:
    """{docstring}"""
    ...
'''
        return typed_enc(module_code=stub_code).model_dump()

    return typing.Annotated[
        callable_type,
        pydantic.PlainValidator(_validate),
        pydantic.PlainSerializer(_serialize),
        # Distinct schemas per direction. Validation (the model *produces* a
        # function -- tool arguments, response_format) carries the synthesis
        # instructions. Serialization (the model *reads* an encoded function --
        # e.g. a tool's output) shows only the shape `_serialize` emits, with no
        # synthesis prose.
        pydantic.WithJsonSchema(
            _inline_refs(pydantic.TypeAdapter(typed_enc).json_schema()),
            mode="validation",
        ),
        pydantic.WithJsonSchema(
            {
                "type": "object",
                "required": ["module_code"],
                "properties": {
                    "module_code": {
                        "type": "string",
                        "description": "Python source defining the function.",
                    }
                },
            },
            mode="serialization",
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


def _serialize_tool(value: Tool) -> ChatCompletionToolParam:
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
                "name": value.__name__,
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
