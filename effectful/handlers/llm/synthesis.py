import collections
import collections.abc
import dataclasses
import inspect
import linecache
import tempfile
import textwrap
import types
import typing
from collections.abc import Callable
from typing import Any

import pydantic
from litellm import OpenAIMessageContentListBlock
from mypy import api as mypy_api
from pydantic import Field

from effectful.handlers.llm import LexicalContext, Template
from effectful.handlers.llm.encoding import EncodableAs, type_to_encodable_type
from effectful.handlers.llm.providers import type_check
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation


class SynthesisError(Exception):
    """Raised when program synthesis fails."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class SynthesizedFunction(pydantic.BaseModel):
    """Structured output for function synthesis.

    Pydantic model representing synthesized code with function name and module code.
    """

    function_name: str = Field(
        ...,
        description="The name of the main function that satisfies the specification",
    )
    module_code: str = Field(
        ...,
        description="Complete Python module code (no imports needed)",
    )


@type_to_encodable_type.register(collections.abc.Callable)
class EncodableSynthesizedFunction(
    EncodableAs[Callable, SynthesizedFunction],
):
    """Encodes Callable to SynthesizedFunction and vice versa."""

    t = SynthesizedFunction

    @classmethod
    def encode(
        cls, vl: Callable, context: LexicalContext | None = None
    ) -> SynthesizedFunction:
        """Encode a Callable to a SynthesizedFunction.

        Extracts the function name and source code.
        """
        func_name = vl.__name__
        try:
            source = inspect.getsource(vl)
        except (OSError, TypeError):
            # If we can't get source, create a minimal representation
            try:
                sig = inspect.signature(vl)
                source = f"def {func_name}{sig}:\n    pass  # Source unavailable"
            except (ValueError, TypeError):
                source = f"def {func_name}(...):\n    pass  # Source unavailable"

        return SynthesizedFunction(
            function_name=func_name, module_code=textwrap.dedent(source).strip()
        )

    # Counter for unique filenames
    _decode_counter: typing.ClassVar[int] = 0

    @classmethod
    def decode(cls, vl: SynthesizedFunction, template: typing.Any = None) -> Callable:
        """Decode a SynthesizedFunction to a Callable.

        Executes the module code and returns the named function.
        The module code becomes the function's lexical context,
        optionally augmented with the template's context.
        """
        # Extract lexical context from template if provided
        context: LexicalContext | None = None
        if template is not None and hasattr(template, "__context__"):
            ctx = template.__context__
            context = ctx if isinstance(ctx, LexicalContext) else LexicalContext(ctx)
        func_name = vl.function_name
        module_code = textwrap.dedent(vl.module_code).strip()

        cls._decode_counter += 1
        filename = f"<synthesized:{func_name}:{cls._decode_counter}>"
        lines = module_code.splitlines(keepends=True)
        # Ensure last line has newline for linecache
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        linecache.cache[filename] = (
            len(module_code),
            None,
            lines,
            filename,
        )

        # Start with provided context or empty dict
        # Include collections module for type hints in synthesized code
        exec_globals: dict[str, typing.Any] = {"collections": collections}
        if context:
            exec_globals.update(context)

        try:
            code_obj = compile(module_code, filename, "exec")
            exec(code_obj, exec_globals)
        except SyntaxError as exc:
            raise SynthesisError(
                f"Syntax error in generated code: {exc}", module_code
            ) from exc
        except Exception as exc:
            raise SynthesisError(f"Evaluation failed: {exc!r}", module_code) from exc

        if func_name not in exec_globals:
            raise SynthesisError(
                f"Function '{func_name}' not found after execution. "
                f"Available names: {[k for k in exec_globals.keys() if not k.startswith('_')]}",
                module_code,
            )

        func = exec_globals[func_name]
        # Also attach source code directly for convenience
        func.__source__ = module_code
        func.__synthesized__ = vl
        return func

    @classmethod
    def serialize(cls, vl: SynthesizedFunction) -> list[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": vl.model_dump_json()}]


# Source representation encoders for LexicalContext items
# These encode objects to their Python source code representation (as strings)


@type_to_encodable_type.register(type)
def _type_to_encodable_type_class[T](ty: type[T]) -> EncodableAs[type, str]:
    """Encoder for class/type objects - produces source code string."""

    class EncodableClass(EncodableAs[type, str]):
        t = str

        @classmethod
        def encode(cls, obj: type) -> str:
            try:
                return inspect.getsource(obj)
            except (OSError, TypeError):
                return f"class {obj.__name__}: ..."

        @classmethod
        def decode(cls, source: str, template: typing.Any = None) -> type:
            exec_globals: dict[str, Any] = {}
            exec(compile(source, "<class>", "exec"), exec_globals)
            # Find the first class defined
            for v in exec_globals.values():
                if isinstance(v, type):
                    return v
            raise ValueError("No class found in source")

    return typing.cast(EncodableAs[type, str], EncodableClass())


@type_to_encodable_type.register(Operation)
def _type_to_encodable_type_operation[T](ty: type[T]) -> EncodableAs[Operation, str]:
    """Encoder for Operation objects - produces signature string."""

    class EncodableOperation(EncodableAs[Operation, str]):
        t = str

        @classmethod
        def encode(cls, obj: Operation) -> str:
            try:
                sig = inspect.signature(obj)
                doc = obj.__doc__ or ""
                first_line = doc.split("\n")[0].strip() if doc else ""
                return (
                    f"def {obj.__name__}{sig}: ...  # {first_line}"
                    if first_line
                    else f"def {obj.__name__}{sig}: ..."
                )
            except (ValueError, TypeError):
                return f"def {obj.__name__}(...): ..."

        @classmethod
        def decode(cls, source: str, template: typing.Any = None) -> Operation:
            raise NotImplementedError("Cannot decode Operation from source")

    return typing.cast(EncodableAs[Operation, str], EncodableOperation())


@type_to_encodable_type.register(Template)
def _type_to_encodable_type_template[T](ty: type[T]) -> EncodableAs[Template, str]:
    """Encoder for Template objects - produces signature string."""

    class EncodableTemplate(EncodableAs[Template, str]):
        t = str

        @classmethod
        def encode(cls, obj: Template) -> str:
            try:
                sig = obj.__signature__
                doc = obj.__prompt_template__.split("\n")[0].strip()
                return (
                    f"def {obj.__name__}{sig}: ...  # {doc}"
                    if doc
                    else f"def {obj.__name__}{sig}: ..."
                )
            except (ValueError, TypeError, AttributeError):
                return f"def {obj.__name__}(...): ..."

        @classmethod
        def decode(cls, source: str, template: typing.Any = None) -> Template:
            raise NotImplementedError("Cannot decode Template from source")

    return typing.cast(EncodableAs[Template, str], EncodableTemplate())


def lexical_context_to_source(context: LexicalContext) -> str:
    """Convert a LexicalContext to Python source code representation.

    Generates source code/signatures for types, functions, Operations, and Templates
    in the lexical context, suitable for including in LLM prompts.
    """
    sources: list[str] = []
    seen_names: set[str] = set()

    for name, obj in context.items():
        # Skip private/dunder names and duplicates
        if name.startswith("_") or name in seen_names:
            continue
        seen_names.add(name)

        # Skip modules
        if isinstance(obj, types.ModuleType):
            continue

        # Use type_to_encodable_type to get encoder for this object's type
        try:
            encoder = type_to_encodable_type(type(obj))
            # Include if encoder produces strings (source repr) or SynthesizedFunction
            if encoder.t is str:
                source = encoder.encode(obj)
                sources.append(textwrap.dedent(source).strip())
            elif encoder.t is SynthesizedFunction:
                synth = encoder.encode(obj)
                # Extract module code from SynthesizedFunction
                sources.append(textwrap.dedent(synth.module_code).strip())
        except (TypeError, NotImplementedError, AttributeError, OSError):
            # No encoder for this type or encoding failed, skip it
            pass

    return "\n\n".join(sources)


def _get_imports_from_lexical_context(
    lexical_context: LexicalContext,
) -> list[str]:
    """Generate import statements for types in the lexical context.

    Only generates imports for types/classes that have a proper module.
    """
    imports = []
    for name, obj in lexical_context.items():
        if isinstance(obj, type):
            module = inspect.getmodule(obj)
            if module is not None and module.__name__ not in ("builtins", "__main__"):
                imports.append(f"from {module.__name__} import {name}")
    return imports


def run_mypy_check(
    code: str,
    lexical_context: LexicalContext,
    function_name: str | None = None,
    expected_type: type | None = None,
) -> tuple[bool, str]:
    """Run mypy on generated code to verify type correctness.

    Args:
        code: The generated function code
        lexical_context: Lexical context containing types for imports
        function_name: Name of the function to type-check
        expected_type: Expected Callable type (e.g., Callable[[str], int])

    Returns:
        A tuple of (success: bool, error_message: str)
    """
    imports = ["from typing import Callable", "import collections.abc"]
    imports.extend(_get_imports_from_lexical_context(lexical_context))

    # Build full module with the generated code
    module_parts = imports + ["", textwrap.dedent(code).strip()]

    # Add type assertion if expected_type is provided
    if function_name and expected_type:
        module_parts.append(f"_: {repr(expected_type)} = {function_name}")

    full_source = "\n".join(module_parts)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete_on_close=False
    ) as f:
        f.write(full_source)
        f.close()  # Close so mypy can read it

        result = mypy_api.run(
            [
                f.name,
                "--no-error-summary",
                "--no-pretty",
                "--hide-error-context",
                "--no-color-output",
            ]
        )
        stdout, stderr, exit_code = result

        if exit_code != 0:
            error_msg = stdout.replace(f.name, "<generated>")
            return False, error_msg.strip()

        return True, ""


class ProgramSynthesis(ObjectInterpretation):
    """Provides a `template` handler to instruct the LLM to generate code of the
    right form and with the right type.
    """

    @implements(Template.__call__)
    def _call(self, template, *args, **kwargs) -> Callable:
        ret_type = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type)
        ret_type_origin = ret_type if origin is None else origin

        # Check if return type is Callable - handle both class and typing special forms
        is_callable = ret_type_origin is collections.abc.Callable
        if not is_callable:
            return fwd()

        # Include the full lexical context - all functions, types, values available to synthesized code
        context_source = lexical_context_to_source(template.__context__)
        escaped_context = context_source.replace("{", "{{").replace("}", "}}")
        context_section = f"""
The following types, functions, and values are available:

```python
{escaped_context}
```
"""

        prompt_ext = textwrap.dedent(f"""
        Implement a Python function with the following specification.

        **Specification:** {template.__prompt_template__}

        **Required function signature:** {repr(ret_type)}
        {context_section}
        **Critical Instructions:**
        1. The function you write MUST have EXACTLY this signature: {repr(ret_type)}
        2. Any values mentioned in the specification (like specific characters or strings) should be hardcoded directly in the function body, NOT as parameters.
        3. Do NOT create a wrapper or factory function. Write the function directly.
        4. You may include helper functions/classes/constants.
        5. Do not redefine provided types - they are already available.
        6. Do not include import statements.
        
        Example: If asked to "count occurrences of 'a'" with signature Callable[[str], int], write:
        def count_a(text: str) -> int:
            return text.count('a')
        NOT:
        def make_counter(char: str) -> Callable[[str], int]:
            def inner(text: str) -> int:
                return text.count(char)
            return inner
        """).strip()

        # NOTE: Only modify the prompt, keep original return type
        # decode_response will use EncodableSynthesizedFunction.decode with template context
        # type_check will be called by LiteLLMProvider with original template
        return fwd(
            dataclasses.replace(template, __prompt_template__=prompt_ext),
            *args,
            **kwargs,
        )


class CallableTypeCheckHandler(ObjectInterpretation):
    """Type check handler for Callable types using mypy.

    Intercepts the type_check operation when the expected type is a Callable,
    and runs mypy to validate the synthesized function's type signature.
    """

    @implements(type_check)
    def _type_check_callable(self, value: Callable, template: Template) -> Callable:
        """Run mypy type checking on Callable values."""
        expected_type = template.__signature__.return_annotation
        origin = typing.get_origin(expected_type)
        type_origin = expected_type if origin is None else origin

        # Only handle Callable types
        if type_origin is not collections.abc.Callable:
            return fwd()

        # Get the synthesized function's source code
        synth: SynthesizedFunction | None = getattr(value, "__synthesized__", None)
        if synth is None:
            # Not a synthesized function, skip type checking
            return fwd()

        # Run mypy type check
        context = template.__context__
        lexical_context = (
            context if isinstance(context, LexicalContext) else LexicalContext({})
        )
        success, error_msg = run_mypy_check(
            synth.module_code,
            lexical_context,
            function_name=synth.function_name,
            expected_type=expected_type,
        )
        if not success:
            raise SynthesisError(f"Type check failed:\n{error_msg}", synth.module_code)

        return value
