import collections
import collections.abc
import inspect
import linecache
import textwrap
import typing
from collections import ChainMap
from collections.abc import Callable
from typing import Any

import pydantic
from litellm.types.utils import ModelResponse
from pydantic import Field

from effectful.handlers.llm import Template
from effectful.handlers.llm.encoding import EncodableAs, type_to_encodable_type
from effectful.handlers.llm.providers import (
    OpenAIMessageContentListBlock,
    _OpenAIPromptFormatter,
    completion,
    compute_response,
    decode_response,
    format_model_input,
)
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


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
        cls, vl: Callable, context: ChainMap[str, Any] | None = None
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
    def _generate_imports_from_context(cls, context: ChainMap[str, Any] | None) -> str:
        """Generate import statements for types/functions in the context."""
        if not context:
            return ""

        imports: set[str] = set()
        for name, obj in context.items():
            if name.startswith("_"):
                continue

            module = getattr(obj, "__module__", None)
            obj_name = getattr(obj, "__name__", name)

            if module and module != "builtins" and module != "__main__":
                # Use the context name if it differs from the object's name (aliased import)
                if obj_name != name:
                    imports.add(f"from {module} import {obj_name} as {name}")
                else:
                    imports.add(f"from {module} import {obj_name}")

        return "\n".join(sorted(imports))

    @classmethod
    def decode(cls, vl: SynthesizedFunction) -> Callable:
        """Decode a SynthesizedFunction to a Callable.

        Executes the module code and returns the named function.
        Uses _decode_context attribute on vl if present (set by ProgramSynthesis).
        """
        context: ChainMap[str, Any] | None = getattr(vl, "_decode_context", None)
        func_name = vl.function_name
        module_code = textwrap.dedent(vl.module_code).strip()

        # Generate imports from context for display purposes
        imports_code = cls._generate_imports_from_context(context)
        full_module_code = (
            f"{imports_code}\n\n{module_code}" if imports_code else module_code
        )

        cls._decode_counter += 1
        filename = f"<synthesized:{func_name}:{cls._decode_counter}>"
        lines = full_module_code.splitlines(keepends=True)
        # Ensure last line has newline for linecache
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        linecache.cache[filename] = (
            len(full_module_code),
            None,
            lines,
            filename,
        )

        # Start with provided context or empty dict
        # Include collections module for type hints in synthesized code
        exec_globals: dict[str, typing.Any] = {}
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
        func.__source__ = full_module_code
        func.__synthesized__ = vl
        return func

    @classmethod
    def serialize(cls, vl: SynthesizedFunction) -> list[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": vl.model_dump_json()}]


def _is_callable_return_type(template: Template) -> bool:
    """Check if template has a Callable return type."""
    ret_type = template.__signature__.return_annotation
    origin = typing.get_origin(ret_type)
    ret_type_origin = ret_type if origin is None else origin
    return ret_type_origin is collections.abc.Callable


class ProgramSynthesis(ObjectInterpretation):
    """A program synthesis handler for Callable return types.

    Intercepts format_model_input, compute_response, and decode_response
    to customize the synthesis flow while reusing the standard template machinery.
    """

    def _build_prompt(self, template: Template) -> str:
        """Build the synthesis prompt for a Callable return type."""
        ret_type = template.__signature__.return_annotation

        # Escape braces in context to avoid format field interpretation
        context_str = str(template.__context__).replace("{", "{{").replace("}", "}}")
        ret_type_repr = repr(ret_type).replace("{", "{{").replace("}", "}}")

        return textwrap.dedent(f"""
        Implement a Python function with the following specification.

        **Specification:** {template.__prompt_template__}

        **Required function signature:** {ret_type_repr}
        The following types, functions, and values are available:

        ```python
        {context_str}
        ```
        **Critical Instructions:**
        1. The function you write MUST have EXACTLY this signature: {ret_type_repr}
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

    @implements(format_model_input)
    def _format_model_input(self, template: Template, *args, **kwargs) -> list:
        """Replace the prompt with synthesis-specific instructions."""
        if not _is_callable_return_type(template):
            return fwd()

        prompt = self._build_prompt(template)

        # Encode arguments for the prompt
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arguments = {}
        for param in bound_args.arguments:
            encoder = type_to_encodable_type(
                template.__signature__.parameters[param].annotation
            )
            encoded = encoder.encode(bound_args.arguments[param])
            arguments[param] = encoder.serialize(encoded)

        formatted_prompt = _OpenAIPromptFormatter().format_as_messages(
            prompt, **arguments
        )
        return [{"type": "message", "content": formatted_prompt, "role": "user"}]

    @implements(compute_response)
    def _compute_response(self, template: Template, model_input: list) -> ModelResponse:
        """Compute response with SynthesizedFunction format and no tools."""
        if not _is_callable_return_type(template):
            return fwd()

        response_format = pydantic.create_model(
            "Response", value=SynthesizedFunction, __config__={"extra": "forbid"}
        )

        return completion(
            messages=model_input,
            response_format=response_format,
            tools=[],  # No tools for synthesis
        )

    @implements(decode_response)
    def _decode_response(self, template: Template, response: ModelResponse) -> Callable:
        """Decode the response with lexical context attached."""
        if not _is_callable_return_type(template):
            return fwd()

        choice = typing.cast(typing.Any, response.choices[0])
        result_str: str = choice.message.content or ""
        Result = pydantic.create_model("Result", value=(SynthesizedFunction, ...))
        synth: SynthesizedFunction = Result.model_validate_json(result_str).value  # type: ignore[attr-defined]
        object.__setattr__(synth, "_decode_context", template.__context__)
        return EncodableSynthesizedFunction.decode(synth)
