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
from pydantic import Field

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import (
    InstructionHandler,
    OpenAIMessageContentListBlock,
)
from effectful.handlers.llm.encoding import EncodableAs, type_to_encodable_type
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements


@defop
def get_synthesis_context() -> ChainMap[str, Any] | None:
    """Get the current synthesis context for decoding synthesized code."""
    return None

class SynthesisContextHandler(ObjectInterpretation):
    """Handler that provides the synthesis context to decode operations."""

    def __init__(self, context: ChainMap[str, Any]):
        self.context = context

    @implements(get_synthesis_context)
    def _get_context(self) -> ChainMap[str, Any] | None:
        return self.context


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
        Uses get_synthesis_context() operation to get the lexical context.
        """
        context: ChainMap[str, Any] | None = get_synthesis_context()
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

    Intercepts Template.__apply__ and uses SynthesisInstructionHandler to inject
    synthesis-specific instructions, following the same pattern as RetryLLMHandler.
    """

    def _build_synthesis_instruction(self, template: Template) -> str:
        """Build the synthesis instruction for a Callable return type."""
        ret_type = template.__signature__.return_annotation

        # Escape braces in context to avoid format field interpretation
        context_str = str(template.__context__).replace("{", "{{").replace("}", "}}")
        ret_type_repr = repr(ret_type).replace("{", "{{").replace("}", "}}")

        return textwrap.dedent(f"""
        You are a code synthesis assistant. Generate Python code based on the user's specification.

        **Required function signature:** {ret_type_repr}
        
        The following types, functions, and values are available in the execution context:

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

        Respond with a JSON object containing:
        - "function_name": the name of the main function
        - "module_code": the complete Python code (no imports needed)
        """).strip()

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Intercept template calls for Callable return types and inject synthesis instructions."""
        if not _is_callable_return_type(template):
            return fwd()

        # Build synthesis instruction
        instruction = self._build_synthesis_instruction(template)

        # Use handlers to inject context and instructions
        with handler(SynthesisContextHandler(template.__context__)):
            with handler(InstructionHandler(instruction)):
                return fwd()
