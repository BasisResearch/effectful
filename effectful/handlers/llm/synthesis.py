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
    decode_response,
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
    def decode(cls, vl: SynthesizedFunction) -> Callable:
        """Decode a SynthesizedFunction to a Callable.

        Executes the module code and returns the named function.
        Uses _decode_context attribute on vl if present (set by ProgramSynthesis).
        """
        context: ChainMap[str, Any] | None = getattr(vl, "_decode_context", None)
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
        func.__source__ = module_code
        func.__synthesized__ = vl
        return func

    @classmethod
    def serialize(cls, vl: SynthesizedFunction) -> list[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": vl.model_dump_json()}]


class ProgramSynthesis(ObjectInterpretation):
    """Provides a `template` handler to instruct the LLM to generate code of the
    right form and with the right type.
    """

    @implements(Template.__apply__)
    def _call(self, template, *args, **kwargs) -> None:
        """Handle synthesis of Callable return types."""
        ret_type = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type)
        ret_type_origin = ret_type if origin is None else origin

        # Check if return type is Callable
        if ret_type_origin is not collections.abc.Callable:
            return fwd()

        prompt_ext = textwrap.dedent(f"""
        Implement a Python function with the following specification.

        **Specification:** {template.__prompt_template__}

        **Required function signature:** {repr(ret_type)}
        The following types, functions, and values are available:

        ```python
        {template.__context__}
        ```
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

        return fwd(
            template.replace(prompt_template=prompt_ext),
            *args,
            **kwargs,
        )

    @implements(decode_response)
    def _decode_response(self, template: Template, response: ModelResponse) -> Callable:
        """Decode a synthesized function response with lexical context."""
        ret_type = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type)
        ret_type_origin = ret_type if origin is None else origin

        # Only handle Callable return types
        if ret_type_origin is not collections.abc.Callable:
            return fwd()

        # Parse JSON and attach context to the value for decode() to use
        choice = typing.cast(typing.Any, response.choices[0])
        result_str: str = choice.message.content or ""
        Result = pydantic.create_model("Result", value=(SynthesizedFunction, ...))
        synth: SynthesizedFunction = Result.model_validate_json(result_str).value  # type: ignore[attr-defined]
        object.__setattr__(synth, "_decode_context", template.__context__)
        return EncodableSynthesizedFunction.decode(synth)
