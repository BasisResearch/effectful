from __future__ import annotations

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

import pydantic
from litellm import OpenAIMessageContentListBlock
from mypy import api as mypy_api
from pydantic import Field

from effectful.handlers.llm import LexicalContext, Template
from effectful.handlers.llm.encoding import EncodableAs, type_to_encodable_type
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
    def decode(
        cls, vl: SynthesizedFunction, context: LexicalContext | None = None
    ) -> Callable:
        """Decode a SynthesizedFunction to a Callable.

        Executes the module code and returns the named function.
        The module code becomes the function's lexical context,
        optionally augmented with provided context.
        """
        func_name = vl.function_name
        module_code = textwrap.dedent(vl.module_code).strip()

        # Create a unique filename and register source with linecache
        # This allows inspect.getsource() to work on the generated function
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
        exec_globals: dict[str, typing.Any] = dict(context) if context else {}

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


@type_to_encodable_type.register(LexicalContext)
class EncodableLexicalContext(
    EncodableAs[LexicalContext, str],
):
    """Encodes LexicalContext to Python source code representation.

    encode: Extracts source code/signatures for types, functions, and values.
    decode: Executes source code to reconstruct a lexical context.
    """

    t = str

    @classmethod
    def encode(cls, context: LexicalContext) -> str:
        """Generate Python source code representations of items in the lexical context."""
        from effectful.ops.types import Operation

        sources: list[str] = []
        seen_names: set[str] = set()

        for name, obj in context.items():
            # Skip private/dunder names and duplicates
            if name.startswith("_") or name in seen_names:
                continue
            seen_names.add(name)

            # Skip modules and common builtins
            if isinstance(obj, types.ModuleType):
                continue

            source = None

            # Try to get source for classes (including dataclasses)
            if isinstance(obj, type):
                try:
                    source = inspect.getsource(obj)
                except (OSError, TypeError):
                    # Fall back to showing class signature
                    source = f"class {name}: ..."

            # Handle functions
            elif callable(obj) and not isinstance(obj, (Operation, Template)):
                try:
                    source = inspect.getsource(obj)
                except (OSError, TypeError):
                    # Fall back to showing function signature
                    try:
                        sig = inspect.signature(obj)
                        source = f"def {name}{sig}: ..."
                    except (ValueError, TypeError):
                        source = f"def {name}(...): ..."

            # Handle Operations - show as function signatures
            elif isinstance(obj, Operation):
                try:
                    sig = inspect.signature(obj)
                    doc = obj.__doc__ or ""
                    first_line = doc.split("\n")[0].strip() if doc else ""
                    source = (
                        f"def {name}{sig}: ...  # {first_line}"
                        if first_line
                        else f"def {name}{sig}: ..."
                    )
                except (ValueError, TypeError):
                    source = f"def {name}(...): ..."

            # Handle Templates - show as function signatures
            elif isinstance(obj, Template):
                try:
                    sig = obj.__signature__
                    doc = obj.__prompt_template__.split("\n")[0].strip()
                    source = (
                        f"def {name}{sig}: ...  # {doc}"
                        if doc
                        else f"def {name}{sig}: ..."
                    )
                except (ValueError, TypeError, AttributeError):
                    source = f"def {name}(...): ..."

            if source:
                sources.append(textwrap.dedent(source).strip())

        return "\n\n".join(sources)

    @classmethod
    def decode(cls, source: str) -> LexicalContext:
        """Execute source code to reconstruct a lexical context."""
        exec_globals: dict[str, typing.Any] = {}
        try:
            code_obj = compile(source, "<lexical_context>", "exec")
            exec(code_obj, exec_globals)
        except SyntaxError:
            # Source may be stubs/signatures that can't be executed
            pass
        except Exception:
            pass
        return LexicalContext(exec_globals)

    @classmethod
    def serialize(cls, source: str) -> list[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": source}]


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
) -> tuple[bool, str]:
    """Run mypy on generated code to verify type correctness.

    Args:
        code: The generated function code
        lexical_context: Lexical context containing types for imports

    Returns:
        A tuple of (success: bool, error_message: str)
    """
    # Always include collections.abc for Callable type assertions
    imports = ["import collections.abc"]
    imports.extend(_get_imports_from_lexical_context(lexical_context))
    source_parts = imports + [textwrap.dedent(code).strip()]

    full_source = "\n".join(source_parts)

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

    def __init__(self, type_check: bool = False):
        """Initialize the program synthesis handler.

        Args:
            type_check: Whether to run mypy to verify the generated code.
                        Even with constrained decoding, this can catch errors
                        in the function body implementation.
        """
        self.type_check = type_check

    def _build_function(
        self,
        result: SynthesizedFunction,
        callable_type: type,
        lexical_context: LexicalContext,
    ) -> Callable:
        """Build and execute a function from the synthesized module.

        Uses EncodableSynthesizedFunction.decode with the template's lexical context.
        Optionally runs mypy type checking if enabled.

        Args:
            result: The structured output from the LLM
            callable_type: The expected Callable type (e.g., Callable[[str], int])
            lexical_context: Dict of lexical context (functions/types) available in scope

        Returns:
            The synthesized callable function
        """
        if self.type_check:
            success, error_msg = run_mypy_check(result.module_code, lexical_context)
            if not success:
                raise SynthesisError(
                    f"Type check failed:\n{error_msg}", result.module_code
                )

        return EncodableSynthesizedFunction.decode(result, context=lexical_context)

    @implements(Template.__call__)
    def _call(self, template, *args, **kwargs) -> Callable:
        ret_type = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type)
        ret_type_origin = ret_type if origin is None else origin

        # Check if return type is Callable - handle both class and typing special forms
        # Also handle string annotations (from __future__ import annotations)
        if isinstance(ret_type_origin, str):
            is_callable = "Callable" in ret_type_origin
        else:
            is_callable = ret_type_origin is collections.abc.Callable or (
                isinstance(ret_type_origin, type)
                and issubclass(ret_type_origin, collections.abc.Callable)
            )
        if not is_callable:
            return fwd()

        # Include the full lexical context - all functions, types, values available to synthesized code
        context_source = EncodableLexicalContext.encode(template.__context__)
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

        **Required signature:** {repr(ret_type)}
        {context_section}
        **Instructions:**
        1. Write a complete Python module with the function.
        2. Choose descriptive function and parameter names.
        3. You may include helper functions/classes/constants.
        4. Do not redefine provided types - they are already available.
        5. Do not include import statements.
        """).strip()

        response: SynthesizedFunction = fwd(
            dataclasses.replace(
                template,
                __prompt_template__=prompt_ext,
                __signature__=template.__signature__.replace(
                    return_annotation=SynthesizedFunction
                ),
            ),
            *args,
            **kwargs,
        )

        # Build the function using lexical context for exec globals
        synthesized_func = self._build_function(response, ret_type, template.__context__)
        
        # Check if the synthesized function should be called with template args
        # or if it's already the result (matches the return type directly)
        try:
            synth_sig = inspect.signature(synthesized_func)
            template_sig = template.__signature__
            
            # If synthesized function has same parameter names as template,
            # it's a factory function - call it with the args
            synth_params = set(synth_sig.parameters.keys())
            template_params = set(template_sig.parameters.keys())
            
            if synth_params == template_params:
                # Factory function - call it to get the actual callable
                result = synthesized_func(*args, **kwargs)
                # Copy synthesis metadata to the result if it's callable
                if callable(result):
                    if hasattr(synthesized_func, "__source__"):
                        result.__source__ = synthesized_func.__source__
                    if hasattr(synthesized_func, "__synthesized__"):
                        result.__synthesized__ = synthesized_func.__synthesized__
                return result
        except (ValueError, TypeError):
            pass
        
        # Otherwise, the synthesized function IS the result
        return synthesized_func
