import collections
import collections.abc
import dataclasses
import inspect
import linecache
import tempfile
import textwrap
import typing
from collections.abc import Callable
from typing import Any

import pydantic
from mypy import api as mypy_api
from pydantic import Field

from effectful.handlers.llm import Template
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class SynthesisError(Exception):
    """Raised when program synthesis fails."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class SynthesizedFunction(pydantic.BaseModel):
    """Structured output for function synthesis.

    LLM provides a complete module and the name of the main function.
    We add a type assertion to verify the function matches the expected signature.
    """

    function_name: str = Field(
        ...,
        description="The name of the main function that satisfies the specification",
    )
    module_code: str = Field(
        ...,
        description="Complete Python module code (no imports needed)",
    )


def _get_imports_from_lexical_context(
    lexical_context: dict[str, tuple[str, Any]],
) -> list[str]:
    """Generate import statements for types in the lexical context.

    Only generates imports for types/classes that have a proper module.
    """
    imports = []
    for name, (_, obj) in lexical_context.items():
        if isinstance(obj, type):
            module = inspect.getmodule(obj)
            if module is not None and module.__name__ not in ("builtins", "__main__"):
                imports.append(f"from {module.__name__} import {name}")
    return imports


def run_mypy_check(
    code: str,
    lexical_context: dict[str, tuple[str, Any]],
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
        lexical_context: dict[str, tuple[str, Any]],
    ) -> typing.Callable:
        """Build and execute a function from the synthesized module.

        Executes the LLM's module code as-is and optionally verifies
        the function matches the expected type using a type assertion.

        Args:
            result: The structured output from the LLM
            callable_type: The expected Callable type (e.g., Callable[[str], int])
            lexical_context: Dict of lexical context (functions/types) available in scope

        Returns:
            The synthesized callable function
        """
        func_name = result.function_name
        module_code = textwrap.dedent(result.module_code).strip()

        # Add type assertion for mypy checking (use repr for the type)
        type_assertion = f"\n\n_: {repr(callable_type)} = {func_name}"
        code_with_assertion = module_code + type_assertion

        # Register in linecache for better tracebacks
        lines = module_code.splitlines(keepends=True)
        filename = f"<generated-{hash(module_code)}>"
        linecache.cache[filename] = (len(module_code), None, lines, filename)

        # Optional mypy type checking with type assertion
        if self.type_check:
            success, error_msg = run_mypy_check(code_with_assertion, lexical_context)
            if not success:
                raise SynthesisError(f"Type check failed:\n{error_msg}", module_code)

        # Build globals dict from lexical context (all functions, types, etc.)
        gs: dict = {}
        for name, (_, obj) in lexical_context.items():
            gs[name] = obj

        try:
            code_obj = compile(module_code, filename, "exec")
            exec(code_obj, gs)
        except SyntaxError as exc:
            raise SynthesisError(
                f"Syntax error in generated code: {exc}", module_code
            ) from exc
        except Exception as exc:
            raise SynthesisError(f"Evaluation failed: {exc!r}", module_code) from exc

        if func_name not in gs:
            raise SynthesisError(
                f"Function '{func_name}' not found after execution. "
                f"Available names: {[k for k in gs.keys() if not k.startswith('_')]}",
                module_code,
            )

        return gs[func_name]

    @implements(Template.__call__)
    def _call(self, template, *args, **kwargs) -> Callable:
        ret_type = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type)
        ret_type_origin = ret_type if origin is None else origin

        if not (issubclass(ret_type_origin, collections.abc.Callable)):  # type: ignore[arg-type]
            return fwd()

        # Get lexical context - contains all functions, types, and values from definition site
        lexical_context = getattr(template, "lexical_context", {})

        # Include the full lexical context - all functions, types, values available to synthesized code
        context_section = ""
        if lexical_context:
            context_source = template.get_lexical_context_source()
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

        # Build and return the function using lexical context for exec globals
        return self._build_function(response, ret_type, lexical_context)
