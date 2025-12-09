import collections
import collections.abc
import dataclasses
import inspect
import linecache
import tempfile
import textwrap
import typing
from collections.abc import Callable
from typing import Any, get_args, get_origin

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

    LLM provides helper code freely, but main function structure is constrained.
    We reconstruct the main function with prescribed types from the signature.
    """

    helper_code: str = Field(
        default="",
        description="Optional helper functions, classes, or constants (no imports needed), will be executed before the main function",
    )
    function_name: str = Field(
        ...,
        description="The name of the main function",
    )
    param_names: list[str] = Field(
        ...,
        description="Parameter names in order (must match number of parameter types)",
    )
    body: str = Field(
        ...,
        description="The indented function body including return statement",
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


def _format_type_for_annotation(t: type) -> str:
    """Format a type for use in a type annotation string.

    Handles Callable types from collections.abc and typing module.
    """
    origin = get_origin(t)

    if origin is not None:
        # handle generic types like Callable[[X], Y], list[X], etc.
        args = get_args(t)

        # get the origin name - handle collections.abc.Callable -> Callable
        if hasattr(origin, "__name__"):
            origin_name = origin.__name__
        else:
            origin_name = str(origin).split(".")[-1]

        if origin_name == "Callable" and args:
            # Format as Callable[[P1, P2], R]
            param_types = args[0]
            return_type = args[-1]

            if param_types is ...:
                params_str = "..."
            else:
                params_str = (
                    "["
                    + ", ".join(_format_type_for_annotation(p) for p in param_types)
                    + "]"
                )

            ret_str = _format_type_for_annotation(return_type)
            return f"Callable[{params_str}, {ret_str}]"
        else:
            # Generic type like list[X], dict[K, V]
            args_str = ", ".join(_format_type_for_annotation(a) for a in args)
            return f"{origin_name}[{args_str}]"

    # Simple type
    if hasattr(t, "__name__"):
        return t.__name__
    return str(t)


def _get_param_types(callable_type: type) -> list[type] | None:
    """Extract parameter types from a Callable type.

    Returns None if the callable uses ellipsis (...) for params.
    """
    args = get_args(callable_type)
    if not args:
        return []

    param_types = args[0]
    if param_types is ...:
        return None

    return list(param_types)


def _format_param_signature(
    callable_type: type, param_names: list[str] | None = None
) -> str:
    """Format the parameter signature from a Callable type.

    E.g., Callable[[str, int], bool] with names ["text", "count"]
    -> "text: str, count: int"
    """
    param_types = _get_param_types(callable_type)
    if param_types is None:
        return "*args, **kwargs"
    if not param_types:
        return ""

    params = []
    for i, param_type in enumerate(param_types):
        type_str = _format_type_for_annotation(param_type)
        name = param_names[i] if param_names and i < len(param_names) else f"arg{i}"
        params.append(f"{name}: {type_str}")

    return ", ".join(params)


def _format_return_type(callable_type: type) -> str:
    """Extract and format the return type from a Callable type."""
    args = get_args(callable_type)
    if not args:
        return "Any"

    return_type = args[-1]
    return _format_type_for_annotation(return_type)


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
    imports = _get_imports_from_lexical_context(lexical_context)
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
        """Build and execute a function from the structured synthesis result.

        Combines helper code with a reconstructed main function using prescribed types.

        Args:
            result: The structured output from the LLM
            callable_type: The expected Callable type (e.g., Callable[[str], int])
            lexical_context: Dict of lexical context (functions/types) available in scope

        Returns:
            The synthesized callable function
        """
        func_name = result.function_name

        # Ensure body is properly indented
        body = result.body
        if body and not body.startswith("    ") and not body.startswith("\t"):
            body = textwrap.indent(body, "    ")

        param_sig = _format_param_signature(callable_type, result.param_names)
        return_type = _format_return_type(callable_type)
        func_code = f"def {func_name}({param_sig}) -> {return_type}:\n{body}"

        helper_code = textwrap.dedent(result.helper_code).strip()
        if helper_code:
            module_code = helper_code + "\n\n" + func_code
        else:
            module_code = func_code

        # Register in linecache for better tracebacks
        lines = module_code.splitlines(keepends=True)
        filename = f"<generated-{hash(module_code)}>"
        linecache.cache[filename] = (len(module_code), None, lines, filename)

        # Optional mypy type checking - uses lexical context for imports
        if self.type_check:
            success, error_msg = run_mypy_check(module_code, lexical_context)
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

        # Get parameter types and return type for the prompt
        param_types = _get_param_types(ret_type)
        return_type_str = _format_return_type(ret_type)

        # Format parameter types for display
        if param_types is None:
            param_types_str = "*args, **kwargs"
        elif not param_types:
            param_types_str = "(no parameters)"
        else:
            param_types_str = ", ".join(
                _format_type_for_annotation(t) for t in param_types
            )

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

        **Required signature:**
        - Parameter types (in order): {param_types_str}
        - Return type: {return_type_str}
        {context_section}
        **Instructions:**
        1. Choose a descriptive function name.
        2. Choose descriptive parameter names (one for each parameter type).
        3. Implement the function body with a return statement.
        4. If needed, include helper functions/classes/constants in helper_code.
        5. Do not redefine provided types - they are already available.
        6. Do not include import statements.
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
