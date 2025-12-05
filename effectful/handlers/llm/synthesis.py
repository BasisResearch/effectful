import collections.abc
import dataclasses
import inspect
import linecache
import tempfile
import textwrap
import typing
from collections.abc import Callable
from typing import get_args, get_origin, get_type_hints

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

    The LLM provides the function name, parameter names, and body.
    The parameter types and return type are prescribed by the prompt.
    """

    function_name: str = Field(..., description="The name of the function")
    param_names: list[str] = Field(
        ..., description="The names of the parameters (in order)"
    )
    body: str = Field(..., description="The indented function body (implementation)")


def collect_referenced_types(t: type, seen: set[type] | None = None) -> set[type]:
    """Collect all non-builtin types referenced in a type annotation.

    Walks through a type annotation (including generic types like
    Callable[[Person], Order]) and collects all user-defined types.

    Args:
        t: The type to analyze
        seen: Set of already-processed types (to avoid infinite recursion)

    Returns:
        A set of non-builtin types referenced in the annotation
    """
    if seen is None:
        seen = set()

    types: set[type] = set()

    # Handle generic types (e.g., Callable[[X], Y], list[X], Optional[X])
    origin = get_origin(t)
    if origin is not None:
        for arg in get_args(t):
            if isinstance(arg, type):
                types.update(collect_referenced_types(arg, seen))
            elif isinstance(arg, list):
                # Handle Callable[[P1, P2], R] where args is a list
                for inner_arg in arg:
                    if isinstance(inner_arg, type):
                        types.update(collect_referenced_types(inner_arg, seen))
        return types

    # Skip non-types, already-seen types, and builtins
    if not isinstance(t, type) or t in seen:
        return types
    if t.__module__ == "builtins":
        return types

    seen.add(t)
    types.add(t)

    # Recursively process type hints from annotations
    try:
        hints = get_type_hints(t)
        for hint in hints.values():
            types.update(collect_referenced_types(hint, seen))
    except Exception:
        pass

    # For dataclasses, also check field types
    if dataclasses.is_dataclass(t):
        for field in dataclasses.fields(t):
            field_type = field.type
            if isinstance(field_type, type):
                types.update(collect_referenced_types(field_type, seen))
            elif not isinstance(field_type, str):
                types.update(collect_referenced_types(field_type, seen))

    # Check base classes (excluding object)
    for base in t.__bases__:
        if base is not object:
            types.update(collect_referenced_types(base, seen))

    return types


def get_type_imports(types: set[type]) -> list[str]:
    """Get import statements for a set of types using inspect.getmodule.

    Args:
        types: Set of types to generate imports for

    Returns:
        List of import statement strings
    """
    imports = []
    for t in types:
        module = inspect.getmodule(t)
        if module is None or module.__name__ == "builtins":
            continue
        imports.append(f"from {module.__name__} import {t.__name__}")
    return imports


def collect_type_sources(t: type) -> dict[type, str]:
    """Collect source code for all types referenced in a type annotation.

    Args:
        t: The type to analyze

    Returns:
        A dict mapping types to their source code strings
    """
    types = collect_referenced_types(t)
    sources: dict[type, str] = {}
    for typ in types:
        try:
            sources[typ] = inspect.getsource(typ)
        except (OSError, TypeError):
            # Can't get source (built-in, C extension, dynamically created, etc.)
            pass
    return sources


def format_type_context(sources: dict[type, str]) -> str:
    """Format collected type sources into a context string for the prompt.

    Args:
        sources: Dict mapping types to their source code

    Returns:
        A formatted string containing all type definitions
    """
    if not sources:
        return ""

    parts = []
    for source in sources.values():
        # Clean up the source (dedent if needed)
        cleaned = textwrap.dedent(source).strip()
        parts.append(cleaned)

    return "\n\n".join(parts)


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
    referenced_types: set[type],
) -> tuple[bool, str]:
    """Run mypy on generated code to verify type correctness.

    Args:
        code: The generated function code
        referenced_types: Set of types referenced in the signature

    Returns:
        A tuple of (success: bool, error_message: str)
    """
    source_parts = get_type_imports(referenced_types)
    source_parts.append(textwrap.dedent(code).strip())

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
        referenced_types: set[type],
    ) -> typing.Callable:
        """Build and execute a function from the structured synthesis result.

        Args:
            result: The structured output from the LLM
            callable_type: The expected Callable type (e.g., Callable[[str], int])
            referenced_types: Set of types referenced in the signature

        Returns:
            The synthesized callable function
        """
        # Build the function with prescribed types and LLM-provided names
        param_sig = _format_param_signature(callable_type, result.param_names)
        return_type = _format_return_type(callable_type)
        func_name = result.function_name

        # Ensure body is properly indented
        body = result.body
        if not body.startswith("    ") and not body.startswith("\t"):
            # Indent the body if not already indented
            body = textwrap.indent(body, "    ")

        # Construct the full function code
        code = f"def {func_name}({param_sig}) -> {return_type}:\n{body}"

        # Register in linecache for better tracebacks
        source_code = code
        lines = code.splitlines(keepends=True)
        filename = f"<generated-{hash(code)}>"
        linecache.cache[filename] = (len(source_code), None, lines, filename)

        # Optional mypy type checking
        if self.type_check:
            success, error_msg = run_mypy_check(code, referenced_types)
            if not success:
                raise SynthesisError(f"Type check failed:\n{error_msg}", code)

        # Build globals dict by importing types from their original modules
        gs: dict = {}
        for typ in referenced_types:
            module = inspect.getmodule(typ)
            if module is not None:
                # Import the type from its module - this brings in all dependencies
                gs[typ.__name__] = typ

        try:
            code_obj = compile(source_code, filename, "exec")
            exec(code_obj, gs)
        except Exception as exc:
            raise SynthesisError("evaluation failed", code) from exc

        return gs[func_name]

    @implements(Template.__call__)
    def _call(self, template, *args, **kwargs) -> Callable:
        ret_type = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type)
        ret_type_origin = ret_type if origin is None else origin

        if not (issubclass(ret_type_origin, collections.abc.Callable)):  # type: ignore[arg-type]
            return fwd()

        # Collect all types referenced in the signature
        referenced_types = collect_referenced_types(ret_type)

        # Get type sources for the prompt (to show LLM the type definitions)
        type_sources = collect_type_sources(ret_type)
        type_context = format_type_context(type_sources)

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

        # Build the type definitions section if there are custom types
        # Escape curly braces in type source code to avoid format string issues
        type_defs_section = ""
        if type_context:
            escaped_type_context = type_context.replace("{", "{{").replace("}", "}}")
            type_defs_section = f"""
The following types are available:

```python
{escaped_type_context}
```
"""

        prompt_ext = textwrap.dedent(f"""
        Implement a Python function with the following specification.

        **Specification:** {template.__prompt_template__}

        **Required types:**
        - Parameter types (in order): {param_types_str}
        - Return type: {return_type_str}
        {type_defs_section}
        **Instructions:**
        1. Choose a descriptive function name.
        2. Choose descriptive parameter names (one for each parameter type).
        3. Implement the function body.
        4. The parameter types and return type are fixed as shown above.
        5. Do not redefine any of the provided types.
        """).strip()

        # Use structured output - the LLM returns JSON with function_name and body
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

        # Build and return the function using imports instead of source injection
        return self._build_function(response, ret_type, referenced_types)
