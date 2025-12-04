import ast
import collections.abc
import dataclasses
import inspect
import linecache
import os
import re
import tempfile
import textwrap
import typing
from typing import get_args, get_origin, get_type_hints

from mypy import api as mypy_api

from effectful.handlers.llm import Template
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class SynthesisError(Exception):
    """Raised when program synthesis fails."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def collect_type_sources(t: type, seen: set[type] | None = None) -> dict[type, str]:
    """Recursively collect source code for a type and all its dependencies.

    This function walks through a type annotation (including generic types like
    Callable[[Person], Order]) and extracts source code for all user-defined
    types it references.

    Args:
        t: The type to analyze
        seen: Set of already-processed types (to avoid infinite recursion)

    Returns:
        A dict mapping types to their source code strings
    """
    if seen is None:
        seen = set()

    sources: dict[type, str] = {}

    # Handle generic types (e.g., Callable[[X], Y], list[X], Optional[X])
    origin = get_origin(t)
    if origin is not None:
        # process type arguments recursively
        for arg in get_args(t):
            if isinstance(arg, type):
                sources.update(collect_type_sources(arg, seen))
            elif isinstance(arg, list):
                # handle Callable[[P1, P2], R] where args is a list
                for inner_arg in arg:
                    if isinstance(inner_arg, type):
                        sources.update(collect_type_sources(inner_arg, seen))
        return sources

    # Skip non-types, already-seen types, and builtins
    if not isinstance(t, type) or t in seen:
        return sources
    if t.__module__ == "builtins":
        return sources

    seen.add(t)

    # Try to get source code for this type, since there might be exceptions...
    try:
        source = inspect.getsource(t)
        sources[t] = source
    except (OSError, TypeError):
        # Can't get source (built-in, C extension, dynamically created, etc.)
        return sources

    # recursive descenting
    try:
        hints = get_type_hints(t)
        for hint in hints.values():
            sources.update(collect_type_sources(hint, seen))
    except Exception:
        pass

    # For dataclasses, also check field types explicitly
    if dataclasses.is_dataclass(t):
        for field in dataclasses.fields(t):
            field_type = field.type
            if isinstance(field_type, type):
                sources.update(collect_type_sources(field_type, seen))
            elif isinstance(field_type, str):
                # Forward reference as string - skip for now
                pass
            else:
                # Could be a generic type
                sources.update(collect_type_sources(field_type, seen))

    # check base classes (excluding object)
    for base in t.__bases__:
        if base is not object:
            sources.update(collect_type_sources(base, seen))

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


def run_mypy_check(
    code: str,
    type_sources: dict[type, str],
    expected_type: type,
) -> tuple[bool, str]:
    """Run mypy on generated code to verify type correctness.

    Args:
        code: The generated function code
        type_sources: Dict mapping types to their source code
        expected_type: The expected Callable type for the function

    Returns:
        A tuple of (success: bool, error_message: str)
    """

    # Build the full source file with imports, type definitions, and generated code
    source_parts = []

    # Add common imports
    source_parts.append("from __future__ import annotations")
    source_parts.append("from typing import *")
    source_parts.append("from collections.abc import Callable")
    source_parts.append("import dataclasses")
    source_parts.append("from dataclasses import dataclass")
    source_parts.append("")

    # Add type definitions
    for type_source in type_sources.values():
        cleaned = textwrap.dedent(type_source).strip()
        source_parts.append(cleaned)
        source_parts.append("")

    # Add the generated code
    source_parts.append(textwrap.dedent(code).strip())
    source_parts.append("")

    # Add a type assertion to verify the function matches the expected type
    # Extract function name from code
    try:
        module_ast = ast.parse(code)
        last_decl = module_ast.body[-1]
        if isinstance(last_decl, ast.FunctionDef):
            func_name = last_decl.name
            # Format the expected type for annotation (using imported names)
            type_annotation = _format_type_for_annotation(expected_type)
            source_parts.append(f"_check: {type_annotation} = {func_name}")
    except Exception:
        pass

    full_source = "\n".join(source_parts)

    # Write to temp file and run mypy
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_source)
        temp_path = f.name

    try:
        # Run mypy with strict settings
        result = mypy_api.run(
            [
                temp_path,
                "--no-error-summary",
                "--no-pretty",
                "--hide-error-context",
                "--no-color-output",
            ]
        )
        stdout, stderr, exit_code = result

        if exit_code != 0:
            # Filter out the temp file path from error messages for cleaner output
            error_msg = stdout.replace(temp_path, "<generated>")
            return False, error_msg.strip()

        return True, ""
    finally:
        os.unlink(temp_path)


class ProgramSynthesis(ObjectInterpretation):
    """Provides a `template` handler to instruct the LLM to generate code of the
    right form and with the right type.

    """

    def __init__(self, type_check: bool = False):
        """Initialize the program synthesis handler.

        Args:
            type_check: Whether to verify the function signature matches the expected type.
        """
        self.type_check = type_check

    def _parse_and_eval[T](
        self, t: type[T], content: str, type_sources: dict[type, str]
    ) -> T:
        pattern = r"<code>(.*?)</code>"
        code_content = re.search(pattern, content, re.DOTALL)
        if code_content is None:
            raise SynthesisError("<code> tags not found", content)
        code = code_content.group(1)

        try:
            module_ast = ast.parse(code)
        except SyntaxError as exc:
            raise SynthesisError("failed to parse", content) from exc

        if not isinstance(module_ast, ast.Module):
            raise SynthesisError("not a module", content)

        last_decl = module_ast.body[-1]
        if not isinstance(last_decl, ast.FunctionDef):
            raise SynthesisError("last definition not a function", content)

        source_code = textwrap.dedent(code)
        lines = code.splitlines(keepends=True)
        filename = f"<generated-{hash(code)}>"

        # register into linecache
        linecache.cache[filename] = (len(source_code), None, lines, filename)

        # Build globals dict with type definitions available for exec
        gs: dict = {}
        for typ in type_sources:
            gs[typ.__name__] = typ

        if self.type_check:
            success, error_msg = run_mypy_check(code, type_sources, t)
            if not success:
                raise SynthesisError(f"Type check failed:\n{error_msg}", content)

        try:
            code_obj = compile(source_code, filename, "exec")
            exec(code_obj, gs)
        except Exception as exc:
            raise SynthesisError("evaluation failed", content) from exc

        return gs[last_decl.name]

    @implements(Template.__call__)
    def _call(self, template, *args, **kwargs) -> None:
        ret_type = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type)
        ret_type_origin = ret_type if origin is None else origin

        if not (issubclass(ret_type_origin, collections.abc.Callable)):  # type: ignore[arg-type]
            return fwd()

        # Collect source code for all types referenced in the signature
        type_sources = collect_type_sources(ret_type)
        type_context = format_type_context(type_sources)

        # Build the type definitions section if there are custom types
        type_defs_section = ""
        if type_context:
            type_defs_section = f"""
        <type_definitions>
{textwrap.indent(type_context, "        ")}
        </type_definitions>
"""

        prompt_ext = textwrap.dedent(f"""
        Generate a Python function satisfying the following specification and type signature.
        
        <specification>{template.__prompt_template__}</specification>
        <signature>{str(ret_type)}</signature>
{type_defs_section}
        <instructions>
        1. Produce one block of Python code.
        2. Do not include usage examples.
        3. Return your response in <code> tags.
        4. Do not return your response in markdown blocks.
        5. Your output function def must be the final statement in the code block.
        6. Do not redefine any types from <type_definitions> - they are already available.
        </instructions>
        """).strip()

        response = fwd(
            dataclasses.replace(
                template,
                __prompt_template__=prompt_ext,
                __signature__=template.__signature__.replace(return_annotation=str),
            ),
            *args,
            **kwargs,
        )

        # Pass full ret_type (with type args) for proper type checking
        functional = self._parse_and_eval(ret_type, response, type_sources)

        return functional
