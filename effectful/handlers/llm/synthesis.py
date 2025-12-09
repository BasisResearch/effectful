import collections
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


class SynthesizedModule(pydantic.BaseModel):
    """Structured output for function synthesis.

    The LLM provides a complete Python module and the name of the main function.
    We extract the function, re-format it with prescribed types, and verify with mypy.
    """

    function_name: str = Field(
        ...,
        description="The name of the main function that satisfies the specification",
    )
    module_code: str = Field(
        ...,
        description="Complete Python module code including the function and any helpers",
    )


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

    # Handle generic types (e.g., Callable[[X], Y], list[X], dict[str, Item])
    origin = get_origin(t)
    if origin is not None:
        for arg in get_args(t):
            if isinstance(arg, list):
                # Handle Callable[[P1, P2], R] where first arg is a list of param types
                for inner_arg in arg:
                    types.update(collect_referenced_types(inner_arg, seen))
            elif arg is not ...:
                # Recursively process all type arguments (including generic aliases)
                types.update(collect_referenced_types(arg, seen))
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


def _types_match(expected: type, actual: type) -> bool:
    """Check if two types match, handling cross-module generic types.

    This is needed because `list[__main__.Product]` != `list[Product]` even
    when they refer to the same class, due to how generic aliases are compared.

    Also handles cases where LLM uses bare type (list) instead of generic (list[X]).

    Args:
        expected: The expected type (from caller's context)
        actual: The actual type (from exec'd code)

    Returns:
        True if the types are structurally equivalent
    """
    # Direct equality check (covers most cases)
    if expected == actual:
        return True

    # Same class by identity
    if expected is actual:
        return True

    # For generic types, compare origin and args recursively
    expected_origin = get_origin(expected)
    actual_origin = get_origin(actual)

    # Handle case where one is generic and one is bare type
    # e.g., list[Product] vs list - accept if origins match
    if expected_origin is not None and actual_origin is None:
        # expected is generic (list[X]), actual is bare (list)
        # Accept if actual is the origin of expected
        if actual is expected_origin:
            return True
        # Also check by name for cross-module cases
        if isinstance(actual, type) and hasattr(expected_origin, "__name__"):
            if actual.__name__ == expected_origin.__name__:
                return True

    if actual_origin is not None and expected_origin is None:
        # actual is generic (list[X]), expected is bare (list)
        if expected is actual_origin:
            return True
        if isinstance(expected, type) and hasattr(actual_origin, "__name__"):
            if expected.__name__ == actual_origin.__name__:
                return True

    if expected_origin is not None and actual_origin is not None:
        # Both are generic - origins must match
        if expected_origin is not actual_origin:
            # Check by name for cross-module cases
            if not (
                hasattr(expected_origin, "__name__")
                and hasattr(actual_origin, "__name__")
                and expected_origin.__name__ == actual_origin.__name__
            ):
                return False

        # Compare type arguments recursively
        expected_args = get_args(expected)
        actual_args = get_args(actual)

        if len(expected_args) != len(actual_args):
            return False

        return all(_types_match(e, a) for e, a in zip(expected_args, actual_args))

    # For non-generic types, compare by name (handles cross-module cases)
    if isinstance(expected, type) and isinstance(actual, type):
        return expected.__name__ == actual.__name__

    return False


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
        result: SynthesizedModule,
        callable_type: type,
        referenced_types: set[type],
        lexical_context: dict[str, tuple[str, typing.Any]] | None = None,
    ) -> typing.Callable:
        """Build and execute a function from the synthesized module.

        Extracts the function from LLM's module, re-formats with prescribed signature,
        and optionally verifies with mypy.

        Args:
            result: The structured output from the LLM containing module code
            callable_type: The expected Callable type (e.g., Callable[[str], int])
            referenced_types: Set of types referenced in the signature
            lexical_context: Dict of lexical context (functions/types) available in scope

        Returns:
            The synthesized callable function
        """
        import ast

        func_name = result.function_name
        original_module_code = textwrap.dedent(result.module_code).strip()

        # Parse the module to extract function body and helpers
        try:
            tree = ast.parse(original_module_code)
        except SyntaxError as exc:
            raise SynthesisError(
                f"Syntax error in generated code: {exc}", original_module_code
            ) from exc

        # Find the target function and separate helpers
        target_func_node = None
        helper_nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                target_func_node = node
            else:
                helper_nodes.append(node)

        if target_func_node is None:
            raise SynthesisError(
                f"Function '{func_name}' not found in generated module.",
                original_module_code,
            )

        # Extract function body using AST - the body starts at first statement's line
        all_lines = original_module_code.splitlines()

        # The body starts at the first statement in the function
        if target_func_node.body:
            body_start_line = target_func_node.body[0].lineno - 1
            body_end_line = target_func_node.end_lineno
            body_lines = all_lines[body_start_line:body_end_line]
            body = "\n".join(body_lines)
        else:
            # Empty function body (just pass or docstring)
            body = "    pass"

        # Get parameter names from the original function
        param_names = [arg.arg for arg in target_func_node.args.args]

        # Build helper code from non-function-def nodes
        helper_code = ""
        if helper_nodes:
            helper_parts = []
            for node in helper_nodes:
                start = node.lineno - 1
                end = node.end_lineno
                helper_parts.append("\n".join(all_lines[start:end]))
            helper_code = "\n\n".join(helper_parts) + "\n\n"

        # Construct the function with PRESCRIBED types and extracted param names
        param_sig = _format_param_signature(callable_type, param_names)
        return_type = _format_return_type(callable_type)
        func_code = f"def {func_name}({param_sig}) -> {return_type}:\n{body}"
        module_code = helper_code + func_code

        # Register in linecache for better tracebacks
        lines = module_code.splitlines(keepends=True)
        filename = f"<generated-{hash(module_code)}>"
        linecache.cache[filename] = (len(module_code), None, lines, filename)

        # Optional mypy type checking - now with guaranteed correct signature
        if self.type_check:
            success, error_msg = run_mypy_check(module_code, referenced_types)
            if not success:
                raise SynthesisError(f"Type check failed:\n{error_msg}", module_code)

        # Build globals dict by importing types from their original modules
        gs: dict = {}
        for typ in referenced_types:
            module = inspect.getmodule(typ)
            if module is not None:
                gs[typ.__name__] = typ

        # Add lexical context (functions and types) from the template's captured context
        if lexical_context:
            for name, (_, obj) in lexical_context.items():
                gs[name] = obj

        try:
            code_obj = compile(module_code, filename, "exec")
            exec(code_obj, gs)
        except Exception as exc:
            raise SynthesisError(
                f"evaluation failed: {exc!r}, source code: {module_code}", module_code
            ) from exc

        if func_name not in gs:
            raise SynthesisError(
                f"Function '{func_name}' not found after execution. "
                f"Available names: {[k for k in gs.keys() if not k.startswith('_')]}",
                module_code,
            )

        return gs[func_name]

    def _verify_signature(
        self, func: typing.Callable, callable_type: type, code: str
    ) -> None:
        """Verify that the function signature matches the expected Callable type.

        Args:
            func: The synthesized function
            callable_type: The expected Callable type
            code: The source code (for error messages)

        Raises:
            SynthesisError: If the signature doesn't match
        """
        expected_param_types = _get_param_types(callable_type)

        sig = inspect.signature(func)
        actual_params = list(sig.parameters.values())

        # Check parameter count (skip if Callable[..., R])
        if expected_param_types is not None:
            if len(actual_params) != len(expected_param_types):
                raise SynthesisError(
                    f"Parameter count mismatch: expected {len(expected_param_types)}, "
                    f"got {len(actual_params)}",
                    code,
                )

        # Get type hints for the function
        hints = typing.get_type_hints(func)

        # Check parameter types (skip if Callable[..., R])
        if expected_param_types is not None:
            for i, (param, expected_type) in enumerate(
                zip(actual_params, expected_param_types)
            ):
                actual_type = hints.get(param.name)
                if actual_type is None:
                    raise SynthesisError(
                        f"Parameter '{param.name}' missing type annotation", code
                    )
                # Use structural comparison to handle cross-module generic types
                if not _types_match(expected_type, actual_type):
                    raise SynthesisError(
                        f"Parameter '{param.name}' type mismatch: "
                        f"expected {expected_type}, got {actual_type}",
                        code,
                    )

        # Check return type
        expected_return_type = (
            get_args(callable_type)[-1] if get_args(callable_type) else None
        )
        if expected_return_type is not None:
            actual_return_type = hints.get("return")
            if actual_return_type is None:
                raise SynthesisError(
                    f"Function missing return type annotation. "
                    f"Expected: {_format_type_for_annotation(expected_return_type)}",
                    code,
                )
            # Use structural comparison to handle cross-module generic types
            if not _types_match(expected_return_type, actual_return_type):
                raise SynthesisError(
                    f"Return type mismatch: expected {_format_type_for_annotation(expected_return_type)}, "
                    f"got {_format_type_for_annotation(actual_return_type)}",
                    code,
                )

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

        # Get lexical context (functions and types) from the template's captured context
        lexical_context = getattr(template, "lexical_context", {})
        lexical_context_source = (
            template.get_lexical_context_source() if lexical_context else ""
        )

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

        # Build the lexical context section (helper functions and types)
        lexical_context_section = ""
        if lexical_context_source:
            escaped_lexical_context = lexical_context_source.replace("{", "{{").replace(
                "}", "}}"
            )
            lexical_context_section = f"""
The following helper functions and types are available for you to use:

```python
{escaped_lexical_context}
```
"""

        prompt_ext = textwrap.dedent(f"""
        Implement a Python module containing a function with the following specification.

        **Specification:** {template.__prompt_template__}

        **Required signature for the main function:**
        - Parameter types (in order): {param_types_str}
        - Return type: {return_type_str}
        {type_defs_section}{lexical_context_section}
        **Instructions:**
        1. Write a complete Python module with the main function and any helper functions/classes you need.
        2. Choose a descriptive name for the main function.
        3. Choose descriptive parameter names (one for each parameter type).
        4. The main function's parameter types and return type must match exactly as specified above.
        5. Do not redefine any of the provided types - they are already imported.
        6. Do not include import statements - all necessary types and helpers are pre-imported.
        7. You may define additional helper functions, classes, or constants in the module.
        8. You may use any of the helper functions and types provided above.
        """).strip()

        # Use structured output - the LLM returns JSON with function_name and module_code
        response: SynthesizedModule = fwd(
            dataclasses.replace(
                template,
                __prompt_template__=prompt_ext,
                __signature__=template.__signature__.replace(
                    return_annotation=SynthesizedModule
                ),
            ),
            *args,
            **kwargs,
        )

        # Build and return the function using imports instead of source injection
        return self._build_function(
            response, ret_type, referenced_types, lexical_context
        )
