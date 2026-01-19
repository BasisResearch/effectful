import collections
import collections.abc
import inspect
import textwrap
import types
import typing
from collections import ChainMap
from collections.abc import Callable
from typing import Any

# Type for symbols that can be included in synthesis context
Symbol = type | Callable[..., Any] | types.ModuleType


def _extract_types_recursive(obj: Any) -> set[type]:
    """Recursively extract all types from a type annotation or object.

    Handles Callable, List, Tuple, Dict, Union, Optional, and other generic types.
    """
    result: set[type] = set()

    # If it's a class/type, add it
    if isinstance(obj, type):
        # Skip builtins
        if obj.__module__ != "builtins":
            result.add(obj)
        return result

    # Get origin and args for generic types
    origin = typing.get_origin(obj)
    args = typing.get_args(obj)

    if origin is not None:
        # Add the origin if it's a real type (not typing constructs)
        if isinstance(origin, type) and origin.__module__ != "builtins":
            result.add(origin)
        # Recursively process type arguments
        for arg in args:
            result.update(_extract_types_recursive(arg))

    return result


def _build_symbols_context(
    symbols: list[Symbol] | tuple[Symbol, ...],
) -> ChainMap[str, Any]:
    """Build a context ChainMap from symbols, including recursively extracted types."""
    context: dict[str, Any] = {}

    for obj in symbols:
        # Add the symbol itself
        name = getattr(obj, "__name__", None)
        if name is not None:
            context[name] = obj

        # For callables, extract types from signature
        if callable(obj) and not isinstance(obj, type):
            try:
                sig = inspect.signature(obj)
                # Extract from return annotation
                if sig.return_annotation is not inspect.Parameter.empty:
                    for t in _extract_types_recursive(sig.return_annotation):
                        context[t.__name__] = t
                # Extract from parameter annotations
                for param in sig.parameters.values():
                    if param.annotation is not inspect.Parameter.empty:
                        for t in _extract_types_recursive(param.annotation):
                            context[t.__name__] = t
            except (ValueError, TypeError):
                pass

        # For types, extract from base classes and type hints
        if isinstance(obj, type):
            # Add base classes (except object and builtins)
            for base in obj.__mro__[1:]:
                if base is not object and base.__module__ != "builtins":
                    context[base.__name__] = base
            # Extract from class annotations
            for annotation in getattr(obj, "__annotations__", {}).values():
                for t in _extract_types_recursive(annotation):
                    context[t.__name__] = t

    return ChainMap(context)


from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import InstructionHandler
from effectful.handlers.llm.encoding import EncodableAs, type_to_encodable_type
from effectful.handlers.llm.synthesized import (
    EncodableSynthesizedFunction,
    SynthesisError,
    SynthesizedFunction,
    get_synthesis_context,
)
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements

__all__ = [
    "BaseSynthesis",
    "EncodableSynthesizedFunction",
    "ProgramSynthesis",
    "SynthesisContextHandler",
    "SynthesisError",
    "SynthesizedFunction",
    "get_synthesis_context",
]


class SynthesisContextHandler(ObjectInterpretation):
    """Handler that provides the synthesis context to decode operations.

    Args:
        context: The full lexical context from the template. Used as fallback
                 if symbols is None.
        symbols: Optional collection of objects (types, functions, modules) to include.
                 If provided, only these objects will be available in the synthesis context.
                 If None, all symbols from context are available.
    """

    def __init__(
        self,
        context: ChainMap[str, Any] | None = None,
        symbols: list[Symbol] | tuple[Symbol, ...] | None = None,
    ):
        if symbols is not None:
            # Build context from provided symbols, recursively extracting dependencies
            self.context = _build_symbols_context(symbols)
        elif context is not None:
            self.context = context
        else:
            self.context = ChainMap({})

    @implements(get_synthesis_context)
    def _get_context(self) -> ChainMap[str, Any] | None:
        return self.context


def _is_callable_return_type(template: Template) -> bool:
    """Check if template has a Callable return type."""
    ret_type = template.__signature__.return_annotation
    origin = typing.get_origin(ret_type)
    ret_type_origin = ret_type if origin is None else origin
    return ret_type_origin is collections.abc.Callable


def _get_context_source(context: ChainMap[str, Any]) -> str:
    """Extract source code for types and callables in the context.

    Uses encodable types (EncodableSynthesizedFunction/Type) when available
    to provide rich source for custom symbols.
    """
    sources: list[str] = []
    seen_names: set[str] = set()

    def _encode_with_context(encodable: EncodableAs[Any, Any], obj: Any) -> Any:
        try:
            return encodable.encode(obj, context=context)
        except TypeError:
            return encodable.encode(obj)

    for name, obj in context.items():
        if name.startswith("_"):
            continue
        if name in seen_names:
            continue
        seen_names.add(name)

        module = getattr(obj, "__module__", None)
        if module and (module == "builtins" or module.startswith("collections")):
            continue

        try:
            if isinstance(obj, type):
                encodable = type_to_encodable_type(type)
                synthesized = _encode_with_context(encodable, obj)
                if hasattr(synthesized, "module_code"):
                    sources.append(textwrap.dedent(synthesized.module_code).strip())
                    continue
                source = inspect.getsource(obj)
                sources.append(textwrap.dedent(source).strip())
            elif callable(obj) and not isinstance(obj, type):
                # Skip non-function callables (e.g., pytest MarkDecorator)
                if not hasattr(obj, "__name__") or not inspect.isroutine(obj):
                    continue
                encodable = type_to_encodable_type(collections.abc.Callable)
                synthesized = _encode_with_context(encodable, obj)
                if hasattr(synthesized, "module_code"):
                    sources.append(textwrap.dedent(synthesized.module_code).strip())
                    continue
                source = inspect.getsource(obj)
                sources.append(textwrap.dedent(source).strip())
        except (OSError, TypeError, AttributeError):
            if isinstance(obj, type):
                sources.append(f"# {name}: class (source unavailable)")
            elif callable(obj):
                try:
                    sig = inspect.signature(obj)
                    sources.append(f"# {name}{sig} (source unavailable)")
                except (ValueError, TypeError):
                    sources.append(f"# {name}: callable (source unavailable)")

    return (
        "\n\n".join(sources) if sources else "# No custom types or functions available"
    )


class BaseSynthesis(ObjectInterpretation):
    """Base class for synthesis handlers.

    Provides common functionality for ProgramSynthesis and TypeSynthesis.

    Args:
        symbols: Optional list/tuple of objects (types, functions, modules) to include
                 in the synthesis context. If provided, only these objects will be
                 available to the synthesized code. If None, all symbols from the
                 template's context are available.
    """

    def __init__(self, symbols: list[Symbol] | tuple[Symbol, ...] | None = None):
        self.symbols = symbols

    def _get_filtered_context(self, template: Template) -> ChainMap[str, Any]:
        """Get the context, optionally filtered by symbols."""
        if self.symbols is not None:
            return _build_symbols_context(self.symbols)
        return template.__context__

    def _should_handle(self, template: Template) -> bool:
        """Return True if this handler should process the template."""
        raise NotImplementedError

    def _build_synthesis_instruction(self, template: Template) -> str:
        """Build the synthesis instruction for the LLM."""
        raise NotImplementedError

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Intercept template calls and inject synthesis instructions."""
        if not self._should_handle(template):
            return fwd()

        instruction = self._build_synthesis_instruction(template)

        with handler(
            SynthesisContextHandler(context=template.__context__, symbols=self.symbols)
        ):
            with handler(InstructionHandler(instruction)):
                return fwd()


class ProgramSynthesis(BaseSynthesis):
    """A program synthesis handler for Callable return types."""

    def _should_handle(self, template: Template) -> bool:
        return _is_callable_return_type(template)

    def _build_synthesis_instruction(self, template: Template) -> str:
        """Build the synthesis instruction for a Callable return type."""
        ret_type = template.__signature__.return_annotation
        context = self._get_filtered_context(template)

        context_str = _get_context_source(context).replace("{", "{{").replace("}", "}}")
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
