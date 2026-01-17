import collections
import collections.abc
import inspect
import linecache
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

        context_str = str(context).replace("{", "{{").replace("}", "}}")
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
