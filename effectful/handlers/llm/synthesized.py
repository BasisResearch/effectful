import ast
import collections
import collections.abc
import ctypes
import inspect
import linecache
import sys
import textwrap
import types
import typing
from collections import ChainMap
from collections.abc import Callable
from typing import Any

import pydantic
from litellm import OpenAIMessageContentListBlock
from pydantic import Field

from effectful.handlers.llm.encoding import EncodableAs, type_to_encodable_type
from effectful.ops.syntax import defop


@defop
def get_synthesis_context() -> ChainMap[str, Any] | None:
    """Get the current synthesis context for decoding synthesized code."""
    return None


class SynthesisError(Exception):
    """Raised when program synthesis fails."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def _safe_getsource(obj: Any, fallback: Callable[[], str]) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return fallback()


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

        def fallback() -> str:
            try:
                sig = inspect.signature(vl)
                return f"def {func_name}{sig}:\n    pass  # Source unavailable"
            except (ValueError, TypeError):
                return f"def {func_name}(...):\n    pass  # Source unavailable"

        source = _safe_getsource(vl, fallback)

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


class SynthesizedType(pydantic.BaseModel):
    """Structured output for type/class synthesis.

    Pydantic model representing synthesized class code with type name and module code.
    """

    type_name: str = Field(
        ...,
        description="The name of the class that satisfies the specification",
    )
    parent_class: str = Field(
        ...,
        description="The name of the parent class that this class inherits from",
    )
    module_code: str = Field(
        ...,
        description="Complete Python module code with ONLY the subclass definition (do NOT redefine the parent class)",
    )


class _PyMappingProxyObject(ctypes.Structure):
    """Internal ctypes structure to access the underlying dict of a mappingproxy."""

    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.py_object),
        ("mapping", ctypes.py_object),
    ]


@type_to_encodable_type.register(type)
class EncodableSynthesizedType(
    EncodableAs[type, SynthesizedType],
):
    """Encodes type to SynthesizedType and vice versa."""

    t = SynthesizedType

    @classmethod
    def encode(
        cls, vl: type, context: ChainMap[str, Any] | None = None
    ) -> SynthesizedType:
        """Encode a type to a SynthesizedType.

        Extracts the type name, parent class, and source code.
        """
        type_name = vl.__name__
        # Get parent class (first non-object base)
        bases = [b for b in vl.__bases__ if b is not object]
        parent_class = bases[0].__name__ if bases else "object"

        def fallback() -> str:
            return f"class {type_name}({parent_class}): pass  # Source unavailable"

        source = _safe_getsource(vl, fallback)

        return SynthesizedType(
            type_name=type_name,
            parent_class=parent_class,
            module_code=textwrap.dedent(source).strip(),
        )

    # Counter for unique filenames
    _decode_counter: typing.ClassVar[int] = 0

    @classmethod
    def decode(
        cls, vl: SynthesizedType, context: ChainMap[str, Any] | None = None
    ) -> type:
        """Decode a SynthesizedType to a type.

        Executes the module code and returns the named class.
        Uses get_synthesis_context() operation for lexical context.
        """
        # Use synthesis context operation if no explicit context provided
        if context is None:
            context = get_synthesis_context()
        type_name = vl.type_name
        module_code = textwrap.dedent(vl.module_code).strip() + "\n"

        # Create a unique filename and register source with linecache
        # This allows inspect.getsource() to work on the generated class
        cls._decode_counter += 1
        # NOTE: adding source to class is more tricky
        # because for function	func.__code__.co_filename (set by compile(..., filename, "exec")) is set automatically
        # We have to do this manually for class (set module name) for inspect.getsource() to work
        module_name = (
            f"_llm_effectful_synthesized_types.{type_name}.{cls._decode_counter}"
        )
        filename = f"<synthesized_type:{module_name}>"

        # Register source for inspect/linecache
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

        # Create a real module and put it to sys.modules
        mod = types.ModuleType(module_name)
        mod.__file__ = filename
        sys.modules[module_name] = mod

        # globals = module.__dict__ + context
        g = mod.__dict__
        g.update({"collections": collections})
        if context:
            g.update(context)
        g.update({"__name__": module_name, "__file__": filename})
        g.setdefault("__package__", module_name.rpartition(".")[0])

        # Get the expected parent class name from the synthesized output
        expected_parent = vl.parent_class

        try:
            # Parse and modify AST
            tree = ast.parse(module_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # 1. Rewrite inheritance for the target class to use parent from context
                    if node.name == type_name:
                        parent_node = ast.Name(
                            id=expected_parent,
                            ctx=ast.Load(),
                            lineno=node.lineno,
                            col_offset=node.col_offset,
                            end_lineno=node.lineno,
                            end_col_offset=node.col_offset + len(expected_parent),
                        )
                        node.bases = [parent_node]

                    # 2. Inject __firstlineno__ for Python 3.13+ compatibility
                    first_body_line = (
                        node.body[0].lineno if node.body else node.lineno + 1
                    )
                    col = node.col_offset + 4

                    name_node = ast.Name(
                        id="__firstlineno__",
                        ctx=ast.Store(),
                        lineno=first_body_line,
                        col_offset=col,
                        end_lineno=first_body_line,
                        end_col_offset=col + 14,
                    )
                    const_node = ast.Constant(
                        value=node.lineno,
                        lineno=first_body_line,
                        col_offset=col + 17,
                        end_lineno=first_body_line,
                        end_col_offset=col + 18,
                    )
                    assign = ast.Assign(
                        targets=[name_node],
                        value=const_node,
                        lineno=first_body_line,
                        col_offset=col,
                        end_lineno=first_body_line,
                        end_col_offset=col + 20,
                    )
                    node.body.insert(0, assign)

            code_obj = compile(tree, filename, "exec")
            exec(code_obj, g, g)
        except SyntaxError as exc:
            raise SynthesisError(
                f"Syntax error in generated code: {exc}", module_code
            ) from exc
        except Exception as exc:
            raise SynthesisError(f"Evaluation failed: {exc!r}", module_code) from exc

        if type_name not in g:
            raise SynthesisError(
                f"Type '{type_name}' not found after execution. "
                f"Available names: {[k for k in g.keys() if not k.startswith('_')]}",
                module_code,
            )

        synthesized_type = g[type_name]

        if not isinstance(synthesized_type, type):
            raise SynthesisError(
                f"'{type_name}' is not a type, got {type(synthesized_type).__name__}",
                module_code,
            )

        # Attach source code and module name
        synthesized_type.__source__ = module_code  # type: ignore[attr-defined]
        synthesized_type.__synthesized__ = vl  # type: ignore[attr-defined]
        synthesized_type.__module__ = module_name

        # NOTE: Set __firstlineno__ AFTER __module__ assignment!
        # In Python 3.13, setting __module__ clears __firstlineno__ from vars().
        # We use ctypes to directly inject it into __dict__ for inspect.getsource().
        if "__firstlineno__" not in vars(synthesized_type):
            firstlineno = next(
                (
                    n.lineno
                    for n in ast.walk(ast.parse(module_code))
                    if isinstance(n, ast.ClassDef) and n.name == type_name
                ),
                1,
            )
            inner_dict = _PyMappingProxyObject.from_address(
                id(vars(synthesized_type))
            ).mapping
            inner_dict["__firstlineno__"] = firstlineno

        return synthesized_type

    @classmethod
    def serialize(cls, vl: SynthesizedType) -> list[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": vl.model_dump_json()}]
