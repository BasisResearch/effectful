"""Encodable type for LLM-synthesized classes."""

import ast
import collections
import ctypes
import inspect
import linecache
import sys
import textwrap
import types
import typing
from collections import ChainMap
from typing import Any

import pydantic
from pydantic import Field

from effectful.handlers.llm.encoding import EncodableAs, type_to_encodable_type
from effectful.handlers.llm.providers import OpenAIMessageContentListBlock
from effectful.handlers.llm.synthesis import SynthesisError


class _PyMappingProxyObject(ctypes.Structure):
    """Internal ctypes structure to access the underlying dict of a mappingproxy."""

    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.py_object),
        ("mapping", ctypes.py_object),
    ]


class SynthesizedType(pydantic.BaseModel):
    """Structured output for type/class synthesis.

    Pydantic model representing synthesized class code with type name and module code.
    """

    type_name: str = Field(
        ...,
        description="The name of the class that satisfies the specification",
    )
    module_code: str = Field(
        ...,
        description="Complete Python module code with the class definition (no imports needed)",
    )


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

        Extracts the type name and source code.
        """
        type_name = vl.__name__
        try:
            source = inspect.getsource(vl)
        except (OSError, TypeError):
            # If we can't get source, create a minimal representation
            source = f"class {type_name}: pass  # Source unavailable"

        return SynthesizedType(
            type_name=type_name, module_code=textwrap.dedent(source).strip()
        )

    # Counter for unique filenames
    _decode_counter: typing.ClassVar[int] = 0

    @classmethod
    def decode(cls, vl: SynthesizedType) -> type:
        """Decode a SynthesizedType to a type.

        Executes the module code and returns the named class.
        Uses _decode_context attribute on vl if present (set by TypeSynthesis handler).
        """
        context: ChainMap[str, Any] | None = getattr(vl, "_decode_context", None)
        type_name = vl.type_name
        module_code = textwrap.dedent(vl.module_code).strip() + "\n"

        # Create a unique filename and register source with linecache
        # This allows inspect.getsource() to work on the generated class
        cls._decode_counter += 1
        # NOTE: adding source to class is more tricky
        # because for function func.__code__.co_filename (set by compile(..., filename, "exec")) is set automatically
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
        if context is not None:
            g.update(context)
        g.update({"__name__": module_name, "__file__": filename})
        g.setdefault("__package__", module_name.rpartition(".")[0])

        try:
            # NOTE: Parse and inject __firstlineno__ into class bodies for Python 3.13+ compatibility
            # inspect.getsource() looks for __firstlineno__ in vars(cls), which requires it to be in the class's __dict__.
            # We inject it via AST before execution.
            tree = ast.parse(module_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Create: __firstlineno__ = <lineno>
                    assign = ast.Assign(
                        targets=[ast.Name(id="__firstlineno__", ctx=ast.Store())],
                        value=ast.Constant(value=node.lineno),
                        lineno=node.lineno,
                        col_offset=0,
                    )
                    ast.fix_missing_locations(assign)
                    node.body.insert(0, assign)
            ast.fix_missing_locations(tree)
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
