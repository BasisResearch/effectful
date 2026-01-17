"""Type/class synthesis for LLM-generated code."""

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
from typing import Any

import pydantic
from litellm import OpenAIMessageContentListBlock
from pydantic import Field

from effectful.handlers.llm import Template
from effectful.handlers.llm.encoding import EncodableAs, type_to_encodable_type
from effectful.handlers.llm.synthesis import (
    BaseSynthesis,
    SynthesisError,
    get_synthesis_context,
)

# Type alias for lexical context
LexicalContext = ChainMap[str, Any]


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
    parent_class: str = Field(
        ...,
        description="The name of the parent class that this class inherits from",
    )
    module_code: str = Field(
        ...,
        description="Complete Python module code with ONLY the subclass definition (do NOT redefine the parent class)",
    )


@type_to_encodable_type.register(type)
class EncodableSynthesizedType(
    EncodableAs[type, SynthesizedType],
):
    """Encodes type to SynthesizedType and vice versa."""

    t = SynthesizedType

    @classmethod
    def encode(cls, vl: type, context: LexicalContext | None = None) -> SynthesizedType:
        """Encode a type to a SynthesizedType.

        Extracts the type name, parent class, and source code.
        """
        type_name = vl.__name__
        # Get parent class (first non-object base)
        bases = [b for b in vl.__bases__ if b is not object]
        parent_class = bases[0].__name__ if bases else "object"

        try:
            source = inspect.getsource(vl)
        except (OSError, TypeError):
            # If we can't get source, create a minimal representation
            source = f"class {type_name}({parent_class}): pass  # Source unavailable"

        return SynthesizedType(
            type_name=type_name,
            parent_class=parent_class,
            module_code=textwrap.dedent(source).strip(),
        )

    # Counter for unique filenames
    _decode_counter: typing.ClassVar[int] = 0

    @classmethod
    def decode(cls, vl: SynthesizedType, context: LexicalContext | None = None) -> type:
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


def _is_type_return_type(template: Template) -> tuple[bool, type | None]:
    """Check if template has a type[BaseClass] return type.

    Returns:
        Tuple of (is_type_return, base_type). base_type is None if not a type return.
    """
    ret_type = template.__signature__.return_annotation
    origin = typing.get_origin(ret_type)
    ret_type_origin = ret_type if origin is None else origin

    if ret_type_origin is not type:
        return False, None

    type_args = typing.get_args(ret_type)
    if not type_args:
        return False, None

    return True, type_args[0]


class TypeSynthesis(BaseSynthesis):
    """A type synthesis handler for type[BaseClass] return types."""

    def _should_handle(self, template: Template) -> bool:
        is_type_return, base_type = _is_type_return_type(template)
        if not is_type_return or base_type is None:
            return False

        # Verify base type is in lexical context
        base_type_name = base_type.__name__
        if base_type_name not in template.__context__:
            raise SynthesisError(
                f"Base type '{base_type_name}' must be in the template's lexical context.",
                None,
            )
        return True

    def _build_synthesis_instruction(self, template: Template) -> str:
        """Build the synthesis instruction for a type[BaseClass] return type."""
        _, base_type = _is_type_return_type(template)
        base_type_name = base_type.__name__  # type: ignore[union-attr]
        context = self._get_filtered_context(template)

        context_str = str(context).replace("{", "{{").replace("}", "}}")

        return textwrap.dedent(f"""
        Generate a Python class that inherits from `{base_type_name}`.

        Available in scope: {context_str}

        Write ONLY your subclass definition (do NOT redefine {base_type_name}).

        Respond with JSON containing:
        - "type_name": your class name
        - "parent_class": "{base_type_name}"
        - "module_code": your subclass code only
        """).strip()
