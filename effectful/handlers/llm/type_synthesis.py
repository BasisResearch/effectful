"""Type/class synthesis for LLM-generated code."""

import ast
import collections
import collections.abc
import ctypes
import dataclasses
import inspect
import linecache
import sys
import textwrap
import types
import typing


class _PyMappingProxyObject(ctypes.Structure):
    """Internal ctypes structure to access the underlying dict of a mappingproxy."""

    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.py_object),
        ("mapping", ctypes.py_object),
    ]


import pydantic
from litellm import OpenAIMessageContentListBlock
from pydantic import Field

from effectful.handlers.llm import LexicalContext, Template
from effectful.handlers.llm.encoding import EncodableAs, type_to_encodable_type
from effectful.handlers.llm.synthesis import (
    EncodableLexicalContext,
    SynthesisError,
    run_mypy_check,
)
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


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
    def encode(cls, vl: type, context: LexicalContext | None = None) -> SynthesizedType:
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
    def decode(cls, vl: SynthesizedType, context: LexicalContext | None = None) -> type:
        """Decode a SynthesizedType to a type.

        Executes the module code and returns the named class.
        The module code becomes the class's definition context,
        optionally augmented with provided context.
        """
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


class TypeSynthesis(ObjectInterpretation):
    """Provides a `template` handler to instruct the LLM to generate a class/type
    that inherits from a specified base type.
    """

    def __init__(self, type_check: bool = False):
        """Initialize the type synthesis handler.

        Args:
            type_check: Whether to run mypy to verify the generated code.
        """
        self.type_check = type_check

    def _build_type(
        self,
        result: SynthesizedType,
        base_type: type,
        lexical_context: LexicalContext,
    ) -> type:
        """Build and execute a type from the synthesized module.

        Uses EncodableSynthesizedType.decode with the template's lexical context.
        Optionally runs mypy type checking if enabled.
        Validates that the synthesized type is a subclass of base_type.

        Args:
            result: The structured output from the LLM
            base_type: The expected base type (e.g., Animal for type[Animal])
            lexical_context: Dict of lexical context (functions/types) available in scope

        Returns:
            The synthesized type (class)
        """
        if self.type_check:
            success, error_msg = run_mypy_check(result.module_code, lexical_context)
            if not success:
                raise SynthesisError(
                    f"Type check failed:\n{error_msg}", result.module_code
                )

        synthesized_type = EncodableSynthesizedType.decode(
            result, context=lexical_context
        )

        # Validate that synthesized type inherits from base_type
        if not issubclass(synthesized_type, base_type):
            raise SynthesisError(
                f"Synthesized type '{synthesized_type.__name__}' does not inherit from '{base_type.__name__}'",
                result.module_code,
            )

        return synthesized_type

    @implements(Template.__call__)
    def _call(self, template, *args, **kwargs) -> type:
        ret_type = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type)
        ret_type_origin = ret_type if origin is None else origin

        # Check if return type is type[BaseClass]
        if ret_type_origin is not type:
            return fwd()

        # Extract the base type from type[BaseClass]
        type_args = typing.get_args(ret_type)
        if not type_args:
            raise SynthesisError(
                "Type synthesis requires a base type, e.g., type[Animal]. "
                "Got bare 'type' without a type parameter.",
                None,
            )

        base_type = type_args[0]

        # Verify base type is in lexical context
        base_type_name = base_type.__name__
        if base_type_name not in template.__context__:
            raise SynthesisError(
                f"Base type '{base_type_name}' must be in the template's lexical context.",
                None,
            )

        # Include the full lexical context
        context_source = EncodableLexicalContext.encode(template.__context__)
        escaped_context = context_source.replace("{", "{{").replace("}", "}}")
        context_section = f"""
The following types, functions, and values are available:

```python
{escaped_context}
```
"""

        prompt_ext = textwrap.dedent(f"""
        Implement a Python class with the following specification.

        **Specification:** {template.__prompt_template__}

        **Required:** The class must inherit from `{base_type_name}` and implement its interface.
        {context_section}
        **Instructions:**
        1. Write a complete Python module with the class definition.
        2. Choose a descriptive class name.
        3. The class MUST inherit from `{base_type_name}`.
        4. Implement all required methods from the base class.
        5. You may include helper functions/classes/constants.
        6. Do not redefine provided types - they are already available.
        7. Do not include import statements.
        """).strip()

        response: SynthesizedType = fwd(
            dataclasses.replace(
                template,
                __prompt_template__=prompt_ext,
                __signature__=template.__signature__.replace(
                    return_annotation=SynthesizedType
                ),
            ),
            *args,
            **kwargs,
        )

        # Build the type using lexical context for exec globals
        synthesized_type = self._build_type(response, base_type, template.__context__)

        return synthesized_type
