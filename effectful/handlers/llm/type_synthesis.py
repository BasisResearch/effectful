"""Type/class synthesis for LLM-generated code."""

import textwrap
import typing

from effectful.handlers.llm import Template
from effectful.handlers.llm.synthesis import BaseSynthesis, _get_context_source
from effectful.handlers.llm.synthesized import EncodableSynthesizedType, SynthesisError

__all__ = ["EncodableSynthesizedType", "SynthesisError", "TypeSynthesis"]


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

        # Get actual source code for types/functions in context
        context_source = (
            _get_context_source(context).replace("{", "{{").replace("}", "}}")
        )

        return textwrap.dedent(f"""
        Generate a Python class that inherits from `{base_type_name}`.

        The following types and functions are available in scope:

        ```python
        {context_source}
        ```

        Write ONLY your subclass definition (do NOT redefine {base_type_name}).

        Respond with JSON containing:
        - "type_name": your class name
        - "parent_class": "{base_type_name}"
        - "module_code": your subclass code only
        """).strip()
