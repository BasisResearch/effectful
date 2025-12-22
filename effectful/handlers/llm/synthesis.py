import collections.abc
import dataclasses
import inspect
import linecache
import textwrap
import typing
from collections.abc import Callable

import pydantic
from pydantic import Field

from effectful.handlers.llm import LexicalContext, Template
from effectful.handlers.llm.encoding import EncodableAs, type_to_encodable_type
from effectful.handlers.llm.providers import OpenAIMessageContentListBlock
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


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
        cls, vl: Callable, context: LexicalContext | None = None
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
    def decode(cls, vl: SynthesizedFunction, template: typing.Any = None) -> Callable:
        """Decode a SynthesizedFunction to a Callable.

        Executes the module code and returns the named function.
        The module code becomes the function's lexical context,
        optionally augmented with the template's context.
        """
        # Extract lexical context from template if provided
        context: LexicalContext | None = None
        if template is not None and hasattr(template, "__context__"):
            ctx = template.__context__
            context = ctx if isinstance(ctx, LexicalContext) else LexicalContext(ctx)
        func_name = vl.function_name
        module_code = textwrap.dedent(vl.module_code).strip()

        cls._decode_counter += 1
        filename = f"<synthesized:{func_name}:{cls._decode_counter}>"
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

        # Start with provided context or empty dict
        # Include collections module for type hints in synthesized code
        exec_globals: dict[str, typing.Any] = {"collections": collections}
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
        func.__source__ = module_code
        func.__synthesized__ = vl
        return func

    @classmethod
    def serialize(cls, vl: SynthesizedFunction) -> list[OpenAIMessageContentListBlock]:
        return [{"type": "text", "text": vl.model_dump_json()}]


class ProgramSynthesis(ObjectInterpretation):
    """Provides a `template` handler to instruct the LLM to generate code of the
    right form and with the right type.

    """

    @implements(Template.__call__)
    def _call(self, template, *args, **kwargs) -> None:
        ret_type = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type)
        ret_type = ret_type if origin is None else origin

        if not (issubclass(ret_type, collections.abc.Callable)):  # type: ignore[arg-type]
            return fwd()

        prompt_ext = textwrap.dedent(f"""
        Generate a Python function satisfying the following specification and type signature.
        
        <specification>{template.__prompt_template__}</specification>
        <signature>{str(ret_type)}</signature>

        <instructions>
        1. Produce one block of Python code.
        2. Do not include usage examples.
        3. Return your response in <code> tags.
        4. Do not return your response in markdown blocks.
        5. Your output function def must be the final statement in the code block.
        </instructions>
        """).strip()

        return fwd(
            dataclasses.replace(
                template,
                __prompt_template__=prompt_ext,
                __signature__=template.__signature__,
            ),
            *args,
            **kwargs,
        )
