import ast
import collections.abc
import dataclasses
import re
import textwrap
import typing

from effectful.handlers.llm import Template
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class SynthesisError(Exception):
    """Raised when program synthesis fails."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class ProgramSynthesis(ObjectInterpretation):
    """Provides a `template` handler to instruct the LLM to generate code of the
    right form and with the right type.

    """

    def _parse_and_eval[T](self, t: type[T], content: str) -> T:
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

        # TODO: assert callable type compatibility
        gs: dict = {}
        try:
            exec(code, gs)
        except Exception as exc:
            raise SynthesisError("evaluation failed", content) from exc

        return gs[last_decl.name]

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

        response = fwd(
            dataclasses.replace(
                template,
                __prompt_template__=prompt_ext,
                __signature__=template.__signature__.replace(return_annotation=str),
            ),
            *args,
            **kwargs,
        )

        return self._parse_and_eval(ret_type, response)
