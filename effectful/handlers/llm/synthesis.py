import ast
import dataclasses
import linecache
import re
import textwrap
import typing

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import (
    compute_response,
    decode_callable,
    decode_response,
    format_model_input,
)
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

    def _wrap_template[**P, T](self, template: Template[P, T]) -> Template[P, str]:
        ret_type = template.__signature__.return_annotation

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

        return dataclasses.replace(
            template,
            __prompt_template__=prompt_ext,
            __signature__=template.__signature__.replace(return_annotation=str),
        )  # type: ignore

    @implements(decode_callable)
    def _decode_callable[T](
        self, _ret_type: type[T], content: str
    ) -> typing.Callable[..., T]:
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

        source_code = textwrap.dedent(code)
        lines = code.splitlines(keepends=True)
        filename = f"<generated-{hash(code)}>"

        # register into linecache
        linecache.cache[filename] = (len(source_code), None, lines, filename)

        # TODO: assert callable type compatibility
        gs: dict = {}
        try:
            code_obj = compile(source_code, filename, "exec")
            exec(code_obj, gs)
        except Exception as exc:
            raise SynthesisError("evaluation failed", content) from exc

        return gs[last_decl.name]

    @implements(Template.__call__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        ret_type_origin = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type_origin)
        ret_type = ret_type_origin if origin is None else origin

        if not issubclass(ret_type, typing.Callable):  # type: ignore
            return fwd()

        str_template = self._wrap_template(template)
        model_input = format_model_input(str_template, *args, **kwargs)
        resp = compute_response(str_template, model_input)

        # decode the response using the decoding mechanism
        return decode_response(template, resp)
