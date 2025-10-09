import ast
import collections.abc
import dataclasses
import re
import textwrap
import typing

from effectful.handlers.llm import Template, decode
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class ProgramSynthesis(ObjectInterpretation):
    """Provides a `decode` handler for callables and a `template` handler to
    instruct the LLM to generate code of the right form and with the right type.

    """

    @implements(decode)
    def _decode[T](self, t: type[T], content: str) -> T:
        origin = typing.get_origin(t)
        t = t if origin is None else origin

        if not (issubclass(t, collections.abc.Callable)):  # type: ignore[arg-type]
            return fwd()

        pattern = r"<code>(.*?)</code>"
        code_content = re.search(pattern, content, re.DOTALL)
        if code_content is None:
            return fwd()
        code = code_content.group(1)

        try:
            module_ast = ast.parse(code)
        except SyntaxError:
            return fwd()

        if not isinstance(module_ast, ast.Module):
            return fwd()

        last_decl = module_ast.body[-1]
        if not isinstance(last_decl, ast.FunctionDef):
            return fwd()

        # TODO: assert callable type compatibility
        gs: dict = {}
        try:
            exec(code, gs)
        except Exception:
            return fwd()

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

        return fwd(
            dataclasses.replace(template, __prompt_template__=prompt_ext),
            *args,
            **kwargs,
        )
