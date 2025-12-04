import ast
import collections.abc
import dataclasses
import inspect
import linecache
import re
import textwrap
import typing
from collections.abc import Callable
from typing import get_args, get_type_hints

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

    def __init__(self, type_check: bool = False):
        """Initialize the program synthesis handler.

        Args:
            type_check: Whether to verify the function signature matches the expected type.
        """
        self.type_check = type_check

    def verify_callable_signature(self, func: Callable, expected_type: type) -> None:
        """Verify that the function signature matches the expected type."""
        type_args = get_args(expected_type)
        if not type_args:
            return

        # For Callable[[P1, P2, ...], R], get_args returns ([P1, P2, ...], R)
        # where the first element is a list of param types (or ... for Callable[..., R])
        expected_param_types, expected_return = type_args[0], type_args[-1]

        sig = inspect.signature(func)
        actual_hints = get_type_hints(func)

        # Verify the return type
        actual_return = actual_hints.get("return", inspect.Parameter.empty)
        if actual_return != expected_return:
            raise SynthesisError(
                f"Return type mismatch: expected {expected_return}, got {actual_return}"
            )

        # Verify the parameter types (if specified and not ellipsis)
        if expected_param_types is not ... and expected_param_types:
            params = list(sig.parameters.values())
            if len(params) != len(expected_param_types):
                raise SynthesisError(
                    f"Parameter count mismatch: expected {len(expected_param_types)}, got {len(params)}"
                )
            for param, expected in zip(params, expected_param_types):
                actual = actual_hints.get(param.name, inspect.Parameter.empty)
                if actual != expected:
                    raise SynthesisError(
                        f"Parameter {param.name} type mismatch: expected {expected}, got {actual}"
                    )

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

        source_code = textwrap.dedent(code)
        lines = code.splitlines(keepends=True)
        filename = f"<generated-{hash(code)}>"

        # register into linecache
        linecache.cache[filename] = (len(source_code), None, lines, filename)

        gs: dict = {}
        try:
            code_obj = compile(source_code, filename, "exec")
            exec(code_obj, gs)
            if self.type_check:
                self.verify_callable_signature(gs[last_decl.name], t)
                # TODO: even more static analysis and type checking, adding type guards, etc.
        except Exception as exc:
            raise SynthesisError(f"evaluation failed: {exc}", content) from exc

        return gs[last_decl.name]

    @implements(Template.__call__)
    def _call(self, template, *args, **kwargs) -> None:
        ret_type = template.__signature__.return_annotation
        origin = typing.get_origin(ret_type)
        ret_type_origin = ret_type if origin is None else origin

        if not (issubclass(ret_type_origin, collections.abc.Callable)):  # type: ignore[arg-type]
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

        # Pass the full ret_type (with type args) for type checking, not just the origin
        functional = self._parse_and_eval(ret_type, response)

        return functional
