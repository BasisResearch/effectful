import ast
import base64
import collections.abc
import dataclasses
import inspect
import io
import os
import re
import string
import textwrap
import typing
import weakref
from collections.abc import Callable
from typing import Any

import anthropic
import openai
from PIL import Image

from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, defop, implements


@dataclasses.dataclass(frozen=True)
class Template[**P, T]:
    __signature__: inspect.Signature
    __prompt_template__: str
    __name__: str

    @defop
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotImplementedError


def template[**P, T](body: Callable[P, T]) -> Callable[P, T]:
    """A prompt template intended to be filled in by an LLM."""
    return typing.cast(
        Callable[P, T],
        Template(
            inspect.signature(body), body.__doc__ if body.__doc__ else "", body.__name__
        ),
    )


class DecodeError(RuntimeError):
    """Raised when decoding an LLM response fails."""

    def __init__(self, t: type, response: str):
        super().__init__()
        self.type_ = t
        self.response = response

    def __repr__(self):
        return f"DecodeError({self.type_}, {self.response})"


@defop
def decode[T](t: type[T], content: str) -> T:
    """Decode `content` as an instance of `t`. Used to consume the output of an
    LLM.

    """
    if t is str:
        return typing.cast(T, content)
    elif t is bool:
        match content.strip().lower():
            case "true":
                return typing.cast(T, True)
            case "false":
                return typing.cast(T, False)
            case _:
                raise DecodeError(t, content)
    elif t in (int, float, complex, bool):
        try:
            result = t(content)  # type: ignore
        except ValueError:
            raise DecodeError(t, content)
        return typing.cast(T, result)

    raise DecodeError(t, content)


class TemplateCache(ObjectInterpretation):
    """Caches prompt template instantiations."""

    @dataclasses.dataclass(frozen=True, eq=True)
    class _ArgsKwargs:
        args: tuple[Any, ...]
        kwargs: tuple[tuple[str, Any], ...]

    _cache: weakref.WeakKeyDictionary[Template, dict[_ArgsKwargs, Any]]

    def __init__(self):
        self._cache = weakref.WeakKeyDictionary()

    @implements(Template.__call__)
    def _call(self, template, *args, **kwargs):
        call_cache = self._cache[template] if template in self._cache else {}
        key = TemplateCache._ArgsKwargs(tuple(args), tuple(kwargs.items()))

        try:
            in_call_cache = key in call_cache
        except TypeError as e:
            if "unhashable type" in str(e):
                return fwd()
            raise e

        if in_call_cache:
            return call_cache[key]

        result = fwd()
        call_cache[key] = result
        self._cache[template] = call_cache
        return result


def _pil_image_to_base64_data(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_image_to_base64_data_uri(pil_image: Image.Image) -> str:
    return f"data:image/png;base64,{_pil_image_to_base64_data(pil_image)}"


class _OpenAIPromptFormatter(string.Formatter):
    def format_as_messages(
        self, format_str: str, /, *args, **kwargs
    ) -> openai.types.responses.ResponseInputMessageContentListParam:
        prompt_parts = []
        current_text = ""

        def push_current_text():
            nonlocal current_text
            if current_text:
                prompt_parts.append({"type": "input_text", "text": current_text})
            current_text = ""

        for literal, field_name, format_spec, conversion in self.parse(format_str):
            current_text += literal

            if field_name is not None:
                obj, _ = self.get_field(field_name, args, kwargs)
                obj = self.convert_field(obj, conversion)

                if isinstance(obj, Image.Image):
                    assert not format_spec, (
                        "image template parameters cannot have format specifiers"
                    )
                    push_current_text()
                    prompt_parts.append(
                        {
                            "type": "input_image",
                            "image_url": _pil_image_to_base64_data_uri(obj),
                        }
                    )
                else:
                    current_text += self.format_field(
                        obj, format_spec if format_spec else ""
                    )

        push_current_text()
        return prompt_parts


class OpenAI(ObjectInterpretation):
    """Implements templates using the OpenAI API."""

    def __init__(self, model_name: str = "gpt-4o", api_key: str | None = None):
        api_key = os.getenv("OPENAI_API_KEY") if api_key is None else api_key
        self._client = openai.OpenAI(api_key=api_key)
        self._model_name = model_name

    @implements(Template.__call__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        prompt = _OpenAIPromptFormatter().format_as_messages(
            template.__prompt_template__, **bound_args.arguments
        )

        # TODO: Support structured outputs https://platform.openai.com/docs/guides/structured-outputs

        # Note: The OpenAI api only seems to accept images in the 'user' role.
        # The effect of different roles on the model's response is currently
        # unclear.
        response = self._client.responses.create(
            model=self._model_name, input=[{"content": prompt, "role": "user"}]
        )

        first_response = response.output[0]
        assert first_response.type == "message"
        first_response_content = first_response.content[0]
        assert first_response_content.type == "output_text"

        ret_type = template.__signature__.return_annotation
        return decode(ret_type, first_response_content.text)


class _AnthropicPromptFormatter(string.Formatter):
    def format_as_messages(
        self, format_str: str, /, *args, **kwargs
    ) -> list[anthropic.types.TextBlockParam | anthropic.types.ImageBlockParam]:
        prompt_parts = []
        current_text = ""

        def push_current_text():
            nonlocal current_text
            if current_text:
                prompt_parts.append({"type": "text", "text": current_text})
            current_text = ""

        for literal, field_name, format_spec, conversion in self.parse(format_str):
            current_text += literal

            if field_name is not None:
                obj, _ = self.get_field(field_name, args, kwargs)
                obj = self.convert_field(obj, conversion)

                if isinstance(obj, Image.Image):
                    assert not format_spec, (
                        "image template parameters cannot have format specifiers"
                    )
                    push_current_text()

                    img_source = {
                        "data": _pil_image_to_base64_data(obj),
                        "media_type": "image/png",
                        "type": "base64",
                    }
                    prompt_parts.append({"type": "image", "source": img_source})
                else:
                    current_text += self.format_field(
                        obj, format_spec if format_spec else ""
                    )

        push_current_text()
        return prompt_parts


class Anthropic(ObjectInterpretation):
    """Implements templates using the Anthropic API."""

    def __init__(
        self, model_name: str = "claude-3-7-sonnet-20250219", api_key: str | None = None
    ):
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY") if not api_key else api_key
        self._client = Anthropic(api_key=api_key)
        self._model_name = model_name

    @implements(Template.__call__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        prompt = _AnthropicPromptFormatter().format_as_messages(
            template.__prompt_template__, **bound_args.arguments
        )

        # TODO: Support structured outputs https://platform.openai.com/docs/guides/structured-outputs
        response = self._client.messages.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2**12,
        )
        content = response.content[0]
        assert content.type == "text"

        ret_type = template.__signature__.return_annotation
        return decode(ret_type, content.text)


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
