import os
import string

import anthropic
from PIL import Image

from effectful.handlers.llm.utils import _pil_image_to_base64_data
from effectful.ops.llm import Template, decode
from effectful.ops.syntax import ObjectInterpretation, implements


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
