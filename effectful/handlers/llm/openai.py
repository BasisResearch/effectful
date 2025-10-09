import os
import string

import openai
from PIL import Image

from effectful.handlers.llm.utils import _pil_image_to_base64_data_uri
from effectful.ops.llm import Template, decode
from effectful.ops.syntax import ObjectInterpretation, implements


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
