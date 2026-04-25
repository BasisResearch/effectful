"""Passing PIL images directly to a template.

Demonstrates:
- Templates accepting ``PIL.Image.Image`` arguments
- Inline base64 image data so the script is self-contained
"""

import argparse
import base64
import io
import os

from PIL import Image

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Inline image (32x32 yellow smiley face)
# ---------------------------------------------------------------------------

IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAhElEQVR4nO2W4QqA"
    "MAiEVXr/VzYWDGoMdk7Cgrt/sUs/DqZTd3EplFU2JwATYAJMoOlAB4bq89s95+Mg"
    "+gyAchsKAYplBBBA43hFhfxnUixDjdEUUL8hpr7R0KLdt9qElzcyiu8As+Kr8zQA"
    "mgLavAl+kIzFZyCRxtsAmWb/voZvqRzgBE1sIDuVFX4eAAAAAElFTkSuQmCC"
)


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


@Template.define
def describe_image(image: Image.Image) -> str:
    """Return a short description of the following image.
    {image}
    """
    raise NotHandled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass a PIL image to a template")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use (must support image inputs)",
    )
    args = parser.parse_args()

    image = Image.open(io.BytesIO(base64.b64decode(IMAGE_BASE64)))

    provider = LiteLLMProvider(model=args.model)
    with handler(provider):
        print(describe_image(image))
