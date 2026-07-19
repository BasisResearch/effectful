"""Passing PIL images directly to a template.

Demonstrates:
- Templates accepting ``PIL.Image.Image`` arguments
- Inline base64 image data so the script is self-contained
"""

import base64
import io

from PIL import Image

from effectful.handlers.llm import Template

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    image = Image.open(io.BytesIO(base64.b64decode(IMAGE_BASE64)))

    print(describe_image(image))


if __name__ == "__main__":
    main()
