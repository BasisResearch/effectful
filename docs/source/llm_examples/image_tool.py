from pprint import pprint

from PIL import Image

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    RetryLLMHandler,
    completion,
)
from effectful.ops.semantics import fwd, handler


class ImageTools(Agent):
    """You are an image processing agent."""

    def __init__(self):
        self._image_to_handle = {}
        self._handle_to_image = {}

    def _encode(self, image: Image) -> int:
        image_id = id(image)
        handle = self._image_to_handle.get(image_id, None)
        if handle is not None:
            return handle

        handle = len(self._image_to_handle)
        self._image_to_handle[image_id] = handle

        assert handle not in self._handle_to_image
        self._handle_to_image[handle] = image
        return handle

    def _decode(self, image_handle: int) -> Image:
        return self._handle_to_image[image_handle]

    @Tool.define
    def rotate(self, image: int, angle: float) -> int:
        """Returns a rotated copy of this image. The copy is rotated by `angle`
        degrees counterclockwise around the image center.

        """
        return self._encode(self._decode(image).rotate(angle))

    @Tool.define
    def concat_horiz(self, i1_h: int, i2_h: int) -> int:
        """Concatenates two images horizontally. The larger image will be
        cropped to the height of the smaller image.

        """
        i1 = self._decode(i1_h)
        i2 = self._decode(i2_h)
        i3 = Image.new("RGB", (i1.width + i2.width, min(i1.height, i2.height)))
        i3.paste(i1, (0, 0))
        i3.paste(i2, (i1.width, 0))
        return self._encode(i3)

    @Template.define
    def _rotate_and_concat(self, i: int) -> int:
        """Create an image consisting of four copies of the image {i}
        concatenated horizontally. Each copy should be rotated 90 degrees from
        the previous.

        """
        pass  # type: ignore

    def rotate_and_concat(self, i: Image) -> Image:
        return self._decode(self._rotate_and_concat(self._encode(i)))


def log_completion(*args, **kwargs):
    pprint((args, kwargs))
    return fwd()


if __name__ == "__main__":
    image_agent = ImageTools()
    img = Image.open("_static/img/chirho_logo_wide.png")

    image_agent._rotate_and_concat.tools
    provider = LiteLLMProvider(model="gpt-5-mini")
    with (
        handler(provider),
        handler({completion: log_completion}),
        handler(RetryLLMHandler()),
    ):
        image_agent.rotate_and_concat(img).show()
