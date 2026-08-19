import collections.abc
import dataclasses
import pathlib
import typing

from effectful.handlers.llm.harness.hooks import call_system
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


def _message_text(content: None | str | collections.abc.Iterable[typing.Any]) -> str:
    """Flatten a message ``content`` to display text.

    ``content`` may be a plain string or a list of content blocks (dicts with a
    ``type`` discriminator, e.g. ``{"type": "text", "text": ...}``, as produced
    by :func:`~effectful.handlers.llm.encoding.to_content_blocks`). Text blocks
    contribute their text; other block types show a ``[type]`` placeholder.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text") or "")
            else:
                parts.append(f"[{block.get('type', 'content')}]")
        else:
            parts.append(str(block))
    return "".join(parts)


@dataclasses.dataclass(frozen=True)
class SystemPromptDumper(ObjectInterpretation):
    """Dump the system prompt produced by `call_system` to a Markdown file.

    Opt-in debugging handler: intercepts `call_system`, forwards to let the
    prompt be assembled and installed as usual, then writes the resulting
    system message content to `path`, overwriting the whole file each time.
    """

    path: pathlib.Path

    @implements(call_system)
    def _call_system(self, harness_prompt, agent_prompt):
        message = fwd()
        self.path.write_text(_message_text(message.get("content")))
        return message
