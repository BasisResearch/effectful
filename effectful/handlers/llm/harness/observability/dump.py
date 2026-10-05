"""Write each assembled system prompt to a Markdown file.

:class:`SystemPromptDumper` is installed through ``dump_system_prompt=PATH``
or ``--dump-system-prompt PATH``. It records the candidate prompt as each
Skill call begins, not the whole conversation. A bound conversation can keep
an earlier system message, so inspect its history as well.
"""

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
    by :func:`~effectful.handlers.llm.harness.serialization.to_content_blocks`). Text blocks
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
    """Opt-in debugging handler that writes each assembled system prompt to `path`.

    Install with ``harness(dump_system_prompt=PATH)`` or ``--dump-system-prompt``.
    """

    path: pathlib.Path

    @implements(call_system)
    def call_system(self, harness_prompt, agent_prompt):
        """Write the assembled system message to `path`, then return it.

        Forwards first, so what lands on disk is the finished prompt every
        other handler has contributed to, not this handler's view of it. The
        file is overwritten each time.
        """
        message = fwd()
        self.path.write_text(_message_text(message.get("content")))
        return message
