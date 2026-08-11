import dataclasses
import pathlib

from effectful.handlers.llm.harness.hooks import call_system
from effectful.handlers.llm.harness.observability.rendering import _message_text
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


@dataclasses.dataclass(frozen=True)
class SystemPromptDumper(ObjectInterpretation):
    """Dump the system prompt produced by `call_system` to a Markdown file.

    Opt-in debugging handler: intercepts `call_system`, forwards to let the
    prompt be assembled and installed as usual, then writes the resulting
    system message content to `path`, overwriting the whole file each time.
    """

    path: pathlib.Path

    @implements(call_system)
    def _call_system(self, prompt):
        message = fwd()
        self.path.write_text(_message_text(message.get("content")))
        return message
