"""A coding assistant served over the Agent Client Protocol.

Demonstrates:
- An ``Agent`` exposed to any ACP editor (Zed, VS Code, Obsidian, Emacs)
- Editor capabilities offered to the model as ordinary ``Tool``\\ s
- Streaming model output and live tool-call status as ``session/update``
- A prompt as typed parts -- prose, ``Attachment`` references, images -- rather
  than one flattened string, so an attached file costs a line of context and a
  screenshot arrives as an image the model actually sees

All the protocol machinery lives in ``library.py``; what is left here is an agent,
four imported tools, and a command line -- which is the point of the example.

The ``prompt`` skill's signature is the server's contract (see
`EffectfulACPAgent`): the name matches the protocol method it answers
(``session/prompt``), and the parameters are what an editor's prompt can carry.
Attachments arrive by reference -- the model reads one through the editor if the
request needs it -- which is what keeps a large attached file from being frozen
into the conversation whole.

``acp_ask_user`` is one of the four tools, and the only one that is a question
rather than an action: it needs the client's ``elicitation.form`` capability, and
reports itself as a failed call in an editor that has none, so importing it costs
nothing where it will not work.

The imports are what put the editor's capabilities in the assistant's lexical scope,
and lexical scope is how a `Skill` finds its tools, so they are load-bearing despite
looking unused. `EffectfulACPAgent` is imported inside `main` instead, to keep the
protocol out of the scope the model is shown.

This is a server: it speaks JSON-RPC over stdin and stdout and expects an editor at
the other end, so it is not useful to run from a terminal by hand. Configure the
editor to launch it, with the command line it would run being::

    python -m effectful.handlers.llm.harness docs/source/llm_examples/acp/assistant.py \\
        --persist-db /tmp/acp_sessions.db

``--persist-db`` is what makes a session survive a restart: the editor's session id
becomes the agent's ``__agent_id__``, and the conversation is checkpointed under it
and replayed when the editor reopens the session.

Set ``ACP_OFFER_MODELS`` to a comma-separated list to put a picker in the editor's
UI, so the session can be switched without editing the editor's configuration.
"""

import argparse
import asyncio
import collections.abc
import dataclasses

import PIL.Image
from library import (  # noqa: F401
    Attachment,
    acp_ask_user,
    acp_read_text_file,
    # acp_run_terminal_command,
    acp_update_plan,
    acp_write_text_file,
)

from effectful.handlers.llm import Skill


@dataclasses.dataclass
class Assistant:
    """A coding assistant working inside the user's editor.

    Your reply is shown to the user in their editor and rendered as Markdown, so
    write it as prose addressed to them; headings, lists, links and fenced code
    blocks all display.
    """

    __agent_id__: str = ""

    @Skill.define
    def prompt(
        self,
        user_input: str,
        attachments: collections.abc.Sequence[Attachment] = (),
        images: collections.abc.Sequence[PIL.Image.Image] = (),
    ) -> str:
        """{user_input}{attachments}{images}"""


def main() -> None:
    from library import EffectfulACPAgent

    argparse.ArgumentParser(description=__doc__).parse_args()
    asyncio.run(EffectfulACPAgent(Assistant).serve())


if __name__ == "__main__":
    main()
