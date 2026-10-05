"""Putting the library's own type docstrings into the system prompt.

Two `PromptInjectingInterpretation` handlers contribute fixed sections to the
``# Harness`` half of the system message, read from
:mod:`effectful.handlers.llm.types` so the model and the reader see one text.
`FrameworkDocumenter` contributes `Skill`'s docstring under "The effectful LLM
framework" (`FrameworkDocumenter.call_system`); it has no docstring of its own,
so it adds no section about itself. `ApiReferenceDocumenter` contributes `Tool`,
`Agent` and `Encodable` under "The effectful API, for code you write"
(`ApiReferenceDocumenter.call_system`); ``harness()`` installs it only with an
eval provider.
"""

import inspect
import typing

from effectful.handlers.llm.harness.hooks import (
    PromptInjectingInterpretation,
    call_system,
)
from effectful.handlers.llm.harness.serialization import (
    PromptSection,
    to_content_blocks,
)
from effectful.handlers.llm.types import Agent, Encodable, Skill, Tool
from effectful.ops.syntax import implements


def _concept(name: str, typ: object) -> PromptSection:
    return PromptSection(
        type="prompt_section",
        title=f"`{name}`",
        content=to_content_blocks(inspect.getdoc(typ) or ""),
    )


class FrameworkDocumenter(PromptInjectingInterpretation):
    # No docstring of its own: the section it contributes is `Skill`'s, which
    # states what the model is answering and where the rest of the package is.

    #: Title of the section `call_system` contributes.
    title: typing.ClassVar[str] = "The effectful LLM framework"

    @implements(call_system)
    def call_system(
        self, harness_prompt: PromptSection, agent_prompt: PromptSection
    ) -> typing.Any:
        """Prepend `Skill`'s docstring, which holds still while the handler stack
        around it varies, so its position is independent of composition order."""
        section = PromptSection(
            type="prompt_section",
            title=self.title,
            content=to_content_blocks(inspect.getdoc(Skill) or ""),
        )
        return super().call_system(
            PromptSection(
                type="prompt_section",
                title=harness_prompt["title"],
                content=[section, *harness_prompt["content"]],
            ),
            agent_prompt,
        )


class ApiReferenceDocumenter(PromptInjectingInterpretation):
    """Code you write in the REPL or submit as a body may define Skills, Tools,
    receivers, and encodings with the API below, which is the library's own
    documentation of those types.
    """

    #: Title of the section `call_system` contributes.
    title: typing.ClassVar[str] = "The effectful API, for code you write"

    @implements(call_system)
    def call_system(
        self, harness_prompt: PromptSection, agent_prompt: PromptSection
    ) -> typing.Any:
        """Append one subsection per type, in a fixed order, then the docstring above."""
        section = PromptSection(
            type="prompt_section",
            title=self.title,
            content=[
                _concept("Tool", Tool),
                _concept("Agent", Agent),
                _concept("Encodable", Encodable),
            ],
        )
        return super().call_system(
            PromptSection(
                type="prompt_section",
                title=harness_prompt["title"],
                content=[*harness_prompt["content"], section],
            ),
            agent_prompt,
        )
