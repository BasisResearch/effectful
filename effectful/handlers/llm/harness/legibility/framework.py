"""Put the LLM type references and example pointers into the model's system prompt.

`FrameworkDocumenter` reads the docstrings in :mod:`effectful.handlers.llm.types`
and :mod:`effectful.handlers.llm.examples`. It always describes `Skill`; when
code execution is enabled, it also describes `Tool`, `Agent`, and `Encodable`
for model-authored Python.
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
from effectful.ops.semantics import fwd
from effectful.ops.syntax import implements


def _concept(name: str, typ: object) -> PromptSection:
    return PromptSection(
        type="prompt_section",
        title=f"`{name}`",
        content=to_content_blocks(inspect.getdoc(typ) or ""),
    )


class FrameworkDocumenter(PromptInjectingInterpretation):
    """Put the LLM type references first in the model's Harness section.

    The class docstring is for Python readers. `call_system` forwards directly
    so it is not added to the model's prompt alongside the type references.
    """

    #: Title of the section `call_system` contributes.
    title: typing.ClassVar[str] = "The effectful LLM framework"

    def __init__(self, *, include_code_api: bool = True) -> None:
        self.include_code_api = include_code_api

    @implements(call_system)
    def call_system(
        self, harness_prompt: PromptSection, agent_prompt: PromptSection
    ) -> typing.Any:
        """Prepend the type references independently of handler order."""
        import effectful.handlers.llm.examples as examples

        concepts = [_concept("Skill", Skill)]
        if self.include_code_api:
            concepts.extend(
                (
                    _concept("Tool", Tool),
                    _concept("Agent", Agent),
                    _concept("Encodable", Encodable),
                )
            )
        concepts.append(_concept("Bundled examples", examples))
        section = PromptSection(
            type="prompt_section",
            title=self.title,
            content=concepts,
        )
        return fwd(
            PromptSection(
                type="prompt_section",
                title=harness_prompt["title"],
                content=[section, *harness_prompt["content"]],
            ),
            agent_prompt,
        )
