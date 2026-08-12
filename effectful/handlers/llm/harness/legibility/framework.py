import inspect
import typing

from effectful.handlers.llm.harness.hooks import call_system
from effectful.handlers.llm.harness.legibility.lexical import _get_qualname
from effectful.handlers.llm.harness.serialization import (
    PromptSection,
    to_content_blocks,
)
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class FrameworkDocumenter(ObjectInterpretation):
    """Fill in the constant framework-concept section of the system prompt.

    Its content is sourced from the real docstrings of
    `effectful.handlers.llm` and the concepts it exports, so what the model is
    told about `Template`, `Tool`, `Agent` and `Encodable` cannot drift from what
    a reader of the package is told.  It is the same for every call in the
    process, which is what makes it worth putting first.
    """

    title: typing.ClassVar[str] = "The effectful LLM framework"

    @implements(call_system)
    def _call_system(
        self, harness_prompt: PromptSection, agent_prompt: PromptSection
    ) -> typing.Any:
        import effectful.handlers.llm as _llm

        content: list[typing.Any] = list(to_content_blocks(inspect.getdoc(_llm) or ""))
        content.extend(
            PromptSection(
                type="prompt_section",
                title=f"`{_get_qualname(typ)}`",
                content=to_content_blocks(inspect.getdoc(typ) or ""),
            )
            for typ in sorted(
                map(lambda name: getattr(_llm, name), _llm.__all__), key=_get_qualname
            )
        )
        section = PromptSection(
            type="prompt_section",
            title=self.title,
            content=content,
        )
        # Prepended, where the capability handlers append: the concepts hold
        # still for the whole process, so they belong at the front of the
        # cached prefix whatever the handler stack around them looks like.
        return fwd(
            PromptSection(
                type="prompt_section",
                title=harness_prompt["title"],
                content=[section, *harness_prompt["content"]],
            ),
            agent_prompt,
        )
