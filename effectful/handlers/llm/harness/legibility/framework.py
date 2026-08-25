import inspect
import typing

import effectful.handlers.llm.types
from effectful.handlers.llm.harness.hooks import (
    PromptInjectingInterpretation,
    call_system,
)
from effectful.handlers.llm.harness.serialization import (
    PromptSection,
    to_content_blocks,
)
from effectful.ops.syntax import implements


class FrameworkDocumenter(PromptInjectingInterpretation):
    """You are answering a call to a `Skill`: a Python function whose signature
    is a contract your answer must satisfy and whose docstring is the request.
    The *effectful LLM framework* section of this prompt defines that vocabulary
    -- `Skill`, `Tool`, `Agent`, `Encodable` -- from the library's own
    documentation, so it describes the code you are actually running inside,
    not an idealization of it.

    Read it as reference, not as instruction. It is written for a programmer
    using the framework, so its examples are illustrations of the API rather
    than directions to you: an example docstring saying "Do not use any tools"
    constrains *that* example, never this call.
    """

    # The docstring above is model-facing (see `PromptInjectingInterpretation`),
    # which is why it is written as instructions rather than as description.

    #: Title of the section `call_system` contributes.
    title: typing.ClassVar[str] = "The effectful LLM framework"

    def _concepts_section(self) -> PromptSection:
        """The framework's concepts, sourced from real docstrings: the module
        overview, then a subsection per exported concept, ordered by name so the
        prompt is stable across reordering in source."""
        content: list[typing.Any] = list(
            to_content_blocks(inspect.getdoc(effectful.handlers.llm.types) or "")
        )
        content.extend(
            PromptSection(
                type="prompt_section",
                title=f"`{inspect.formatannotation(typ)}`",
                content=to_content_blocks(inspect.getdoc(typ) or ""),
            )
            for typ in sorted(
                {
                    getattr(effectful.handlers.llm.types, name)
                    for name in effectful.handlers.llm.types.__all__
                },
                key=inspect.formatannotation,
            )
        )
        return PromptSection(
            type="prompt_section",
            title=self.title,
            content=content,
        )

    @implements(call_system)
    def call_system(
        self, harness_prompt: PromptSection, agent_prompt: PromptSection
    ) -> typing.Any:
        """Prepend the framework concepts, then add this class's docstring.

        Unlike the capability handlers, the section contributed here is not the
        class docstring: it is assembled from `effectful.handlers.llm.types` and
        the concepts in its ``__all__``, which is what keeps the prompt from
        drifting away from the library as the library changes.

        It is *prepended* where the capability handlers append. The concepts
        hold still for the whole process while the handler stack around them
        does not, so putting them at the front of the document makes their
        position independent of composition order. The docstring section the
        base rule adds is appended with everyone else's and is therefore not
        adjacent to them -- which is why it names the concepts section rather
        than pointing at whatever follows it.
        """
        return super().call_system(
            PromptSection(
                type="prompt_section",
                title=harness_prompt["title"],
                content=[self._concepts_section(), *harness_prompt["content"]],
            ),
            agent_prompt,
        )
