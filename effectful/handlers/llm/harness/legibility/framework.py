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

    # The docstring above is model-facing (see `PromptInjectingInterpretation`);
    # notes for a reader of the code belong in comments like this one.
    #
    # Unlike the capability handlers, the section this contributes is not its own
    # docstring: it is assembled from `effectful.handlers.llm.types` and the
    # concepts in its `__all__`, which is what keeps the prompt from drifting
    # away from the library. It is the same for every call in the process, which
    # is what makes it worth putting first.

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
        # Prepended, where the capability handlers append: the concepts hold
        # still for the whole process, so they belong at the front of the
        # document whatever the handler stack around them looks like.  The
        # docstring section the base rule adds is appended with everyone else's,
        # so it is not adjacent to the concepts -- which is why it names that
        # section rather than pointing at what follows it.
        return super().call_system(
            PromptSection(
                type="prompt_section",
                title=harness_prompt["title"],
                content=[self._concepts_section(), *harness_prompt["content"]],
            ),
            agent_prompt,
        )
