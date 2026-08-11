import builtins
import collections.abc
import inspect
import json
import types
import typing

import pydantic

from effectful.handlers.llm.harness.hooks import (
    AssistantResult,
    Message,
    call_assistant,
    call_system,
)
from effectful.handlers.llm.harness.serialization import (
    PromptSection,
    SystemPrompt,
    add_prompt_content,
    to_content_blocks,
)
from effectful.handlers.llm.types import Agent, Encodable, Template, Tool
from effectful.internals.unification import nested_type
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements

HARNESS_SECTION: typing.Final[str] = "Harness"


class LexicalReaders(ObjectInterpretation):
    """Some of the tools below take no arguments and simply return the current
    value of a named variable from this Template's lexical scope (see the
    *Lexical scope* table for the available names and their types). Call such a
    reader when your answer depends on the concrete value of an in-scope
    variable that has not already been spliced into the prompt — it lets you
    fetch that value on demand instead of guessing it. Each reader's description
    names the variable it reads.
    """

    @typing.final
    class _LexicalVariableTool[T](Tool[[], T]):
        """A synthetic zero-argument reader for one lexically scoped value.

        A distinct type only so `LexicalReaders` can recognize its own readers
        among the tools in a request; the capability is described to the model by
        the handler's docstring.
        """

        @classmethod
        def define(cls, value: typing.Any, *, name: str) -> "Tool[[], typing.Any]":
            """Construct a synthetic reader Tool that returns `value`.

            Raises if `Encodable[nested_type(value)]` cannot be generated.
            The caller is responsible for catching the failure and deciding
            whether to skip the symbol.
            """
            assert not isinstance(value, Tool), (
                "Tools are real tools and must not be re-wrapped as lexical readers."
            )
            typ: typing.Any = nested_type(value).value
            # Probe schema generation; raises if `Encodable[typ]` is not implemented.
            pydantic.TypeAdapter(Encodable[typ]).json_schema()

            def tool_fn():
                return value

            tool_fn.__name__ = name
            tool_fn.__qualname__ = name
            tool_fn.__module__ = type(value).__module__
            tool_fn.__doc__ = "Reads lexical variable of the same name"
            tool_fn.__annotations__ = {"return": typ}
            return super().define(tool_fn)

    @implements(call_system)
    def _call_system(self, prompt: SystemPrompt) -> typing.Any:
        return fwd(
            add_prompt_content(
                prompt,
                PromptSection(
                    type="prompt_section",
                    title=type(self).__name__,
                    content=to_content_blocks(inspect.getdoc(type(self)) or ""),
                ),
                under=HARNESS_SECTION,
            )
        )

    @implements(call_assistant)
    def _call_assistant[T](
        self,
        messages: collections.abc.Sequence[Message],
        env: collections.abc.Mapping[str, typing.Any],
        response_type: type[T],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult[T]:
        readers: set[Tool] = set(tools)
        taken = {t.__name__ for t in tools}
        for name, obj in env.items():
            if (
                name in taken
                or not name.isidentifier()
                or isinstance(obj, Tool)
                or (name.startswith("__") and name.endswith("__"))
            ):
                continue
            try:
                readers.add(self._LexicalVariableTool.define(obj, name=name))
                taken.add(name)
            except Exception:
                continue
        return fwd(messages, env, response_type, readers)


def _get_qualname(cls) -> str:
    """Module-qualified name of a type, dropping the ``builtins`` prefix."""
    if not isinstance(cls, type):
        return str(cls)
    module = getattr(cls, "__module__", None)
    name = (
        getattr(cls, "__qualname__", None) or getattr(cls, "__name__", None) or str(cls)
    )
    return name if module in (None, "builtins") else f"{module}.{name}"


def _vars_section(
    env: collections.abc.Mapping[str, typing.Any],
) -> PromptSection | None:
    """Markdown table of the non-module bindings in scope (name -> type).

    Excludes dunder names (``__main__`` etc.) and names already bound to their
    standard builtin (which the model knows).
    """
    rows = {
        name: _get_qualname(type(value))
        for name, value in env.items()
        if not (name.startswith("__") and name.endswith("__"))
        and value not in vars(builtins).values()
        and not isinstance(value, types.ModuleType)
    }
    if not rows:
        return None
    body = "\n".join(f"| `{n}` | `{t}` |" for n, t in sorted(rows.items()))
    return PromptSection(
        type="prompt_section",
        title="Lexical scope",
        content=to_content_blocks(f"| name | type |\n| --- | --- |\n{body}"),
    )


def _imports_section(
    env: collections.abc.Mapping[str, typing.Any],
) -> PromptSection | None:
    """Markdown table of the imported modules in scope (name -> module name).

    Excludes dunder names and names already bound to their standard builtin.
    """
    rows = {
        name: value.__name__
        for name, value in env.items()
        if not (name.startswith("__") and name.endswith("__"))
        and value not in vars(builtins).values()
        and isinstance(value, types.ModuleType)
    }
    if not rows:
        return None
    body = "\n".join(f"| `{n}` | `{m}` |" for n, m in sorted(rows.items()))
    return PromptSection(
        type="prompt_section",
        title="Imported modules",
        content=to_content_blocks(f"| name | module |\n| --- | --- |\n{body}"),
    )


def _template_section(template: Template) -> PromptSection:
    """Spec for a single `Template`: header, prompt, arg schemas.

    A subsection of the enclosing agent/template section (see `_agent_section`),
    so it renders at ``##`` and the prompt's own headings below that.
    """
    parts = []
    prompt = inspect.getdoc(template.__default__) or ""
    if prompt:
        parts.append(prompt)
    args = [
        f"- `{name}` — `{_get_qualname(p.annotation)}`\n\n"
        f"    ```json\n    {json.dumps(pydantic.TypeAdapter(Encodable[p.annotation]).json_schema())}\n    ```"  # type: ignore[name-defined]
        for name, p in template.__signature__.parameters.items()
    ]
    if args:
        parts.append("**Arguments**\n\n" + "\n".join(args))
    return PromptSection(
        type="prompt_section",
        title=f"`{template.__name__}{template.__signature__}`",
        content=to_content_blocks("\n\n".join(parts)),
    )


def _agent_section(template: Template) -> PromptSection:
    """The section for the task: the Agent's docstring (if any) followed by a
    subsection for every Template sharing the current history (an Agent's
    methods, or just ``template`` for a free-function template)."""
    inst = (
        template.__default__.__self__
        if isinstance(template.__default__, types.MethodType)
        else None
    )
    if isinstance(inst, Agent):
        agent_doc = inspect.getdoc(type(inst)) or ""
        title = f"Agent `{_get_qualname(type(inst))}`"
        templates = set()
        for cls in type(inst).__mro__:
            for attr in vars(cls):
                try:
                    value = getattr(inst, attr)
                except Exception:
                    continue
                if isinstance(value, Template):
                    templates.add(value)
    else:
        agent_doc = ""
        title = "Template"
        templates = {template}

    # The agent docstring is intro prose for the section, ahead of the specs;
    # order the specs by name so the prompt is stable across method reordering
    # in source.
    return PromptSection(
        type="prompt_section",
        title=title,
        content=[
            *to_content_blocks(agent_doc),
            *(
                _template_section(t)
                for t in sorted(templates, key=lambda t: t.__name__)
            ),
        ],
    )


def _module_section(mod: types.ModuleType | None) -> PromptSection | None:
    """The section carrying the source (or docstring fallback) of a module."""
    if mod is None:
        return None
    try:
        src = inspect.getsource(mod)
        body = f"```python\n{src}\n```"
    except (OSError, TypeError):
        body = inspect.getdoc(mod) or ""
        if not body:
            return None
    return PromptSection(
        type="prompt_section",
        title=f"Module `{mod.__name__}`",
        content=to_content_blocks(body),
    )


def template_system_prompt(template: Template) -> SystemPrompt:
    """The initial system prompt for a call to `template`, before any handler
    has added to it.

    The document is laid out most-constant-first, so that it caches well as the
    conversation grows: the two sections handlers contribute to come first and
    start empty, followed by the sections introspected
    here, which vary per module, per instance and per scope.
    """
    sections = (
        PromptSection(type="prompt_section", title=HARNESS_SECTION, content=[]),
        _module_section(inspect.getmodule(template)),
        _agent_section(template),
        _imports_section(template.__context__),
        _vars_section(template.__context__),
    )
    return SystemPrompt(
        type="system_prompt",
        title=f"`{template.__name__}{template.__signature__}`",
        content=[
            *(s for s in sections if s is not None),
        ],
    )


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
    def _call_system(self, prompt: SystemPrompt) -> typing.Any:
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
        new_prompt = SystemPrompt(
            type=prompt["type"],
            title=prompt["title"],
            content=[section, *prompt["content"]],
        )
        return fwd(new_prompt)
