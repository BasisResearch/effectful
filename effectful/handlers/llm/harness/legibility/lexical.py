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
    to_content_blocks,
)
from effectful.handlers.llm.types import Agent, Encodable, Skill, Tool
from effectful.internals.unification import nested_type
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class LexicalReaders(ObjectInterpretation):
    """Some of the tools below take no arguments and simply return the current
    value of a named variable from this Skill's lexical scope (see the
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
    def _call_system(
        self, harness_prompt: PromptSection, agent_prompt: PromptSection
    ) -> typing.Any:
        return fwd(
            PromptSection(
                type="prompt_section",
                title=harness_prompt["title"],
                content=[
                    *harness_prompt["content"],
                    PromptSection(
                        type="prompt_section",
                        title=type(self).__name__,
                        content=to_content_blocks(inspect.getdoc(type(self)) or ""),
                    ),
                ],
            ),
            agent_prompt,
        )

    @implements(call_assistant)
    def _call_assistant[T](
        self,
        messages: collections.abc.Sequence[Message],
        response_type: type[T],
        env: collections.abc.Mapping[str, typing.Any],
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
        return fwd(messages, response_type, env, readers)


class LexicalToolExtractor(ObjectInterpretation):
    """Offer the model the tools reachable from a `Skill`'s lexical scope.

    Unions `_tools_in_scope(env)` into the request's `tools` and forwards, so a
    `Skill` is offered the `Tool`/`Skill` values bound in its context (and those
    its in-scope `Agent`s hold) without naming them itself.

    Install it below anything else that contributes tools: the anchor `Skill` is
    dropped from the set by `call_assistant`'s default rule, so this must be the
    innermost `call_assistant` handler for that subtraction to see it. Without
    this handler installed a `Skill` is simply offered no lexical tools -- the
    request is well-formed either way, so the omission surfaces only as a model
    that never calls a tool it was supposed to have.
    """

    @implements(call_assistant)
    def _call_assistant(
        self,
        messages: collections.abc.Sequence[Message],
        response_type: type,
        env: collections.abc.Mapping[str, typing.Any],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult:
        return fwd(messages, response_type, env, tools | _tools_in_scope(env))


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


def _skill_section(skill: Skill) -> PromptSection:
    """Spec for a single `Skill`: header, prompt, arg schemas.

    A subsection of the enclosing agent/skill section (see `_agent_section`),
    so it renders at ``##`` and the prompt's own headings below that.
    """
    parts = []
    prompt = inspect.getdoc(skill.__default__) or ""
    if prompt:
        parts.append(prompt)
    args = [
        f"- `{name}` — `{_get_qualname(p.annotation)}`\n\n"
        f"    ```json\n    {json.dumps(pydantic.TypeAdapter(Encodable[p.annotation]).json_schema())}\n    ```"  # type: ignore[name-defined]
        for name, p in skill.__signature__.parameters.items()
    ]
    if args:
        parts.append("**Arguments**\n\n" + "\n".join(args))
    return PromptSection(
        type="prompt_section",
        title=f"`{skill.__name__}{skill.__signature__}`",
        content=to_content_blocks("\n\n".join(parts)),
    )


def _agent_section(skill: Skill) -> PromptSection:
    """The section for the task: the Agent's docstring (if any) followed by a
    subsection for every Skill sharing the current history (an Agent's
    methods, or just ``skill`` for a free-function skill)."""
    inst = (
        skill.__default__.__self__
        if isinstance(skill.__default__, types.MethodType)
        else None
    )
    if isinstance(inst, Agent):
        agent_doc = inspect.getdoc(type(inst)) or ""
        title = f"Agent `{_get_qualname(type(inst))}`"
        skills = set()
        for cls in type(inst).__mro__:
            for attr in vars(cls):
                try:
                    value = getattr(inst, attr)
                except Exception:
                    continue
                if isinstance(value, Skill):
                    skills.add(value)
    else:
        agent_doc = ""
        title = "Skill"
        skills = {skill}

    # The agent docstring is intro prose for the section, ahead of the specs;
    # order the specs by name so the prompt is stable across method reordering
    # in source.
    return PromptSection(
        type="prompt_section",
        title=title,
        content=[
            *to_content_blocks(agent_doc),
            *(_skill_section(t) for t in sorted(skills, key=lambda t: t.__name__)),
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


def _tools_in_scope(
    env: collections.abc.Mapping[str, typing.Any],
    *,
    seen: frozenset[int] = frozenset(),
) -> collections.abc.Set[Tool]:
    """
    Return the tools available to a Skill given its lexical context.

    Default rule: `Tool` and `Skill` values bound directly in `env`, plus
    those reachable through any `Agent` instance in `env` -- whatever is bound
    on the instance or declared on its class, and, recursively, the tools of any
    `Agent` those in turn hold.  `seen` guards that recursion against reference
    cycles; it is internal, and callers pass only `env`.

    Reaching through a nested `Agent` flattens its whole toolset into the result.
    That is what makes holding one as an attribute a way to compose tools, and it
    is why a specialised sub-agent is better left a bare `Skill`, whose own
    scope stays its own.

    Tools are identified by object, so the same `Tool` visible under several
    bindings appears once.  The name each one is offered under is assigned by
    :func:`_advertised_names`, not taken from the binding name.
    """
    result: set[Tool] = set()

    for name, obj in env.items():
        if not name.isidentifier():
            continue
        if isinstance(obj, Tool | Skill):
            result.add(obj)
        elif isinstance(obj, Agent) and id(obj) not in seen:
            seen |= {id(obj)}
            result |= _tools_in_scope(
                vars(obj)
                | {k: getattr(obj, k) for cls in type(obj).__mro__ for k in vars(cls)},
                seen=seen,
            )

    return frozenset(result)
