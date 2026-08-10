import builtins
import collections.abc
import inspect
import json
import re
import types
import typing

import pydantic

from effectful.handlers.llm.harness.hooks import (
    AssistantResult,
    call_assistant,
    call_system,
)
from effectful.handlers.llm.types import Agent, Encodable, Template, Tool
from effectful.internals.unification import nested_type
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class LexicalReaders(ObjectInterpretation):
    """Intercept `call_assistant` to also expose plain values from the
    lexical context as zero-argument read-only Tools.  Each non-Tool,
    non-Template, non-Agent value bound to a valid identifier is
    wrapped via `_LexicalVariableTool` if `Encodable[T]` accepts it;
    schema-generation failures cause the symbol to be skipped.
    """

    @typing.final
    class _LexicalVariableTool[T](Tool[[], T]):
        """## Reading lexical variables

        Some of the tools below take no arguments and simply return the current
        value of a named variable from this Template's lexical scope (see the
        *Lexical scope* table for the available names and their types). Call such a
        reader when your answer depends on the concrete value of an in-scope
        variable that has not already been spliced into the prompt — it lets you
        fetch that value on demand instead of guessing it. Each reader's description
        names the variable it reads.
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
    def _call_system(self, template, tool_types=frozenset()):
        return fwd(template, tool_types=tool_types | {self._LexicalVariableTool})

    @implements(call_assistant)
    def _call_assistant[T](
        self,
        env: collections.abc.Mapping[str, typing.Any],
        response_type: type[T],
        tools: collections.abc.Set[Tool] = frozenset(),
        force_tool: bool = False,
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
        return fwd(env, response_type, readers, force_tool=force_tool)


def _tools_in_scope(
    env: collections.abc.Mapping[str, typing.Any],
) -> collections.abc.Set[Tool]:
    """Return the tools available to a Template given its lexical context.

    Default rule: real `Tool` and `Template` values bound directly in
    `env`, plus `Tool` methods discovered through the MRO of any
    `Agent` instance in `env`.

    Tools are identified by object, so the same `Tool` visible under
    several bindings appears once.  Names are derived from each tool's
    `__name__` by :func:`call_assistant`, not from the binding name.
    """
    result: set[Tool] = set()

    for obj in env.values():
        if isinstance(obj, Tool | Template):
            result.add(obj)
        elif isinstance(obj, Agent):
            for cls in type(obj).__mro__:
                for attr_name in vars(cls):
                    attr = getattr(obj, attr_name)
                    if isinstance(attr, Tool):
                        result.add(attr)

    return result


def _get_qualname(cls) -> str:
    """Module-qualified name of a type, dropping the ``builtins`` prefix."""
    if not isinstance(cls, type):
        return str(cls)
    module = getattr(cls, "__module__", None)
    name = (
        getattr(cls, "__qualname__", None) or getattr(cls, "__name__", None) or str(cls)
    )
    return name if module in (None, "builtins") else f"{module}.{name}"


# Matches an ATX heading's leading ``#``s (1-6, followed by whitespace) at the
# start of a line, e.g. ``## Foo``. The lookahead avoids matching ``#!`` or a
# ``#tag`` that is not a heading.
_ATX_HEADING = re.compile(r"^(#{1,6})(?=\s)")


def _shift_headings(md: str, by: int) -> str:
    """Shift every ATX heading in `md` by `by` levels (clamped to 1..6).

    Fenced code blocks (``` ``` ``` / ``` ~~~ ```) are skipped so ``#`` inside code --
    Python comments, shell shebangs -- is left untouched.
    """
    if by == 0 or not md:
        return md
    out: list[str] = []
    fence: str | None = None
    for line in md.splitlines():
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
        elif fence is not None and stripped.startswith(fence):
            fence = None
        elif fence is None:
            m = _ATX_HEADING.match(line)
            if m:
                level = max(1, min(6, len(m.group(1)) + by))
                line = "#" * level + line[m.end(1) :]
        out.append(line)
    return "\n".join(out)


def _rebase_headings(md: str, top: int) -> str:
    """Renumber the headings in `md` so its shallowest one sits at level `top`,
    preserving relative nesting; text with no headings is returned unchanged.

    Used to nest a docstring that was authored with its own ``##``-rooted
    heading hierarchy beneath a deeper section heading when the system prompt is
    assembled, so the composed document has a single coherent outline.
    """
    if not md:
        return md
    fence: str | None = None
    levels: list[int] = []
    for line in md.splitlines():
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
        elif fence is not None and stripped.startswith(fence):
            fence = None
        elif fence is None:
            m = _ATX_HEADING.match(line)
            if m:
                levels.append(len(m.group(1)))
    if not levels:
        return md
    return _shift_headings(md, top - min(levels))


def _section(title: str, body: str) -> str:
    """Wrap `body` as a top-level ``# title`` section, or ``""`` if body is empty.

    Callers pass a `body` whose own headings already start at ``##`` (rebasing
    incorporated docstrings with `_rebase_headings` as needed), so every section
    is a self-contained subtree rooted at its ``#`` heading.
    """
    body = body.strip()
    return f"# {title}\n\n{body}" if body else ""


def _system_vars_block(env: collections.abc.Mapping[str, typing.Any]) -> str:
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
        return ""
    body = "\n".join(f"| `{n}` | `{t}` |" for n, t in sorted(rows.items()))
    return _section("Lexical scope", f"| name | type |\n| --- | --- |\n{body}")


def _system_imports_block(env: collections.abc.Mapping[str, typing.Any]) -> str:
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
        return ""
    body = "\n".join(f"| `{n}` | `{m}` |" for n, m in sorted(rows.items()))
    return _section("Imported modules", f"| name | module |\n| --- | --- |\n{body}")


def _system_template_block(template: Template) -> str:
    """Markdown spec for a single `Template`: header, prompt, arg schemas.

    Emitted at ``##`` so each template reads as a subsection of the enclosing
    agent/template ``#`` section (see `_system_agent_block`).
    """
    parts = [f"## `{template.__name__}{template.__signature__}`"]
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
    return "\n\n".join(parts)


def _system_agent_block(template: Template) -> str:
    """The ``#`` section for the task: the Agent's docstring (if any) followed by
    a ``##`` spec for every Template sharing the current history (an Agent's
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

    # Order by name so the prompt is stable across method reordering in source.
    specs = "\n\n".join(
        _system_template_block(t) for t in sorted(templates, key=lambda t: t.__name__)
    )
    # The agent docstring is intro prose for the section; rebase its own headings
    # to sit at ``##`` alongside the per-template specs.
    body = "\n\n".join(p for p in [_rebase_headings(agent_doc, 2), specs] if p)
    return _section(title, body)


def _system_module_block(mod: types.ModuleType | None) -> str:
    """The ``#`` section carrying the source (or docstring fallback) of a module."""
    if mod is None:
        return ""
    try:
        src = inspect.getsource(mod)
        body = f"```python\n{src}\n```"
    except (OSError, TypeError):
        doc = inspect.getdoc(mod)
        if not doc:
            return ""
        body = _rebase_headings(doc, 2)
    return _section(f"Module `{mod.__name__}`", body)


def _system_global_block(tool_types: collections.abc.Set[type[Tool]]) -> str:
    """The constant ``#`` framework-concept section, sourced from real docstrings.

    The module overview and each concept nest as ``##`` subsections. Core
    concept classes carry a synthesized ``## `Name``` heading and their own
    docstring subsections are demoted to ``###``; the synthetic tool docstrings
    already open with a descriptive ``##`` heading, so they are used verbatim
    (rebased if needed) rather than labelled with their private class names.
    """
    import effectful.handlers.llm as _llm

    assert all(issubclass(t, Tool) and t not in {Tool, Template} for t in tool_types)
    parts = [_rebase_headings(inspect.getdoc(_llm) or "", 2)]
    for typ in sorted(
        map(lambda name: getattr(_llm, name), _llm.__all__), key=_get_qualname
    ):
        parts += [
            f"## `{_get_qualname(typ)}`\n\n{_rebase_headings(inspect.getdoc(typ) or '', 3)}"
        ]
    for t in sorted(tool_types, key=_get_qualname):
        parts += [_rebase_headings(inspect.getdoc(t) or "", 2)]
    body = "\n\n".join(p for p in parts if p.strip())
    return _section("The effectful LLM framework", body)
