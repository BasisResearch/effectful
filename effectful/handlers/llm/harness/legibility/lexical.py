"""Making a `Skill`'s lexical scope legible to the model.

`LexicalToolExtractor` unions `_tools_in_scope(env)` into a request's `tools`
and forwards, so a `Skill` is offered the `Tool`/`Skill` values bound in its
context (and those its in-scope `Agent`\\ s hold) without naming them itself.
The section builders it shares with the rest of the harness render the
surrounding scope as prompt tables.

Install `LexicalToolExtractor` below anything else that contributes tools: the
anchor `Skill` is dropped from the set by `call_assistant`'s default rule, so
this must be the innermost `call_assistant` handler for that subtraction to see
it. Without it a `Skill` is simply offered no lexical tools -- the request is
well-formed either way, so the omission surfaces only as a model that never
calls a tool it was supposed to have.

A *polymorphic* tool advertises through `LexicalToolExtractor` in degraded form:
its TypeVar-carrying parameters render as untyped ``{}`` schemas and its JSON
arguments cannot be decoded to typed values (issues #489/#505). Polymorphic
tools, and tools whose advertisement cannot be encoded at all, are both fully
supported by the code-generation pathway
(`~effectful.handlers.llm.harness.synthesis.toolcall.ExpressionToolCaller`),
which this handler is the JSON alternative to.
"""

import builtins
import collections.abc
import inspect
import json
import logging
import types
import typing

import pydantic

from effectful.handlers.llm.harness.hooks import (
    AssistantResult,
    Message,
    PromptInjectingInterpretation,
    call_assistant,
)
from effectful.handlers.llm.harness.serialization import (
    PromptSection,
    _best_effort_schema,
    _NameAndTool,
    to_content_blocks,
)
from effectful.handlers.llm.types import Agent, Encodable, Skill, Tool
from effectful.ops.semantics import fwd
from effectful.ops.syntax import implements
from effectful.ops.types import INSTANCE_OP_PREFIX

logger = logging.getLogger(__name__)


class LexicalToolExtractor(PromptInjectingInterpretation):
    """The tools you are offered are the ones this Skill can actually reach:
    the `Tool` and `Skill` values bound in its lexical scope, plus those held by
    any `Agent` in that scope. Nobody chose them for you by hand -- they are
    what the surrounding code has in view -- so the set is worth reading as
    evidence of what the caller expects this task to need.

    That has a practical consequence: a capability you might expect is missing
    from the list because it is not in scope here, not because it is forbidden.
    Do not try to name or invoke a tool that is not offered. If the work seems
    to require one, do what you can with what is offered and say plainly what
    was missing.

    The *Lexical scope* and *Imported modules* tables list the same scope's
    non-callable bindings, so the tools and those tables describe one
    environment together.
    """

    @implements(call_assistant)
    def call_assistant(
        self,
        messages: collections.abc.Sequence[Message],
        response_type: type,
        env: collections.abc.Mapping[str, typing.Any],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult:
        """Union the in-scope tools into the request, skipping unencodable ones.

        Each candidate's *advertisement* is probed before it is offered. A tool
        whose parameters have no `Encodable` schema would otherwise fail the
        encoding of the whole request, taking down every call it happens to be
        in scope for; skipping it with a warning keeps the failure proportional
        to the tool, and it remains callable through
        `~effectful.handlers.llm.harness.synthesis.toolcall.ExpressionToolCaller`,
        which does not need a JSON schema.
        """
        offered: set[Tool] = set(tools)
        probe: pydantic.TypeAdapter[_NameAndTool] = pydantic.TypeAdapter(
            Encodable[_NameAndTool]
        )
        for tool in _tools_in_scope(env):
            try:
                probe.dump_python(_NameAndTool(tool.__name__, tool), mode="json")
            except Exception:
                logger.warning(
                    "skipping lexical tool %r: its advertisement cannot be "
                    "encoded; it remains callable via ExpressionToolCaller",
                    tool.__name__,
                    exc_info=True,
                )
                continue
            offered.add(tool)
        return fwd(messages, response_type, env, offered)


def _binding_type(value: typing.Any) -> str:
    """How to describe a binding in the lexical scope table.

    A binding that *is* a type -- a class, a ``type`` alias, a parameterized
    generic, a union, a special form -- is described as ``type[X]``, naming what
    it denotes.  Its own type, which is what every other binding is described
    by, is a metaclass here, and a metaclass says nothing a reader could use:
    described that way every dataclass in scope is ``type``, every enum
    ``enum.EnumType``, and every PEP 695 alias ``typing.TypeAliasType`` -- which
    `inspect.formatannotation`, meant for annotations rather than classes,
    renders as the raw ``<class 'TypeAliasType'>``.
    """
    if (
        isinstance(value, type | types.GenericAlias | types.UnionType)
        # Type aliases, special forms and type variables are not instances of
        # any of those, and are only recognizable by where their type lives.
        or type(value).__module__ == "typing"
    ):
        return f"type[{inspect.formatannotation(value)}]"
    return inspect.formatannotation(type(value))


def _vars_section(
    env: collections.abc.Mapping[str, typing.Any],
) -> PromptSection | None:
    """Markdown table of the non-module bindings in scope (name -> type).

    Excludes dunder names (``__main__`` etc.) and names already bound to their
    standard builtin (which the model knows).

    Built from the Skill's *lexical context* and nothing else -- in particular
    not from the arguments of a call, which are bound in the same namespace but
    vary call to call. This section lives in the system message, which is written
    once per conversation and must therefore hold still. The arguments are named
    per call by the request's own heading instead, which
    `~effectful.handlers.llm.harness.synthesis.snippet.StatefulReplSynthesizer`
    points the model at.
    """
    rows = {
        name: _binding_type(value)
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
        title="Variables already in scope that do not need to be re-defined:",
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
        title="Modules already in scope that do not need to be re-imported:",
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
        f"- `{name}` — `{inspect.formatannotation(p.annotation)}`\n\n"
        f"    ```json\n    {json.dumps(_best_effort_schema(p.annotation, 'validation'))}\n    ```"
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
        title = f"Agent `{inspect.formatannotation(type(inst))}`"
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
    return frozenset(_tool_paths(env, seen=seen))


def _tool_paths(
    env: collections.abc.Mapping[str, typing.Any],
    *,
    seen: frozenset[int] = frozenset(),
) -> dict[Tool, str]:
    """The tools of `_tools_in_scope`, each mapped to the expression that names it.

    Same reachability rule as `_tools_in_scope` (which is defined in terms of
    this), but remembering *how* each tool was reached: ``name`` for a `Tool`
    bound directly in ``env``, ``name.attr`` (recursively) for one held by an
    in-scope `Agent`. This is the reference a code-writing model must use to
    call the tool -- ``self.retrieve(...)`` for a method tool of the Skill's
    own agent, bare ``story_funny(...)`` for a module-level skill -- and
    `~effectful.handlers.llm.harness.synthesis.toolcall.ExpressionToolCaller`
    puts it in each wrapper's advertisement so the model need not discover the
    distinction by trial and error.

    A tool visible under several bindings keeps the first path found, in
    ``env`` order -- so for a Skill call, whose env lists bound arguments
    before the enclosing context, a tool on the receiver is named through
    ``self`` rather than through some outer alias.
    """
    paths: dict[Tool, str] = {}

    for name, obj in env.items():
        # `vars(agent)` also lists each cached instance-bound operation under
        # its internal `__instanceop_*` key -- the same object `getattr` hands
        # back for the public attribute name, so skipping the mangled alias
        # loses no tool and keeps the path the one a model can actually write.
        if not name.isidentifier() or name.startswith(INSTANCE_OP_PREFIX):
            continue
        if isinstance(obj, Tool | Skill):
            paths.setdefault(obj, name)
        elif isinstance(obj, Agent) and id(obj) not in seen:
            seen |= {id(obj)}
            sub_env = vars(obj) | {
                k: getattr(obj, k) for cls in type(obj).__mro__ for k in vars(cls)
            }
            for tool, sub_path in _tool_paths(sub_env, seen=seen).items():
                paths.setdefault(tool, f"{name}.{sub_path}")

    return paths
