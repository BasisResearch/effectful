"""Making a `Skill`'s lexical scope legible to the model.

`LexicalToolExtractor` unions `_tools_in_scope(env)` into a request's `tools`
and forwards, so a `Skill` is offered the `Tool`/`Skill` values bound in its
context (and those its in-scope `Agent`\\ s hold) without naming them itself.
`ImplicitToolExtractor` widens that discovery: ordinary functions and methods
in scope that look deliberately published -- public name, docstring, complete
annotations (see `ImplicitToolExtractor._implicit_tool_candidate`) -- are
wrapped with `Tool.define` and offered too, no decorator required. The section
builders this module shares with the rest of the harness render the
surrounding scope as prompt tables.

The extractors are the *discovery* stage of the tool pipeline, and the only
one: the tool callers in
`~effectful.handlers.llm.harness.synthesis.toolcall` *transform* the tools an
extractor discovered (replacing lexical tools with expression wrappers) rather
than re-walking the scope themselves. Install exactly one extractor per stack,
below the tool caller and anything else that contributes tools, passing
``json_only=False`` whenever a caller is installed above (see
`LexicalToolExtractor.__init__`): the anchor `Skill` is dropped from the set
by `call_assistant`'s default rule, and a tool caller without an extractor
beneath it has no lexical tools to offer -- the request is well-formed either
way, so the omission surfaces only as a model that never calls a tool it was
supposed to have.

A *polymorphic* tool advertises in degraded form under the JSON pathway: its
TypeVar-carrying parameters render as untyped ``{}`` schemas and its JSON
arguments cannot be decoded to typed values (issues #489/#505). Polymorphic
tools, and tools whose advertisement cannot be encoded at all, are both fully
supported by the code-generation pathway
(`~effectful.handlers.llm.harness.synthesis.toolcall.ExpressionToolCaller`),
which composes above either extractor.
"""

import builtins
import collections.abc
import inspect
import json
import logging
import sys
import types
import typing
import weakref

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

    def __init__(self, json_only: bool = True):
        """``json_only`` says whether this extractor feeds the JSON pathway
        directly, with no tool caller stacked above it.

        When true (the default, and how ``tool_calling="json"`` installs it),
        a discovered tool whose JSON advertisement cannot be encoded is
        dropped, with a warning, before it can fail the encoding of the whole
        request. When a caller *is* installed above
        (`~effectful.handlers.llm.harness.synthesis.toolcall.MixedToolCaller`
        or ``ExpressionToolCaller``), pass ``json_only=False`` -- the caller
        replaces exactly those tools with expression wrappers (which always
        encode), so dropping them here would starve the expression pathway of
        the tools it exists for.
        """
        super().__init__()
        self._json_only = json_only

    def _lexical_tool_paths(
        self, env: collections.abc.Mapping[str, typing.Any]
    ) -> dict[Tool, str]:
        """The lexical tools this handler discovers, each mapped to the
        expression that names it in the Skill's scope.

        The extension point for widening discovery: `ImplicitToolExtractor`
        overrides this to also wrap qualifying plain functions and methods.
        """
        return _tool_paths(env)

    @implements(call_assistant)
    def call_assistant(
        self,
        messages: collections.abc.Sequence[Message],
        response_type: type,
        env: collections.abc.Mapping[str, typing.Any],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult:
        """Union the in-scope tools into the request.

        Discovery: this handler decides *which* tools the Skill's scope
        offers; a tool caller stacked above it may then replace lexical tools
        with expression wrappers. Under ``json_only`` (no caller above), each
        discovered tool's advertisement is probed first, and one that cannot
        be encoded is skipped with a warning -- implicitly wrapped tools (see
        `ImplicitToolExtractor`) log at debug instead, since a blanket scan of
        a scope legitimately picks up functions that don't advertise.
        """
        offered = set(tools) | set(self._lexical_tool_paths(env))
        if self._json_only:
            probe: pydantic.TypeAdapter[_NameAndTool] = pydantic.TypeAdapter(
                Encodable[_NameAndTool]
            )
            for tool in offered - set(tools):
                try:
                    probe.dump_python(_NameAndTool(tool.__name__, tool), mode="json")
                except Exception:
                    logger.log(
                        logging.DEBUG
                        if hasattr(tool, "__implicit_target__")
                        else logging.WARNING,
                        "skipping lexical tool %r: its advertisement cannot be "
                        "encoded; it remains callable via ExpressionToolCaller "
                        "(install one above an extractor with json_only=False)",
                        tool.__name__,
                        exc_info=True,
                    )
                    offered.discard(tool)
        return fwd(messages, response_type, env, frozenset(offered))


class ImplicitToolExtractor(LexicalToolExtractor):
    """The tools you are offered are the ones this Skill can actually reach in
    the surrounding code: the `Tool` and `Skill` values bound in its lexical
    scope, those held by any `Agent` in that scope -- and, beyond the ones
    declared as tools, the ordinary functions and methods of that scope that
    are public, documented, and fully type-annotated, wrapped and offered as
    tools automatically. Nobody chose them by hand: the set is what the
    surrounding code has in view, so read it as evidence of what the caller
    expects this task to need, and read each tool's own docstring as its
    contract.

    That has a practical consequence: a capability you might expect is missing
    from the list because it is not in scope here (or is private, undocumented,
    or unannotated), not because it is forbidden. Do not try to name or invoke
    a tool that is not offered. If the work seems to require one, do what you
    can with what is offered and say plainly what was missing.

    The *Lexical scope* and *Imported modules* tables list the same scope's
    non-callable bindings, so the tools and those tables describe one
    environment together.
    """

    def __init__(
        self,
        predicate: collections.abc.Callable[[types.FunctionType], bool] | None = None,
        json_only: bool = True,
    ):
        super().__init__(json_only=json_only)
        self._predicate = predicate
        # fn -> its class-level wrapper Tool (or None for a rejected fn), so
        # the same function maps to the same Tool object across requests
        # through this handler -- downstream handlers resolve calls against
        # the offered set by identity. Instance-owned: the cache dies with
        # the handler.
        self._wrapped: weakref.WeakKeyDictionary[types.FunctionType, Tool | None] = (
            weakref.WeakKeyDictionary()
        )
        # Classmethods bind to a *class*, and subclasses share the underlying
        # function while needing distinct bound wrappers -- so their cache is
        # keyed per (function, bound class). See `_wrap_bound_classmethod`.
        self._wrapped_classmethods: weakref.WeakKeyDictionary[
            types.FunctionType, weakref.WeakKeyDictionary[type, Tool | None]
        ] = weakref.WeakKeyDictionary()

    _SKIPPED_MODULES: typing.ClassVar[frozenset[str]] = frozenset(
        [
            "builtins",
            "effectful",
            "pydantic",
            "pydantic_core",
            "pytest",
            "annotated_types",
            "typing_extensions",
            *sys.stdlib_module_names,
        ]
    )

    @classmethod
    def _implicit_tool_candidate(cls, obj: typing.Any) -> bool:
        """Whether ``obj`` (a plain function, or a bound method reached through
        an in-scope `Agent`) should be offered as an implicit tool.

        The heuristic errs toward exclusion -- every criterion is a way a
        function signals it was not written to be called by a model:

        - a plain Python function (or a bound method of one): no classes,
          partials, callable instances, or C-level callables;
        - a public, deliberate name: not a lambda, not ``_``/``test_``-prefixed,
          and not ``main`` -- the de facto entry-point name, whose exclusion is
          deliberately unconditional (a ``__name__ == "__main__"`` gate would
          make the same skill's tool set vary with how the process was started);
        - documented: a nonempty docstring (also what `Tool.define` requires);
        - fully annotated, resolvably (`typing.get_type_hints` succeeds), with
          no ``*args``/``**kwargs`` -- for a bound method the signature has
          already dropped ``self`` (or ``cls``, for a classmethod), which is
          how the unannotated receiver of every method is exempted;
        - synchronous: not a coroutine/generator/async-generator function,
          whose call result the JSON pathway could not do anything with;
        - not from the standard library or from ``effectful`` itself.
        """
        fn = obj.__func__ if isinstance(obj, types.MethodType) else obj
        if not isinstance(fn, types.FunctionType):
            return False
        name = fn.__name__
        if not name.isidentifier() or name.startswith(("_", "test_")) or name == "main":
            return False
        if not (fn.__doc__ or "").strip():
            return False
        if (
            inspect.iscoroutinefunction(fn)
            or inspect.isgeneratorfunction(fn)
            or inspect.isasyncgenfunction(fn)
        ):
            return False
        top = (fn.__module__ or "").partition(".")[0]
        if not top or top in cls._SKIPPED_MODULES:
            return False
        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            return False
        if "return" not in hints:
            return False
        return all(
            p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and p.name in hints
            for p in inspect.signature(obj).parameters.values()
        )

    def _wrap(self, obj: typing.Any) -> Tool | None:
        """Wrap a qualifying function or bound method as an implicit `Tool`.

        Each wrapper carries ``__implicit_target__`` -- the raw callable it
        stands for (the function, a bound method of the same instance, or a
        bound classmethod of the same class) -- which is how the expression
        pathway's decoder recognizes a model-written call to the *raw* name as
        a call to this tool (the env binds the raw callable, not the wrapper;
        see `~effectful.handlers.llm.harness.synthesis.toolcall`).

        Staticmethods need no case of their own: instance access hands back
        the plain underlying function, so they take the free-function branch.
        """
        fn = obj.__func__ if isinstance(obj, types.MethodType) else obj
        if not isinstance(fn, types.FunctionType):
            return None
        if isinstance(obj, types.MethodType) and isinstance(obj.__self__, type):
            return self._wrap_bound_classmethod(obj)
        if fn in self._wrapped:
            cls_tool = self._wrapped[fn]
        else:
            ok = self._implicit_tool_candidate(obj) and (
                self._predicate is None or self._predicate(fn)
            )
            cls_tool = typing.cast(Tool, Tool.define(fn)) if ok else None
            self._wrapped[fn] = cls_tool
        if cls_tool is None:
            return None
        if not isinstance(obj, types.MethodType):
            if not hasattr(cls_tool, "__implicit_target__"):
                cls_tool.__implicit_target__ = fn  # type: ignore[attr-defined]
            return cls_tool
        # A bound method: reuse `Operation.__get__`'s instance-op machinery by
        # binding the wrapper's name manually (class creation would normally
        # have done this), so the *bound* tool is created once per instance and
        # cached in the instance's own __dict__ under `_name_on_instance`. If
        # two handler instances wrap the same method, the second reuses the
        # first's cached instance op through that shared key -- an equivalent
        # wrapper of the same function.
        owner = type(obj.__self__)
        if not hasattr(cls_tool, "_name_on_instance"):
            cls_tool.__set_name__(owner, fn.__name__)
        bound = cls_tool.__get__(obj.__self__, owner)
        if not isinstance(bound, Tool):
            return None  # e.g. an instance with free variables; nothing to offer
        if not hasattr(bound, "__implicit_target__"):
            bound.__implicit_target__ = obj  # type: ignore[attr-defined]
        return bound

    def _wrap_bound_classmethod(self, obj: types.MethodType) -> Tool | None:
        """Wrap a qualifying bound classmethod as an implicit `Tool`.

        Symmetric with instance methods, but with its own cache: a classmethod
        binds to the *class*, so `Operation.__get__`'s instance-op caching is
        unusable (a class's ``__dict__`` is a read-only mappingproxy), and
        subclasses share the underlying function while binding it to different
        classes -- so the wrapper wraps the bound method directly and is
        cached per (function, bound class).
        """
        fn, owner = obj.__func__, obj.__self__
        assert isinstance(fn, types.FunctionType) and isinstance(owner, type)
        per_class = self._wrapped_classmethods.get(fn)
        if per_class is None:
            per_class = self._wrapped_classmethods[fn] = weakref.WeakKeyDictionary()
        if owner in per_class:
            return per_class[owner]
        ok = self._implicit_tool_candidate(obj) and (
            self._predicate is None or self._predicate(fn)
        )
        tool: Tool | None = None
        if ok:
            tool = typing.cast(Tool, Tool.define(obj))
            tool.__implicit_target__ = obj  # type: ignore[attr-defined]
        per_class[owner] = tool
        return tool

    def _lexical_tool_paths(
        self, env: collections.abc.Mapping[str, typing.Any]
    ) -> dict[Tool, str]:
        """Explicit tools plus implicit wrappers, minus wrappers that
        duplicate an explicit tool's own callable (a function both bound
        directly in scope and already wrapped by a `Tool` there)."""
        paths = _tool_paths(env, wrap=self._wrap)
        explicit = {
            t.__default__ for t in paths if not hasattr(t, "__implicit_target__")
        }
        # A fresh sentinel can never be in `explicit`, so explicit tools (which
        # have no `__implicit_target__`) always survive the filter.
        missing = object()
        return {
            t: p
            for t, p in paths.items()
            if getattr(t, "__implicit_target__", missing) not in explicit
        }


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
    wrap: collections.abc.Callable[[typing.Any], Tool | None] | None = None,
    seen: frozenset[int] = frozenset(),
) -> collections.abc.Set[Tool]:
    """
    Return the tools available to a Skill given its lexical context.

    Default rule: `Tool` and `Skill` values bound directly in `env`, plus
    those reachable through any `Agent` instance in `env` -- whatever is bound
    on the instance or declared on its class, and, recursively, the tools of any
    `Agent` those in turn hold.  `seen` guards that recursion against reference
    cycles; it is internal, and callers pass only `env`.

    ``wrap``, if given, widens discovery: it is offered every binding that is
    not already a tool or an `Agent` (including, through the `Agent`
    recursion, bound methods), and may return an implicitly wrapped `Tool` for
    it (see `ImplicitToolExtractor`) or ``None`` to pass.

    Reaching through a nested `Agent` flattens its whole toolset into the result.
    That is what makes holding one as an attribute a way to compose tools, and it
    is why a specialised sub-agent is better left a bare `Skill`, whose own
    scope stays its own.

    Tools are identified by object, so the same `Tool` visible under several
    bindings appears once.  The name each one is offered under is assigned by
    :func:`_advertised_names`, not taken from the binding name.
    """
    return frozenset(_tool_paths(env, wrap=wrap, seen=seen))


def _readable_attrs(obj: typing.Any) -> dict[str, typing.Any]:
    """``obj``'s class attributes that could be a tool, bound to ``obj``.

    Bound, because that is the form the rest of this module wants: a `Tool`
    method is a descriptor whose `__get__` produces the instance's operation,
    and a plain method is a candidate for `ImplicitToolExtractor` only once it
    is a bound method. An `Agent` held as a class attribute is not a descriptor
    and is taken as it is.

    Everything else is left alone rather than read, so discovery cannot run a
    property's side effects, and an attribute that refuses instance access --
    a ``pydantic.dataclasses.dataclass`` publishes ``__signature__`` as
    class-only -- is simply never asked for.
    """
    attrs: dict[str, typing.Any] = {}
    for cls in type(obj).__mro__:
        for name, attr in vars(cls).items():
            if name in attrs or name in vars(obj):  # a nearer one already won
                continue
            if isinstance(attr, Agent):
                attrs[name] = attr
            elif isinstance(
                attr, Tool | types.FunctionType | classmethod | staticmethod
            ):
                # Bound to `type(obj)`, not to `cls`: an inherited classmethod
                # takes the instance's class, as attribute access would give it.
                attrs[name] = attr.__get__(obj, type(obj))
    return attrs


def _tool_paths(
    env: collections.abc.Mapping[str, typing.Any],
    *,
    wrap: collections.abc.Callable[[typing.Any], Tool | None] | None = None,
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
            sub_env = vars(obj) | _readable_attrs(obj)
            for tool, sub_path in _tool_paths(sub_env, wrap=wrap, seen=seen).items():
                paths.setdefault(tool, f"{name}.{sub_path}")
        elif wrap is not None and (wrapped := wrap(obj)) is not None:
            paths.setdefault(wrapped, name)

    return paths
