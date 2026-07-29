import abc
import builtins
import collections
import collections.abc
import contextlib
import dataclasses
import functools
import inspect
import json
import re
import traceback
import types
import typing
import uuid

import litellm
import pydantic

from effectful.handlers.llm.harness.encoding import (
    _TOOLS_KEY,
    REPL_ANCHOR_KEY,
    TYPE_CHECK_ANCHOR_KEY,
    DecodedToolCall,
    format_as_content_blocks,
    to_content_blocks,
)
from effectful.handlers.llm.types import (
    Agent,
    Encodable,
    FinalTool,
    Template,
    Tool,
)
from effectful.internals.unification import nested_type
from effectful.ops.semantics import fwd, handler
from effectful.ops.types import Operation


class AssistantMessage(litellm.OpenAIChatCompletionAssistantMessage):
    id: str


class ToolMessage(litellm.ChatCompletionToolMessage):
    id: str


class FunctionMessage(litellm.ChatCompletionFunctionMessage):
    id: str


class SystemMessage(litellm.OpenAIChatCompletionSystemMessage):
    id: str


class UserMessage(litellm.OpenAIChatCompletionUserMessage):
    id: str


Message = AssistantMessage | ToolMessage | FunctionMessage | SystemMessage | UserMessage


class _NoActiveHistoryException(Exception):
    """Raised when there is no active message history to append to."""


@Operation.define
def _get_history() -> collections.OrderedDict[str, Message]:
    raise _NoActiveHistoryException(
        "No active message history. This operation should only be used within a handler that provides a message history."
    )


def append_message(message: Message, last: bool = True) -> None:
    try:
        _get_history()[message["id"]] = message
        if not last:
            _get_history().move_to_end(message["id"], last=False)
    except _NoActiveHistoryException:
        pass


def _make_message(content: dict) -> Message:
    m_id = content.get("id") or str(uuid.uuid1())
    message = typing.cast(Message, {**content, "id": m_id})
    return message


class DecodingError[E: Exception](abc.ABC, Exception):
    """Base class for decoding errors that can occur during LLM response processing."""

    original_error: E

    @abc.abstractmethod
    def to_feedback_message(self, include_traceback: bool) -> Message:
        """Convert the decoding error into a feedback message to be sent back to the LLM."""
        raise NotImplementedError


@dataclasses.dataclass
class ToolCallDecodingError[E: Exception](DecodingError[E]):
    """Error raised when decoding a tool call fails."""

    original_error: E
    raw_message: Message
    raw_tool_call: litellm.ChatCompletionMessageToolCall

    def __str__(self) -> str:
        return f"Error decoding tool call '{self.raw_tool_call.function.name}': {self.original_error}. Please provide a valid response and try again."

    def to_feedback_message(self, include_traceback: bool) -> Message:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return _make_message(
            {
                "role": "tool",
                "tool_call_id": self.raw_tool_call.id,
                "content": error_message,
            },
        )


@dataclasses.dataclass
class ResultDecodingError[E: Exception](DecodingError[E]):
    """Error raised when decoding the LLM response result fails."""

    original_error: E
    raw_message: Message

    def __str__(self) -> str:
        return f"Error decoding response: {self.original_error}. Please provide a valid response and try again."

    def to_feedback_message(self, include_traceback: bool) -> Message:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return _make_message(
            {"role": "user", "content": error_message},
        )


@dataclasses.dataclass
class ToolCallExecutionError[E: Exception, T](DecodingError[E]):
    """Error raised when a tool execution fails at runtime."""

    original_error: E
    raw_tool_call: DecodedToolCall[T]

    def __str__(self) -> str:
        return f"Tool execution failed: Error executing tool '{self.raw_tool_call.name}': {self.original_error}"

    def to_feedback_message(self, include_traceback: bool) -> Message:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return _make_message(
            {
                "role": "tool",
                "tool_call_id": self.raw_tool_call.id,
                "content": error_message,
            },
        )


type MessageResult[T] = tuple[Message, typing.Sequence[DecodedToolCall], T | None]


@Operation.define
@functools.wraps(litellm.completion)
def completion(*args, **kwargs) -> typing.Any:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    """
    return litellm.completion(*args, **kwargs)


class _BoxedResponse[T](pydantic.BaseModel):
    value: T


type AssistantResult[T] = tuple[Message, typing.Sequence[DecodedToolCall], T | None]


@Operation.define
def call_assistant[T](
    env: collections.abc.Mapping[str, typing.Any],
    response_type: type[T],
    tools: collections.abc.Set[Tool] = frozenset(),
    anchor: "Template | None" = None,
    force_tool: bool = False,
) -> AssistantResult[T]:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    The available `tools` are passed explicitly as a set; handlers that expose
    additional tools (synthetic readers, REPL access, synthesis) intercept this
    operation and union them into `tools` before forwarding.  Each tool's
    model-visible name is derived from its `__name__`, so collection and
    decoding agree on a single naming scheme.

    `force_tool` is set when the request requires the model to call a tool (the
    provider derives it from a ``tool_choice="required"`` config) so that a
    response which nonetheless comes back with no tool call — some
    OpenAI-compatible servers treat ``tool_choice`` as advisory — is reported as
    the protocol violation it is, rather than being misdecoded as a bare
    structured result.

    Raises:
        ToolCallDecodingError: If a tool call cannot be decoded. The error
            includes the raw assistant message for retry handling.
        ResultDecodingError: If the result cannot be decoded. The error
            includes the raw assistant message for retry handling.
    """
    name2tool = {t.__name__: t for t in tools}
    assert len(tools) == len(name2tool), "Tool name collision detected"
    env = {_TOOLS_KEY: name2tool, REPL_ANCHOR_KEY: anchor, **env}
    tool_specs = []
    for name, t in sorted(name2tool.items()):
        spec = typing.cast(
            pydantic.TypeAdapter[typing.Any],
            pydantic.TypeAdapter(Encodable[type(t)]),  # type: ignore[misc]
        ).dump_python(t, mode="json", context={name: t})
        tool_specs.append(spec)

    # The OpenAI API requires a wrapper object for non-object structured output types,
    # so we create one on the fly here. Using a Pydantic model offloads JSON schema
    # generation and validation logic to litellm, and offers better error messages.
    response_format: type[_BoxedResponse[T]] = pydantic.create_model(
        "BoxedResponse",
        value=Encodable[response_type],  # type: ignore[valid-type]
        __base__=_BoxedResponse,
    )

    response: litellm.types.utils.ModelResponse = completion(
        messages=list(_get_history().values()),
        response_format=None if response_type is str else response_format,
        tools=tool_specs,
    )
    choice = response.choices[0]
    assert isinstance(choice, litellm.types.utils.Choices)

    message: litellm.Message = choice.message
    assert message.role == "assistant"

    raw_message = _make_message({**message.model_dump(mode="json")})
    append_message(raw_message)

    raw_tool_calls = message.get("tool_calls") or []
    if force_tool and not raw_tool_calls:
        raise ResultDecodingError(
            ValueError(
                "tool_choice='required' but the model returned no tool call."
                "**IMPORTANT: YOU MUST GENERATE A TOOL CALL IN YOUR NEXT RESPONSE.**"
            ),
            raw_message=raw_message,
        )

    tool_calls: list[DecodedToolCall] = []
    encoding: pydantic.TypeAdapter[DecodedToolCall] = pydantic.TypeAdapter(
        Encodable[DecodedToolCall]
    )
    for raw_tool_call in raw_tool_calls:
        try:
            tool_calls += [encoding.validate_python(raw_tool_call, context=env)]
            if isinstance(tool_calls[-1].tool, FinalTool):
                if not (
                    tool_calls[-1].result_type == response_type
                    or issubclass(tool_calls[-1].result_type, response_type)
                ):
                    raise TypeError(
                        f"FinalTool '{tool_calls[-1].name}' returns {tool_calls[-1].result_type!r}, "
                        f"which does not match the Template's result type {response_type!r}."
                    )
                if len(raw_tool_calls) > 1:
                    raise TypeError(
                        f"A FinalTool call must be the only tool call in its turn, but "
                        f"{len(raw_tool_calls)} tool calls were requested."
                    )
        except Exception as e:
            raise ToolCallDecodingError(
                raw_tool_call=raw_tool_call,
                original_error=e,
                raw_message=raw_message,
            ) from e

    result = None
    if not tool_calls:
        # return response
        serialized_result = message.get("content") or message.get("reasoning_content")
        assert isinstance(serialized_result, str), (
            "final response from the model should be a string"
        )
        if response_type is str:
            result = typing.cast(T, serialized_result)
        else:
            try:
                # Add the type-check anchor to the decode context only (not `env`,
                # which is exposed as tools), so a synthesized result is checked
                # against the Template's source.
                result = response_format.model_validate(
                    json.loads(serialized_result),
                    context={**env, TYPE_CHECK_ANCHOR_KEY: anchor},
                ).value
            except Exception as e:
                raise ResultDecodingError(e, raw_message=raw_message) from e

    return (raw_message, tool_calls, result)


type ToolResult[T] = tuple[Message, T | None, bool]


@Operation.define
def call_tool[T](tool_call: DecodedToolCall[T]) -> ToolResult[T]:
    """Implements a roundtrip call to a python function. Input is a json
    string representing an LLM tool call request parameters. The output is
    the serialised response to the model.

    Returns the appended tool message, the tool's return value, and whether the
    call was a finalizing one (a :class:`FinalTool` call, whose value becomes the
    Template's result and terminates the completion loop).
    """
    # call tool with python types
    try:
        result = tool_call.tool(
            *tool_call.bound_args.args, **tool_call.bound_args.kwargs
        )
    except Exception as e:
        raise ToolCallExecutionError(raw_tool_call=tool_call, original_error=e) from e

    return_type: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
        Encodable[nested_type(result).value]  # type: ignore[misc]
    )
    encoded_result = to_content_blocks(
        return_type.dump_python(result, mode="json", context={})
    )
    message = _make_message(
        dict(role="tool", content=encoded_result, tool_call_id=tool_call.id),
    )
    append_message(message)
    return (message, result, isinstance(tool_call.tool, FinalTool))


@Operation.define
def call_user(
    template: Template,
    env: collections.abc.Mapping[str, typing.Any],
) -> Message:
    """
    Format a `Template`'s prompt applied to arguments into a user message.

    The prompt is the template's header (``name(signature)``, with braces
    escaped so it is not itself formatted) followed by its docstring; its
    ``{...}`` fields are filled from `env`.
    """
    assert template.__default__.__doc__ is not None
    header = f"{template.__name__}{template.__signature__}".replace("{", "{{").replace(
        "}", "}}"
    )
    prompt = f"{header}\n\n{template.__default__.__doc__}"
    parts = format_as_content_blocks(prompt, env)
    message = _make_message(dict(role="user", content=parts))
    append_message(message)
    return message


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


@Operation.define
def call_system(
    template: Template, *, tool_types: collections.abc.Set[type[Tool]] = frozenset()
) -> Message:
    """Assemble and install the system message (a Markdown document)."""
    sections = [
        _system_global_block(tool_types),
        _system_module_block(inspect.getmodule(template)),
        _system_agent_block(template),
        _system_imports_block(template.__context__),
        _system_vars_block(template.__context__),
    ]
    content = "\n\n".join(s for s in sections if s)
    message = _make_message(
        dict(role="system", content=content, cache_control={"type": "ephemeral"})
    )
    append_message(message, last=False)
    return message


def new_agent_call_scope():
    """Create an independent, per-agent nesting tracker.

    Returns a context manager `scope(agent_id)` yielding whether this is the
    outermost call for `agent_id` on the current call stack (`agent_id=None`
    always yields `True`, with no tracking installed -- for callers with no
    agent to key on). Each call to this factory produces its own private
    `Operation`, so composed handlers that each need their own "am I
    outermost at my layer" notion (e.g. `LiteLLMProvider`'s history
    write-back tracking and `SQLitePersister`'s checkpoint tracking,
    which sit at different layers in the handler stack) don't interfere
    with each other: installing a marker for agent X at one layer never
    makes a *different* agent, or the *same* agent at a *different* layer,
    look nested.
    """

    @Operation.define
    def _active(agent_id: str) -> bool:
        """Whether a call for `agent_id` is already in progress at this scope's layer."""
        return False

    @contextlib.contextmanager
    def scope(agent_id: str | None):
        if agent_id is None:
            yield True
            return
        is_outermost = not _active(agent_id)
        with handler({_active: lambda aid, _id=agent_id: aid == _id or fwd(aid)}):
            yield is_outermost

    return scope
