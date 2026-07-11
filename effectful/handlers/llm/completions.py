import abc
import ast
import builtins
import collections
import collections.abc
import dataclasses
import functools
import inspect
import json
import pathlib
import re
import time
import traceback
import types
import typing
import uuid

import langfuse
import litellm
import pydantic
import rich.console
import rich.live
import rich.markdown
import rich.panel
import rich.spinner
import rich.styled
import rich.syntax
import rich.text
import tenacity
from litellm import (
    ChatCompletionFunctionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessage,
    OpenAIChatCompletionAssistantMessage,
    OpenAIChatCompletionSystemMessage,
    OpenAIChatCompletionUserMessage,
)

from effectful.handlers.llm.encoding import (
    _TOOLS_KEY,
    TYPE_CHECK_ANCHOR_KEY,
    DecodedToolCall,
    Encodable,
    _callable_type_from_signature,
    _SynthesisSpec,
    format_as_content_blocks,
    to_content_blocks,
)
from effectful.handlers.llm.evaluation import ReplSession
from effectful.handlers.llm.template import (
    Agent,
    FinalTool,
    Template,
    Tool,
)
from effectful.internals.unification import nested_type
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation


class AssistantMessage(OpenAIChatCompletionAssistantMessage):
    id: str


class ToolMessage(ChatCompletionToolMessage):
    id: str


class FunctionMessage(ChatCompletionFunctionMessage):
    id: str


class SystemMessage(OpenAIChatCompletionSystemMessage):
    id: str


class UserMessage(OpenAIChatCompletionUserMessage):
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
    raw_tool_call: ChatCompletionMessageToolCall

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
    anchor: types.FunctionType | None = None,
) -> AssistantResult[T]:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    The available `tools` are passed explicitly as a set; handlers that expose
    additional tools (synthetic readers, REPL access, synthesis) intercept this
    operation and union them into `tools` before forwarding.  Each tool's
    model-visible name is derived from its `__name__`, so collection and
    decoding agree on a single naming scheme.

    Raises:
        ToolCallDecodingError: If a tool call cannot be decoded. The error
            includes the raw assistant message for retry handling.
        ResultDecodingError: If the result cannot be decoded. The error
            includes the raw assistant message for retry handling.
    """
    name2tool = {t.__name__: t for t in tools}
    assert len(name2tool) == len(tools)
    env = {_TOOLS_KEY: name2tool, **env}
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
    message = _make_message(dict(role="system", content=content))
    append_message(message, last=False)
    return message


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
        return fwd(env, response_type, readers)


class SynthesizeAndCall(ObjectInterpretation):
    """Answer a Template by synthesizing a function and calling it.

    Instead of asking the LLM to generate an instance of the Template's return
    type directly, this handler exposes a :class:`FinalTool` that lets the model
    "answer" by writing a Python function with the Template's signature.  The
    harness applies that function to the original arguments and its return value
    becomes the Template's result.  This is the declarative "CodeAdapt" workflow:
    the LLM writes code implementing the body of the Template rather than
    reasoning out the answer itself.

    The synthesis tool is offered *alongside* the Template's normal completion
    paths rather than replacing them: across turns the model may freely call any
    other tool in scope (their results are fed back as usual), and it may still
    answer the return type directly via structured output.  The loop terminates
    when it either answers directly or calls the synthesis :class:`FinalTool`.
    To force the synthesis path, pass ``tool_choice="required"`` (handler config
    is forwarded to the model request).  The function is synthesized by reusing
    the existing ``Callable`` synthesis machinery: the tool's argument is typed
    as ``Callable[[params], ret]``, so :func:`call_assistant`'s tool-call
    decoding parses, type-checks, compiles and executes the model's code into a
    real function before it is applied.

    Failures compose with :class:`RetryLLMHandler`: a function that fails to
    synthesize surfaces as a :class:`ToolCallDecodingError`, and one that raises
    when applied to the inputs as a :class:`ToolCallExecutionError`; both are fed
    back to the model as a tool message and the loop continues so it can revise::

        with (
            handler(LiteLLMProvider(model="gpt-5-mini")),
            handler(SynthesizeAndCall()),
            handler(RetryLLMHandler()),
        ):
            ...

    Requires an eval provider (e.g. :class:`UnsafeEvalProvider` or
    :class:`RestrictedEvalProvider`) to be installed so the synthesized code can
    be compiled and executed.
    """

    @typing.final
    class _SynthesisFinalTool[T](FinalTool[[collections.abc.Callable[..., T]], T]):
        """## Code synthesis

        You may "answer" a Template by writing code instead of producing the value
        directly. A final tool (typically `submit_solution`) accepts a single
        argument: a Python function whose signature matches the Template's signature
        (see its spec below). The harness applies that function to the original
        inputs and its return value becomes the answer, so write the function body
        as a drop-in implementation of the Template. The function may reference
        names from the lexical scope (see the *Lexical scope* table).

        Give the function a docstring containing `>>>` doctests that demonstrate
        its intended behavior on examples. On submission the harness runs those
        doctests: a solution whose doctests fail (or that errors when applied) is
        rejected and fed back to you to revise, so the answer only stands once the
        function's own doctests pass. Calling this tool terminates the completion.
        """

        __toolname__: typing.ClassVar[typing.Literal["submit_solution"]] = (
            "submit_solution"
        )

        @classmethod
        def define(
            cls,
            template: Template[..., T],
            bound_args: inspect.BoundArguments,
        ) -> FinalTool[[collections.abc.Callable[..., T]], T]:
            # Synthesize a drop-in syntactic replacement for the Template body, so the
            # function carries the Template's full signature -- including `self` for
            # Agent-method Templates (whose `__default__` is a bound method).
            if isinstance(template.__default__, types.MethodType):
                signature = inspect.signature(template.__default__.__func__)
                args, kwargs = (
                    (template.__default__.__self__,) + bound_args.args,
                    bound_args.kwargs,
                )
            else:
                signature = inspect.signature(template)
                args, kwargs = bound_args.args, bound_args.kwargs

            callable_type = _callable_type_from_signature(signature)
            callable_type = typing.Annotated[callable_type, _SynthesisSpec(template)]  # type: ignore
            return_type = signature.return_annotation

            def submit_solution(implementation: callable_type) -> return_type:  # type: ignore
                """
                Submit your final answer as a Python function implementing the task.
                The function must have the required signature; it is applied to the
                original inputs and its return value is your final answer.
                """
                return implementation(*args, **kwargs)  # type: ignore

            return super().define(submit_solution, name=cls.__toolname__)

    @implements(call_system)
    def _call_system(self, template, tool_types=frozenset()):
        return fwd(template, tool_types=tool_types | {self._SynthesisFinalTool})

    @implements(Template.__apply__)
    def _apply[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        tool = self._SynthesisFinalTool.define(template, bound_args)

        def _add_synthesis_tool(env, response_type, tools=frozenset()):
            return fwd(env, response_type, tools | {tool})

        with handler({call_assistant: _add_synthesis_tool}):
            return fwd()


class PythonRepl(ObjectInterpretation):
    """Expose a persistent Python session to the LLM as an `exec_code` Tool.

    Off by default; install it where the LLM should be able to run code whose
    state (variables, imports, definitions) survives across tool calls within a
    single Template invocation.

    Scoping mirrors how `__history__` is managed for Template calls: `PythonRepl`
    handles `Template.__apply__` to introduce a fresh `_repl_session` handler for
    the duration of the call, and intercepts `call_assistant` to inject an
    `exec_code` Tool routed to that session.  The session is therefore introduced and
    eliminated by its own handler, bounded to the Template call by construction --
    there is no global registry of sessions, and nested Template calls get their
    own isolated sessions.

    The session is seeded from the Template's lexical context and routes execution
    through the `parse`/`compile`/`exec` effect operations, so it works under any
    installed eval provider (`UnsafeEvalProvider` or `RestrictedEvalProvider`).
    """

    @typing.final
    class _ReplInteractionTool[**P, T](Tool[P, T]):
        """## Python REPL

        You may run arbitrary Python code in a persistent session. The code is
        executed in the context of this Template's lexical scope (see the *Lexical
        scope* table for the available names and their types). The session persists
        across turns, so you may define variables, functions, and classes that are
        used in later turns. The return value of the code is returned to you as the
        result of the tool call.
        """

    @typing.final
    @_ReplInteractionTool.define
    @classmethod
    @functools.wraps(ReplSession.exec_code)
    def exec_code(cls, code: types.CodeType) -> str:
        raise NotImplementedError("No handler")

    @typing.final
    @_ReplInteractionTool.define
    @classmethod
    def read_lexical_variable(cls, name: str) -> typing.Any:
        """
        Read the value of lexical variable ``name`` into the LLM context.
        """
        raise NotImplementedError("No handler")

    @implements(call_system)
    def _call_system(self, template, tool_types=frozenset()):
        return fwd(template, tool_types=tool_types | {self._ReplInteractionTool})

    @implements(Template.__apply__)
    def _apply[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = collections.ChainMap(bound_args.arguments, template.__context__)
        session = ReplSession(env=env)
        with handler(
            {
                self.exec_code: session.exec_code,
                self.read_lexical_variable: env.get,
            }
        ):
            return fwd()

    @implements(call_assistant)
    def _call_assistant[T](
        self,
        env: collections.abc.Mapping[str, typing.Any],
        response_type: type[T],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult[T]:
        return fwd(
            env,
            response_type,
            tools | {self.exec_code, self.read_lexical_variable},
        )


class RetryLLMHandler(ObjectInterpretation):
    """Retries LLM requests if tool call or result decoding fails.

    This handler intercepts `call_assistant` and catches `ToolCallDecodingError`
    and `ResultDecodingError`. When these errors occur, it appends error feedback
    to the messages and retries the request. Malformed messages from retry attempts
    are pruned from the final result.

    For runtime tool execution failures (handled via `call_tool`), errors are
    captured and returned as tool response messages.

    Args:
        include_traceback: If True, include full traceback in error feedback
            for better debugging context (default: True).
        catch_tool_errors: Exception type(s) to catch during tool execution.
            Can be a single exception class or a tuple of exception classes.
            Defaults to Exception (catches all exceptions).
        stop: tenacity stop condition for retrying `call_assistant`. Defaults to
            `tenacity.stop_after_attempt(4)`, which stops after 4 attempts.
        **kwargs: Additional keyword arguments forwarded to `tenacity.Retrying`.
    """

    call_assistant_retryer: tenacity.Retrying

    _user_before_sleep: collections.abc.Callable[[tenacity.RetryCallState], None] | None

    def __init__(
        self,
        include_traceback: bool = True,
        catch_tool_errors: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
        stop: tenacity.stop.stop_base = tenacity.stop_after_attempt(4),
        **kwargs,
    ):
        self.include_traceback = include_traceback
        self.catch_tool_errors = catch_tool_errors
        assert "retry" not in kwargs, "Cannot override retry logic of RetryLLMHandler"
        assert "reraise" not in kwargs, (
            "Cannot override reraise logic of RetryLLMHandler"
        )
        self._user_before_sleep = kwargs.pop("before_sleep", None)
        self.call_assistant_retryer = tenacity.Retrying(
            retry=tenacity.retry_if_exception_type(
                (ToolCallDecodingError, ResultDecodingError)
            ),
            reraise=True,
            before_sleep=self._before_sleep,
            stop=stop,
            **kwargs,
        )

    def _before_sleep(self, retry_state: tenacity.RetryCallState) -> None:
        e = retry_state.outcome.exception()  # type: ignore
        assert isinstance(e, (ToolCallDecodingError, ResultDecodingError))
        append_message(e.raw_message)
        append_message(e.to_feedback_message(self.include_traceback))
        if self._user_before_sleep is not None:
            self._user_before_sleep(retry_state)

    @implements(call_assistant)
    def _call_assistant[T](
        self,
        env: collections.abc.Mapping[str, typing.Any],
        response_type: type[T],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult[T]:
        _message_sequence = _get_history().copy()

        with handler({_get_history: lambda: _message_sequence}):
            message, tool_calls, result = self.call_assistant_retryer(fwd)

        append_message(message)
        return (message, tool_calls, result)

    @implements(call_tool)
    def _call_tool[T](self, tool_call: DecodedToolCall[T]) -> ToolResult[T]:
        """Handle tool execution with runtime error capture.

        Runtime errors from tool execution are captured and returned as
        error messages to the LLM. Only exceptions matching `catch_tool_errors`
        are caught; others propagate up.

        A captured failure is reported as ``is_final=False`` so that the
        completion loop continues even when a :class:`FinalTool` call raised:
        the model sees the error message and gets another turn to retry.
        """
        try:
            return fwd(tool_call)
        except ToolCallExecutionError as e:
            if isinstance(e.original_error, self.catch_tool_errors):
                message = e.to_feedback_message(self.include_traceback)
                append_message(message)
                return (message, None, False)
            else:
                raise


def _message_text(content: None | str | collections.abc.Iterable[typing.Any]) -> str:
    """Flatten a message ``content`` to display text.

    ``content`` may be a plain string or a list of content blocks (dicts with a
    ``type`` discriminator, e.g. ``{"type": "text", "text": ...}``, as produced
    by :func:`~effectful.handlers.llm.encoding.to_content_blocks`). Text blocks
    contribute their text; other block types show a ``[type]`` placeholder.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text") or "")
            else:
                parts.append(f"[{block.get('type', 'content')}]")
        else:
            parts.append(str(block))
    return "".join(parts)


class _PartialToolCall(typing.TypedDict):
    """A tool call being assembled from streamed deltas (name + raw JSON args)."""

    name: str
    args: str


class _PartialAssistant(typing.TypedDict):
    """The in-progress assistant turn accumulated from streaming deltas.

    ``tool_calls`` is keyed by each tool call's streaming ``index`` so fragments
    for the same call (which arrive across many chunks) coalesce.
    """

    content: str
    reasoning_content: str
    tool_calls: dict[int, _PartialToolCall]


def _accumulate(
    partial: _PartialAssistant, delta: litellm.types.utils.Delta | None
) -> None:
    """Fold one streaming ``delta`` into the in-progress assistant ``partial``.

    Concatenates ``content`` and ``reasoning_content``, and accumulates each
    streamed tool-call fragment (``function.name`` / ``function.arguments``) into
    ``partial["tool_calls"]`` keyed by the tool call's ``index``.
    """
    if delta is None:
        return
    partial["content"] += delta.content or ""
    # `reasoning_content` is absent (not just None) on deltas that carry none.
    partial["reasoning_content"] += getattr(delta, "reasoning_content", None) or ""
    tc: litellm.types.utils.ChatCompletionDeltaToolCall
    for tc in delta.tool_calls or []:
        slot = partial["tool_calls"].setdefault(tc.index, {"name": "", "args": ""})
        if tc.function is not None:
            slot["name"] = tc.function.name or slot["name"]
            slot["args"] += tc.function.arguments or ""


# Panel border colors keyed by message role.
_ROLE_STYLES = {
    "system": "grey50",
    "user": "cyan",
    "assistant": "green",
    "tool": "yellow",
}


# Longest field body rendered in a panel before it is truncated, keeping
# large-but-static messages (notably the system prompt) from dominating the
# frame. Each renderable truncates with its own native mechanism: `Syntax` by
# whole lines (`line_range`); `Markdown` has none, so its source is clipped by
# whole lines.
_MAX_LINES = 40


def _syntax(code: str, lexer: str) -> rich.console.RenderableType:
    """Syntax-highlight `code` using the terminal palette, truncated to
    `_MAX_LINES` via `Syntax.line_range` (no parsing, safe on partial input)."""
    syntax = rich.syntax.Syntax(
        code,
        lexer,
        theme="ansi_dark",
        word_wrap=True,
        background_color="default",
        line_range=(1, _MAX_LINES),
    )
    total = code.count("\n") + 1
    if total <= _MAX_LINES:
        return syntax
    note = rich.text.Text(f"… (+{total - _MAX_LINES} more lines)", style="dim")
    return rich.console.Group(syntax, note)


def _render_markdown(text: str, *, clip: bool = True) -> rich.console.RenderableType:
    """Render prose (system/user/assistant content) as Markdown.

    `Markdown` -- unlike `Syntax` -- has no native length limit, so when ``clip``
    the source is truncated to `_MAX_LINES` whole lines first. The live streaming
    panel passes ``clip=False`` so the growing tail stays fully visible.
    """
    lines = text.splitlines()
    if clip and len(lines) > _MAX_LINES:
        text = (
            "\n".join(lines[:_MAX_LINES])
            + f"\n\n*… (+{len(lines) - _MAX_LINES} more lines)*"
        )
    return rich.markdown.Markdown(text, code_theme="ansi_dark")


def _render_reasoning(text: str, *, clip: bool = True) -> rich.console.RenderableType:
    """Render reasoning as dimmed Markdown.

    `Markdown` takes no ``style=``, so `rich.styled.Styled` applies a ``dim``
    base -- the Markdown-compatible analog of the old dim-italic plain text. A
    base ``italic`` interferes with Markdown's own paragraph styling (dropping
    the dim too), so only ``dim`` is used. ``clip`` is forwarded to
    `_render_markdown`.
    """
    return rich.styled.Styled(_render_markdown(text, clip=clip), "dim")


def _render_data(value: typing.Any) -> rich.console.RenderableType:
    """Render an already-parsed JSON value (tool result / structured-output
    answer / tool-call arguments) as pretty, highlighted, line-truncated JSON."""
    return _syntax(json.dumps(value, indent=2), "json")


# Structured-output answers are wrapped by `_BoxedResponse` as `{"value": ...}`
# (call_assistant); the wrapper is display noise. Sourced from the model.
_BOX_FIELD = next(iter(_BoxedResponse.model_fields))


def _render_content(text: str, *, unwrap: bool = False) -> rich.console.RenderableType:
    """Render message content, choosing by shape rather than role: JSON
    objects/arrays (tool results, direct structured-output answers) as pretty
    JSON, everything else (prose, the Markdown system/user prompts) as Markdown.

    When ``unwrap`` (for a direct structured-output answer), a lone
    ``_BoxedResponse`` ``{"value": ...}`` wrapper is stripped to its payload.
    """
    if text.lstrip()[:1] in ("{", "["):
        try:
            value = json.loads(text)
        except ValueError:
            pass
        else:
            if unwrap and isinstance(value, dict) and set(value) == {_BOX_FIELD}:
                value = value[_BOX_FIELD]
            return _render_data(value)
    return _render_markdown(text)


def _is_python(text: str) -> bool:
    """Whether `text` looks like a Python source snippet worth highlighting.

    Detects code by *content* rather than schema/field name, so it covers every
    `Encodable` type that serializes Python as a string -- the synthesis
    `SynthesizedFunction.module_code` field, `exec_code`'s `types.CodeType`
    argument, and any future code-carrying tool -- uniformly. Requires a
    multi-line string that parses as a module with at least one real statement
    (not a lone expression), which excludes prose and JSON-as-string.
    """
    if "\n" not in text:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(not isinstance(node, ast.Expr) for node in tree.body)


def _extract_code(args: typing.Any) -> str | None:
    """Return an embedded Python source string from parsed tool-call arguments.

    Walks nested dicts (a synthesized callable is ``{"implementation":
    {"module_code": ...}}``; `exec_code` is a flat ``{"code": ...}``) and returns
    the first string value that :func:`_is_python` recognizes.
    """
    if isinstance(args, str):
        return args if _is_python(args) else None
    if isinstance(args, dict):
        for value in args.values():
            found = _extract_code(value)
            if found is not None:
                return found
    return None


def _render_tool_call(
    name: str, args: str, *, streaming: bool
) -> rich.console.RenderableType:
    """Render one tool call: a ``→ name`` header over its arguments.

    Synthesized code is shown as Python; ordinary arguments as pretty JSON. While
    ``streaming`` the argument JSON is still partial (unparseable), so the raw
    fragment is highlighted as JSON instead.
    """
    header = rich.text.Text(f"→ {name}", style="bold magenta")
    parsed: typing.Any = None
    if not streaming:
        try:
            parsed = json.loads(args)
        except ValueError:
            parsed = None
    code = _extract_code(parsed)
    if code is not None:
        body: rich.console.RenderableType = _syntax(code, "python")
    elif parsed is not None:
        body = _render_data(parsed)
    else:
        body = _syntax(args, "json") if args else rich.text.Text("…", style="dim")
    return rich.console.Group(header, body)


def _message_panel(message: Message) -> rich.panel.Panel:
    """Render a single completed history message as a titled panel.

    Every role (including ``system``) is shown; long field bodies are truncated
    to `_MAX_LINES` lines so the frame stays readable.
    """
    # A loose view for reads of keys not declared across the whole `Message`
    # union (`reasoning_content`, `tool_calls`), which typecheckers infer as
    # `object`; these messages are dynamically built dicts (see `_make_message`).
    msg = typing.cast("collections.abc.Mapping[str, typing.Any]", message)
    role = msg.get("role", "?")
    renderables: list[rich.console.RenderableType] = []
    reasoning = _message_text(msg.get("reasoning_content"))
    if reasoning:
        renderables.append(_render_reasoning(reasoning))
    content = _message_text(msg.get("content"))
    if content:
        renderables.append(_render_content(content, unwrap=role == "assistant"))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        renderables.append(
            _render_tool_call(
                fn.get("name") or "?", fn.get("arguments") or "", streaming=False
            )
        )
    body = rich.console.Group(*renderables) if renderables else rich.text.Text("")
    return rich.panel.Panel(
        body,
        title=role,
        title_align="left",
        border_style=_ROLE_STYLES.get(role, "white"),
    )


def _partial_panel(
    partial: _PartialAssistant, ttft: float | None = None, *, streaming: bool = True
) -> rich.panel.Panel:
    """Render the in-progress (or just-finished) assistant turn as a live panel.

    When ``ttft`` (time-to-first-token, seconds) is known it is shown as a
    subtitle -- how long the model spent prefilling before the first delta.

    ``streaming`` is forwarded to `_render_tool_call`: the caller passes
    ``streaming=False`` for the final frame, once the stream is exhausted and the
    tool-call arguments form complete JSON, so they render as pretty JSON /
    synthesized code rather than the raw partial payload. This is the only chance
    the *terminating* tool call gets to settle -- there is no later `completion`
    to re-render it as a history message.
    """
    # Content and reasoning render as Markdown even mid-stream (Markdown never
    # raises on incomplete text) and are shown in full (clip=False) so the
    # growing tail stays visible.
    renderables: list[rich.console.RenderableType] = []
    if partial["reasoning_content"]:
        renderables.append(_render_reasoning(partial["reasoning_content"], clip=False))
    if partial["content"]:
        renderables.append(_render_markdown(partial["content"], clip=False))
    for _, slot in sorted(partial["tool_calls"].items()):
        renderables.append(
            _render_tool_call(slot["name"], slot["args"], streaming=streaming)
        )
    body = (
        rich.console.Group(*renderables)
        if renderables
        else rich.text.Text("…", style="dim")
    )
    subtitle = (
        rich.text.Text(f"TTFT {ttft:.1f}s", style="dim") if ttft is not None else None
    )
    return rich.panel.Panel(
        body,
        title="assistant",
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style="green",
    )


class _PrefillStatus:
    """Live "prefilling…" line shown until the first streamed chunk arrives.

    litellm/provider APIs report no prompt-processing progress, so there is no
    true prefill percentage. Instead this shows the (locally counted) prompt
    size and a ticking elapsed timer, which is what a large prompt's
    time-to-first-token latency actually reflects.

    It is re-rendered by :class:`rich.live.Live`'s background refresh thread
    while the main thread blocks on the first chunk, so the spinner animates and
    the timer ticks on their own -- :meth:`__rich__` recomputes elapsed each call.
    """

    def __init__(self, prompt_tokens: int | None, start: float):
        self._spinner = rich.spinner.Spinner("dots", style="cyan")
        self._prompt_tokens = prompt_tokens
        self._start = start

    def __rich__(self) -> rich.spinner.Spinner:
        elapsed = time.monotonic() - self._start
        size = (
            f"{self._prompt_tokens:,} tokens"
            if self._prompt_tokens is not None
            else "prompt"
        )
        self._spinner.update(
            text=rich.text.Text(f" prefilling {size}… {elapsed:.1f}s", style="cyan")
        )
        return self._spinner


def _render_frame(
    history: collections.abc.Sequence[Message],
    partial: _PartialAssistant,
    *,
    status: _PrefillStatus | None = None,
    ttft: float | None = None,
    streaming: bool = True,
) -> rich.console.Group:
    """Build the full frame: one panel per history message, then either the live
    ``status`` line (while prefilling, before any token) or the partial turn."""
    tail = (
        status
        if status is not None
        else _partial_panel(partial, ttft=ttft, streaming=streaming)
    )
    return rich.console.Group(*[_message_panel(m) for m in history], tail)


@dataclasses.dataclass(frozen=True)
class TerminalRenderer(ObjectInterpretation):
    """Stream `completion` and live-render the whole message sequence.

    Opt-in debugging handler: forces streaming, redraws the entire (partial)
    message history from scratch on every chunk via :class:`rich.live.Live`
    (reasoning, generation, and tool-call arguments appear as they are produced),
    then reassembles a normal ``ModelResponse`` via
    :func:`litellm.stream_chunk_builder` so the rest of the pipeline is unchanged.
    """

    console: rich.console.Console = dataclasses.field(
        default_factory=rich.console.Console
    )

    @implements(completion)
    def _completion(self, *args, **kwargs) -> typing.Any:
        kwargs = {
            **kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream: litellm.CustomStreamWrapper = fwd(*args, **kwargs)

        # The request already carries the full message history as `messages`.
        history: list[Message] = list(kwargs.get("messages") or [])

        chunks: list[litellm.types.utils.ModelResponseStream] = []
        partial: _PartialAssistant = {
            "content": "",
            "reasoning_content": "",
            "tool_calls": {},
        }

        # Count prompt tokens locally to size the prefill wait. `model` is injected
        # downstream by LiteLLMProvider, so it may be absent here -- token_counter
        # falls back to a default tokenizer, giving an approximate count.
        try:
            prompt_tokens: int | None = litellm.token_counter(
                model=kwargs.get("model", ""),
                messages=history,
                tools=kwargs.get("tools"),
            )
        except Exception:
            prompt_tokens = None

        start = time.monotonic()
        status: _PrefillStatus | None = _PrefillStatus(prompt_tokens, start)
        ttft: float | None = None

        with rich.live.Live(
            _render_frame(history, partial, status=status),
            console=self.console,
            vertical_overflow="visible",
        ) as live:
            for chunk in stream:
                chunks.append(chunk)
                if chunk.choices:
                    _accumulate(partial, chunk.choices[0].delta)
                # The first chunk carrying any content ends prefill; record TTFT
                # and drop the status line in favor of the streaming panel.
                if status is not None and (
                    partial["content"]
                    or partial["reasoning_content"]
                    or partial["tool_calls"]
                ):
                    ttft = time.monotonic() - start
                    status = None
                live.update(_render_frame(history, partial, status=status, ttft=ttft))

            # The args are now complete JSON; re-render settled so the final
            # (loop-terminating) tool call shows as pretty JSON / synthesized
            # code rather than the raw streaming payload.
            live.update(_render_frame(history, partial, ttft=ttft, streaming=False))

        return litellm.stream_chunk_builder(chunks, messages=kwargs.get("messages"))


@dataclasses.dataclass(frozen=True)
class SystemPromptDumper(ObjectInterpretation):
    """Dump the system prompt produced by `call_system` to a Markdown file.

    Opt-in debugging handler: intercepts `call_system`, forwards to let the
    prompt be assembled and installed as usual, then writes the resulting
    system message content to `path`, overwriting the whole file each time.
    """

    path: pathlib.Path

    @implements(call_system)
    def _call_system(self, template, tool_types=frozenset()):
        message = fwd()
        self.path.write_text(_message_text(message.get("content")))
        return message


class LiteLLMProvider(ObjectInterpretation):
    """Implements templates using the LiteLLM API."""

    config: collections.abc.Mapping[str, typing.Any]

    def __init__(self, model="gpt-4o", **config):
        self.config = {
            "model": model,
            **inspect.signature(litellm.completion).bind_partial(**config).kwargs,
        }

    @implements(completion)
    def _completion(self, *args, **kwargs):
        """Inject the provider's configuration (model and bound litellm kwargs)
        into the low-level request before delegating."""
        return fwd(*args, **{**self.config, **kwargs})

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        # encode arguments
        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(bound_args.arguments)

        history: collections.OrderedDict[str, Message] = getattr(
            template, "__history__", collections.OrderedDict()
        )  # type: ignore
        history_copy = history.copy()

        with handler({_get_history: lambda: history_copy}):
            if (
                not _get_history()
                or next(iter(_get_history().values()))["role"] != "system"
            ):
                message: Message = call_system(template)

            message = call_user(template, env)

            # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
            result: T | None = None
            is_final: bool = False
            while not is_final:
                message, tool_calls, result = call_assistant(
                    env,
                    template.__signature__.return_annotation,
                    _tools_in_scope(env) - {template},
                    anchor=template.__default__,
                )
                if tool_calls:
                    for tool_call in tool_calls:
                        message, result, is_final = call_tool(tool_call)
                else:
                    is_final = True

        try:
            _get_history()
        except _NoActiveHistoryException:
            history.clear()
            history.update(history_copy)
        return typing.cast(T, result)


@dataclasses.dataclass(frozen=True)
class LangfuseTracer(ObjectInterpretation):
    """Traces Tool, Template, and completion calls with Langfuse.

    Compose with a provider via :func:`~effectful.ops.semantics.handler`
    to add tracing::

        with handler(provider), handler(LangfuseTracer()):
            print(limerick(theme))
    """

    client: langfuse.Langfuse = dataclasses.field(default_factory=langfuse.get_client)

    @implements(completion)
    def completion(self, *args, **kwargs):
        messages = kwargs.get("messages")
        if kwargs.get("tools") is not None:
            gen_input = {"tools": kwargs["tools"], "messages": messages}
        else:
            gen_input = messages

        model_parameters = {
            k: kwargs[k]
            for k in ("tool_choice", "temperature", "max_tokens", "top_p")
            if kwargs.get(k) is not None
        }
        metadata = {}
        response_format = kwargs.get("response_format")
        if response_format is not None:
            metadata["response_format"] = (
                response_format.model_json_schema()
                if isinstance(response_format, type)
                and issubclass(response_format, pydantic.BaseModel)
                else response_format
            )
        with self.client.start_as_current_observation(
            as_type="generation",
            name="completion",
            input=gen_input,
            model_parameters=model_parameters or None,
            metadata=metadata or None,
        ) as gen:
            response = fwd()
            gen.update(model=response.model)
            usage = getattr(response, "usage", None)
            if usage is not None:
                gen.update(
                    usage_details={
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    }
                )
            gen.update(output=response.choices[0].message)
            return response

    @implements(call_tool)
    def call_tool(self, tool_call: DecodedToolCall):
        input = {
            name: pydantic.TypeAdapter(
                Encodable[nested_type(value).value]  # type: ignore[misc]
            ).dump_python(value, mode="json", context={})
            for name, value in tool_call.bound_args.arguments.items()
        }
        with self.client.start_as_current_observation(
            as_type="tool",
            name=tool_call.name,
            input=input,
            metadata={"tool_call_id": tool_call.id},
        ) as obs:
            message, result, is_final = fwd()
            obs.update(output=message["content"], metadata={"is_final": is_final})
            return message, result, is_final

    @implements(Template.__apply__)
    def call_template(self, template: Template, *args, **kwargs):
        bound = inspect.signature(template).bind(*args, **kwargs)
        bound.apply_defaults()
        agent_input = {
            name: pydantic.TypeAdapter(
                Encodable[nested_type(value).value]  # type: ignore[misc]
            ).dump_python(value, mode="json", context={})
            for name, value in bound.arguments.items()
        }
        with self.client.start_as_current_observation(
            as_type="agent", name=template.__name__, input=agent_input
        ) as obs:
            result = fwd()
            obs.update(
                output=pydantic.TypeAdapter(
                    Encodable[nested_type(result).value]  # type: ignore[misc]
                ).dump_python(result, mode="json", context={})
            )
            return result
