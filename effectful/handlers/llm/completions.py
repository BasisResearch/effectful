import abc
import builtins
import collections
import collections.abc
import dataclasses
import functools
import inspect
import json
import traceback
import types
import typing
import uuid

import langfuse
import litellm
import pydantic
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
    _is_recursive_signature,
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


@Operation.define
def collect_tools(
    env: collections.abc.Mapping[str, typing.Any],
) -> collections.abc.Mapping[str, Tool]:
    """Return the tools available to a Template given its lexical context.

    Default rule: real `Tool` and `Template` values bound directly in
    `env`, plus `Tool` methods discovered through the MRO of any
    `Agent` instance in `env`.  Same-Tool-under-different-names is
    deduped so each Tool appears exactly once.

    Handlers (see :class:`LexicalReaders`) may override this to add
    synthetic readers, hide tools, etc.
    """
    result: dict[str, Tool] = {}

    for name, obj in env.items():
        if isinstance(obj, Tool | Template):
            result[name] = obj
        elif isinstance(obj, Agent):
            for cls in type(obj).__mro__:
                for attr_name in vars(cls):
                    if isinstance(getattr(obj, attr_name), Tool):
                        result[f"{name}__{attr_name}"] = getattr(obj, attr_name)

    # Same Tool can appear under multiple names when visible both in the
    # enclosing scope and via an Agent instance's MRO.  Keep only the
    # last name for each unique tool object.
    tool2name = {tool: name for name, tool in sorted(result.items())}
    for name, tool in tuple(result.items()):
        if tool2name[tool] != name:
            del result[name]

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
    model: str,
    **kwargs,
) -> AssistantResult[T]:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    Raises:
        ToolCallDecodingError: If a tool call cannot be decoded. The error
            includes the raw assistant message for retry handling.
        ResultDecodingError: If the result cannot be decoded. The error
            includes the raw assistant message for retry handling.
    """
    tools = collect_tools(env)
    # Decode tool calls (and the code synthesized for them) against the lexical
    # context, so they resolve names from the Template's scope.
    env = collections.ChainMap(
        typing.cast("collections.abc.MutableMapping[str, typing.Any]", env),
        typing.cast("collections.abc.MutableMapping[str, typing.Any]", tools),
    )
    tool_specs = {
        k: typing.cast(
            pydantic.TypeAdapter[typing.Any],
            pydantic.TypeAdapter(Encodable[type(t)]),  # type: ignore[misc]
        ).dump_python(t, mode="json", context={k: t})
        for k, t in tools.items()
    }

    # The OpenAI API requires a wrapper object for non-object structured output types,
    # so we create one on the fly here. Using a Pydantic model offloads JSON schema
    # generation and validation logic to litellm, and offers better error messages.
    response_format: type[_BoxedResponse[T]] = pydantic.create_model(
        "BoxedResponse",
        value=Encodable[response_type],  # type: ignore[valid-type]
        __base__=_BoxedResponse,
    )

    response: litellm.types.utils.ModelResponse = completion(
        model,
        messages=list(_get_history().values()),
        response_format=None if response_type is str else response_format,
        tools=list(tool_specs.values()),
        **kwargs,
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
                result = response_format.model_validate(
                    json.loads(serialized_result), context=env
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
    template: str,
    env: collections.abc.Mapping[str, typing.Any],
) -> Message:
    """
    Format a template applied to arguments into a user message.
    """
    parts = format_as_content_blocks(template, env)
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


def _render_vars_block(env: collections.abc.Mapping[str, typing.Any]) -> str:
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
    return f"## Lexical scope\n\n| name | type |\n| --- | --- |\n{body}"


def _render_imports_block(env: collections.abc.Mapping[str, typing.Any]) -> str:
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
    return f"## Imported modules\n\n| name | module |\n| --- | --- |\n{body}"


def _render_template_block(template: Template) -> str:
    """Markdown spec for a single `Template`: header, prompt, arg schemas."""
    parts = [f"### `{template.__name__}{template.__signature__}`"]
    prompt = inspect.getdoc(template.__default__) or ""
    if prompt:
        parts.append(prompt)
    args = [
        f"- `{name}` — `{_get_qualname(p.annotation)}`\n\n"
        f"    ```json\n    {json.dumps(pydantic.TypeAdapter(Encodable[p.annotation]).json_schema())}\n    ```"
        for name, p in template.__signature__.parameters.items()
    ]
    if args:
        parts.append("**Arguments**\n\n" + "\n".join(args))
    return "\n\n".join(parts)


def _render_agent_block(template: Template) -> str:
    """One lexical inventory plus the spec of every Template
    sharing the current history (an Agent's methods, or just ``template``)."""
    inst = (
        template.__default__.__self__
        if isinstance(template.__default__, types.MethodType)
        else None
    )
    if isinstance(inst, Agent):
        agent_doc = inspect.getdoc(type(inst)) or ""
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
        templates = {template}

    # Order by name so the prompt is stable across method reordering in source.
    specs = "\n\n".join(
        _render_template_block(t) for t in sorted(templates, key=lambda t: t.__name__)
    )
    sections = [
        f"## Agent `{_get_qualname(type(inst))}`\n\n{agent_doc}" if agent_doc else "",
        f"## Templates\n\n{specs}",
    ]
    return "\n\n".join(s for s in sections if s)


def _render_module_block(mod: types.ModuleType | None) -> str:
    """Markdown section with the source (or docstring fallback) of a module."""
    if mod is None:
        return ""
    try:
        src = inspect.getsource(mod)
        return f"## Module `{mod.__name__}`\n\n```python\n{src}\n```"
    except (OSError, TypeError):
        doc = inspect.getdoc(mod)
        return f"## Module `{mod.__name__}`\n\n{doc}" if doc else ""


def _render_global_block(tool_types: collections.abc.Set[type[Tool]]) -> str:
    """Constant framework-concept prefix, sourced from real docstrings."""
    import effectful.handlers.llm as _llm

    assert all(issubclass(t, Tool) and t not in {Tool, Template} for t in tool_types)
    parts = [inspect.getdoc(_llm) or ""]
    for obj in [
        Template,
        Tool,
        Agent,
        Encodable,
        *sorted(tool_types, key=_get_qualname),
    ]:
        parts += [f"## `{obj.__name__}`\n\n{inspect.getdoc(obj)}"]
    return "\n\n".join(p for p in parts if p)


@Operation.define
def call_system(
    template: Template, *, tool_types: collections.abc.Set[type[Tool]] = frozenset()
) -> Message:
    """Assemble and install the system message (a Markdown document)."""
    sections = [
        _render_global_block(tool_types),
        _render_module_block(inspect.getmodule(template)),
        _render_agent_block(template),
        _render_imports_block(template.__context__),
        _render_vars_block(template.__context__),
    ]
    content = "\n\n".join(s for s in sections if s)
    message = _make_message(dict(role="system", content=content))
    append_message(message, last=False)
    return message


class LexicalReaders(ObjectInterpretation):
    """Override `collect_tools` to also expose plain values from the
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

    @implements(collect_tools)
    def _collect(
        self, env: collections.abc.Mapping[str, typing.Any]
    ) -> collections.abc.Mapping[str, Tool]:
        result = dict(fwd())
        for name, obj in env.items():
            if name in result or not name.isidentifier() or isinstance(obj, Tool):
                continue
            try:
                result[name] = self._LexicalVariableTool.define(obj, name=name)
            except (
                pydantic.errors.PydanticSchemaGenerationError,
                pydantic.errors.PydanticInvalidForJsonSchema,
                pydantic.errors.PydanticUserError,
            ):
                continue
        return result


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
        names from the lexical scope (see the *Lexical scope* table). Calling this
        tool terminates the completion.
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
        with handler({collect_tools: lambda _: {**fwd(), tool.__name__: tool}}):  # type: ignore
            return fwd()


class PythonRepl(ObjectInterpretation):
    """Expose a persistent Python session to the LLM as an `exec_code` Tool.

    Off by default; install it where the LLM should be able to run code whose
    state (variables, imports, definitions) survives across tool calls within a
    single Template invocation.

    Scoping mirrors how `__history__` is managed for Template calls: `PythonRepl`
    handles `Template.__apply__` to introduce a fresh `_repl_session` handler for
    the duration of the call, and handles `collect_tools` to inject an `exec_code`
    Tool routed to that session.  The session is therefore introduced and
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

    @implements(collect_tools)
    def _collect(
        self, env: collections.abc.Mapping[str, typing.Any]
    ) -> collections.abc.Mapping[str, Tool]:
        tools = dict(fwd())
        tools[self.exec_code.__name__] = self.exec_code
        tools[self.read_lexical_variable.__name__] = self.read_lexical_variable
        return tools


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
        model: str,
        **kwargs,
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


class LiteLLMProvider(ObjectInterpretation):
    """Implements templates using the LiteLLM API."""

    config: collections.abc.Mapping[str, typing.Any]

    def __init__(self, model="gpt-4o", **config):
        self.config = {
            "model": model,
            **inspect.signature(litellm.completion).bind_partial(**config).kwargs,
        }

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        # encode arguments
        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(bound_args.arguments)

        if not _is_recursive_signature(template.__signature__):
            env = env.new_child({k: None for k, v in env.items() if v is template})

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

            message = call_user(template.__prompt_template__, env)

            # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
            result: T | None = None
            is_final: bool = False
            while not is_final:
                message, tool_calls, result = call_assistant(
                    env, template.__signature__.return_annotation, **self.config
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
    def completion(self, model, *args, **kwargs):
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
            model=model,
            input=gen_input,
            model_parameters=model_parameters or None,
            metadata=metadata or None,
        ) as gen:
            response = fwd()
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
