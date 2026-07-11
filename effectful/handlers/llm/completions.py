import abc
import collections
import collections.abc
import dataclasses
import functools
import inspect
import json
import string
import textwrap
import traceback
import typing
import uuid

import litellm
import pydantic
import tenacity
from litellm import (
    ChatCompletionFunctionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionTextObject,
    ChatCompletionToolMessage,
    OpenAIChatCompletionAssistantMessage,
    OpenAIChatCompletionSystemMessage,
    OpenAIChatCompletionUserMessage,
    OpenAIMessageContentListBlock,
)

from effectful.handlers.llm.encoding import (
    TYPE_CHECK_ANCHOR_KEY,
    DecodedToolCall,
    Encodable,
    to_content_blocks,
)
from effectful.handlers.llm.evaluation import ReplSession
from effectful.handlers.llm.template import (
    Agent,
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

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant, you need to follow user's instruction"
)


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


type MessageResult[T] = tuple[Message, typing.Sequence[DecodedToolCall], T | None]


class _LexicalVariableTool[T](Tool[[], T]):
    """A zero-arg `Tool` that returns the captured value of a variable
    from a `Template`'s lexical context.

    Tools are constructed fresh each `call_assistant` invocation, so
    the reader closes over the snapshot `value` rather than the
    surrounding `env` — in-place mutation of a mutable value is still
    visible (same object reference), but rebinding the source name is
    not.
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
        tool_fn.__doc__ = (
            f"Reads the value of lexical variable `{name}` from the "
            f"enclosing scope where this Template was defined.  Takes "
            f"no arguments; returns the current value."
        )
        tool_fn.__annotations__ = {"return": typ}
        return super().define(tool_fn)


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


class LexicalReaders(ObjectInterpretation):
    """Override `collect_tools` to also expose plain values from the
    lexical context as zero-argument read-only Tools.  Each non-Tool,
    non-Template, non-Agent value bound to a valid identifier is
    wrapped via `_LexicalVariableTool` if `Encodable[T]` accepts it;
    schema-generation failures cause the symbol to be skipped.
    """

    @implements(collect_tools)
    def _collect(
        self, env: collections.abc.Mapping[str, typing.Any]
    ) -> collections.abc.Mapping[str, Tool]:
        result = dict(fwd())
        for name, obj in env.items():
            if name in result or not name.isidentifier():
                continue
            try:
                result[name] = _LexicalVariableTool.define(obj, name=name)
            # `TypeError` joins the three Pydantic errors because the
            # `Encodable[T]` registry raises `TypeError` to signal
            # "no schema possible" — e.g. `_pydantic_type_operation`,
            # `_pydantic_type_term`, and `_pydantic_callable`'s
            # incomplete-signature path. Same intent as the Pydantic
            # cases, different exception class.
            except (
                pydantic.errors.PydanticSchemaGenerationError,
                pydantic.errors.PydanticInvalidForJsonSchema,
                pydantic.errors.PydanticUserError,
                TypeError,
            ):
                continue
        return result


@Operation.define
def _repl_session(env: collections.abc.MutableMapping[str, typing.Any]) -> ReplSession:
    """Return the REPL session for the current Template call, seeded from `env`.

    `PythonRepl` installs a fresh handler for this inside each `Template.__apply__`
    (mirroring how `__history__` is managed), giving the session a lifetime of
    exactly one Template call.  Outside such a scope there is no managed session,
    so this falls back to a fresh one -- e.g. when tools are listed outside a
    Template call.
    """
    return ReplSession(env)


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

    @implements(Template.__apply__)
    def _apply[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        # One session per Template call, created lazily on first use (the call's
        # `env`, supplied by `collect_tools`/`exec_code`, seeds it).  The
        # enclosing `handler(...)` bounds the session's lifetime to this call, so
        # nested Template calls introduce their own fresh session.
        session: ReplSession | None = None

        def session_for(
            env: collections.abc.MutableMapping[str, typing.Any],
        ) -> ReplSession:
            nonlocal session
            if session is None:
                session = ReplSession(env)
            return session

        with handler({_repl_session: session_for}):
            return fwd()

    @implements(collect_tools)
    def _collect(
        self, env: collections.abc.Mapping[str, typing.Any]
    ) -> collections.abc.Mapping[str, Tool]:
        tools = dict(fwd())
        # `collect_tools` only promises a `Mapping`, but the per-call `env` is the
        # writable `ChainMap` the session splices its shared scope layer into, so
        # narrow it for `_repl_session`/`ReplSession`.
        tools["exec_code"] = _repl_session(
            typing.cast(collections.abc.MutableMapping[str, typing.Any], env)
        ).exec_code
        return tools


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


@Operation.define
def call_assistant[T](
    env: collections.abc.Mapping[str, typing.Any],
    response_type: type[T],
    model: str,
    **kwargs,
) -> MessageResult[T]:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    Raises:
        ToolCallDecodingError: If a tool call cannot be decoded. The error
            includes the raw assistant message for retry handling.
        ResultDecodingError: If the result cannot be decoded. The error
            includes the raw assistant message for retry handling.
    """
    anchor = kwargs.pop("anchor", None)  # ride in kwargs; pop before the LLM call
    tools = dict(collect_tools(env))
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

    tool_calls: list[DecodedToolCall] = []
    encoding: pydantic.TypeAdapter[DecodedToolCall] = pydantic.TypeAdapter(
        Encodable[DecodedToolCall]
    )
    # Tool arguments decode with `context=tools` (no anchor key), so a synthesized
    # tool-argument Callable gets no type-check anchor -- correctly: its contract
    # is the tool parameter's type, not the enclosing Template's return type, so
    # the Template anchor would be the wrong splice target.
    for raw_tool_call in message.get("tool_calls") or []:
        try:
            tool_calls += [encoding.validate_python(raw_tool_call, context=tools)]
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


@Operation.define
def call_tool(tool_call: DecodedToolCall) -> Message:
    """Implements a roundtrip call to a python function. Input is a json
    string representing an LLM tool call request parameters. The output is
    the serialised response to the model.

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
    return message


@Operation.define
def call_user(
    template: str,
    env: collections.abc.Mapping[str, typing.Any],
) -> Message:
    """
    Format a template applied to arguments into a user message.
    """
    formatter = string.Formatter()
    parts: list[OpenAIMessageContentListBlock] = []

    buf: list[str] = []

    def flush_text() -> None:
        if buf:
            parts.append(ChatCompletionTextObject(type="text", text="".join(buf)))
            buf.clear()

    for literal, field_name, format_spec, conversion in formatter.parse(
        textwrap.dedent(template)
    ):
        if literal:
            buf.append(literal)

        if field_name is None:
            continue

        obj, _ = formatter.get_field(field_name, (), env)
        encoder: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
            Encodable[nested_type(obj).value]  # type: ignore[misc]
        )
        encoded_obj = encoder.dump_python(obj, mode="json", context=env)
        for part in to_content_blocks(encoded_obj):
            if part["type"] == "text":
                text = (
                    formatter.convert_field(part["text"], conversion)
                    if conversion
                    else part["text"]
                )
                buf.append(formatter.format_field(text, format_spec or ""))
            else:
                flush_text()
                parts.append(part)

    flush_text()

    # Note: The OpenAI api only seems to accept images in the 'user' role. The
    # effect of different roles on the model's response is currently unclear.
    message = _make_message(dict(role="user", content=parts))
    append_message(message)
    return message


@Operation.define
def call_system(template: Template) -> Message:
    """Get system instruction message(s) to prepend to all LLM prompts."""
    system_prompt = template.__system_prompt__ or DEFAULT_SYSTEM_PROMPT
    message = _make_message(dict(role="system", content=system_prompt))
    append_message(message, last=False)
    return message


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
    ) -> MessageResult[T]:
        _message_sequence = _get_history().copy()

        with handler({_get_history: lambda: _message_sequence}):
            message, tool_calls, result = self.call_assistant_retryer(fwd)

        append_message(message)
        return (message, tool_calls, result)

    @implements(call_tool)
    def _call_tool(self, tool_call: DecodedToolCall) -> Message:
        """Handle tool execution with runtime error capture.

        Runtime errors from tool execution are captured and returned as
        error messages to the LLM. Only exceptions matching `catch_tool_errors`
        are caught; others propagate up.
        """
        try:
            return fwd(tool_call)
        except ToolCallExecutionError as e:
            if isinstance(e.original_error, self.catch_tool_errors):
                message = e.to_feedback_message(self.include_traceback)
                append_message(message)
                return message
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
                call_system(template)

            message: Message = call_user(template.__prompt_template__, env)

            # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
            tool_calls: list[DecodedToolCall] = []
            result: T | None = None
            while message["role"] != "assistant" or tool_calls:
                message, tool_calls, result = call_assistant(
                    env,
                    template.__signature__.return_annotation,
                    anchor=template.__default__,
                    **self.config,
                )
                for tool_call in tool_calls:
                    message = call_tool(tool_call)

        try:
            _get_history()
        except _NoActiveHistoryException:
            history.clear()
            history.update(history_copy)
        return typing.cast(T, result)
