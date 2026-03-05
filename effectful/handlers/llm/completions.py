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
import openai.lib._pydantic
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
    DecodedToolCall,
    Encodable,
    to_content_blocks,
)
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


@Operation.define
def _get_history() -> collections.OrderedDict[str, Message]:
    raise NotImplementedError


def append_message(message: Message):
    try:
        _get_history()[message["id"]] = message
    except NotImplementedError:
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


def _collect_tools(
    env: collections.abc.Mapping[str, typing.Any],
) -> collections.abc.Mapping[str, Tool]:
    """Operations and Templates available as tools. Auto-capture from lexical context."""
    result = {}

    for name, obj in env.items():
        # Collect tools directly in context
        if isinstance(obj, Tool):
            result[name] = obj

        # Collect tools as methods on Agent instances in context
        elif isinstance(obj, Agent):
            for cls in type(obj).__mro__:
                for attr_name in vars(cls):
                    if isinstance(getattr(obj, attr_name), Tool):
                        result[f"{name}__{attr_name}"] = getattr(obj, attr_name)

    # The same Tool can appear under multiple names when it is both
    # visible in the enclosing scope *and* discovered via an Agent
    # instance's MRO.  Since Tools are hashable Operations and
    # instance-method Tools are cached per instance, we keep only
    # the last name for each unique tool object.
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


@Operation.define
def call_assistant[T](
    env: collections.abc.Mapping[str, typing.Any],
    response_format: pydantic.TypeAdapter[T],
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
    tools = _collect_tools(env)
    tool_specs = {
        k: typing.cast(
            pydantic.TypeAdapter[typing.Any],
            pydantic.TypeAdapter(Encodable[type(t)]),  # type: ignore[misc]
        ).dump_python(t, mode="json", context=tools)
        for k, t in tools.items()
    }
    _schema = openai.lib._pydantic.to_strict_json_schema(response_format)
    _is_str = _schema.get("type") == "string"
    _needs_wrap = not _is_str and _schema.get("type") != "object"
    if _needs_wrap:
        _wire_schema: dict[str, typing.Any] = {
            "type": "object",
            "properties": {"value": _schema},
            "required": ["value"],
            "additionalProperties": False,
        }
    else:
        _wire_schema = _schema
    response: litellm.types.utils.ModelResponse = completion(
        model,
        messages=list(_get_history().values()),
        response_format=None
        if _is_str
        else {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": _wire_schema,
                "strict": True,
            },
        },
        tools=list(tool_specs.values()),
        **kwargs,
    )
    choice = response.choices[0]
    assert isinstance(choice, litellm.types.utils.Choices)

    message: litellm.Message = choice.message
    assert message.role == "assistant"

    # Transparently unwrap the {"value": ...} envelope so the rest of the
    # pipeline (including message history and validation) never sees it.
    if _needs_wrap and message.get("content"):
        raw_content = message["content"]
        try:
            _envelope = json.loads(raw_content)
            if isinstance(_envelope, dict) and "value" in _envelope:
                message["content"] = json.dumps(_envelope["value"])
        except (json.JSONDecodeError, TypeError):
            pass

    raw_message = _make_message({**message.model_dump(mode="json")})
    append_message(raw_message)

    tool_calls: list[DecodedToolCall] = []
    encoding: pydantic.TypeAdapter[DecodedToolCall] = pydantic.TypeAdapter(
        Encodable[DecodedToolCall]
    )
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
        try:
            decoded = serialized_result if _is_str else json.loads(serialized_result)
            result = response_format.validate_python(decoded, context=env)
        except (pydantic.ValidationError, TypeError, ValueError, SyntaxError) as e:
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
    try:
        history: collections.OrderedDict[str, Message] = _get_history()
        if any(m["role"] == "system" for m in history.values()):
            assert sum(1 for m in history.values() if m["role"] == "system") == 1, (
                "There should be at most one system message in the history"
            )
            assert history[next(iter(history))]["role"] == "system", (
                "The system message should be the first message in the history"
            )
            history.popitem(last=False)  # remove existing system message
        history[message["id"]] = message
        history.move_to_end(message["id"], last=False)
        return message
    except NotImplementedError:
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
        response_format: pydantic.TypeAdapter[T],
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

        # Create response_model with env so tools passed as arguments are available
        response_model: pydantic.TypeAdapter[T] = pydantic.TypeAdapter(
            Encodable[template.__signature__.return_annotation]  # type: ignore[name-defined]
        )

        history: collections.OrderedDict[str, Message] = getattr(
            template, "__history__", collections.OrderedDict()
        )  # type: ignore
        history_copy = history.copy()

        with handler({_get_history: lambda: history_copy}):
            call_system(template)

            message: Message = call_user(template.__prompt_template__, env)

            # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
            tool_calls: list[DecodedToolCall] = []
            result: T | None = None
            while message["role"] != "assistant" or tool_calls:
                message, tool_calls, result = call_assistant(
                    env, response_model, **self.config
                )
                for tool_call in tool_calls:
                    message = call_tool(tool_call)

        try:
            _get_history()
        except NotImplementedError:
            history.update(history_copy)
        return typing.cast(T, result)
