import abc
import collections
import collections.abc
import dataclasses
import functools
import inspect
import json
import traceback
import typing

import litellm
import pydantic

from effectful.handlers.llm.harness.serialization import (
    _IS_FINAL_KEY,
    _TOOLS_KEY,
    _TYPE_CHECK_ANCHOR_KEY,
    DecodedToolCall,
    format_as_content_blocks,
    to_content_blocks,
)
from effectful.handlers.llm.types import (
    Encodable,
    FinalTool,
    Template,
    Tool,
)
from effectful.internals.unification import nested_type
from effectful.ops.types import Operation

Message = (
    litellm.ChatCompletionAssistantMessage
    | litellm.ChatCompletionToolMessage
    | litellm.ChatCompletionSystemMessage
    | litellm.ChatCompletionUserMessage
)


class DecodingError[E: Exception](abc.ABC, Exception):
    """Base class for decoding errors that can occur during LLM response processing."""

    original_error: E

    @abc.abstractmethod
    def to_feedback_message(self, *, include_traceback: bool = True) -> Message:
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

    def to_feedback_message(
        self, *, include_traceback: bool = True
    ) -> litellm.ChatCompletionToolMessage:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return {
            "role": "tool",
            "tool_call_id": self.raw_tool_call.id,
            "content": error_message,
        }


@dataclasses.dataclass
class ResultDecodingError[E: Exception](DecodingError[E]):
    """Error raised when decoding the LLM response result fails."""

    original_error: E
    raw_message: Message

    def __str__(self) -> str:
        return f"Error decoding response: {self.original_error}. Please provide a valid response and try again."

    def to_feedback_message(
        self, *, include_traceback: bool = True
    ) -> litellm.ChatCompletionUserMessage:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return {"role": "user", "content": error_message}


@dataclasses.dataclass
class ToolCallExecutionError[E: Exception, T](DecodingError[E]):
    """Error raised when a tool execution fails at runtime."""

    original_error: E
    raw_tool_call: DecodedToolCall[T]

    def __str__(self) -> str:
        return f"Tool execution failed: Error executing tool '{self.raw_tool_call.name}': {self.original_error}"

    def to_feedback_message(
        self, *, include_traceback: bool = True
    ) -> litellm.ChatCompletionToolMessage:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return {
            "role": "tool",
            "tool_call_id": self.raw_tool_call.id,
            "content": error_message,
        }


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


type AssistantResult[T] = tuple[
    litellm.ChatCompletionAssistantMessage,
    typing.Sequence[DecodedToolCall],
    T | None,
]


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
    from effectful.handlers.llm.harness.transaction import HistoryBuilder

    name2tool = {t.__name__: t for t in tools}
    assert len(tools) == len(name2tool), "Tool name collision detected"
    env = {
        _TOOLS_KEY: name2tool,
        _IS_FINAL_KEY: False,
        _TYPE_CHECK_ANCHOR_KEY: anchor,
        **env,
    }
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
        messages=list(HistoryBuilder.get_history().values()),
        response_format=None if response_type is str else response_format,
        tools=tool_specs,
    )
    choice = response.choices[0]
    assert isinstance(choice, litellm.types.utils.Choices)

    message: litellm.Message = choice.message

    raw_message = typing.cast(
        litellm.ChatCompletionAssistantMessage, message.model_dump(mode="json")
    )
    assert raw_message["role"] == "assistant"

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
        serialized_result = message.get("content") or message.get("reasoning_content")
        assert isinstance(serialized_result, str)
        try:
            if force_tool:
                raise ValueError(
                    "tool_choice='required' but the model returned no tool call."
                    "**IMPORTANT: YOU MUST GENERATE A TOOL CALL IN YOUR NEXT RESPONSE.**"
                )

            if response_type is str:
                result = typing.cast(T, serialized_result)
            else:
                # Add the type-check anchor to the decode context only (not `env`,
                # which is exposed as tools), so a synthesized result is checked
                # against the Template's source.
                result = response_format.model_validate(
                    json.loads(serialized_result),
                    context={
                        **env,
                        _IS_FINAL_KEY: True,
                    },
                ).value
        except Exception as e:
            raise ResultDecodingError(e, raw_message=raw_message) from e

    return (raw_message, tool_calls, result)


type ToolResult[T] = tuple[litellm.ChatCompletionToolMessage, T | None, bool]


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
    message = litellm.ChatCompletionToolMessage(
        role="tool",
        content=encoded_result,  # type: ignore
        tool_call_id=tool_call.id,
    )
    return (message, result, isinstance(tool_call.tool, FinalTool))


@Operation.define
def call_user(
    prompt_template: str,
    env: collections.abc.Mapping[str, typing.Any],
) -> litellm.ChatCompletionUserMessage:
    """
    Format a `Template`'s prompt applied to arguments into a user message.
    """
    parts = format_as_content_blocks(prompt_template, env)
    message = litellm.ChatCompletionUserMessage(role="user", content=parts)
    return message


@Operation.define
def call_system(
    template: Template, *, tool_types: collections.abc.Set[type[Tool]] = frozenset()
) -> litellm.ChatCompletionSystemMessage:
    """Assemble and install the system message (a Markdown document)."""
    from effectful.handlers.llm.harness.context import (
        _system_agent_block,
        _system_global_block,
        _system_imports_block,
        _system_module_block,
        _system_vars_block,
    )

    sections = [
        _system_global_block(tool_types),
        _system_module_block(inspect.getmodule(template)),
        _system_agent_block(template),
        _system_imports_block(template.__context__),
        _system_vars_block(template.__context__),
    ]
    content = "\n\n".join(s for s in sections if s)
    message = litellm.ChatCompletionSystemMessage(
        role="system", content=content, cache_control={"type": "ephemeral"}
    )
    return message
