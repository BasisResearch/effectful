import abc
import collections
import collections.abc
import dataclasses
import functools
import json
import traceback
import typing

import litellm
import pydantic

from effectful.handlers.llm.harness.serialization import (
    _IS_FINAL_KEY,
    _NAME2TOOL_KEY,
    _TYPE_CHECK_ANCHOR_KEY,
    DecodedToolCall,
    PromptSection,
    _advertised_names,
    _BoxedResponse,
    _NameAndTool,
    _render_system_prompt,
    format_as_content_blocks,
    to_content_blocks,
)
from effectful.handlers.llm.types import (
    Agent,
    Encodable,
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


type AssistantResult[T] = tuple[
    litellm.ChatCompletionAssistantMessage,
    typing.Sequence[DecodedToolCall],
    T | None,
]


def _tools_in_scope(
    env: collections.abc.Mapping[str, typing.Any],
    *,
    seen: frozenset[int] = frozenset(),
) -> collections.abc.Set[Tool]:
    """
    Return the tools available to a Template given its lexical context.

    Default rule: `Tool` and `Template` values bound directly in `env`, plus
    those reachable through any `Agent` instance in `env` -- whatever is bound
    on the instance or declared on its class, and, recursively, the tools of any
    `Agent` those in turn hold.  `seen` guards that recursion against reference
    cycles; it is internal, and callers pass only `env`.

    Reaching through a nested `Agent` flattens its whole toolset into the result.
    That is what makes holding one as an attribute a way to compose tools, and it
    is why a specialised sub-agent is better left a bare `Template`, whose own
    scope stays its own.

    Tools are identified by object, so the same `Tool` visible under several
    bindings appears once.  The name each one is offered under is assigned by
    :func:`_advertised_names`, not taken from the binding name.
    """
    result: set[Tool] = set()

    for name, obj in env.items():
        if not name.isidentifier():
            continue
        if isinstance(obj, Tool | Template):
            result.add(obj)
        elif isinstance(obj, Agent) and id(obj) not in seen:
            seen |= {id(obj)}
            result |= _tools_in_scope(
                vars(obj)
                | {k: getattr(obj, k) for cls in type(obj).__mro__ for k in vars(cls)},
                seen=seen,
            )

    return frozenset(result)


@Operation.define
def call_assistant[T](
    messages: collections.abc.Sequence[Message],
    response_type: type[T],
    env: collections.abc.Mapping[str, typing.Any],
    tools: collections.abc.Set[Tool] = frozenset(),
) -> AssistantResult[T]:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    The request is fully determined by the arguments: `messages` is the
    conversation sent to the model, so the rule reads no ambient history and a
    caller (or an intercepting handler) decides exactly what the model sees.

    The available `tools` are passed explicitly as a set; handlers that expose
    additional tools (synthetic readers, REPL access, synthesis) intercept this
    operation and union them into `tools` before forwarding.

    Raises:
        ToolCallDecodingError: If a tool call cannot be decoded. The error
            includes the raw assistant message for retry handling.
        ResultDecodingError: If the result cannot be decoded. The error
            includes the raw assistant message for retry handling.
    """

    tools = tools | _tools_in_scope(env)
    if _TYPE_CHECK_ANCHOR_KEY in env:
        tools = tools - {env[_TYPE_CHECK_ANCHOR_KEY]}

    name2tool = _advertised_names(tools)
    env = {_NAME2TOOL_KEY: name2tool, **env}
    spec_encoding: pydantic.TypeAdapter[_NameAndTool] = pydantic.TypeAdapter(
        Encodable[_NameAndTool]
    )
    tool_specs = [
        spec_encoding.dump_python(_NameAndTool(name, t), mode="json")
        for name, t in sorted(name2tool.items())
    ]

    # The OpenAI API requires a wrapper object for non-object structured output types,
    # so we create one on the fly here. Using a Pydantic model offloads JSON schema
    # generation and validation logic to litellm, and offers better error messages.
    response_format: type[_BoxedResponse[T]] = pydantic.create_model(
        "BoxedResponse",
        value=Encodable[response_type],  # type: ignore[valid-type]
        __base__=_BoxedResponse,
    )

    response: litellm.types.utils.ModelResponse = completion(
        messages=list(messages),
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
            if response_type is str:
                result = typing.cast(T, serialized_result)
            else:
                result = response_format.model_validate(
                    json.loads(serialized_result),
                    context={**env, _IS_FINAL_KEY: True},
                ).value
        except Exception as e:
            raise ResultDecodingError(e, raw_message=raw_message) from e

    return (raw_message, tool_calls, result)


type ToolResult[T] = tuple[
    litellm.ChatCompletionToolMessage, T | ToolCallExecutionError, bool
]


@Operation.define
def call_tool[T](tool_call: DecodedToolCall[T]) -> ToolResult[T]:
    """Implements a roundtrip call to a python function. Input is a json
    string representing an LLM tool call request parameters. The output is
    the serialised response to the model.

    Returns the appended tool message, the tool's return value, and whether the
    call finalizes the Template -- always ``False`` here. Finalization is a policy
    a handler of this operation applies to its own tools, not a property of the
    tool's type: see
    `effectful.handlers.llm.harness.synthesis.body.FinalBodySynthesizer`, which
    marks its ``submit_solution`` call final so that value becomes the Template's
    result and the completion loop stops.

    The returned value is a :class:`ToolCallExecutionError` rather than the tool's
    result when a handler captured a failed call (see
    `effectful.handlers.llm.harness.durability.TenacityRetryer`); this rule itself
    raises instead.
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
    return (message, result, False)


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
    harness_prompt: PromptSection, agent_prompt: PromptSection
) -> litellm.ChatCompletionSystemMessage:
    """Assemble the system message from the two halves of the system prompt.

    `agent_prompt` describes the task: the caller (`LiteLLMProvider._call`)
    introspects it from the `Template` being called.  `harness_prompt` describes
    the machinery the task runs under, and arrives empty: each installed handler
    with something to say about the harness intercepts this operation and adds the
    section documenting the capability it provides.  This rule only has to put the
    two together and flatten the result.

    Handing the handlers their own argument is what keeps them independent.  A
    handler appends to `harness_prompt` and forwards; it never looks a section up
    by title, never has to create one that isn't there, and cannot disturb the
    other half of the document.  Installation order therefore decides only the
    order the sections appear in -- innermost first, since it intercepts first --
    and never whether one of them lands.
    """
    document = PromptSection(
        type="prompt_section",
        title=f"System Prompt: {agent_prompt['title']}",
        content=[harness_prompt, agent_prompt],
    )
    return litellm.ChatCompletionSystemMessage(
        role="system", content=list(_render_system_prompt(document))
    )
