"""Send the agent loop's model requests through any_llm.

`AnyLLMConfigurer` configures each request. The rest of this module serves the
requests OpenAI's newer reasoning models refuse on ``/v1/chat/completions`` --
function tools while reasoning is on, which for some is always -- through its
Responses API instead: `needs_responses_api` says which requests those are, and
`completion_via_responses` sends one there and answers in Chat Completions
types, so nothing that calls or intercepts `completion` can tell.
"""

import collections
import collections.abc
import inspect
import os
import re
import typing
import urllib.parse

import any_llm
import pydantic
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    CompletionTokensDetails,
    CompletionUsage,
    Function,
    PromptTokensDetails,
    Reasoning,
    ReasoningEffort,
)
from any_llm.types.completion import Choice as ChatChoice
from any_llm.utils.streaming import accumulate_chunks
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseIncompleteEvent,
    ResponseInputContentParam,
    ResponseInputItemParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ResponseTextDeltaEvent,
    ResponseUsage,
    ToolChoiceFunctionParam,
    ToolChoiceOptions,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.shared_params import (
    Reasoning as ReasoningParam,
)
from openai.types.shared_params import (
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatText,
)

from effectful.handlers.llm.harness.hooks import (
    Message,
    ResultDecodingError,
    assistant_message,
    completion,
)
from effectful.handlers.llm.harness.serialization import ContentBlock, TextBlock
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


def any_llm_model(model: str) -> str:
    """`model` in any_llm's ``provider:model`` form.

    A ``provider/model`` string has its first ``/`` read as the provider
    separator, and a bare model name is taken to be OpenAI's.
    """
    colon, slash = model.find(":"), model.find("/")
    if colon != -1 and (slash == -1 or colon < slash):
        return model
    if slash != -1:
        return f"{model[:slash]}:{model[slash + 1 :]}"
    return f"openai:{model}"


def _with_retries(
    model: str, num_retries: int, client_args: dict[str, typing.Any] | None
) -> dict[str, typing.Any] | None:
    """`client_args` with ``max_retries`` set, where the provider has one.

    Providers without a client-level retry count reject ``max_retries``
    outright, so for them the value is left out.
    """
    provider, _ = any_llm.AnyLLM.split_model_provider(model)
    if any_llm.AnyLLM.get_provider_class(provider).MAX_RETRIES_SUPPORT == "unsupported":
        return client_args
    return {"max_retries": num_retries, **(client_args or {})}


class AnyLLMConfigurer(ObjectInterpretation):
    """Configures requests sent through any_llm, and enforces the parts of that
    configuration that a provider may ignore (see `_enforce_tool_choice`)."""

    config: collections.abc.Mapping[str, typing.Any]

    def __init__(self, model="openai:gpt-4o", **config):
        self.config = {
            "model": model,
            **inspect.signature(any_llm.completion).bind_partial(**config).kwargs,
        }

    @staticmethod
    def _mark(msg: Message) -> Message:
        """`msg`, with a `cache_control` breakpoint on its last content block.

        String content is first made a single text block: any_llm reads a
        breakpoint off a message's content parts on every provider, but some
        (Anthropic) drop a message-level one.
        """
        content = msg.get("content")
        if isinstance(content, str):
            content = [TextBlock(type="text", text=content)]
        if not isinstance(content, list) or not content:
            return msg
        last_block = content[-1]
        if not isinstance(last_block, dict) or "cache_control" in last_block:
            return msg
        return typing.cast(
            Message,
            {
                **msg,
                "content": [
                    *content[:-1],
                    {**last_block, "cache_control": {"type": "ephemeral"}},
                ],
            },
        )

    def _add_cache_control(
        self,
        messages: collections.abc.Sequence[Message],
    ) -> list[Message]:
        """Mark the system message and the last user/tool message of a request
        for prompt caching.

        A `cache_control` breakpoint caches the request prefix -- tools, system,
        and messages, in that order -- up to and including the block it sits on,
        and a later request that shares that prefix reads it back at a fraction
        of the input price. Putting a breakpoint on the *last* input message
        therefore caches as much of a growing conversation as possible: an
        agent's accumulated history, or the rounds of a single tool-use loop.
        The system prompt gets one of its own because it is the largest constant
        part of every request and is worth caching even on the first round of a
        conversation, before there is any history to extend.

        Exactly two breakpoints are added, and the second moves to the newest
        message on every request. That matters in both directions: providers cap
        how many breakpoints a request may carry (Anthropic allows four), and a
        breakpoint pinned to an old position stops extending the cached prefix
        as the conversation grows.

        Returns a new list, leaving the caller's messages untouched, so these
        transport-level annotations never reach the stored history -- and so
        never reach an `Agent`'s checkpointed transcript.
        """
        out = list(messages)
        for i in reversed(range(len(out))):
            if out[i]["role"] in ("user", "tool"):
                out[i] = self._mark(out[i])
                break
        for i in range(len(out)):
            if out[i]["role"] == "system":
                out[i] = self._mark(out[i])
                break
        return out

    @staticmethod
    def _enforce_tool_choice(
        tool_choice: typing.Any,
        response: ChatCompletion,
    ) -> None:
        """Reject a response that disobeys the ``tool_choice`` it was sent.

        Some OpenAI-compatible servers treat ``tool_choice`` as advisory, in
        either direction: a ``"required"`` request answered in prose, whose reply
        would otherwise be decoded as a bare structured result, or a ``"none"``
        request answered with a tool call, whose reply would otherwise be
        executed. Both are silent protocol violations, so both are raised as a
        `ResultDecodingError`, which `HistoryBuilder` turns into feedback and
        `TenacityRetryer` retries, like any other malformed response.

        The other `tool_choice` values are `None`/``"auto"``, which permit either
        shape, and the ``{"type": "function", ...}`` form, whose fidelity is a
        question about *which* tool was called and is left to decoding.
        """
        tool_calls = response.choices[0].message.tool_calls or []
        if tool_choice == "required" and not tool_calls:
            error = ValueError(
                "tool_choice='required' but the model returned no tool call."
                "**IMPORTANT: YOU MUST GENERATE A TOOL CALL IN YOUR NEXT RESPONSE.**"
            )
        elif tool_choice == "none" and tool_calls:
            error = ValueError(
                f"tool_choice='none' but the model returned {len(tool_calls)} tool call(s)."
                "**IMPORTANT: YOU MUST ANSWER DIRECTLY, WITHOUT CALLING A TOOL, "
                "IN YOUR NEXT RESPONSE.**"
            )
        else:
            return
        raise ResultDecodingError(error, raw_message=assistant_message(response))

    @classmethod
    def _enforce_tool_choice_on_stream(
        cls,
        tool_choice: typing.Any,
        stream: collections.abc.Iterable[ChatCompletionChunk],
    ) -> collections.abc.Iterator[ChatCompletionChunk]:
        """`stream`, re-yielded, with `_enforce_tool_choice` applied once it ends.

        A consumer that streams -- `RichTerminalRenderer` -- is the only thing
        that ever sees a whole response here, since the chunks are all this
        handler gets back. Reassembling them the same way it does keeps the check
        on the request it belongs to, and the error surfaces from the consumer's
        own iteration rather than from a later, unrelated place. That second
        reassembly is why this is worth doing only on the streaming path, which
        is a debugging one.
        """
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
            yield chunk
        cls._enforce_tool_choice(tool_choice, accumulate_chunks(chunks))

    @implements(completion)
    def completion(self, *args, **kwargs):
        """Inject the provider's configuration into the low-level request before
        delegating, and hold the response to the ``tool_choice`` that request
        carries.

        The merge below lets a value already in `kwargs` stand, so an enclosed
        configurer's ``tool_choice`` is the one any_llm is sent -- and, being the
        one sent, the only one there is anything to enforce about. Reading it
        back off the merged request rather than off `self.config` is what keeps
        those two in agreement.
        """
        kwargs = {**self.config, **kwargs}
        kwargs["model"] = any_llm_model(kwargs["model"])
        # Provider failures surface as any_llm's own exception types.
        kwargs["client_args"] = {
            "unified_exceptions": True,
            **(kwargs.get("client_args") or {}),
        }
        if (num_retries := kwargs.pop("num_retries", None)) is not None:
            kwargs["client_args"] = _with_retries(
                kwargs["model"], num_retries, kwargs.get("client_args")
            )
        if kwargs.get("messages"):
            kwargs["messages"] = self._add_cache_control(kwargs["messages"])
        response = fwd(*args, **kwargs)
        if not isinstance(response, ChatCompletion):
            return self._enforce_tool_choice_on_stream(
                kwargs.get("tool_choice"), response
            )
        self._enforce_tool_choice(kwargs.get("tool_choice"), response)
        return response


_EXTRA_CONTENT_KEY = "openai_responses"
"""The `extra_content` namespace an assistant message keeps its Responses
reasoning items under, to be sent back with the rest of the conversation."""

_GPT_VERSION = re.compile(r"^gpt-(\d+)(?:\.(\d+))?(?=[.-]|$)")

type _ResponseFormat = (
    ResponseFormatJSONSchema | ResponseFormatJSONObject | ResponseFormatText
)


def _is_openai_endpoint(api_base: str | None) -> bool:
    """Whether requests are served by OpenAI itself, rather than by another
    server that speaks its API."""
    base = api_base or os.environ.get("OPENAI_BASE_URL")
    if not base:
        return True
    host = urllib.parse.urlsplit(base).hostname or ""
    return host == "api.openai.com" or host.endswith(".api.openai.com")


def needs_responses_api(
    model: str,
    messages: typing.Any = None,
    *,
    tools: collections.abc.Sequence[ChatCompletionFunctionToolParam] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    api_base: str | None = None,
    **kwargs,
) -> bool:
    """Whether OpenAI refuses this `completion` request on Chat Completions.

    It refuses function tools on GPT-5.4 and later whenever reasoning is on,
    and on OpenAI's own endpoints reasoning is on unless asked off. Another
    server that speaks OpenAI's API may serve these models' tools without
    reasoning, so there only an explicit effort counts. The rule is litellm's
    (``litellm.main.responses_api_bridge_check``), keyed on model names because
    OpenAI exposes nothing to ask instead.
    """
    if ":" not in model and "/" not in model:
        return False
    provider, name = any_llm.AnyLLM.split_model_provider(model)
    if provider not in ("openai", "azureopenai") or "search" in name:
        return False
    version = _GPT_VERSION.match(name)
    if version is None or (int(version[1]), int(version[2] or 0)) < (5, 4):
        return False
    if not any(tool.get("type") == "function" for tool in tools or ()):
        return False
    if reasoning_effort == "none":
        return False
    explicit = reasoning_effort not in (None, "auto")
    return explicit or provider == "azureopenai" or _is_openai_endpoint(api_base)


def _input_parts(
    content: collections.abc.Sequence[ContentBlock],
) -> list[ResponseInputContentParam]:
    """Chat content parts as Responses input content parts."""
    parts: list[ResponseInputContentParam] = []
    for block in content:
        if block["type"] == "text":
            parts.append({"type": "input_text", "text": block["text"]})
        elif block["type"] == "image_url":
            image = block["image_url"]
            parts.append(
                {
                    "type": "input_image",
                    "image_url": image["url"],
                    "detail": image.get("detail", "auto"),
                }
            )
        elif block["type"] == "file":
            parts.append({"type": "input_file", **block["file"]})
        else:
            raise ValueError(f"No Responses API form for a {block['type']!r} part")
    return parts


def _input_content(
    content: str | collections.abc.Sequence[ContentBlock],
) -> str | list[ResponseInputContentParam]:
    return content if isinstance(content, str) else _input_parts(content)


def _input_items(
    messages: collections.abc.Sequence[Message],
) -> list[ResponseInputItemParam]:
    """Chat messages as Responses input items."""
    items: list[ResponseInputItemParam] = []
    for message in messages:
        if message["role"] == "tool":
            items.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=message["tool_call_id"],
                    # The tool-output part types are the input part types,
                    # under other names.
                    output=typing.cast(typing.Any, _input_content(message["content"])),
                )
            )
        elif message["role"] == "assistant":
            stored = message.get("extra_content", {}).get(_EXTRA_CONTENT_KEY, {})
            reasoning: list[ResponseReasoningItemParam] = stored.get("reasoning", [])
            items += reasoning
            if content := message.get("content"):
                items.append(
                    EasyInputMessageParam(
                        type="message", role="assistant", content=content
                    )
                )
            for call in message.get("tool_calls", []):
                items.append(
                    ResponseFunctionToolCallParam(
                        type="function_call",
                        call_id=call["id"],
                        name=call["function"]["name"],
                        arguments=call["function"]["arguments"],
                    )
                )
        else:
            items.append(
                EasyInputMessageParam(
                    type="message",
                    role=message["role"],
                    content=_input_content(message["content"]),
                )
            )
    return items


def _tool(tool: ChatCompletionFunctionToolParam) -> FunctionToolParam:
    function = tool["function"]
    converted = FunctionToolParam(
        type="function",
        name=function["name"],
        parameters=dict(function.get("parameters") or {}),
        strict=function.get("strict"),
    )
    if "description" in function:
        converted["description"] = function["description"]
    return converted


def _tool_choice(
    choice: ChatCompletionToolChoiceOptionParam,
) -> ToolChoiceOptions | ToolChoiceFunctionParam:
    if isinstance(choice, str):
        return choice
    elif choice["type"] == "function":
        return ToolChoiceFunctionParam(type="function", name=choice["function"]["name"])
    else:
        raise ValueError(f"No Responses API form for tool_choice {choice!r}")


def _text_config(response_format: _ResponseFormat) -> ResponseTextConfigParam:
    if response_format["type"] == "json_schema":
        schema = response_format["json_schema"]
        return {
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": dict(schema.get("schema") or {}),
                "strict": schema.get("strict"),
            }
        }
    elif response_format["type"] == "json_object":
        return {"format": {"type": "json_object"}}
    else:
        return {"format": {"type": "text"}}


def _usage(usage: ResponseUsage | None) -> CompletionUsage | None:
    if usage is None:
        return None
    return CompletionUsage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=usage.input_tokens_details.cached_tokens,
            cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
        ),
        completion_tokens_details=CompletionTokensDetails(
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens
        ),
    )


def _finish_reason(
    response: Response, calls: bool
) -> typing.Literal["stop", "length", "tool_calls", "content_filter"]:
    if response.status == "incomplete":
        details = response.incomplete_details
        if details is not None and details.reason == "content_filter":
            return "content_filter"
        return "length"
    return "tool_calls" if calls else "stop"


def _reasoning_param(item: ResponseReasoningItem) -> ResponseReasoningItemParam:
    """A reasoning output item in the form it is sent back as input."""
    param = ResponseReasoningItemParam(
        type="reasoning",
        id=item.id,
        summary=[{"type": "summary_text", "text": s.text} for s in item.summary],
    )
    if item.encrypted_content:
        param["encrypted_content"] = item.encrypted_content
    return param


def _extra_content(
    reasoning: collections.abc.Sequence[ResponseReasoningItemParam],
) -> dict[str, typing.Any]:
    return {_EXTRA_CONTENT_KEY: {"reasoning": list(reasoning)}}


def _as_response(value: pydantic.BaseModel) -> Response:
    """`value` as OpenAI's own `Response`, whichever form any_llm returned."""
    if isinstance(value, Response):
        return value
    return Response.model_validate(value.model_dump())


def _chat_completion(response: Response) -> ChatCompletion:
    """A finished Responses reply as a `ChatCompletion` with one choice."""
    content: list[str] = []
    refusal: list[str] = []
    summaries: list[str] = []
    reasoning: list[ResponseReasoningItemParam] = []
    calls: list[ChatCompletionMessageFunctionToolCall] = []
    for item in response.output:
        if isinstance(item, ResponseReasoningItem):
            reasoning.append(_reasoning_param(item))
            summaries += [s.text for s in item.summary]
        elif isinstance(item, ResponseOutputMessage):
            for part in item.content:
                if isinstance(part, ResponseOutputText):
                    content.append(part.text)
                elif isinstance(part, ResponseOutputRefusal):
                    refusal.append(part.refusal)
        elif isinstance(item, ResponseFunctionToolCall):
            calls.append(
                ChatCompletionMessageFunctionToolCall(
                    id=item.call_id,
                    type="function",
                    function=Function(name=item.name, arguments=item.arguments),
                )
            )
    message = ChatCompletionMessage(
        role="assistant",
        content="".join(content) or None,
        refusal="".join(refusal) or None,
        tool_calls=list(calls) or None,
        reasoning=Reasoning(content="\n\n".join(summaries)) if summaries else None,
        extra_content=_extra_content(reasoning) if reasoning else None,
    )
    return ChatCompletion(
        id=response.id,
        object="chat.completion",
        created=int(response.created_at),
        model=response.model,
        choices=[
            ChatChoice(
                index=0,
                message=message,
                finish_reason=_finish_reason(response, bool(calls)),
            )
        ],
        usage=_usage(response.usage),
    )


def _chunks(
    events: collections.abc.Iterable[ResponseStreamEvent],
) -> collections.abc.Iterator[ChatCompletionChunk]:
    """Responses stream events as Chat Completions chunks.

    Only the final event carries a `finish_reason` and the usage, as a Chat
    Completions stream does.
    """
    opened: Response | None = None
    tool_index: dict[str, int] = {}

    def chunk(
        delta: ChoiceDelta,
        finish_reason: str | None = None,
        usage: CompletionUsage | None = None,
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=opened.id if opened else "",
            created=int(opened.created_at) if opened else 0,
            model=opened.model if opened else "",
            object="chat.completion.chunk",
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
            usage=usage,
        )

    for event in events:
        if isinstance(event, ResponseCreatedEvent):
            opened = event.response
            yield chunk(ChoiceDelta(role="assistant"))
        elif isinstance(event, ResponseOutputItemAddedEvent) and isinstance(
            event.item, ResponseFunctionToolCall
        ):
            item = event.item
            tool_index[item.id or ""] = index = len(tool_index)
            call = ChoiceDeltaToolCall(
                index=index,
                id=item.call_id,
                type="function",
                function=ChoiceDeltaToolCallFunction(name=item.name, arguments=""),
            )
            yield chunk(ChoiceDelta(tool_calls=[call]))
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            call = ChoiceDeltaToolCall(
                index=tool_index[event.item_id],
                function=ChoiceDeltaToolCallFunction(arguments=event.delta),
            )
            yield chunk(ChoiceDelta(tool_calls=[call]))
        elif isinstance(event, ResponseTextDeltaEvent):
            yield chunk(ChoiceDelta(content=event.delta))
        elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
            yield chunk(ChoiceDelta(reasoning=Reasoning(content=event.delta)))
        elif isinstance(event, ResponseOutputItemDoneEvent) and isinstance(
            event.item, ResponseReasoningItem
        ):
            extra = _extra_content([_reasoning_param(event.item)])
            yield chunk(ChoiceDelta(extra_content=extra))
        elif isinstance(event, ResponseCompletedEvent | ResponseIncompleteEvent):
            yield chunk(
                ChoiceDelta(),
                finish_reason=_finish_reason(event.response, bool(tool_index)),
                usage=_usage(event.response.usage),
            )
        elif isinstance(event, ResponseFailedEvent | ResponseErrorEvent):
            error = (
                event.response.error
                if isinstance(event, ResponseFailedEvent)
                else event
            )
            raise any_llm.ProviderError(
                error.message if error else "The Responses API stream failed.",
                provider_name="openai",
            )


def completion_via_responses(
    model: str,
    messages: collections.abc.Sequence[Message],
    *,
    tools: collections.abc.Sequence[ChatCompletionFunctionToolParam] | None = None,
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
    response_format: _ResponseFormat | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    stream: bool | None = None,
    stream_options: typing.Any = None,
    **kwargs,
) -> ChatCompletion | collections.abc.Iterator[ChatCompletionChunk]:
    """Answer a Chat Completions request through the Responses API, in Chat
    Completions types.

    A reply's reasoning items are kept in its message's ``extra_content`` and
    sent back with the conversation, so a tool loop keeps its reasoning across
    turns. ``stream_options`` is dropped: a Responses stream always reports its
    usage.
    """
    request: dict[str, typing.Any] = {
        "model": model,
        "input_data": _input_items(messages),
        **kwargs,
    }
    if tools:
        request["tools"] = [_tool(tool) for tool in tools]
    if tool_choice is not None:
        request["tool_choice"] = _tool_choice(tool_choice)
    if response_format is not None:
        request["text"] = _text_config(response_format)
    if reasoning_effort not in (None, "auto"):
        request["reasoning"] = ReasoningParam(
            effort=typing.cast(typing.Any, reasoning_effort)
        )
    if (max_output := max_completion_tokens or max_tokens) is not None:
        request["max_output_tokens"] = max_output
    if stream:
        events = typing.cast(
            collections.abc.Iterator[ResponseStreamEvent],
            any_llm.responses(**request, stream=True),
        )
        return _chunks(events)
    reply = any_llm.responses(**request)
    assert isinstance(reply, pydantic.BaseModel)
    return _chat_completion(_as_response(reply))
