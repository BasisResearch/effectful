import collections
import collections.abc
import inspect
import typing

import litellm
from litellm.types.llms.openai import ChatCompletionToolChoiceValues

from effectful.handlers.llm.harness.context import template_system_prompt
from effectful.handlers.llm.harness.hooks import (
    Message,
    ResultDecodingError,
    call_assistant,
    call_system,
    call_tool,
    call_user,
    completion,
)
from effectful.handlers.llm.harness.serialization import _TYPE_CHECK_ANCHOR_KEY
from effectful.handlers.llm.harness.transaction import HistoryBuilder
from effectful.handlers.llm.types import Template
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class LiteLLMConfigurer(ObjectInterpretation):
    """Configures the LiteLLM API, and enforces the parts of that configuration
    that a provider may ignore (see `_enforce_tool_choice`)."""

    config: collections.abc.Mapping[str, typing.Any]

    def __init__(self, model="gpt-4o", **config):
        self.config = {
            "model": model,
            **inspect.signature(litellm.completion).bind_partial(**config).kwargs,
        }

    @staticmethod
    def _mark(msg: Message) -> Message:
        """`msg`, with a `cache_control` breakpoint on its last content block.

        A message whose content is a plain string -- which the assembled system
        prompt is not, but a hand-written or externally supplied message may be
        -- takes the message-level key instead, the only form litellm reads a
        breakpoint from for string content.
        """
        content = msg.get("content")
        if isinstance(content, str):
            return typing.cast(Message, {**msg, "cache_control": {"type": "ephemeral"}})
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
        tool_choice: ChatCompletionToolChoiceValues | None,
        response: litellm.types.utils.ModelResponse,
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
        choice = response.choices[0]
        assert isinstance(choice, litellm.types.utils.Choices)
        tool_calls = choice.message.get("tool_calls") or []
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
        raise ResultDecodingError(
            error,
            raw_message=typing.cast(
                litellm.ChatCompletionAssistantMessage,
                choice.message.model_dump(mode="json"),
            ),
        )

    @classmethod
    def _enforce_tool_choice_on_stream(
        cls,
        tool_choice: ChatCompletionToolChoiceValues | None,
        stream: collections.abc.Iterable[typing.Any],
        messages: list[Message] | None,
    ) -> collections.abc.Iterator[typing.Any]:
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
        assembled = litellm.stream_chunk_builder(chunks, messages=messages)
        if isinstance(assembled, litellm.types.utils.ModelResponse):
            cls._enforce_tool_choice(tool_choice, assembled)

    @implements(completion)
    def _completion(self, *args, **kwargs):
        """Inject the provider's configuration (model and bound litellm kwargs)
        into the low-level request before delegating, and hold the response to
        the ``tool_choice`` that request carries.

        The merge below lets a value already in `kwargs` stand, so an enclosed
        configurer's ``tool_choice`` is the one litellm is sent -- and, being the
        one sent, the only one there is anything to enforce about. Reading it
        back off the merged request rather than off `self.config` is what keeps
        those two in agreement.
        """
        kwargs = {**self.config, **kwargs}
        if kwargs.get("messages"):
            kwargs["messages"] = self._add_cache_control(kwargs["messages"])
        response = fwd(*args, **kwargs)
        if not isinstance(response, litellm.types.utils.ModelResponse):
            return self._enforce_tool_choice_on_stream(
                kwargs.get("tool_choice"), response, kwargs.get("messages")
            )
        self._enforce_tool_choice(kwargs.get("tool_choice"), response)
        return response


class LiteLLMProvider(LiteLLMConfigurer):
    """Implements templates using the LiteLLM API."""

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        message: Message = call_system(template_system_prompt(template))

        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(
            bound_args.arguments | {_TYPE_CHECK_ANCHOR_KEY: template}
        )

        header = f"{template.__name__}{template.__signature__}".replace(
            "{", "{{"
        ).replace("}", "}}")
        assert template.__doc__ is not None
        prompt_template = header + "\n\n" + template.__doc__
        message = call_user(prompt_template, env)

        result: T | None = None
        is_final: bool = False
        while not is_final:
            message, tool_calls, result = call_assistant(
                list(HistoryBuilder.get_history().values()),
                env,
                template.__signature__.return_annotation,
            )
            if tool_calls:
                for tool_call in tool_calls:
                    message, result, is_final = call_tool(tool_call)
                    if is_final:
                        # Stop here rather than run the rest of the turn: a later
                        # call would overwrite `result` and `is_final`, silently
                        # discarding the answer. Which call in a mixed turn is the
                        # answer is genuinely ambiguous, so say so instead of
                        # picking one -- a finalizing tool tells the model to call
                        # it alone (see `FinalBodySynthesizer`), and a turn that
                        # ignores that is a defect worth surfacing.
                        assert len(tool_calls) == 1, (
                            f"a finalizing tool call must be the only call in its "
                            f"turn, but {len(tool_calls)} were requested"
                        )
                        break
            else:
                is_final = True

        return typing.cast(T, result)
