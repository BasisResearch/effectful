"""Threshold-driven compaction of an agent transcript: lossy, by the harness, or
forced on the model."""

import collections.abc
import dataclasses
import enum
import json
import typing
from collections.abc import Sequence

import litellm

from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
from effectful.handlers.llm.harness.durability.truncation import _truncate_content
from effectful.handlers.llm.harness.hooks import (
    AssistantResult,
    Message,
    PromptInjectingInterpretation,
    ResultDecodingError,
    ToolCallDecodingError,
    call_assistant,
    completion,
)
from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
from effectful.handlers.llm.harness.serialization import ToolCallID
from effectful.handlers.llm.types import Tool
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements


class CompactionScope(enum.StrEnum):
    """How much of the conversation a compacting tool call drops.

    ``"none"`` compacts nothing. ``"turn"`` drops the current call's earlier rounds,
    keeping every previous call. ``"conversation"`` additionally drops those previous
    calls, leaving the system message, the request and the asking round.
    """

    NONE = "none"
    TURN = "turn"
    CONVERSATION = "conversation"


def compact_(
    history: collections.abc.MutableSequence[Message],
    tool_call_id: ToolCallID,
    scope: CompactionScope,
) -> None:
    """Compact a history in-place, keeping the request and the asking round.

    `tool_call_id` identifies the call that asked, and so the round to keep: the
    assistant message advertising it, and everything after (which is exactly the
    tool messages answering it and its siblings, whether they were appended
    before this one or are still to come -- truncation only ever removes messages
    *ahead* of that assistant message, so no tool message is ever orphaned from
    the call it answers).

    The request kept is the last user message before that round -- the one this
    call opened -- carried over untouched.

    A no-op for ``scope="none"``, and whenever the shape this reads off the
    history is not the one it expects: no assistant message advertising
    `tool_call_id`, or no user message ahead of it. Declining is the right
    failure here; a compaction is a courtesy, and a wrong guess about the shape
    would corrupt the history the call still has to finish over. A conversation
    that opens with something other than a system message simply has no head to
    keep, which is not a failure.
    """
    asking, request = None, None
    for i, message in reversed(list(enumerate(history))):
        if message["role"] == "assistant" and any(
            call["id"] == tool_call_id for call in message.get("tool_calls") or []
        ):
            for j in reversed(range(i)):
                if history[j]["role"] == "user":
                    asking, request = i, j
                    break
            break

    if scope == CompactionScope.NONE or asking is None or request is None:
        return
    elif scope == CompactionScope.CONVERSATION:
        history[:] = [history[0], history[request], *history[asking:]]
    elif scope == CompactionScope.TURN:
        history[:] = [*history[:request], history[request], *history[asking:]]


_SUMMARY_PREFIX = "[Earlier conversation summary]\n"


def _size(messages: collections.abc.Sequence[Message]) -> int:
    """Provider-independent, rough approximation of input tokens."""
    return sum((len(json.dumps(message, default=str)) + 3) // 4 for message in messages)


class MiddleCompactor(ObjectInterpretation):
    """Compact stale tool outputs, then summarize older rounds under context pressure.

    The system prompt, first task message and a budget-sized recent window of at
    least two assistant rounds remain verbatim. Above ``soft_tokens``, bulky
    tool output in between is truncated (not stored or recallable). If still
    above ``hard_tokens``, the remaining middle is folded into a running summary
    by a separate tool-free model call. Budgets use a provider-independent
    character/4 token estimate; they do not guarantee a request will fit a
    provider's exact tokenizer or tool schemas.

    """

    soft_tokens: int
    hard_tokens: int
    max_stale_output_chars: int
    recent_tokens: int

    def __init__(
        self,
        soft_tokens: int,
        hard_tokens: int,
        *,
        max_stale_output_chars: int = 256,
        recent_tokens: int | None = None,
    ):
        if not 0 < soft_tokens < hard_tokens:
            raise ValueError("require 0 < soft_tokens < hard_tokens")
        if max_stale_output_chars <= 0 or (
            recent_tokens is not None and recent_tokens <= 0
        ):
            raise ValueError("output, summary and recent budgets must be positive")
        self.soft_tokens = soft_tokens
        self.hard_tokens = hard_tokens
        self.max_stale_output_chars = max_stale_output_chars
        self.recent_tokens = (
            recent_tokens if recent_tokens is not None else hard_tokens // 4
        )
        if self.recent_tokens >= hard_tokens:
            raise ValueError("recent_tokens must be less than hard_tokens")

    def _recent(self, history: Sequence[Message]) -> int:
        """Keep a budget-sized recent window of at least two complete rounds.

        A round begins with an assistant message and includes all its tool answers.
        If a new user request follows, keep that request with the recent rounds.
        Never cut between a tool call and its answer.
        """
        rounds = [
            i for i in range(1, len(history)) if history[i]["role"] == "assistant"
        ]
        recent = rounds[-2] if len(rounds) >= 2 else 1
        if recent > 1 and history[recent - 1]["role"] == "user":
            recent -= 1
        while recent > 1 and _size(history[recent:]) < self.recent_tokens:
            earlier = [i for i in rounds if i < recent]
            recent = earlier[-1] if earlier else 1
            if recent > 1 and history[recent - 1]["role"] == "user":
                recent -= 1
        return recent

    def _compact_history(self, history: Sequence[Message]) -> Sequence[Message]:
        if _size(history) < self.soft_tokens:
            return history

        recent = self._recent(history)

        # truncate tool calls in the compaction window
        truncated_history = list(history)
        for i in range(1, recent):
            message = truncated_history[i]
            if message["role"] != "tool":
                continue
            content = message.get("content")
            shortened = _truncate_content(content, self.max_stale_output_chars)
            if shortened is not content:
                truncated_history[i] = typing.cast(
                    Message, {**message, "content": shortened}
                )

        if _size(truncated_history) < self.hard_tokens:
            return truncated_history

        middle = truncated_history[1:recent]
        if not middle:
            raise RuntimeError(
                "history is too large, but there are no compactable messages"
            )

        response = completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the conversation for continuing the task. "
                        "Preserve goals, decisions, files changed, errors and pending work. "
                        "The task system prompt is provided for context. "
                        "It will be retained, so do not include it in the summary. "
                        "Do not call tools. Return only the concise updated summary."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"system": history[0], "conversation": middle}, default=str
                    ),
                },
            ],
            tools=[],
            tool_choice="none",
        )
        summary = response.choices[0].message.content
        if isinstance(summary, str) and summary.strip():
            replacement: Message = {
                "role": "user",
                "content": _SUMMARY_PREFIX + summary.strip(),
            }
            summarized_history = [history[0], replacement] + list(history[recent:])
            return summarized_history
        else:
            raise RuntimeError(
                f"Expected a nonempty summary string, but got {summary!r}"
            )

    @implements(call_assistant)
    def call_assistant(self, messages, response_type, env, tools=frozenset()):
        # The loop passes a snapshot, but changes must reach the transaction's
        # buffer so subsequent requests and persisted agent history agree.
        history = HistoryBuilder.get_history()
        compact_history = self._compact_history(history)
        if compact_history is history:
            return fwd()
        history.clear()
        history.extend(compact_history)
        return fwd(list(compact_history), response_type, env, tools)


@dataclasses.dataclass
class ReplCompactor(PromptInjectingInterpretation):
    """
    This conversation has a token budget. When a request would exceed it, you
    are told so at the end of that request and must call `exec_code` with the
    `compact` scope you are given; any other reply is rejected and you are asked
    again. Use the snippet to promote anything you still need onto `self`;
    your message with the call, the snippet and its output survive the compaction.
    """

    hard_tokens: int
    scope: CompactionScope = CompactionScope.CONVERSATION

    def _over_budget(self, messages: Sequence[Message]) -> bool:
        """Whether `messages` is over budget with at least two rounds to drop.

        One round is what a previous forced compaction kept, so requiring a second
        keeps a floor above ``hard_tokens`` from forcing every round. The count uses
        litellm's default tokenizer and leaves out tool schemas, so it is approximate
        and low.
        """
        dropped = messages
        if self.scope is CompactionScope.TURN:
            request = max(
                (i for i, m in enumerate(messages) if m["role"] == "user"), default=-1
            )
            dropped = messages[request + 1 :]
        return (
            sum(m["role"] == "assistant" for m in dropped) >= 2
            and litellm.token_counter(messages=list(messages)) >= self.hard_tokens
        )

    @implements(call_assistant)
    def call_assistant[T](
        self,
        messages: Sequence[Message],
        response_type: type[T],
        env: collections.abc.Mapping[str, typing.Any],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult[T]:
        """Nudge an over-budget request to compact through `exec_code`, force the
        call, and reject any other reply.

        The tools are left as they are and the nudge is appended after the history,
        outside the stored transcript, so the request reuses the cached prefix.
        Installed inside `HistoryBuilder`, so a rejected reply is recorded with its
        feedback before the retry.
        """
        from effectful.handlers.llm.harness.synthesis.snippet import (
            StatefulReplSynthesizer,
        )

        exec_code = StatefulReplSynthesizer.exec_code
        if exec_code not in tools or not self._over_budget(messages):
            return fwd()
        name = exec_code.__name__
        nudge: Message = {
            "role": "user",
            "content": (
                f"This conversation is over its token budget. Call `{name}` with "
                f'`compact="{self.scope.value}"` now, saving anything you still '
                "need onto `self` first. ANY OTHER ACTION WILL BE REJECTED."
            ),
        }
        forced = LiteLLMConfigurer(
            model=None,
            tool_choice={"type": "function", "function": {"name": name}},
            # Lets litellm downgrade the choice to "auto" for models that reject
            # forced tool use; the check below still holds.
            drop_params=True,
        )
        with handler(forced):
            message, tool_calls, result = fwd(
                [*messages, nudge], response_type, env, tools
            )
        if not tool_calls:
            raise ResultDecodingError(
                ValueError(
                    "the conversation is over its token budget, so this round must "
                    f'call `{name}` with `compact="{self.scope.value}"`, not answer'
                ),
                raw_message=message,
            )
        raw_calls = {raw["id"]: raw for raw in message.get("tool_calls") or []}
        for call in tool_calls:
            compact = call.bound_args.arguments.get("compact")
            if call.tool is not exec_code or compact != self.scope:
                raise ToolCallDecodingError(
                    original_error=ValueError(
                        "the conversation is over its token budget, so this round "
                        f'must call `{name}` with `compact="{self.scope.value}"`, '
                        f"not `{call.name}` with `compact={compact!r}`"
                    ),
                    raw_message=message,
                    raw_tool_call=litellm.types.utils.ChatCompletionMessageToolCall(
                        **raw_calls[call.id]
                    ),
                )
        return message, tool_calls, result
