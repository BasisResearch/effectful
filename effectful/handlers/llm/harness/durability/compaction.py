"""Threshold-driven, lossy compaction of the middle of an agent transcript."""

import collections.abc
import json
import typing
from collections.abc import Sequence

from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
from effectful.handlers.llm.harness.durability.truncation import _truncate_content
from effectful.handlers.llm.harness.hooks import Message, call_assistant, completion
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements

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
