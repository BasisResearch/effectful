"""Bounds tool results before they enter the model's conversation history."""

import collections.abc
import typing

from effectful.handlers.llm.harness.hooks import ToolResult, call_tool
from effectful.handlers.llm.harness.serialization import DecodedToolCall
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements

DEFAULT_TOOL_OUTPUT_MAX_CHARS = 50_000
"""Default maximum number of text characters in one tool result."""


def _truncation_notice(omitted: int) -> str:
    return f"\n\n[tool output truncated: {omitted} characters omitted]\n\n"


def _budgets(total: int, max_chars: int) -> tuple[int, int, str]:
    """Return head/tail budgets and a notice that together fit ``max_chars``."""
    kept = max_chars
    while True:
        notice = _truncation_notice(total - kept)
        if len(notice) >= max_chars:
            return 0, 0, _truncation_notice(total)[:max_chars]
        new_kept = max_chars - len(notice)
        if new_kept == kept:
            break
        kept = new_kept
    head = (kept + 1) // 2
    return head, kept - head, notice


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head, tail, notice = _budgets(len(text), max_chars)
    return text[:head] + notice + (text[-tail:] if tail else "")


def _truncate_content(content: typing.Any, max_chars: int) -> typing.Any:
    """Truncate text across a tool message while preserving non-text blocks."""
    if isinstance(content, str):
        return _truncate_text(content, max_chars)
    if not isinstance(content, list):
        return content

    total = sum(
        len(block.get("text", ""))
        for block in content
        if isinstance(block, collections.abc.Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    if total <= max_chars:
        return content

    head, tail, notice = _budgets(total, max_chars)
    tail_start = total - tail
    position = 0
    notice_inserted = False
    truncated: list[typing.Any] = []
    for block in content:
        if not (
            isinstance(block, collections.abc.Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            truncated.append(block)
            continue

        text = block["text"]
        start, end = position, position + len(text)
        prefix = text[: max(0, min(end, head) - start)] if start < head else ""
        suffix_from = max(0, tail_start - start)
        suffix = text[suffix_from:] if end > tail_start else ""
        replacement = prefix
        if not notice_inserted and end > head and start < tail_start:
            replacement += notice
            notice_inserted = True
        replacement += suffix
        if replacement:
            truncated.append({**block, "text": replacement})
        position = end

    return truncated


class ToolOutputTruncator(ObjectInterpretation):
    """Keep any one tool result from consuming the model's context window.

    Textual output longer than ``max_chars`` is replaced by its beginning and
    end with an omission notice between them. Keeping both sides preserves the
    command or document header as well as errors and summaries commonly written
    at the end. Non-text content blocks, such as images, are left untouched.
    """

    def __init__(self, max_chars: int = DEFAULT_TOOL_OUTPUT_MAX_CHARS):
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars

    @implements(call_tool)
    def call_tool[T](self, tool_call: DecodedToolCall[T]) -> ToolResult[T]:
        message, result, is_final = fwd(tool_call)
        content = message.get("content")
        truncated = _truncate_content(content, self.max_chars)
        if truncated is not content:
            message = typing.cast(typing.Any, {**message, "content": truncated})
        return message, result, is_final
