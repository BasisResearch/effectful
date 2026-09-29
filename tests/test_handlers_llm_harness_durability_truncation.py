import inspect
import json

import pytest

from effectful.handlers.llm import Skill, Tool
from effectful.handlers.llm.harness import harness
from effectful.handlers.llm.harness.durability.truncation import (
    ToolOutputTruncator,
    _truncate_content,
)
from effectful.handlers.llm.harness.hooks import call_tool
from effectful.handlers.llm.harness.serialization import DecodedToolCall
from effectful.ops.semantics import handler

from .conftest import (
    MockCompletionHandler,
    make_text_response,
    make_tool_call_response,
)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if block.get("type") == "text")


def test_truncates_tool_output_and_preserves_python_result():
    output = "H" * 150 + "T" * 150

    @Tool.define
    def verbose() -> str:
        """Return a verbose result."""
        return output

    tool_call = DecodedToolCall(
        verbose, inspect.signature(verbose).bind(), "call_1", "verbose"
    )
    with handler(ToolOutputTruncator(max_chars=100)):
        message, result, is_final = call_tool(tool_call)

    text = _text(message["content"])
    assert result == output
    assert not is_final
    assert len(text) == 100
    assert text.startswith("H") and text.endswith("T")
    assert "tool output truncated" in text
    assert "characters omitted" in text


def test_short_tool_output_is_unchanged():
    content = [{"type": "text", "text": "small"}]
    assert _truncate_content(content, 100) is content


def test_truncates_text_across_blocks_but_preserves_attachments():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
    content = [
        {"type": "text", "text": "A" * 90},
        image,
        {"type": "text", "text": "Z" * 90},
    ]

    truncated = _truncate_content(content, 100)

    assert image in truncated
    text = _text(truncated)
    assert len(text) == 100
    assert text.startswith("A") and text.endswith("Z")
    assert "tool output truncated" in text


def test_harness_truncates_before_recording_and_resending():
    output = "begin:" + "x" * 200 + ":end"

    @Tool.define
    def verbose() -> str:
        """Return a verbose result."""
        return output

    @Skill.define
    def ask() -> str:
        """Call verbose, then answer."""

    mock = MockCompletionHandler(
        [
            make_tool_call_response("verbose", json.dumps({})),
            make_text_response("done"),
        ]
    )
    with (
        handler(
            harness(
                model="test",
                eval_provider="none",
                type_checker="none",
                tool_calling="json",
                max_tool_output_chars=100,
            )
        ),
        handler(mock),
    ):
        assert ask() == "done"

    tool_messages = [
        message for message in mock.received_messages[1] if message["role"] == "tool"
    ]
    assert len(tool_messages) == 1
    text = _text(tool_messages[0]["content"])
    assert len(text) == 100
    assert text.startswith("begin:") and text.endswith(":end")
    assert "tool output truncated" in text


def test_harness_can_disable_tool_output_truncation():
    output = "x" * 150

    @Tool.define
    def verbose() -> str:
        """Return a verbose result."""
        return output

    @Skill.define
    def ask() -> str:
        """Call verbose, then answer."""

    mock = MockCompletionHandler(
        [make_tool_call_response("verbose", "{}"), make_text_response("done")]
    )
    with (
        handler(
            harness(
                model="test",
                eval_provider="none",
                type_checker="none",
                tool_calling="json",
                max_tool_output_chars=None,
            )
        ),
        handler(mock),
    ):
        assert ask() == "done"

    tool_message = next(
        message for message in mock.received_messages[1] if message["role"] == "tool"
    )
    assert _text(tool_message["content"]) == output


def test_rejects_nonpositive_limits():
    with pytest.raises(ValueError, match="positive"):
        ToolOutputTruncator(max_chars=0)
