import copy
import json
from types import SimpleNamespace

import pytest

from effectful.handlers.llm import Skill, Tool
from effectful.handlers.llm.harness import harness
from effectful.handlers.llm.harness.durability.compaction import MiddleCompactor
from effectful.handlers.llm.harness.durability.transaction import transaction
from effectful.handlers.llm.harness.hooks import call_assistant, completion
from effectful.ops.semantics import handler

from .conftest import MockCompletionHandler, make_text_response, make_tool_call_response


def _history():
    return [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "x" * 800},
        {"role": "assistant", "content": "old reasoning" * 20},
        {"role": "user", "content": "followup"},
        {"role": "assistant", "tool_calls": [{"id": "new"}]},
        {"role": "tool", "tool_call_id": "new", "content": "y" * 800},
    ]


def _run(history, compactor, summary=None):
    seen = []
    summaries = []

    def respond(messages, *_args):
        seen.append(copy.deepcopy(messages))
        return {"role": "assistant", "content": "done"}, [], "done"

    def summarize(*_args, **kwargs):
        summaries.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=summary))]
        )

    with (
        handler({call_assistant: respond, completion: summarize}),
        handler(compactor),
        transaction(history),
    ):
        call_assistant(list(history), str, {})
    return seen[0], summaries


def test_soft_threshold_truncates_only_middle_tool_output():
    history = _history()
    recent = copy.deepcopy(history[-4:])
    seen, summaries = _run(history, MiddleCompactor(1, 10000, recent_tokens=1))
    assert not summaries
    assert "truncated" in str(history[3]["content"])
    assert history[-4:] == recent
    assert history == seen
    assert history[:3] == _history()[:3]


def test_hard_threshold_summarizes_complete_old_rounds_and_updates_summary():
    history = _history()
    recent = copy.deepcopy(history[-4:])
    seen, summaries = _run(
        history,
        MiddleCompactor(1, 400, recent_tokens=1, summary_tokens=64),
        summary="old work",
    )
    assert len(summaries) == 1
    assert summaries[0]["tools"] == []
    assert summaries[0]["tool_choice"] == "none"
    assert history[1] == {
        "role": "user",
        "content": "[Earlier conversation summary]\nold work",
    }
    assert history[2:] == recent
    assert seen == history

    history.extend(
        [
            {"role": "assistant", "tool_calls": [{"id": "latest"}]},
            {"role": "tool", "tool_call_id": "latest", "content": "z" * 800},
        ]
    )
    _, summaries = _run(
        history,
        MiddleCompactor(1, 400, recent_tokens=1, summary_tokens=64),
        summary="new work",
    )
    assert len(summaries) == 1
    assert "old work" in summaries[0]["messages"][1]["content"]
    assert history[1]["content"] == "[Earlier conversation summary]\nnew work"


def test_preserves_history_if_summary_fails():
    history = _history()
    before = copy.deepcopy(history)

    def fail(*args, **kwargs):
        raise RuntimeError("model down")

    with (
        handler({call_assistant: lambda *a: None, completion: fail}),
        handler(MiddleCompactor(1, 400, recent_tokens=1, summary_tokens=64)),
        pytest.raises(RuntimeError, match="model down"),
        transaction(history),
    ):
        call_assistant(history, str, {})
    assert history == before


def test_recent_budget_can_keep_older_rounds():
    history = _history()
    seen, summaries = _run(history, MiddleCompactor(1, 100000, recent_tokens=10000))
    assert not summaries
    assert seen == _history()


def test_harness_compacts_before_resending_and_retains_tool_results():
    @Tool.define
    def verbose() -> str:
        """Produce a long observation."""
        return "x" * 8000

    @Skill.define
    def ask() -> str:
        """Call verbose three times, then answer."""

    mock = MockCompletionHandler(
        [
            *[make_tool_call_response("verbose", json.dumps({})) for _ in range(3)],
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
                max_tool_output_chars=None,
                compaction_soft_tokens=1,
                compaction_hard_tokens=100000,
                compaction_recent_tokens=1,
            )
        ),
        handler(mock),
    ):
        assert ask() == "done"
    tools = [m for m in mock.received_messages[-1] if m["role"] == "tool"]
    assert len(tools) == 3
    assert "truncated" in str(tools[0]["content"])
    assert "x" * 8000 in str(tools[-1]["content"])


def test_invalid_thresholds():
    with pytest.raises(ValueError, match="soft_tokens"):
        MiddleCompactor(5, 5)
    with pytest.raises(ValueError, match="summary_tokens \\+ recent_tokens"):
        MiddleCompactor(1, 1000, summary_tokens=750)  # default recent: 250
    with pytest.raises(ValueError, match="summary_tokens \\+ recent_tokens"):
        MiddleCompactor(1, 1000, recent_tokens=900, summary_tokens=100)
    MiddleCompactor(1, 1000, recent_tokens=899, summary_tokens=100)
