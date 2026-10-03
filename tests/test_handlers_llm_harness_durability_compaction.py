import copy
import json
from types import SimpleNamespace

import pytest

from effectful.handlers.llm import Skill, Tool
from effectful.handlers.llm.harness import harness
from effectful.handlers.llm.harness.durability.compaction import (
    CompactionScope,
    MiddleCompactor,
)
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
        MiddleCompactor(1, 400, recent_tokens=1),
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
        MiddleCompactor(1, 400, recent_tokens=1),
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
        handler(MiddleCompactor(1, 400, recent_tokens=1)),
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
    with pytest.raises(ValueError, match="recent_tokens must be less"):
        MiddleCompactor(1, 1000, recent_tokens=1000)
    MiddleCompactor(1, 1000, recent_tokens=999)


class Counter:
    """Count things, keeping notes on `self`."""

    def __init__(self):
        self.note = None

    @Skill.define
    def count(self, up_to: int) -> str:
        """Count to {up_to}, then say how far you got."""

    @Tool.define
    def tally(self) -> int:
        """Return the count so far."""
        return 0


def _exec(code: str, call_id: str = "call_1"):
    return make_tool_call_response("exec_code", json.dumps({"code": code}), call_id)


def _compact(code: str, call_id: str = "call_c", scope=CompactionScope.CONVERSATION):
    return make_tool_call_response(
        "exec_code", json.dumps({"code": code, "compact": scope}), call_id
    )


FORCED = {"type": "function", "function": {"name": "exec_code"}}


def _nudged(messages: list) -> bool:
    return "over its token budget" in json.dumps(messages[-1])


def _tool_names(request: dict) -> list[str]:
    return sorted(t["function"]["name"] for t in request["tools"])


def _count(responses, *, limit: int, scope=CompactionScope.CONVERSATION, retries=3):
    """A `Counter` driven through the harness by scripted responses, with the mock
    *under* the stack so the forcing rule sees every request on its way out."""
    mock = MockCompletionHandler(responses)
    agent = Counter()
    with (
        handler(mock),
        handler(
            harness(
                model="test",
                type_checker="none",
                tool_calling="json",
                num_retries=retries,
                compaction_hard_tokens=limit,
                compaction_scope=scope,
            )
        ),
    ):
        result = agent.count(3)
    return mock, agent, result


def test_under_budget_neither_nudges_nor_forces():
    mock, agent, result = _count(
        [_exec("x = 1"), _exec("x += 1"), make_text_response("two")], limit=10**9
    )
    assert result == "two"
    for request, messages in zip(mock.received_kwargs, mock.received_messages):
        assert "tool_choice" not in request
        assert not _nudged(messages)


def test_over_budget_nudges_forces_and_compacts_on_success():
    mock, agent, result = _count(
        [
            _exec("x = 1"),
            _exec("x += 1", "call_2"),
            _compact("self.note = x\nprint('saved', x)"),
            make_text_response("two"),
        ],
        limit=1,
    )
    assert result == "two"
    first, second, forced, after = mock.received_kwargs
    # Not forced until there are two rounds to drop.
    assert "tool_choice" not in first and "tool_choice" not in second
    assert forced["tool_choice"] == FORCED
    assert forced["drop_params"] is True
    assert "drop_params" not in second
    # The tools are unchanged, and the nudge is the request's last message only.
    assert _tool_names(forced) == _tool_names(second)
    assert _nudged(mock.received_messages[2])
    # The snippet ran in the REPL session, where `x` was bound.
    assert agent.note == 2
    # The request after a compaction has one round to drop, so it is not forced.
    assert "tool_choice" not in after
    assert not _nudged(mock.received_messages[3])
    roles = [m["role"] for m in agent.__history__]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert agent.__history__[2]["tool_calls"][0]["id"] == "call_c"
    assert "saved 2" in json.dumps(agent.__history__[3]["content"])
    # The system message embeds this module's source, so only the rest is checked.
    assert not any(_nudged([m]) for m in agent.__history__[1:])


def test_conversation_scope_drops_prior_turns_and_turn_scope_keeps_them():
    def two_turns(scope):
        script = [
            _exec("x = 1"),
            _exec("x += 1", "call_2"),
            _compact("self.note = x", scope=scope),
            make_text_response("two"),
        ]
        mock = MockCompletionHandler([make_text_response("zero"), *script])
        agent = Counter()
        with (
            handler(mock),
            handler(
                harness(
                    model="test",
                    type_checker="none",
                    tool_calling="json",
                    compaction_hard_tokens=1,
                    compaction_scope=scope,
                )
            ),
        ):
            agent.count(0)
            agent.count(3)
        return [m["role"] for m in agent.__history__]

    assert two_turns(CompactionScope.CONVERSATION) == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert two_turns(CompactionScope.TURN) == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_failed_snippet_compacts_nothing_and_is_forced_again():
    mock, agent, result = _count(
        [
            _exec("x = 1"),
            _exec("x += 1", "call_2"),
            _compact("raise RuntimeError('not yet')"),
            _compact("self.note = x", "call_d"),
            make_text_response("two"),
        ],
        limit=1,
    )
    assert result == "two"
    assert mock.received_kwargs[2]["tool_choice"] == FORCED
    assert mock.received_kwargs[3]["tool_choice"] == FORCED
    assert agent.note == 2
    history = agent.__history__
    # Only the successful compaction's round survives, the failed one went with
    # the rest.
    assert [m["role"] for m in history] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history[2]["tool_calls"][0]["id"] == "call_d"


def test_prose_under_forcing_is_retried():
    mock, agent, result = _count(
        [
            _exec("x = 1"),
            _exec("x += 1", "call_2"),
            make_text_response("I would rather answer: two"),
            _compact("self.note = x"),
            make_text_response("two"),
        ],
        limit=1,
    )
    assert result == "two"
    assert mock.received_kwargs[2]["tool_choice"] == FORCED
    assert mock.received_kwargs[3]["tool_choice"] == FORCED
    # The retry saw the refusal and the feedback on it, then the nudge again.
    assert [m["role"] for m in mock.received_messages[3][-3:]] == [
        "assistant",
        "user",
        "user",
    ]
    assert _nudged(mock.received_messages[3])
    assert agent.note == 2
    assert all(
        "rather answer" not in json.dumps(m.get("content"))
        for m in agent.__history__
        if m["role"] == "assistant"
    )


def test_prose_under_forcing_fails_the_call_once_retries_run_out():
    with pytest.raises(Exception, match="must call `exec_code`"):
        _count(
            [
                _exec("x = 1"),
                _exec("x += 1", "call_2"),
                make_text_response("no"),
            ],
            limit=1,
            retries=2,
        )


@pytest.mark.parametrize(
    "wrong",
    [
        _exec("y = x", "call_3"),
        _compact("y = x", "call_3", scope=CompactionScope.TURN),
        make_tool_call_response("tally", json.dumps({}), "call_3"),
    ],
    ids=["no-scope", "wrong-scope", "other-tool"],
)
def test_a_forced_round_rejects_anything_but_the_scoped_exec_code(wrong):
    mock, agent, result = _count(
        [
            _exec("x = 1"),
            _exec("x += 1", "call_2"),
            wrong,
            _compact("self.note = x"),
            make_text_response("two"),
        ],
        limit=1,
    )
    assert result == "two"
    assert mock.received_kwargs[2]["tool_choice"] == FORCED
    assert mock.received_kwargs[3]["tool_choice"] == FORCED
    # The retry saw the rejected call answered, then the nudge again.
    retry = mock.received_messages[3]
    assert [m["role"] for m in retry[-3:]] == ["assistant", "tool", "user"]
    assert retry[-2]["tool_call_id"] == "call_3"
    assert "must call `exec_code`" in json.dumps(retry[-2]["content"])
    assert agent.note == 2
    assert [m["role"] for m in agent.__history__] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert agent.__history__[2]["tool_calls"][0]["id"] == "call_c"
