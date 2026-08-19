"""Tests for the opt-in observability handlers.

These handlers are only installed behind a `harness` flag (``langfuse=True``,
``render=True``, ``dump_system_prompt=...``), so nothing else in the suite
exercises them and a break in one surfaces only when someone turns the flag on.
Each test here is a smoke test standing in for that missing coverage.

All of it runs offline. The model is replaced by `MockCompletionHandler`, and
Langfuse -- whose SDK would otherwise POST spans to a collector -- is pointed at
an in-memory OpenTelemetry exporter (see :func:`langfuse_client`), which is the
Langfuse SDK's own recommended way to unit-test instrumentation. No API key, no
Langfuse server, no network.
"""

import io
import json

import langfuse
import litellm
import pydantic
import pytest
import rich.console
import rich.text
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from effectful.handlers.llm import Skill
from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
from effectful.handlers.llm.harness.hooks import (
    AgentLoop,
    PromptSection,
    call_system,
    completion,
)
from effectful.handlers.llm.harness.observability.dumping import SystemPromptDumper
from effectful.handlers.llm.harness.observability.rendering import RichTerminalRenderer
from effectful.handlers.llm.harness.observability.tracing import LangfuseTracer
from effectful.handlers.llm.harness.provision import LiteLLMConfigurer
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled
from tests.conftest import (
    MockCompletionHandler,
    add_numbers,
    make_text_response,
    make_tool_call_response,
)

# ============================================================================
# Langfuse tracing
# ============================================================================


@pytest.fixture
def langfuse_client(request):
    """A `langfuse.Langfuse` whose spans are captured in memory.

    ``span_exporter`` replaces the SDK's default OTLP exporter, so spans are
    collected locally instead of being sent to a collector; the explicit
    ``tracer_provider`` keeps the SDK from installing itself as the *global*
    OpenTelemetry provider, so tests don't leak tracing state into each other.

    The ``public_key`` must be unique per test: the SDK caches its resources
    (including the exporter) per key, so a second client sharing a key silently
    reuses the first one's exporter and this test's own stays empty.

    Yields the client paired with the exporter holding whatever it recorded.
    """
    exporter = InMemorySpanExporter()
    client = langfuse.Langfuse(
        public_key=f"pk-{request.node.name}",
        secret_key="sk-test",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
        flush_at=1,
    )
    yield client, exporter
    client.shutdown()


def attr(span, suffix):
    """Read one ``langfuse.observation.*`` attribute off an exported span."""
    return span.attributes[f"langfuse.observation.{suffix}"]


def by_type(spans):
    """Group exported spans by their Langfuse observation type.

    Spans are exported in *completion* order -- children before the parent that
    encloses them -- so tests must find them by type or name rather than by
    position.
    """
    grouped: dict[str, list] = {}
    for span in spans:
        grouped.setdefault(attr(span, "type"), []).append(span)
    return grouped


@Skill.define
def compute(x: int, y: int) -> str:
    """Add {x} and {y} using the tool, then report the result."""
    raise NotHandled


# `compute` offers `add_numbers` because it is in scope where the skill is
# defined -- the import is load-bearing even though the body never names it.


def test_tracer_traces_skill_completion_and_tool(langfuse_client):
    """One traced call covers all three operations `LangfuseTracer` implements.

    A skill that calls a tool produces an ``agent`` span (the `Skill`), two
    ``generation`` spans (the `completion` before and after the tool ran) and a
    ``tool`` span, all in one trace.

    The payload assertions are the point: the tracer serializes every input and
    output through ``Encodable[nested_type(value).value]``, so they pin the
    repo's own encoding as much as the Langfuse SDK's span API. Checking only
    that spans exist would pass straight through a break in either.
    """
    client, exporter = langfuse_client
    responses = [
        make_tool_call_response(add_numbers.__name__, json.dumps({"a": 1, "b": 2})),
        make_text_response(
            "The result is 3",
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        ),
    ]

    with (
        handler(MockCompletionHandler(responses)),
        handler(AgentLoop()),
        handler(LiteLLMConfigurer(model="test-model")),
        handler(HistoryBuilder()),
        handler(LangfuseTracer(client=client)),
    ):
        assert compute(1, 2) == "The result is 3"

    client.flush()
    spans = exporter.get_finished_spans()
    grouped = by_type(spans)

    # A single trace, with every observation hanging off the skill's span.
    (agent,) = grouped["agent"]
    assert agent.name == "compute"
    assert agent.parent is None
    assert len({s.context.trace_id for s in spans}) == 1
    assert all(
        s.parent.span_id == agent.context.span_id for s in spans if s is not agent
    )

    assert json.loads(attr(agent, "input")) == {"x": 1, "y": 2}
    assert attr(agent, "output") == "The result is 3"

    (tool,) = grouped["tool"]
    assert tool.name == add_numbers.__name__
    assert json.loads(attr(tool, "input")) == {"a": 1, "b": 2}
    assert attr(tool, "metadata.tool_call_id") == "call_1"
    assert attr(tool, "metadata.is_final") is False

    first, second = grouped["generation"]
    assert [g.name for g in (first, second)] == ["completion", "completion"]
    assert all(attr(g, "model.name") == "test-model" for g in (first, second))
    # The tools offered are traced as part of the request.
    assert add_numbers.__name__ in attr(first, "input")
    # Only the second response carried usage, and it is reported as-is.
    assert json.loads(attr(second, "usage_details")) == {
        "input": 11,
        "output": 7,
        "total": 18,
    }
    assert json.loads(attr(second, "output"))["content"] == "The result is 3"


def test_tracer_records_response_format_metadata(langfuse_client):
    """A pydantic ``response_format`` is traced as its JSON schema.

    The structured-output branch is unreachable from a plain string-returning
    skill, so `completion` is driven directly here.
    """

    class Answer(pydantic.BaseModel):
        value: int

    client, exporter = langfuse_client
    with (
        handler(MockCompletionHandler([make_text_response('{"value": 3}')])),
        handler(LangfuseTracer(client=client)),
    ):
        completion(
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            response_format=Answer,
        )

    client.flush()
    (span,) = exporter.get_finished_spans()
    assert json.loads(attr(span, "metadata.response_format")) == (
        Answer.model_json_schema()
    )


# ============================================================================
# Terminal rendering
# ============================================================================


def _delta_chunk(**delta):
    """One streamed chunk carrying an assistant `Delta`."""
    return litellm.types.utils.ModelResponseStream(
        id="test",
        model="test-model",
        choices=[
            litellm.types.utils.StreamingChoices(
                index=0,
                delta=litellm.types.utils.Delta(role="assistant", **delta),
            )
        ],
    )


def _tool_call_delta(name, arguments):
    """One streamed tool-call fragment for the tool call at index 0."""
    return [
        litellm.types.utils.ChatCompletionDeltaToolCall(
            index=0,
            id="call_1",
            type="function",
            function=litellm.types.utils.Function(name=name, arguments=arguments),
        )
    ]


# Model-authored Python, streamed as a tool call's arguments: the renderer should
# recognize it as code (`_is_python`) and highlight it rather than dumping JSON.
_CODE = "def f():\n    return 1\n"
_ARGS = json.dumps({"code": _CODE})


class _StreamingCodeHandler(ObjectInterpretation):
    """Streams reasoning, prose, and a tool call whose arguments are split.

    The argument JSON arrives in two fragments broken mid-string, which is the
    case `_accumulate` exists to handle: neither half is parseable alone, so the
    renderer can only show highlighted code once they have been coalesced.
    """

    @implements(completion)
    def _completion(self, *args, **kwargs):
        assert kwargs.get("stream"), "expected the renderer to force streaming"
        return iter(
            [
                _delta_chunk(reasoning_content="thinking about it"),
                _delta_chunk(content="here goes"),
                _delta_chunk(tool_calls=_tool_call_delta("exec_code", _ARGS[:12])),
                _delta_chunk(tool_calls=_tool_call_delta(None, _ARGS[12:])),
            ]
        )


def _plain(rendered: str) -> str:
    """The visible text of `rendered`, with the styling escapes removed.

    Assertions have to run against this rather than the raw capture: syntax
    highlighting styles each token separately, so escape sequences land *inside*
    a phrase (``return 1`` is a keyword then a number) and it never appears
    literally in the output.
    """
    return rich.text.Text.from_ansi(rendered).plain


def test_renderer_streams_reasoning_and_tool_calls():
    """`RichTerminalRenderer` renders a whole streamed turn and reassembles it.

    The console is injected because the default is deliberately pinned to
    ``sys.__stdout__`` (so rendered panels can't be captured back into the
    model's context), which a test could not otherwise read.
    """
    history = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "compute something"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "add_numbers", "arguments": '{"a": 1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": '{"result": 1}'},
    ]

    buffer = io.StringIO()
    console = rich.console.Console(file=buffer, force_terminal=True, width=100)
    with (
        handler(_StreamingCodeHandler()),
        handler(RichTerminalRenderer(console=console)),
    ):
        response = completion(messages=history, model="test-model")

    # The stream is reassembled into an ordinary response, with the tool call's
    # split arguments coalesced back into complete JSON.
    message = response.choices[0].message
    assert message.content == "here goes"
    (tool_call,) = message.tool_calls
    assert tool_call.function.name == "exec_code"
    assert json.loads(tool_call.function.arguments) == {"code": _CODE}

    output = _plain(buffer.getvalue())
    # The streamed turn: reasoning, prose, and the tool call's arguments shown as
    # highlighted Python rather than as JSON.
    assert "thinking about it" in output
    assert "here goes" in output
    assert "exec_code" in output
    assert "def f():" in output
    assert "return 1" in output
    # ... rendered beneath a panel per message of the history preceding it.
    for role in ("system", "user", "assistant", "tool"):
        assert role in output
    assert "you are a helpful assistant" in output
    assert "add_numbers" in output


# ============================================================================
# System prompt dumping
# ============================================================================


def test_dumper_writes_system_prompt(tmp_path):
    """`SystemPromptDumper` writes the assembled system prompt to its path.

    `call_system` has a default rule that assembles the two halves, so this needs
    no provider -- just the handler under test.
    """
    path = tmp_path / "prompt.md"
    with handler(SystemPromptDumper(path=path)):
        call_system(
            PromptSection(
                type="prompt_section",
                title="Harness",
                content=[{"type": "text", "text": "how the machinery works"}],
            ),
            PromptSection(
                type="prompt_section",
                title="Task",
                content=[{"type": "text", "text": "what to do"}],
            ),
        )

    dumped = path.read_text()
    assert "how the machinery works" in dumped
    assert "what to do" in dumped
    assert "Harness" in dumped and "Task" in dumped


def test_dumper_overwrites_on_each_call(tmp_path):
    """Each dump replaces the file rather than appending to it."""
    path = tmp_path / "prompt.md"
    with handler(SystemPromptDumper(path=path)):
        for text in ("first prompt", "second prompt"):
            call_system(
                PromptSection(type="prompt_section", title="Harness", content=[]),
                PromptSection(
                    type="prompt_section",
                    title="Task",
                    content=[{"type": "text", "text": text}],
                ),
            )

    dumped = path.read_text()
    assert "second prompt" in dumped
    assert "first prompt" not in dumped
