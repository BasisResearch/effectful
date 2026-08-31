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

import concurrent.futures
import io
import json
import re
import threading

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
from effectful.handlers.llm.harness.legibility.lexical import LexicalToolExtractor
from effectful.handlers.llm.harness.observability.dump import SystemPromptDumper
from effectful.handlers.llm.harness.observability.langfuse import LangfuseTracer
from effectful.handlers.llm.harness.observability.rich import (
    RichTerminalRenderer,
    _is_python,
    _message_text,
    _panel_key,
    _partial_panel,
    _render_content,
    _render_tool_call,
)
from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
from effectful.handlers.llm.harness.synthesis.snippet import (
    StatefulReplSynthesizer,
)
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
        handler(LexicalToolExtractor()),
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


def test_renderer_prints_each_message_once():
    """A conversation is printed as it happens, not re-printed per request.

    Every request carries the whole history, so a renderer that draws all of it
    draws the same panels again on each turn. That is not merely redundant: the
    frame outgrows the terminal, `rich.live.Live` erases a frame by rewinding the
    cursor over it, and a rewind is clamped at the top of the screen -- so past
    that size each refresh leaves a full copy of the conversation behind. A
    three-turn example run emitted 7,208 lines carrying 335 distinct ones.
    """
    renderer = RichTerminalRenderer(
        console=rich.console.Console(file=io.StringIO(), force_terminal=True, width=100)
    )
    history = [
        {"role": "system", "content": "SYSTEM_PROMPT_MARKER"},
        {"role": "user", "content": "USER_PROMPT_MARKER"},
    ]
    with handler(_StreamingCodeHandler()), handler(renderer):
        first = completion(messages=history, model="test-model")
        # The next request carries what the loop appends: this turn, verbatim
        # (`call_assistant` dumps the same message), and its tool result.
        history = [
            *history,
            first.choices[0].message.model_dump(mode="json"),
            {"role": "tool", "tool_call_id": "call_1", "content": "TOOL_RESULT_MARKER"},
        ]
        completion(messages=history, model="test-model")

    output = _plain(renderer.console.file.getvalue())
    assert output.count("SYSTEM_PROMPT_MARKER") == 1
    assert output.count("USER_PROMPT_MARKER") == 1
    assert output.count("TOOL_RESULT_MARKER") == 1


def test_messages_are_told_apart_by_what_they_render_as():
    """A turn re-serialized on its way into the next request is the same panel.

    Which is why "already printed" is decided on the rendering and not on the
    dict: a turn reaches the next request's history through handlers that may
    rebuild it -- the decoding-error path replays a rejected turn into the
    retry's history -- and bookkeeping no panel displays must not make the same
    panel print twice.
    """
    turn = {
        "role": "assistant",
        "content": "here goes",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "exec_code", "arguments": _ARGS},
            }
        ],
    }
    replayed = {**turn, "provider_specific_fields": {"trace": "abc"}, "audio": None}
    assert _panel_key(turn) == _panel_key(replayed)

    # What a panel *does* show still tells two messages apart.
    assert _panel_key(turn) != _panel_key({**turn, "content": "here goes again"})
    assert _panel_key(turn) != _panel_key({**turn, "role": "user"})


def test_the_live_window_reports_everything_it_has_hidden():
    """The tally counts what the turn has put out of view, not what the window did.

    The source is clipped to a fixed number of lines before it is laid out, so
    everything the window itself hides past that point is the same handful of
    wrapped rows however long the turn grows. Reporting only that -- as this did
    -- leaves a count that stalls and dips instead of climbing.
    """
    console = rich.console.Console(
        file=io.StringIO(), force_terminal=True, width=100, height=20
    )
    counts = []
    for total in (10, 50, 100, 200, 400):
        partial = {
            "content": "\n".join(f"line {i}" for i in range(total)),
            "reasoning_content": "",
            "tool_calls": {},
        }
        with console.capture() as capture:
            console.print(_partial_panel(partial, height=12))
        found = re.search(r"\(\+(\d+) earlier lines\)", _plain(capture.get()))
        counts.append(int(found.group(1)) if found else 0)

    assert counts == sorted(counts), f"the tally went backwards: {counts}"
    assert counts[-1] > counts[0]
    # It tracks the turn: nearly all of a 400-line one is out of view by then.
    assert counts[-1] >= 400 - 20, counts


def test_renderer_keeps_the_live_region_within_the_screen():
    """No frame may be taller than the terminal it is redrawn on.

    This is the invariant the whole layout serves, and it is checked where it is
    decided: `rich` erases the previous frame by emitting one cursor-up per row
    of it, so a frame that asks to rewind further than the screen is exactly the
    frame the terminal cannot erase.

    The payload is one enormous *single line*, because that is the shape a source
    line limit cannot bound and the streamed one actually takes: tool-call
    arguments arrive as one line of JSON that word-wraps across dozens of rows.
    Only the console can say how tall that is.
    """
    height = 20
    buffer = io.StringIO()
    console = rich.console.Console(
        file=buffer, force_terminal=True, width=100, height=height
    )
    args = json.dumps({"note": f"FIRST_ROW_MARKER{'x' * 4000}LAST_ROW_MARKER"})

    class _WideTurn(ObjectInterpretation):
        @implements(completion)
        def _completion(self, *args_, **kwargs):
            return iter([_delta_chunk(tool_calls=_tool_call_delta("note_it", args))])

    with handler(_WideTurn()), handler(RichTerminalRenderer(console=console)):
        completion(messages=[{"role": "user", "content": "q"}], model="test-model")

    # Each erase is a carriage return and one `cursor up` per row above the last.
    erases = re.findall(r"\r\x1b\[2K(?:\x1b\[1A\x1b\[2K)*", buffer.getvalue())
    rewound = [erase.count("\x1b[1A") + 1 for erase in erases]
    assert rewound, "expected the live region to have been redrawn at all"
    assert max(rewound) <= height

    # The window keeps the end of what is arriving, which is what is watched.
    live = _plain(buffer.getvalue()[: buffer.getvalue().index("LAST_ROW_MARKER")])
    assert "FIRST_ROW_MARKER" not in live


def test_partial_python_is_recognised_as_code_at_every_prefix():
    """Source still arriving is recognised as source, the way partial JSON parses.

    Mid-stream a snippet is a syntax error by construction -- a function whose
    body has not arrived, a docstring not yet closed -- so the question asked of
    finished source, *does this parse*, answers no for most of the time a model
    spends writing one. Asked instead whether it could still become Python, as
    `codeop` answers it for a REPL, it holds throughout.
    """
    code = (
        "import itertools\n\n"
        "def can_make(numbers, target):\n"
        '    """Whether target is reachable from numbers."""\n'
        "    for a, b in itertools.combinations(numbers, 2):\n"
        "        if a + b == target:\n"
        "            return True\n"
        "    return False\n"
    )
    prefixes = [code[:n] for n in range(1, len(code) + 1) if "\n" in code[:n]]
    assert all(_is_python(prefix, partial=True) for prefix in prefixes)
    # Whereas most of them do not parse, which is what the settled test asks.
    assert sum(_is_python(prefix) for prefix in prefixes) < len(prefixes) // 2

    # What is not source stays not source, at every prefix of it. A `{` opening
    # a dict literal is the interesting one: it *is* a valid Python prefix.
    for other in (
        "Let me think about this.\nI will brute force it.\nThen answer.\n",
        '{\n  "numbers": [3, 6, 25, 50],\n  "target": 147\n}\n',
        "# Plan\n\n- brute force it\n- check the result\n",
    ):
        assert not any(
            _is_python(other[:n], partial=True)
            for n in range(1, len(other) + 1)
            if "\n" in other[:n]
        ), other


def test_streamed_tool_call_arguments_render_as_code_before_they_are_complete():
    """Half-arrived arguments are shown as what they are becoming.

    A tool call's arguments are a JSON document delivered a few characters at a
    time, so for the whole of the interesting part -- a model writing a function
    -- the raw fragment is an unterminated string of escaped source. Closing what
    the fragment left open recovers the payload as it stands.
    """
    args = json.dumps({"code": "import itertools\n\ndef f():\n    return 1\n"})
    console = rich.console.Console(file=io.StringIO(), force_terminal=True, width=100)
    call, _ = _render_tool_call("exec_code", args[:-20], streaming=True)
    console.print(call)

    output = _plain(console.file.getvalue())
    assert "import itertools" in output
    assert "\\n" not in output, "arguments shown as escaped JSON rather than as code"


class _ConcurrentProbe(ObjectInterpretation):
    """Records the ``stream`` each request carried, and holds the first caller
    inside the provider until a second one has arrived.

    The barrier is what makes the test deterministic: without it the first call
    could finish and release the live region before the second ever entered, and
    the contention this is about would never occur.
    """

    def __init__(self, parties: int):
        self.barrier = threading.Barrier(parties, timeout=10)
        self.streamed: list[bool] = []
        self.lock = threading.Lock()

    @implements(completion)
    def _completion(self, *args, **kwargs):
        with self.lock:
            self.streamed.append(bool(kwargs.get("stream")))
        self.barrier.wait()
        if kwargs.get("stream"):
            return iter([_delta_chunk(content="streamed")])
        return make_text_response("settled")


def test_renderer_serializes_the_live_region_without_serializing_the_calls():
    """Only one concurrent completion may drive a ``Live``; the rest run unstreamed.

    A ``Live`` owns the console for its duration and there is one console, so
    overlapping regions would interleave two redraws into the same rows. The
    guard is non-blocking, so the losing callers proceed immediately -- the point
    is to keep the fan-out parallel while keeping the terminal legible.
    """
    probe = _ConcurrentProbe(parties=3)
    buffer = io.StringIO()
    console = rich.console.Console(file=buffer, force_terminal=True, width=100)
    renderer = RichTerminalRenderer(console=console)

    def call(i):
        with handler(probe), handler(renderer):
            return completion(
                messages=[{"role": "user", "content": f"q{i}"}], model="test-model"
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(call, range(3)))

    # All three ran -- the barrier could only be cleared by three callers inside
    # the provider at once, so none of them queued behind the live one.
    assert len(responses) == 3
    # Exactly one took the live path; the others were not forced onto streaming.
    assert sorted(probe.streamed) == [False, False, True]

    output = _plain(buffer.getvalue())
    assert "streamed" in output
    # The settled turns are still rendered, just as finished panels.
    assert output.count("settled") == 2


class _BrokenStream(ObjectInterpretation):
    """Fails the streamed request, answers the unstreamed one."""

    def __init__(self, error: Exception):
        self.error = error
        self.streamed_attempts = 0
        self.settled_attempts = 0

    @implements(completion)
    def _completion(self, *args, **kwargs):
        if kwargs.get("stream"):
            self.streamed_attempts += 1
            raise self.error
        self.settled_attempts += 1
        return make_text_response("recovered")


@pytest.mark.parametrize(
    "error",
    [
        litellm.exceptions.MidStreamFallbackError(
            message="boom", model="m", llm_provider="openai"
        ),
        litellm.APIConnectionError(message="boom", model="m", llm_provider="openai"),
        litellm.Timeout(message="boom", model="m", llm_provider="openai"),
    ],
    ids=["mid-stream", "connection", "timeout"],
)
def test_renderer_retries_a_broken_stream_without_streaming(error):
    """A transport failure on the streamed read is retried unstreamed.

    Streaming is something this handler adds to a request that did not ask for
    it, so a call that would have succeeded unstreamed must not fail merely
    because it was being rendered.
    """
    provider = _BrokenStream(error)
    buffer = io.StringIO()
    console = rich.console.Console(file=buffer, force_terminal=True, width=100)

    with handler(provider), handler(RichTerminalRenderer(console=console)):
        response = completion(
            messages=[{"role": "user", "content": "q"}], model="test-model"
        )

    assert response.choices[0].message.content == "recovered"
    assert provider.streamed_attempts == 1
    assert provider.settled_attempts == 1
    assert "retrying unstreamed" in _plain(buffer.getvalue())


def test_renderer_does_not_retry_a_rejected_request():
    """A request the provider refuses fails identically unstreamed, so re-issuing
    it would only pay for the same error twice."""
    provider = _BrokenStream(
        litellm.BadRequestError(message="nope", model="m", llm_provider="openai")
    )
    renderer = RichTerminalRenderer(
        console=rich.console.Console(file=io.StringIO(), force_terminal=True)
    )
    with handler(provider), handler(renderer):
        with pytest.raises(litellm.BadRequestError):
            completion(messages=[{"role": "user", "content": "q"}], model="test-model")

    assert provider.settled_attempts == 0
    assert renderer._live_lock.acquire(blocking=False), "live region left held"
    renderer._live_lock.release()


def test_renderer_releases_the_live_region_after_a_failure():
    """A completion that raises must not leave the live region held, or every
    later call in the process would silently fall back to the settled path."""

    class _Boom(ObjectInterpretation):
        @implements(completion)
        def _completion(self, *args, **kwargs):
            raise RuntimeError("provider exploded")

    renderer = RichTerminalRenderer(
        console=rich.console.Console(file=io.StringIO(), force_terminal=True)
    )
    with handler(_Boom()), handler(renderer):
        with pytest.raises(RuntimeError, match="provider exploded"):
            completion(messages=[{"role": "user", "content": "q"}], model="test-model")

    assert renderer._live_lock.acquire(blocking=False), "live region left held"
    renderer._live_lock.release()


def test_renderer_shows_the_repl_session_section():
    """What a handler appends to a request must reach the operator, not just the
    model.

    The `REPL session` section is delimited by a Markdown heading, and that is
    not a style choice: an earlier annotation on the user message used a column-0
    ``<consolidated>`` tag, which CommonMark classifies as an ``html_block``.
    `rich.markdown.Markdown` registers no element for that token and drops it
    silently, so the text reached the model while vanishing from ``--render``.
    """

    @Skill.define
    def decide(observation: str) -> str:
        """Decide what to do about {observation}."""

    console = rich.console.Console(file=io.StringIO(), force_terminal=True, width=100)
    mock = MockCompletionHandler([make_text_response("press B")])
    with (
        handler(AgentLoop()),
        handler(LiteLLMConfigurer(model="test-model")),
        handler(HistoryBuilder()),
        handler(StatefulReplSynthesizer()),
        handler(mock),
    ):
        assert decide("the next presses") == "press B"

    # Off the wire rather than out of a history: a free `Skill` keeps none.
    request = next(m for m in mock.received_messages[0] if m["role"] == "user")
    console.print(_render_content(_message_text(request["content"])))
    rendered = console.file.getvalue()

    assert "the next presses" in rendered
    assert "REPL session" in rendered
    assert "observation" in rendered


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
