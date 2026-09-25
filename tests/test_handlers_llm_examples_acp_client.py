"""Offline tests for the AG-UI bridge in ``docs/source/llm_examples/acp/client.py``.

A scripted ACP agent talks to the bridge over the SDK's in-memory transport, so
every test crosses the real wire; the last one drives the effectful ACP server
itself, with the model mocked.
"""

import asyncio
import base64
import collections.abc
import contextlib
import dataclasses
import functools
import json
import pathlib
import sys
import typing

import httpx
import pytest
from PIL import Image

from effectful.handlers.llm import Agent, Skill
from effectful.handlers.llm.harness import harness
from effectful.ops.semantics import coproduct, handler
from tests.conftest import (
    MockCompletionHandler,
    make_text_response,
    make_tool_call_response,
)

EXAMPLE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "docs"
    / "source"
    / "llm_examples"
    / "acp"
)
sys.path.insert(0, str(EXAMPLE_DIR))

pytest.importorskip("acp", reason="the ACP example needs agent-client-protocol")
pytest.importorskip("ag_ui", reason="the AG-UI bridge needs ag-ui-protocol")

import acp  # noqa: E402
import acp.schema  # noqa: E402
import client  # noqa: E402
import library  # noqa: E402
from acp._transport import memory_transport_pair  # noqa: E402
from ag_ui.core import (  # noqa: E402
    BaseEvent,
    DataSource,
    EventType,
    ImagePart,
    ResumeEntry,
    RunAgentInput,
    TextPart,
    UserMessage,
)
from ag_ui.encoder import EventEncoder  # noqa: E402

# In scope for `_Writer`'s skill, which finds its tools lexically.
from library import acp_write_text_file  # noqa: E402, F401

pytestmark = pytest.mark.timeout(60)

OPTIONS = [
    acp.schema.PermissionOption(
        option_id="allow_once", name="Allow", kind="allow_once"
    ),
    acp.schema.PermissionOption(
        option_id="reject_once", name="Reject", kind="reject_once"
    ),
]

FORM = acp.schema.ElicitationSchema.model_validate(
    {
        "type": "object",
        "properties": {"name": {"type": "string", "title": "Name"}},
        "required": ["name"],
    }
)

IMAGE = base64.b64encode(b"not really a png").decode()


def asynchronous(test):
    """Run an async test on a fresh event loop."""

    @functools.wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return run


class FakeAgent:
    """An ACP agent whose turns are scripts, and which logs what it is told."""

    def __init__(self, *turns, image=False, on_new_session=None):
        self.turns = list(turns)
        self.image = image
        self.on_new_session = on_new_session
        self.log: list[str] = []
        self.prompts: list[list[typing.Any]] = []
        self.cancelled = asyncio.Event()
        self.sessions = 0
        self.capabilities: acp.schema.ClientCapabilities | None = None

    def on_connect(self, conn):
        self.conn = conn

    async def initialize(self, protocol_version, client_capabilities=None, **kwargs):
        self.capabilities = client_capabilities
        return acp.schema.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=acp.schema.AgentCapabilities(
                prompt_capabilities=acp.schema.PromptCapabilities(image=self.image)
            ),
        )

    async def new_session(self, cwd, mcp_servers=None, **kwargs):
        self.sessions += 1
        session_id = f"s{self.sessions}"
        if self.on_new_session is not None:
            await self.on_new_session(self, session_id)
        return acp.schema.NewSessionResponse(session_id=session_id)

    async def prompt(self, session_id, prompt, **kwargs):
        self.log.append("prompt")
        self.prompts.append(prompt)
        self.cancelled.clear()
        stop_reason = await self.turns.pop(0)(self, session_id)
        self.log.append(f"end:{stop_reason}")
        return acp.schema.PromptResponse(stop_reason=stop_reason)

    async def cancel(self, session_id, **kwargs):
        self.log.append("cancel")
        self.cancelled.set()

    async def send(self, session_id, update):
        await self.conn.session_update(session_id=session_id, update=update)

    async def say(self, session_id, text):
        await self.send(session_id, acp.update_agent_message_text(text))

    async def think(self, session_id, text):
        await self.send(session_id, acp.update_agent_thought_text(text))

    async def ask(self, session_id, call_id, title="write /x"):
        response = await self.conn.request_permission(
            session_id=session_id,
            tool_call=acp.schema.ToolCallUpdate(
                tool_call_id=call_id, title=title, raw_input={"path": "/x"}
            ),
            options=OPTIONS,
        )
        outcome = response.outcome
        chosen = (
            outcome.option_id
            if isinstance(outcome, acp.schema.AllowedOutcome)
            else "cancelled"
        )
        self.log.append(f"permission:{chosen}")


async def say_ok(agent, session_id):
    await agent.say(session_id, "ok")
    return "end_turn"


@contextlib.asynccontextmanager
async def connected(agent, cwd="/tmp"):
    """A started bridge talking to `agent` over the in-memory transport."""
    agent_end, client_end = memory_transport_pair()
    serving = asyncio.create_task(
        acp.run_agent(agent, agent_end, use_unstable_protocol=True)
    )
    bridge = client.Bridge(cwd)
    connection = acp.connect_to_agent(bridge, client_end, use_unstable_protocol=True)
    try:
        await bridge.start()
        yield bridge
    finally:
        await connection.close()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


def user(text, id="u1"):
    return UserMessage(id=id, content=text)


def request(*messages, run="r1", resume=None, version=None, thread="t1"):
    return RunAgentInput(
        thread_id=thread,
        run_id=run,
        messages=list(messages),
        resume=resume,
        protocol_version=version,
    )


def answer(interrupt, status="resolved", payload=None):
    return [ResumeEntry(interrupt_id=interrupt.id, status=status, payload=payload)]


ALLOW = OPTIONS[0].model_dump(mode="json", by_alias=True, exclude_none=True)


def kinds(events):
    return [event.type.value for event in events]


def of(events, kind):
    return [event for event in events if event.type.value == kind]


def texts(events):
    return [event.delta for event in of(events, "TEXT_MESSAGE_CONTENT")]


def wire(event: BaseEvent) -> dict:
    return json.loads(EventEncoder().encode(event).removeprefix("data: "))


PAIRS = {
    "TEXT_MESSAGE_START": ("text", "open"),
    "TEXT_MESSAGE_CONTENT": ("text", "use"),
    "TEXT_MESSAGE_END": ("text", "close"),
    "REASONING_START": ("span", "open"),
    "REASONING_END": ("span", "close"),
    "REASONING_MESSAGE_START": ("reasoning", "open"),
    "REASONING_MESSAGE_CONTENT": ("reasoning", "use"),
    "REASONING_MESSAGE_END": ("reasoning", "close"),
    "TOOL_CALL_START": ("tool", "open"),
    "TOOL_CALL_ARGS": ("tool", "use"),
    "TOOL_CALL_END": ("tool", "close"),
}


def check(events):
    """The rules @ag-ui/client enforces on a run, and no nulls on the wire."""
    ends = {EventType.RUN_FINISHED, EventType.RUN_ERROR}
    assert events[0].type == EventType.RUN_STARTED
    assert events[-1].type in ends
    assert not {e.type for e in events[1:-1]} & (ends | {EventType.RUN_STARTED})
    if events[-1].type == EventType.RUN_FINISHED:
        assert events[-1].run_id == events[0].run_id
    open_ids: set[tuple[str, str]] = set()
    for event in events:
        assert "null" not in json.dumps(_values(wire(event)))
        kind, action = PAIRS.get(event.type.value, (None, None))
        if kind is None:
            continue
        key = (kind, getattr(event, "tool_call_id", None) or event.message_id)
        if action == "open":
            assert key not in open_ids
            open_ids.add(key)
        elif action == "use":
            assert key in open_ids
        else:
            open_ids.remove(key)
    assert not open_ids


def _values(value):
    """`value` with its strings blanked, so only structural nulls remain.

    A ``responseSchema`` is free-form JSON, whose own nulls are content.
    """
    if isinstance(value, dict):
        return {k: _values(v) for k, v in value.items() if k != "responseSchema"}
    if isinstance(value, list):
        return [_values(item) for item in value]
    return "" if isinstance(value, str) else value


async def collect(bridge, input):
    events = [event async for event in bridge.run(input)]
    check(events)
    return events


# ============================================================================
# Streaming a turn
# ============================================================================


@asynchronous
async def test_a_reply_streams_as_one_assistant_message():
    async def turn(agent, session_id):
        await agent.say(session_id, "Hello, ")
        await agent.say(session_id, "world.")
        return "end_turn"

    agent = FakeAgent(turn)
    async with connected(agent) as bridge:
        events = await collect(bridge, request(user("hi")))

    assert kinds(events) == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    assert "".join(texts(events)) == "Hello, world."
    assert events[0].protocol_version == "1.0"
    assert events[-1].result == {"stopReason": "end_turn"}
    assert agent.prompts == [[acp.text_block("hi")]]
    capabilities = agent.capabilities
    assert capabilities is not None and capabilities.fs is not None
    assert capabilities.fs.read_text_file and capabilities.fs.write_text_file
    assert not capabilities.terminal
    assert capabilities.elicitation is not None
    assert capabilities.elicitation.form is not None


@asynchronous
async def test_reasoning_and_text_alternate_as_separate_messages():
    async def turn(agent, session_id):
        await agent.think(session_id, "Let me see.")
        await agent.say(session_id, "First.")
        await agent.think(session_id, "More.")
        await agent.say(session_id, "Second.")
        return "end_turn"

    async with connected(FakeAgent(turn)) as bridge:
        events = await collect(bridge, request(user("hi")))

    assert kinds(events)[1:6] == [
        "REASONING_START",
        "REASONING_MESSAGE_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END",
        "REASONING_END",
    ]
    assert texts(events) == ["First.", "Second."]
    assert len({e.message_id for e in of(events, "TEXT_MESSAGE_START")}) == 2


@asynchronous
async def test_a_tool_call_is_shown_once_with_its_final_arguments_and_result():
    async def turn(agent, session_id):
        await agent.say(session_id, "Editing.")
        await agent.send(
            session_id,
            acp.start_tool_call("c1", "write", status="pending", raw_input={"pa": 1}),
        )
        await agent.send(
            session_id,
            acp.update_tool_call("c1", raw_input={"path": "/x", "content": "new\n"}),
        )
        await agent.send(
            session_id,
            acp.update_tool_call("c1", status="in_progress", title="write(/x)"),
        )
        await agent.send(
            session_id,
            acp.update_tool_call(
                "c1",
                status="completed",
                content=[acp.tool_diff_content("/x", "new\n", "old\n")],
            ),
        )
        return "end_turn"

    async with connected(FakeAgent(turn)) as bridge:
        events = await collect(bridge, request(user("edit")))

    (start,) = of(events, "TOOL_CALL_START")
    (args,) = of(events, "TOOL_CALL_ARGS")
    (result,) = of(events, "TOOL_CALL_RESULT")
    (text,) = of(events, "TEXT_MESSAGE_START")
    assert start.tool_call_name == "write(/x)"
    assert start.parent_message_id == text.message_id
    assert json.loads(args.delta) == {"path": "/x", "content": "new\n"}
    assert "-old" in result.content and "+new" in result.content


# ============================================================================
# Questions for the user, as interrupts
# ============================================================================


@pytest.mark.parametrize(
    "status, payload, logged",
    [
        ("resolved", ALLOW, "permission:allow_once"),
        ("cancelled", None, "permission:cancelled"),
        (
            "resolved",
            {"optionId": "sudo", "name": "Sudo", "kind": "allow_always"},
            "permission:cancelled",
        ),
    ],
)
@asynchronous
async def test_a_permission_request_is_an_interrupt_the_next_run_answers(
    status, payload, logged
):
    async def turn(agent, session_id):
        await agent.say(session_id, "Checking.")
        await agent.send(session_id, acp.start_tool_call("c1", "write"))
        await agent.ask(session_id, "c1")
        await agent.say(session_id, "Done.")
        return "end_turn"

    agent = FakeAgent(turn)
    async with connected(agent) as bridge:
        first = await collect(bridge, request(user("go")))
        (interrupt,) = first[-1].outcome.interrupts
        second = await collect(
            bridge, request(user("go"), resume=answer(interrupt, status, payload))
        )

    assert kinds(first)[-4:] == [
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "RUN_FINISHED",
    ]
    assert of(first, "TOOL_CALL_START")[0].tool_call_name == "write /x"
    shown = wire(first[-1])["outcome"]
    assert shown["type"] == "interrupt"
    assert shown["interrupts"][0]["reason"] == "confirmation"
    assert shown["interrupts"][0]["toolCallId"] == "c1"
    assert shown["interrupts"][0]["responseSchema"]["title"] == "PermissionOption"
    assert [o["optionId"] for o in shown["interrupts"][0]["metadata"]["options"]] == [
        "allow_once",
        "reject_once",
    ]
    assert texts(second) == ["Done."]
    assert agent.log == ["prompt", logged, "end:end_turn"]


@pytest.mark.parametrize(
    "status, payload, expected",
    [
        ("resolved", {"name": "Ada"}, "accept:{'name': 'Ada'}"),
        ("resolved", None, "decline:None"),
        ("cancelled", None, "cancel:None"),
    ],
)
@asynchronous
async def test_a_form_is_an_interrupt_whose_answer_is_the_reply(
    status, payload, expected
):
    async def turn(agent, session_id):
        response = await agent.conn.create_elicitation(
            message="Who?",
            mode=acp.schema.ElicitationFormSessionMode(
                session_id=session_id, requested_schema=FORM
            ),
        )
        agent.log.append(f"{response.action}:{getattr(response, 'content', None)}")
        return "end_turn"

    agent = FakeAgent(turn)
    async with connected(agent) as bridge:
        first = await collect(bridge, request(user("go")))
        (interrupt,) = first[-1].outcome.interrupts
        await collect(
            bridge, request(user("go"), resume=answer(interrupt, status, payload))
        )

    assert interrupt.reason == "input_required"
    assert interrupt.message == "Who?"
    assert interrupt.response_schema == FORM.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    assert agent.log == ["prompt", expected, "end:end_turn"]


@asynchronous
async def test_parallel_requests_are_shown_one_at_a_time_in_order():
    async def turn(agent, session_id):
        await agent.say(session_id, "Two things.")
        first = asyncio.create_task(agent.ask(session_id, "a", title="write a"))
        await asyncio.sleep(0.05)
        await agent.say(session_id, "Also:")
        second = asyncio.create_task(agent.ask(session_id, "b", title="write b"))
        await asyncio.gather(first, second)
        await agent.say(session_id, "Done.")
        return "end_turn"

    agent = FakeAgent(turn)
    async with connected(agent) as bridge:
        stream = bridge.run(request(user("go")))
        first = [await anext(stream), await anext(stream)]
        # Hold the run still until both requests wait, so it could see B too.
        while len(bridge.threads["t1"].turn.interrupts) < 2:
            await asyncio.sleep(0.01)
        first += [event async for event in stream]
        check(first)
        (a,) = first[-1].outcome.interrupts
        second = await collect(
            bridge, request(user("go"), resume=answer(a, payload=ALLOW))
        )
        (b,) = second[-1].outcome.interrupts
        third = await collect(
            bridge, request(user("go"), resume=answer(b, payload=ALLOW))
        )

    assert (a.tool_call_id, b.tool_call_id) == ("a", "b")
    assert [e.tool_call_id for e in of(first, "TOOL_CALL_START")] == ["a"]
    assert texts(first) == ["Two things."]
    assert [e.tool_call_id for e in of(second, "TOOL_CALL_START")] == ["b"]
    assert texts(second) == ["Also:"]
    assert texts(third) == ["Done."]
    assert agent.log == [
        "prompt",
        "permission:allow_once",
        "permission:allow_once",
        "end:end_turn",
    ]


@asynchronous
async def test_an_unanswered_interrupt_is_sent_again_even_with_a_new_message():
    async def turn(agent, session_id):
        await agent.send(session_id, acp.start_tool_call("c1", "write"))
        await agent.ask(session_id, "c1")
        return "end_turn"

    agent = FakeAgent(turn)
    async with connected(agent) as bridge:
        first = await collect(bridge, request(user("go")))
        again = await collect(
            bridge, request(user("go"), user("never mind", id="u2"), run="r2")
        )
        log_while_waiting = list(agent.log)
        (interrupt,) = again[-1].outcome.interrupts
        await collect(
            bridge, request(user("go"), resume=answer(interrupt, payload=ALLOW))
        )

    assert kinds(again) == ["RUN_STARTED", "RUN_FINISHED"]
    assert interrupt.id == first[-1].outcome.interrupts[0].id
    assert log_while_waiting == ["prompt"]
    assert agent.log == ["prompt", "permission:allow_once", "end:end_turn"]


# ============================================================================
# One run per thread, and ending a turn early
# ============================================================================


@asynchronous
async def test_a_second_run_on_a_busy_thread_is_refused():
    release = asyncio.Event()

    async def turn(agent, session_id):
        await agent.say(session_id, "working")
        await release.wait()
        return "end_turn"

    async with connected(FakeAgent(turn)) as bridge:
        stream = bridge.run(request(user("go")))
        head = [await anext(stream) for _ in range(3)]
        busy = await collect(bridge, request(user("again", id="u2"), run="r2"))
        release.set()
        rest = [event async for event in stream]

    assert kinds(busy) == ["RUN_STARTED", "RUN_ERROR"]
    assert busy[-1].code == "THREAD_BUSY"
    check(head + rest)
    assert rest[-1].result == {"stopReason": "end_turn"}


@asynchronous
async def test_a_dropped_stream_cancels_its_turn_before_a_new_one_starts():
    async def dropped(agent, session_id):
        await agent.say(session_id, "partial")
        await agent.ask(session_id, "c1")
        await agent.say(session_id, "tail")
        return "cancelled"

    async def fresh(agent, session_id):
        await agent.say(session_id, "fresh")
        return "end_turn"

    agent = FakeAgent(dropped, fresh)
    async with connected(agent) as bridge:
        stream = bridge.run(request(user("go")))
        while (await anext(stream)).type != EventType.TEXT_MESSAGE_CONTENT:
            pass
        await stream.aclose()
        second = await collect(
            bridge, request(user("go"), user("again", id="u2"), run="r2")
        )

    assert agent.log == [
        "prompt",
        "cancel",
        "permission:cancelled",
        "end:cancelled",
        "prompt",
        "end:end_turn",
    ]
    assert texts(second) == ["fresh"]


@pytest.mark.parametrize("version", [None, "1.0"])
@asynchronous
async def test_a_cancelled_turn_is_reported_in_the_clients_protocol_version(
    version, capsys
):
    async def turn(agent, session_id):
        await agent.send(session_id, acp.start_tool_call("c1", "write"))
        await agent.ask(session_id, "c1")
        return "cancelled"

    agent = FakeAgent(turn)
    async with connected(agent) as bridge:
        first = await collect(bridge, request(user("go"), version=version))
        (interrupt,) = first[-1].outcome.interrupts
        resume = answer(interrupt, "cancelled")
        last = await collect(
            bridge, request(user("go"), resume=resume, version=version)
        )

    assert agent.log == ["prompt", "permission:cancelled", "end:cancelled"]
    assert last[0].protocol_version == "1.0"
    warned = "predates the cancelled outcome" in capsys.readouterr().err
    if version is None:
        assert last[-1].outcome is None
        assert last[-1].result == {"stopReason": "cancelled"}
        assert warned
    else:
        assert wire(last[-1])["outcome"] == {"type": "cancelled"}
        assert last[-1].result is None
        assert not warned


# ============================================================================
# The prompt, and session state
# ============================================================================


@pytest.mark.parametrize("image", [False, True])
@asynchronous
async def test_parts_the_agent_cannot_take_are_skipped(image, capsys):
    agent = FakeAgent(say_ok, image=image)
    picture = ImagePart(source=DataSource(value=IMAGE, mime_type="image/png"))
    message = UserMessage(id="u1", content=[TextPart(text="look"), picture])
    async with connected(agent) as bridge:
        await collect(bridge, request(message))

    expected = [acp.text_block("look")]
    if image:
        expected.append(acp.image_block(IMAGE, "image/png"))
    assert agent.prompts == [expected]
    assert ("skipping a image part" in capsys.readouterr().err) is not image


@asynchronous
async def test_a_message_with_nothing_the_agent_can_take_is_not_sent():
    agent = FakeAgent(say_ok)
    picture = ImagePart(source=DataSource(value=IMAGE, mime_type="image/png"))
    async with connected(agent) as bridge:
        events = await collect(bridge, request(UserMessage(id="u1", content=[picture])))

    assert kinds(events) == ["RUN_STARTED", "RUN_FINISHED"]
    assert agent.log == []


@asynchronous
async def test_session_updates_become_state_snapshots():
    async def announce(agent, session_id):
        await agent.send(
            session_id,
            acp.schema.AvailableCommandsUpdate(
                session_update="available_commands_update",
                available_commands=[
                    acp.schema.AvailableCommand(name="clear", description="Forget.")
                ],
            ),
        )

    async def turn(agent, session_id):
        await agent.send(
            session_id, acp.update_plan([acp.plan_entry("Read it", status="pending")])
        )
        await agent.say(session_id, "ok")
        return "end_turn"

    async with connected(FakeAgent(turn, on_new_session=announce)) as bridge:
        events = await collect(bridge, request(user("go")))

    first, last = of(events, "STATE_SNAPSHOT")
    commands = first.snapshot["available_commands_update"]["availableCommands"]
    assert commands[0]["name"] == "clear"
    assert last.snapshot["plan"]["entries"][0]["content"] == "Read it"
    assert "available_commands_update" in last.snapshot


@asynchronous
async def test_a_failed_prompt_is_a_run_error_and_the_thread_goes_on():
    async def broken(agent, session_id):
        raise acp.RequestError.invalid_params({"reason": "no such model"})

    async with connected(FakeAgent(broken, say_ok)) as bridge:
        failed = await collect(bridge, request(user("go")))
        after = await collect(bridge, request(user("again", id="u2"), run="r2"))

    assert failed[-1].type == EventType.RUN_ERROR
    assert "no such model" in failed[-1].message
    assert texts(after) == ["ok"]


# ============================================================================
# Files, served for the agent
# ============================================================================


@asynchronous
async def test_files_are_read_and_written_on_disk(tmp_path):
    agent = FakeAgent()
    target = tmp_path / "nested" / "dir" / "a.txt"
    async with connected(agent):
        await agent.conn.write_text_file(
            session_id="s1", path=str(target), content="one\ntwo\nthree\n"
        )
        read = await agent.conn.read_text_file(
            session_id="s1", path=str(target), line=2, limit=1
        )
        with pytest.raises(acp.RequestError) as missing:
            await agent.conn.read_text_file(
                session_id="s1", path=str(tmp_path / "nope.txt")
            )

    assert target.read_text() == "one\ntwo\nthree\n"
    assert read.content == "two\n"
    assert missing.value.code == acp.RequestError.resource_not_found().code


@asynchronous
async def test_terminals_are_refused():
    agent = FakeAgent()
    async with connected(agent):
        with pytest.raises(acp.RequestError) as refused:
            await agent.conn.create_terminal(session_id="s1", command="ls")

    assert refused.value.code == acp.RequestError.method_not_found("x").code


# ============================================================================
# HTTP
# ============================================================================


def _http(bridge, base_url="http://localhost"):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=client.make_app(bridge)), base_url=base_url
    )


def _body(input: RunAgentInput) -> dict:
    return input.model_dump(mode="json", by_alias=True, exclude_none=True)


@asynchronous
async def test_a_run_streams_as_server_sent_events():
    async with connected(FakeAgent(say_ok)) as bridge, _http(bridge) as http:
        response = await http.post(
            "/",
            json=_body(request(user("hi"))),
            headers={"accept": "text/event-stream", "origin": "http://localhost:3000"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    *frames, rest = response.text.split("\n\n")
    assert rest == ""
    assert all(frame.startswith("data: ") for frame in frames)
    events = [json.loads(frame.removeprefix("data: ")) for frame in frames]
    assert [events[0]["type"], events[-1]["type"]] == ["RUN_STARTED", "RUN_FINISHED"]


@asynchronous
async def test_browser_pages_cannot_drive_the_agent():
    agent = FakeAgent(say_ok)
    page = {"origin": "http://localhost:3000"}
    preflight = {"access-control-request-method": "POST"}
    async with (
        connected(agent) as bridge,
        _http(bridge) as http,
        _http(bridge, base_url="http://attacker.example") as rebound,
    ):
        body = _body(request(user("hi")))
        # A JSON POST from a page needs this preflight to succeed first.
        checked = await http.options("/", headers=page | preflight)
        # The one POST a page can make without a preflight.
        simple = await http.post(
            "/", content=json.dumps(body), headers=page | {"content-type": "text/plain"}
        )
        rebound_response = await rebound.post("/", json=body)
        reached = (agent.sessions, list(agent.log))
        from_server = await http.post("/", json=body)

    assert "access-control-allow-origin" not in checked.headers
    assert simple.status_code == 422
    assert rebound_response.status_code == 400
    assert reached == (0, [])
    assert from_server.status_code == 200
    assert agent.log == ["prompt", "end:end_turn"]


# ============================================================================
# End to end, with the effectful ACP server
# ============================================================================


@dataclasses.dataclass
class _Writer(Agent):
    """Writes the files it is asked to write."""

    __agent_id__: str = ""

    @Skill.define
    def prompt(
        self,
        user_input: str,
        attachments: collections.abc.Sequence[library.Attachment] = (),
        images: collections.abc.Sequence[Image.Image] = (),
    ) -> str:
        """{user_input}"""
        raise NotImplementedError


def _stack(*handlers):
    intp = harness(
        model="mock/model",
        eval_provider="none",
        type_checker="none",
        tool_calling="json",
    )
    for h in handlers:
        intp = coproduct(intp, h)
    return intp


def test_the_effectful_server_writes_through_the_bridge_once_allowed(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(library.OFFER_MODELS_ENV, raising=False)
    target = tmp_path / "hello.txt"
    mock = MockCompletionHandler(
        [
            make_tool_call_response(
                "acp_write_text_file",
                json.dumps({"path": str(target), "content": "hello\n"}),
            ),
            make_text_response("Wrote it."),
        ]
    )

    async def drive():
        agent_end, client_end = memory_transport_pair()
        server = library.EffectfulACPAgent(_Writer)
        serving = asyncio.create_task(
            acp.run_agent(server, agent_end, use_unstable_protocol=True)
        )
        bridge = client.Bridge(str(tmp_path))
        connection = acp.connect_to_agent(
            bridge, client_end, use_unstable_protocol=True
        )
        try:
            await bridge.start()
            first = await collect(bridge, request(user("write hello.txt")))
            (interrupt,) = first[-1].outcome.interrupts
            second = await collect(
                bridge,
                request(
                    user("write hello.txt"), resume=answer(interrupt, payload=ALLOW)
                ),
            )
            return first, second
        finally:
            await connection.close()
            serving.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await serving

    with handler(_stack(mock)):
        first, second = asyncio.run(drive())

    assert target.read_text() == "hello\n"
    (call,) = of(first, "TOOL_CALL_START")
    assert call.tool_call_name.startswith("acp_write_text_file")
    (result,) = of(second, "TOOL_CALL_RESULT")
    assert "+hello" in result.content
    assert "".join(texts(second)) == "Wrote it."
    assert second[-1].result["stopReason"] == "end_turn"
