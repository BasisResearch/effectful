"""Offline tests for the ACP example under ``docs/source/llm_examples/acp``.

This is the example's only coverage, and the reason it needs its own file. Every
other example is run for real by `test_handlers_llm_examples.py`, which launches it
as a subprocess and checks it exits zero; this one is a *server*, whose stdio is a
JSON-RPC conversation with an editor rather than a terminal, so there is nothing for
that suite to launch it against and it is skipped there.

What stands in for the editor is `_FakeClient`, and for the model
`MockCompletionHandler`, so none of this touches the network. Between them they cover
the three translations the example is made of -- harness effects to `session/update`
notifications, `session/request_permission` to a decision about a tool call, and the
editor's own methods to tools -- plus the contracts the examples tree imposes on any
script.

Every test runs against the same threading arrangement the server uses -- an event
loop on its own thread, the harness call on another -- because that split is the
thing most likely to be got wrong. A session whose loop is constructed but never run
deadlocks the first time a tool blocks on the editor, and only a test that runs the
loop can catch it.
"""

import argparse
import ast
import asyncio
import contextlib
import dataclasses
import os
import pathlib
import re
import sys
import threading
import time
import typing

import litellm
import pydantic
import pytest

from effectful.handlers.llm import Agent, Encodable, Skill, Tool
from effectful.handlers.llm.harness import harness
from effectful.handlers.llm.harness.hooks import call_tool, completion
from effectful.handlers.llm.harness.legibility.lexical import (
    _tool_paths,
    _tools_in_scope,
)
from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
from effectful.handlers.llm.harness.serialization import (
    DecodedToolCall,
    _NameAndTool,
    _serialize_name_and_tool,
    to_content_blocks,
)
from effectful.ops.semantics import coproduct, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from tests.conftest import (
    MockCompletionHandler,
    make_text_response,
    make_tool_call_response,
)

# The example is a script rather than an installed package, so it is imported the way
# the launcher makes it importable: with its own directory on the path.
EXAMPLE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "docs"
    / "source"
    / "llm_examples"
    / "acp"
)
sys.path.insert(0, str(EXAMPLE_DIR))

# Skip the module rather than fail collection where the optional dependency is
# absent, then import for real -- `importorskip`'s return value is opaque to a type
# checker, and these names are used as base classes and in annotations below.
pytest.importorskip("acp", reason="the ACP example needs agent-client-protocol")

import acp  # noqa: E402
import assistant  # noqa: E402
import library  # noqa: E402
from acp import schema  # noqa: E402

# Bound here for the same reason `assistant.py` binds them: a `Skill` finds its tools
# in its own lexical scope, so the skills defined in this module see these three.
from library import (  # noqa: E402
    acp_ask_user,
    acp_read_text_file,
    acp_run_terminal_command,
    acp_update_plan,
    acp_write_text_file,
)

# A hang in this file means a session whose loop is not running, or a queue nobody
# drains -- a deadlock rather than a slow test. Fail it quickly and say so.
pytestmark = pytest.mark.timeout(60)


@pytest.fixture(autouse=True)
def _no_inherited_model_offers(monkeypatch):
    """Keep the developer's own environment out of every server these tests build.

    `EffectfulACPAgent` reads `OFFER_MODELS_ENV` when told nothing about models, so
    without this a machine configured to run the example for real would give every
    session a model picker and quietly fail the tests that assert there is none.
    """
    monkeypatch.delenv(library.OFFER_MODELS_ENV, raising=False)


# ============================================================================
# Doubles
# ============================================================================


class _FakeClient:
    """An editor that records notifications and answers requests from a script."""

    def __init__(
        self,
        permission: str | None = "allow_once",
        files: dict[str, str] | None = None,
        terminal: tuple[str, int] = ("", 0),
        elicitation: typing.Any = None,
    ):
        self.updates: list[typing.Any] = []
        self.permission = permission
        self.files = files or {}
        self.asked: list[str] = []
        self.ran: list[tuple[str, list[str], str | None]] = []
        self.released: list[str] = []
        self.output, self.exit_code = terminal
        # An editor that accepts every form, unless a test scripts otherwise. The
        # answers are keyed by whatever was asked for, which `elicited` records.
        self.elicitation = elicitation
        self.elicited: list[tuple[str, typing.Any]] = []

    async def session_update(self, session_id, update, **kw) -> None:
        self.updates.append(update)

    async def request_permission(self, session_id, tool_call, options, **kw):
        self.asked.append(tool_call.title)
        if self.permission is None:
            return schema.RequestPermissionResponse(
                outcome=schema.DeniedOutcome(outcome="cancelled")
            )
        return schema.RequestPermissionResponse(
            outcome=schema.AllowedOutcome(outcome="selected", option_id=self.permission)
        )

    async def read_text_file(self, session_id, path, line=None, limit=None, **kw):
        return schema.ReadTextFileResponse(content=self.files[path])

    async def write_text_file(self, session_id, path, content, **kw):
        self.files[path] = content
        return schema.WriteTextFileResponse()

    async def create_terminal(self, session_id, command, args=None, cwd=None, **kw):
        self.ran.append((command, list(args or []), cwd))
        return schema.CreateTerminalResponse(terminal_id="t1")

    async def wait_for_terminal_exit(self, session_id, terminal_id, **kw):
        return schema.WaitForTerminalExitResponse(exit_code=self.exit_code, signal=None)

    async def terminal_output(self, session_id, terminal_id, **kw):
        return schema.TerminalOutputResponse(output=self.output, truncated=False)

    async def release_terminal(self, session_id, terminal_id, **kw):
        self.released.append(terminal_id)
        return schema.ReleaseTerminalResponse()

    async def create_elicitation(self, message, mode, **kw):
        self.elicited.append((message, mode))
        if self.elicitation is not None:
            return self.elicitation
        # Accepting with a plausible answer per field means a test that only cares
        # that the form was asked for does not have to script one.
        properties = (mode.requested_schema.properties or {}).keys()
        return schema.AcceptElicitationResponse(
            action="accept", content={name: "yes" for name in properties}
        )


@dataclasses.dataclass
class _Bot(Agent):
    """A minimal agent, shaped like the example's: the id is the only field.

    That shape is what makes the class itself the ``make_agent`` callable
    `EffectfulACPAgent` takes, so passing ``_Bot`` below exercises the real wiring.
    """

    __agent_id__: str = ""

    @Skill.define
    def respond(self, user_input: str) -> str:
        """Answer: {user_input}"""


@Tool.define
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@Tool.define
def explode() -> str:
    """Always fails."""
    raise RuntimeError("boom")


_CWD = "/tmp/project"


@contextlib.contextmanager
def _session(
    client: _FakeClient,
    caps: schema.ClientCapabilities | None = None,
    *,
    cwd: str = _CWD,
    **kwargs,
):
    """A session arranged the way the server arranges one, for the test's duration.

    The event loop runs on its own thread and the test body plays the worker, which
    is the split production has. It matters: `ACPSession.call` blocks on
    `asyncio.run_coroutine_threadsafe`, so a loop that is merely *constructed* and
    never run deadlocks the first time a tool asks the editor anything.

    The session is built on the loop thread because it starts its own writer task
    there, and that task running is what lets a test assert on what the client
    actually received rather than on what was queued. Waiting for the queue to drain
    on the way out is what makes those assertions deterministic.
    """
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    def on_loop(coro, timeout=10):
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

    async def build():
        return library.ACPSession(
            agent=_Bot("s1"),
            client=typing.cast(typing.Any, client),
            client_capabilities=caps or schema.ClientCapabilities(),
            cwd=cwd,
            **kwargs,
        )

    session = on_loop(build())
    try:
        yield session
        on_loop(session.updates.join())
    finally:
        writer = typing.cast(asyncio.Task, session.writer)
        loop.call_soon_threadsafe(writer.cancel)
        # Let the loop actually process the cancellation before it is stopped;
        # otherwise the task is collected mid-cancel and asyncio complains.
        with contextlib.suppress(Exception):
            on_loop(asyncio.sleep(0))
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()


def _stack(*handlers):
    """The real handler stack with the model mocked, plus `handlers` on top.

    `harness` rather than a hand-picked set, because the order handlers are installed
    in is part of what these tests are checking and a stack assembled here would only
    ever be a guess at it. The ACP handlers go on top, which is where
    `EffectfulACPAgent._answer` puts them -- above `TenacityRetryer`, so a failing
    tool reaches them as a *result* rather than as the exception it started as.

    The code paths that need a model, a type checker or an interpreter are all
    switched off; what is left is the ordering.
    """
    intp = harness(
        model="mock/model",
        eval_provider="none",
        type_checker="none",
        tool_calling="json",
    )
    for h in handlers:
        intp = coproduct(intp, h)
    return intp


def _tool_schema(tool: Tool) -> dict:
    """The JSON Schema for `tool`'s parameters, as the model is shown it.

    Through the harness's own encoder rather than pydantic's, because the difference
    is the point: `_ensure_strict_json_schema` is what closes the objects and moves
    every property into ``required``, and both are what the tests below are about.
    """
    return _serialize_name_and_tool(_NameAndTool(tool.__name__, tool))["function"][
        "parameters"
    ]


def _plan_tool_taking(step_type: type) -> Tool:
    """A plan tool built on `step_type`, for comparing against the real one."""

    @Tool.define
    def show_plan(steps: list[step_type]) -> str:  # type: ignore[valid-type]
        """Show the user a plan."""
        raise RuntimeError("never called; only its schema is read")

    return show_plan


async def _serve(client: _FakeClient, **kwargs) -> library.EffectfulACPAgent:
    """A connected, initialized server over `_Bot`. Call from inside a running loop."""
    server = library.EffectfulACPAgent(_Bot, skill_name="respond", **kwargs)
    server.on_connect(typing.cast(typing.Any, client))
    await server.initialize(protocol_version=acp.PROTOCOL_VERSION)
    return server


class _RecordingModel(ObjectInterpretation):
    """A model that records the kwargs of every request it is asked to answer.

    `MockCompletionHandler` keeps only the messages, and what matters below is the
    rest of the request -- specifically whether anything named a model.
    """

    def __init__(self, response=None):
        self.response = response if response is not None else make_text_response("ok")
        self.requests: list[dict] = []

    @implements(completion)
    def _completion(self, messages=None, **kwargs):
        self.requests.append(kwargs)
        return self.response


class _StreamingModel(ObjectInterpretation):
    """A model that answers in chunks, the way a real provider does.

    `MockCompletionHandler` hands back a settled response, which `ACPSessionReporter`
    notices and reports in one go -- so it exercises `_settled` and never `_streamed`,
    which is the path every real turn takes.
    """

    def __init__(
        self,
        *pieces: str,
        model: str = "openrouter/z-ai/glm-5.3-flash",
        reports: litellm.types.utils.Usage | None = None,
    ):
        self.pieces = pieces or ("ok",)
        self.model = model
        self.reports = reports
        self.requests: list[dict] = []

    def _chunk(self, **kwargs) -> litellm.types.utils.ModelResponseStream:
        return litellm.types.utils.ModelResponseStream(
            id="chunked", model=self.model, object="chat.completion.chunk", **kwargs
        )

    @implements(completion)
    def _completion(self, messages=None, stream=False, **kwargs):
        assert stream, "the reporter asks for a stream; ignoring that proves nothing"
        self.requests.append(kwargs)

        def chunks():
            for piece in (*self.pieces, None):
                yield self._chunk(
                    choices=[
                        litellm.types.utils.StreamingChoices(
                            index=0,
                            delta=litellm.types.utils.Delta(content=piece),
                            finish_reason=None if piece else "stop",
                        )
                    ]
                )
            # A provider asked for `include_usage` sends its counts in a final chunk
            # that carries no choices at all. One that was not, or that does not
            # support it, simply ends -- which is what `reports=None` stands for.
            if self.reports is not None:
                yield self._chunk(choices=[], usage=self.reports)

        return chunks()


def _stopped(content: str, finish_reason: str, **kwargs):
    """A reply the provider cut short, or refused, rather than finished.

    `make_text_response` always says ``stop``; the whole point of the stop reasons
    below is telling the other endings apart from that one.
    """
    response = make_text_response(content, **kwargs)
    response.choices[0].finish_reason = typing.cast(typing.Any, finish_reason)
    return response


def _kinds(updates) -> list[str]:
    return [u.session_update for u in updates]


def _conversation(updates) -> list:
    """Just the messages, without the session's own housekeeping notifications.

    Opening or reopening a session announces its slash commands, which is not part of
    the conversation and would otherwise have to be spelled out in every test that
    asserts on what the editor saw.
    """
    return [u for u in updates if u.session_update != "available_commands_update"]


def _statuses(updates) -> list[str]:
    return [u.status for u in updates if u.session_update == "tool_call_update"]


# ============================================================================
# Reporting activity as session/update
# ============================================================================


def test_a_text_answer_is_reported_as_an_agent_message():
    """The skill's answer reaches the editor as an `agent_message_chunk`.

    `MockCompletionHandler` ignores the `stream=True` the reporter adds and hands
    back a settled response, which is exactly the fallback the reporter has to cope
    with -- any handler is free to answer from a cache or a fixture.
    """
    client = _FakeClient()
    with _session(client) as session:
        with handler(
            _stack(
                MockCompletionHandler([make_text_response("hi there")]),
                library.ACPSessionReporter(session),
            )
        ):
            assert session.agent.respond("hello") == "hi there"

    assert [u.content.text for u in client.updates] == ["hi there"]


def test_a_tool_call_is_bracketed_by_status_updates():
    """An editor sees the call announced, then running, then completed."""
    client = _FakeClient()

    @Skill.define
    def use_tool(x: int) -> str:
        """Add {x} to 1 using `add_numbers`."""

    with _session(client) as session:
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
                        make_text_response("3"),
                    ]
                ),
                library.ACPSessionReporter(session),
            )
        ):
            assert use_tool(2) == "3"

    assert _kinds(client.updates) == [
        "tool_call",
        "tool_call_update",
        "tool_call_update",
        "agent_message_chunk",
    ]
    assert client.updates[0].title == "add_numbers"
    assert _statuses(client.updates) == ["in_progress", "completed"]
    assert "3" in client.updates[2].content[0].content.text


def test_a_running_call_is_titled_with_its_arguments():
    """The bare tool name is not enough, for two reasons that compound.

    A client may treat a title that is only an identifier as a placeholder and
    substitute phrasing of its own -- Poolside tests it against
    ``/^[a-z][a-z0-9_]*$/`` -- and it may prefer tool *output* over `rawInput` when
    it has both, which for our calls it always does. Between them the arguments
    disappear, and a turn reads as a list of tool names with no idea what was run.
    """
    client = _FakeClient()

    @Skill.define
    def use_tool(x: int) -> str:
        """Add {x} to 1 using `add_numbers`."""

    with _session(client) as session:
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
                        make_text_response("3"),
                    ]
                ),
                library.ACPSessionReporter(session),
            )
        ):
            use_tool(2)

    running = next(
        u
        for u in client.updates
        if u.session_update == "tool_call_update" and u.status == "in_progress"
    )
    assert running.title == "add_numbers(a=1, b=2)"
    assert not re.fullmatch(r"[a-z][a-z0-9_]*", running.title), (
        "a title that is only an identifier is discarded by some clients"
    )
    # And the arguments are still sent structurally, for clients that show them.
    assert running.raw_input == {"a": 1, "b": 2}


def test_the_title_and_the_permission_prompt_describe_a_call_the_same_way():
    """The same call, so the same sentence: approving it and watching it run agree."""
    tool_call = DecodedToolCall(
        tool=add_numbers,
        bound_args=add_numbers.__signature__.bind(1, 2),
        id="call_1",
        name="add_numbers",
    )
    raw_input = library._raw_input(tool_call)
    assert raw_input == {"a": 1, "b": 2}
    assert library._call_title("add_numbers", raw_input) == "add_numbers(a=1, b=2)"


def test_a_failing_tool_is_reported_as_failed():
    """A tool that raises is reported `failed`, and the model still gets its turn.

    On the real stack the reporter never sees the exception: it sits above
    `TenacityRetryer`, whose whole job is to turn a raising tool into that call's
    result. So the status has to be read off the *result*, and a reporter that only
    watched for an exception would show every failed call as completed with its
    traceback rendered as the output.
    """
    client = _FakeClient()

    @Skill.define
    def use_tool() -> str:
        """Call `explode`."""

    with _session(client) as session:
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response("explode", "{}"),
                        make_text_response("it failed"),
                    ]
                ),
                library.ACPSessionReporter(session),
            )
        ):
            assert use_tool() == "it failed"

    assert _statuses(client.updates) == ["in_progress", "failed"]


def test_a_structured_answer_is_not_streamed_as_prose():
    """A non-`str` return travels as JSON, which would be noise to stream.

    `call_assistant` sets a `response_format` for exactly those, so the reporter has
    a reliable signal; the decoded value is reported once, by the server.
    """
    client = _FakeClient()

    @Skill.define
    def count() -> int:
        """Return 7."""

    with _session(client) as session:
        with handler(
            _stack(
                MockCompletionHandler([make_text_response('{"value": 7}')]),
                library.ACPSessionReporter(session),
            )
        ):
            assert count() == 7

    assert client.updates == [], "structured output should not be streamed as prose"


def test_a_settled_response_is_handed_back_unchanged():
    """Pin the fallback for a handler that ignores the reporter's `stream=True`."""
    response = make_text_response("whole")
    with _session(_FakeClient()) as session:
        reporter = library.ACPSessionReporter(session)
        assert reporter._settled(response, is_prose=True) is response
    assert isinstance(response, litellm.types.utils.ModelResponse)


def test_an_editor_tool_gets_the_icon_its_kind_deserves():
    """`_tool_kind` is keyed by advertised name, so a rename must be made in both.

    A stale key is invisible in every other way: the call still runs, the editor just
    draws the wrong icon.
    """
    assert library._tool_kind(acp_read_text_file.__name__) == "read"
    assert library._tool_kind(acp_write_text_file.__name__) == "edit"
    assert library._tool_kind(acp_run_terminal_command.__name__) == "execute"
    assert library._tool_kind("something_else") is None


def test_a_call_of_no_known_kind_claims_none_rather_than_other():
    """``"other"`` reads as neutral and is not: it is a value in the same vocabulary as
    ``execute``, and editors treat it as a claim. Poolside maps both to the terminal
    icon, which drew `acp_ask_user` -- a form, running nothing -- as a shell command.

    ``kind`` is optional in every message carrying it, so omitting it is the protocol's
    own way to say nothing about a call.
    """
    assert acp_ask_user.__name__ not in library._TOOL_KINDS
    assert library._tool_kind(acp_ask_user.__name__) is None

    client = _FakeClient()
    with _session(client) as session:
        session.reporter.begin_turn()
        session.reporter._start("call-1", acp_ask_user.__name__, status="pending")

    (start,) = [u for u in client.updates if u.session_update == "tool_call"]
    assert start.kind is None, "a kind we do not know must not be asserted"


# ============================================================================
# The permission gate
# ============================================================================


def test_a_denied_tool_call_does_not_run():
    """A refusal reaches the model as the call's result, and the tool never runs."""
    calls: list[int] = []

    @Tool.define
    def record(x: int) -> str:
        """Record a number."""
        calls.append(x)
        return "recorded"

    @Skill.define
    def use_tool() -> str:
        """Call `record`."""

    client = _FakeClient(permission="reject_once")
    with _session(client) as session:
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response("record", '{"x": 1}'),
                        make_text_response("could not"),
                    ]
                ),
                library.ACPPermissionGate(session),
            )
        ):
            assert use_tool() == "could not"

    assert calls == [], "a declined tool must not run"
    assert client.asked == ["record(x=1)"]


def test_a_denied_tool_call_leaves_a_well_formed_history():
    """Every advertised call is answered, so the conversation can be sent again.

    `HistoryBuilder.append_message` asserts this, and both OpenAI APIs require one
    output per advertised call -- so a refusal that merely raised would make the next
    request unsendable rather than informative.
    """
    mock = MockCompletionHandler(
        [
            make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
            make_text_response("declined"),
        ]
    )

    @Skill.define
    def use_tool() -> str:
        """Call `add_numbers`."""

    with _session(_FakeClient(permission="reject_once")) as session:
        with handler(_stack(mock, library.ACPPermissionGate(session))):
            use_tool()

    replied = [m for m in mock.received_messages[-1] if m.get("role") == "tool"]
    assert len(replied) == 1
    assert "declined" in replied[0]["content"]


def test_always_decisions_are_remembered():
    """`allow_always` is asked once and applied to every later call of that tool."""

    @Skill.define
    def use_tool() -> str:
        """Call `add_numbers` twice."""

    client = _FakeClient(permission="allow_always")
    with _session(client) as session:
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response("add_numbers", '{"a": 1, "b": 1}'),
                        make_tool_call_response("add_numbers", '{"a": 2, "b": 2}'),
                        make_text_response("done"),
                    ]
                ),
                library.ACPPermissionGate(session),
            )
        ):
            use_tool()

    assert client.asked == ["add_numbers(a=1, b=1)"], (
        "the second call should not re-ask"
    )


def test_the_permission_prompt_says_which_call_is_being_approved():
    """`title` is the only field ACP calls human-readable, so it must be informative.

    A bare tool name there asks the user to approve ``acp_write_text_file`` without
    telling them which file. `rawInput` carries the same detail structurally, but
    the spec says nothing about clients rendering it.
    """
    client = _FakeClient(permission="reject_once")

    @Skill.define
    def use_tool() -> str:
        """Call `add_numbers`."""

    with _session(client) as session:
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
                        make_text_response("declined"),
                    ]
                ),
                library.ACPPermissionGate(session),
            )
        ):
            use_tool()

    assert client.asked == ["add_numbers(a=1, b=2)"]


def test_a_dismissed_permission_prompt_cancels_the_turn():
    """A `cancelled` outcome is the user dismissing the prompt, not rejecting."""
    tool_call = DecodedToolCall(
        tool=add_numbers,
        bound_args=add_numbers.__signature__.bind(1, 2),
        id="call_1",
        name="add_numbers",
    )
    with _session(_FakeClient(permission=None)) as session:
        with pytest.raises(library.SessionCancelled):
            with handler(library.ACPPermissionGate(session)):
                call_tool(tool_call)


def test_the_gate_decides_before_the_reporter_announces_the_call():
    """Order within the session's stack: nothing is shown as running unapproved.

    `ACPSession.intp` puts the gate outermost for this reason. Reversed, the editor
    would show a call as `in_progress` while the user was still being asked whether
    it may run at all.
    """
    client = _FakeClient(permission="reject_once")

    @Skill.define
    def use_tool() -> str:
        """Call `add_numbers`."""

    with _session(client) as session:
        with handler(coproduct(_stack(), session.intp)):
            with handler(
                MockCompletionHandler(
                    [
                        make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
                        make_text_response("declined"),
                    ]
                )
            ):
                use_tool()

    assert client.asked, "the user should have been asked"
    assert _statuses(client.updates) == ["in_progress", "failed"], (
        "a declined call is reported as failed, not completed"
    )


# ============================================================================
# The editor's capabilities, as tools the model may call
# ============================================================================


_FS = schema.ClientCapabilities(
    fs=schema.FileSystemCapabilities(read_text_file=True, write_text_file=True)
)


def test_the_assistant_offers_every_tool_it_imports():
    """Importing the tools is the whole of the wiring, and is easy to undo.

    A `Skill` finds its tools by lexical scope, so ``assistant.py``'s imports of
    them -- which no other line in that file mentions, and which carry a ``noqa`` --
    are what put them in front of the model. Delete one as unused and the assistant
    silently loses that hand.

    Read from the imports rather than checked against a list written here, because
    which tools an assistant wants is its own business: an editor with no terminal
    makes `acp_run_terminal_command` worth dropping, and that is a decision this test
    should record rather than veto. What it pins is the wiring -- whatever is
    imported reaches the model, and at least one thing is.
    """
    imported = {
        alias.name
        for node in ast.walk(ast.parse((EXAMPLE_DIR / "assistant.py").read_text()))
        if isinstance(node, ast.ImportFrom) and node.module == "library"
        for alias in node.names
        if alias.name.startswith("acp_")
    }
    names = {
        t.__name__ for t in _tools_in_scope(assistant.Assistant().prompt.__context__)
    }
    assert imported, "the assistant should offer the editor's capabilities"
    assert imported <= names


def test_an_editor_tool_is_named_by_a_bare_call():
    """The recorded path is what a code-writing model must type to call the tool.

    Module-level, so a bare name -- not reached through the agent, which is the
    difference from a tool defined as one of its methods.
    """
    paths = {
        t.__name__: p
        for t, p in _tool_paths(assistant.Assistant().prompt.__context__).items()
    }
    assert paths["acp_read_text_file"] == "acp_read_text_file"


def test_reading_and_writing_go_through_the_editor(tmp_path):
    """The point of these tools: the edit lands in the editor, not on the disk.

    The fake client keeps files in a dict, so a tool that quietly used `open()`
    instead would fail here.
    """
    target = str(tmp_path / "x.py")
    client = _FakeClient(files={target: "print(1)\n"})
    with _session(client, _FS) as session:
        with handler(library.ACPToolRuntime(session)):
            assert acp_read_text_file(target) == "print(1)\n"
            acp_write_text_file(target, "print(2)\n")

    assert client.files[target] == "print(2)\n"
    assert not pathlib.Path(target).exists(), "the tool must not touch the disk"


def test_writing_answers_with_something_the_model_can_read():
    """ACP's own write response is empty, but the tool is declared to return `str`.

    Whatever it returns is what the model is shown as that call's result, and `null`
    reads as a call that did nothing.
    """
    client = _FakeClient(files={})
    with _session(client, _FS) as session:
        with handler(library.ACPToolRuntime(session)):
            answer = acp_write_text_file("/tmp/x.py", "print(2)\n")
    assert isinstance(answer, str) and "/tmp/x.py" in answer


def test_running_a_command_goes_through_the_users_terminal():
    """The command runs in the user's own terminal, and the terminal is released."""
    client = _FakeClient(terminal=("hello\n", 0))
    with _session(client, schema.ClientCapabilities(terminal=True)) as session:
        with handler(library.ACPToolRuntime(session)):
            output = acp_run_terminal_command("echo", ["hello"])

    assert client.ran == [("echo", ["hello"], _CWD)]
    assert "hello" in output and "exit code 0" in output
    assert client.released == ["t1"], "a terminal left open is a leak in the editor"


def test_a_terminal_command_is_rendered_as_a_live_terminal():
    """ACP's one construct that renders *while* it happens, rather than after.

    The editor "displays live output as it's generated and continues to display it
    even after the terminal is released" -- so the terminal id is handed over as soon
    as the editor returns it, not once the command exits. Waiting for the exit before
    saying anything is a spinning row for however long the command takes, and then the
    whole output at once.
    """
    client = _FakeClient(terminal=("hello\n", 0), permission="allow_once")

    @Skill.define
    def use_tool() -> str:
        """Run `echo hello` with `acp_run_terminal_command`."""

    mock = MockCompletionHandler(
        [
            make_tool_call_response(
                "acp_run_terminal_command",
                '{"command": "echo", "args": ["hello"]}',
            ),
            make_text_response("it printed hello"),
        ]
    )
    with _session(client, schema.ClientCapabilities(terminal=True)) as session:
        with handler(coproduct(_stack(mock), session.intp)):
            assert use_tool() == "it printed hello"

    updates = [u for u in client.updates if u.session_update == "tool_call_update"]
    # in_progress, then the terminal, then the terminal again as the call completes.
    assert [u.status for u in updates] == ["in_progress", None, "completed"]
    embedded = [u.content[0] for u in updates if u.content]
    assert [c.type for c in embedded] == ["terminal", "terminal"]
    assert {c.terminal_id for c in embedded} == {"t1"}


def test_a_terminal_backed_call_does_not_repeat_its_output_as_text():
    """`ToolCallUpdate.content` replaces the collection, so the final update decides.

    Sending the captured text there would both drop the terminal the user is watching
    and show them the same output a second time underneath.
    """
    client = _FakeClient(terminal=("hello\n", 0), permission="allow_once")

    @Skill.define
    def use_tool() -> str:
        """Run `echo hello` with `acp_run_terminal_command`."""

    mock = MockCompletionHandler(
        [
            make_tool_call_response(
                "acp_run_terminal_command",
                '{"command": "echo", "args": ["hello"]}',
            ),
            make_text_response("done"),
        ]
    )
    with _session(client, schema.ClientCapabilities(terminal=True)) as session:
        with handler(coproduct(_stack(mock), session.intp)):
            use_tool()

    completed = [
        u
        for u in client.updates
        if u.session_update == "tool_call_update" and u.status == "completed"
    ]
    assert [c.type for c in completed[0].content] == ["terminal"]
    # The model still gets the text: the terminal is what the *user* reads.
    replied = [m for m in mock.received_messages[-1] if m.get("role") == "tool"]
    assert "hello" in library._as_text(replied[0])


def test_a_non_text_tool_result_is_described_rather_than_dropped():
    """A tool may return something that is not text, and the editor must show *that*.

    Reading only the ``text`` key across a result's content blocks turned an image
    into the empty string, so the editor rendered the call as having produced nothing
    -- indistinguishable from a tool that printed nothing.
    """
    from PIL import Image

    encoded = pydantic.TypeAdapter(Encodable[Image.Image]).dump_python(
        Image.new("RGB", (4, 4), "red"), mode="json", context={}
    )
    blocks = list(to_content_blocks(encoded))
    assert library._as_text({"content": blocks}) == "[image/png]"
    # Text around it is kept, so a result that is partly prose still reads.
    assert library._as_text(
        {"content": [{"type": "text", "text": "here: "}, *blocks]}
    ) == ("here: [image/png]")
    # An unknown block kind is named rather than silently skipped.
    assert library._as_text({"content": [{"type": "audio"}]}) == "[audio]"


def test_an_image_returning_tool_shows_something_in_the_editor():
    """The end-to-end symptom: an empty tool-call row for a call that did work."""
    from PIL import Image

    client = _FakeClient()

    @Tool.define
    def render_chart() -> Image.Image:
        """Draw a chart."""
        return Image.new("RGB", (4, 4), "blue")

    @Skill.define
    def use_tool() -> str:
        """Call `render_chart`."""

    with _session(client) as session:
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response("render_chart", "{}"),
                        make_text_response("drawn"),
                    ]
                ),
                library.ACPSessionReporter(session),
            )
        ):
            assert use_tool() == "drawn"

    completed = [
        u
        for u in client.updates
        if u.session_update == "tool_call_update" and u.status == "completed"
    ]
    assert completed[0].content[0].content.text == "[image/png]"


def test_a_file_tool_call_says_which_file_it_is_about():
    """`locations` is where an editor reads the file a call touched.

    It will guess from the raw arguments if it must, but only this field can carry a
    line, and only this field is unambiguous.
    """
    client = _FakeClient(files={"/tmp/x.py": "print(1)\n"})

    @Skill.define
    def use_tool() -> str:
        """Read `/tmp/x.py`."""

    with _session(client, _FS) as session:
        with handler(
            coproduct(
                _stack(
                    MockCompletionHandler(
                        [
                            make_tool_call_response(
                                "acp_read_text_file",
                                '{"path": "/tmp/x.py", "line": 3}',
                            ),
                            make_text_response("read it"),
                        ]
                    )
                ),
                session.intp,
            )
        ):
            use_tool()

    running = next(
        u
        for u in client.updates
        if u.session_update == "tool_call_update" and u.status == "in_progress"
    )
    assert running.locations is not None
    assert (running.locations[0].path, running.locations[0].line) == ("/tmp/x.py", 3)


def test_a_call_about_no_file_carries_no_locations():
    assert library._locations({"a": 1, "b": 2}) == []
    assert library._locations({"path": ""}) == []
    assert [(x.path, x.line) for x in library._locations({"path": "/x"})] == [
        ("/x", None)
    ]


def test_writing_a_file_is_rendered_as_a_diff():
    """A write reported as a line of prose is a log entry; a diff is the change."""
    client = _FakeClient(files={"/tmp/x.py": "print(1)\n"})

    @Skill.define
    def use_tool() -> str:
        """Rewrite `/tmp/x.py`."""

    with _session(client, _FS) as session:
        with handler(
            coproduct(
                _stack(
                    MockCompletionHandler(
                        [
                            make_tool_call_response(
                                "acp_write_text_file",
                                '{"path": "/tmp/x.py", "content": "print(2)\\n"}',
                            ),
                            make_text_response("done"),
                        ]
                    )
                ),
                session.intp,
            )
        ):
            use_tool()

    completed = next(
        u
        for u in client.updates
        if u.session_update == "tool_call_update" and u.status == "completed"
    )
    (diff,) = completed.content
    assert diff.type == "diff"
    assert (diff.path, diff.old_text, diff.new_text) == (
        "/tmp/x.py",
        "print(1)\n",
        "print(2)\n",
    )


def test_a_new_file_is_a_diff_with_nothing_before_it():
    """Reading fails for a file that does not exist yet; the write must not."""
    client = _FakeClient(files={})
    with _session(client, _FS) as session:
        session.reporter.begin_turn()
        session.reporter.running = "call_1"
        with handler(library.ACPToolRuntime(session)):
            acp_write_text_file("/tmp/new.py", "print(1)\n")

    assert client.files["/tmp/new.py"] == "print(1)\n", "the write still happened"
    (diff,) = [
        u.content[0]
        for u in client.updates
        if u.session_update == "tool_call_update" and u.content
    ][-1:]
    assert (diff.path, diff.old_text, diff.new_text) == (
        "/tmp/new.py",
        None,
        "print(1)\n",
    )


def test_an_editor_that_will_not_read_still_gets_the_write_and_a_diff():
    """No `fs.readTextFile` means no before-text, and no reason to fail the write."""
    client = _FakeClient(files={})
    caps = schema.ClientCapabilities(
        fs=schema.FileSystemCapabilities(read_text_file=False, write_text_file=True)
    )
    with _session(client, caps) as session:
        session.reporter.begin_turn()
        session.reporter.running = "call_1"
        with handler(library.ACPToolRuntime(session)):
            acp_write_text_file("/tmp/y.py", "print(1)\n")

    assert client.files["/tmp/y.py"] == "print(1)\n"
    diff = [u.content[0] for u in client.updates if u.content][-1]
    assert diff.old_text is None


def test_a_tool_that_opens_no_terminal_still_renders_as_text():
    """The terminal is a special case, and must stay one."""
    client = _FakeClient()

    @Skill.define
    def use_tool() -> str:
        """Call `add_numbers`."""

    with _session(client) as session:
        with handler(
            coproduct(
                _stack(
                    MockCompletionHandler(
                        [
                            make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
                            make_text_response("3"),
                        ]
                    )
                ),
                session.intp,
            )
        ):
            use_tool()

    completed = [
        u
        for u in client.updates
        if u.session_update == "tool_call_update" and u.status == "completed"
    ]
    assert [c.type for c in completed[0].content] == ["content"]


def test_a_terminal_opened_outside_a_tool_call_has_no_row_to_attach_to():
    """The tools are reachable directly, and that is not an error."""
    client = _FakeClient(terminal=("", 0))
    with _session(client, schema.ClientCapabilities(terminal=True)) as session:
        assert session.reporter.running is None
        with handler(library.ACPToolRuntime(session)):
            acp_run_terminal_command("echo", ["hi"])

    assert client.updates == []


def test_a_command_runs_in_the_sessions_directory_not_the_processs():
    """ACP's `cwd` is the project the user opened; this process's is meaningless.

    An editor spawns the agent from wherever it happens to be running, so a command
    that inherited the process's directory would run somewhere the user never chose --
    and `git status` would answer about the wrong repository, convincingly.
    """
    client = _FakeClient(terminal=("", 0))
    with _session(
        client, schema.ClientCapabilities(terminal=True), cwd="/work/other-project"
    ) as session:
        with handler(library.ACPToolRuntime(session)):
            acp_run_terminal_command("git", ["status"])

    assert client.ran[0][2] == "/work/other-project"
    assert client.ran[0][2] != os.getcwd()


def test_the_root_set_leads_with_the_working_directory():
    """ACP requires `cwd` to be part of the session's effective root set."""
    with _session(_FakeClient(), cwd="/a") as session:
        session.additional_directories = ("/b", "/c")
        assert session.roots == ("/a", "/b", "/c")


def test_the_model_is_told_which_directories_the_session_is_about():
    """Otherwise "use absolute paths" is advice the model cannot act on.

    Checked through a real skill call, because the delivery is the part that breaks:
    the directories are computed per session, so they cannot ride on a docstring and
    have to be appended to the harness prompt by `call_system`.
    """
    mock = MockCompletionHandler([make_text_response("ok")])
    with _session(_FakeClient(), cwd="/work/thing") as session:
        session.additional_directories = ("/work/vendor",)
        with handler(_stack(mock, library.ACPToolRuntime(session))):
            session.agent.respond("hello")

    system = "".join(
        part.get("text", "")
        for message in mock.received_messages[0]
        if message.get("role") == "system"
        for part in message.get("content", [])
    )
    assert "/work/thing" in system
    assert "/work/vendor" in system


def test_an_editor_that_claims_no_filesystem_at_all_is_read_as_claiming_nothing():
    """`fs` is optional in `ClientCapabilities`, so a client may send it as null.

    Reached straight through, that is an `AttributeError` from inside a tool rather
    than the "this editor cannot" the model is meant to be told.
    """
    with _session(_FakeClient(), schema.ClientCapabilities(fs=None)) as session:
        assert session.fs_capabilities.read_text_file is False
        with handler(library.ACPToolRuntime(session)):
            with pytest.raises(NotImplementedError):
                acp_read_text_file("/tmp/x")


def test_a_missing_capability_is_reported_to_the_model_not_raised_at_the_caller():
    """A tool the editor cannot service fails as a *tool*, which the model can read.

    This is what pays for offering the tools unconditionally instead of choosing them
    per connection: `TenacityRetryer` turns a raising tool into that call's result and
    the loop goes on -- without consuming a retry -- so the model can say what it
    could not do.
    """

    @Skill.define
    def use_tool() -> str:
        """Read `/tmp/x` with `acp_read_text_file`."""

    # Default capabilities: no filesystem at all.
    with _session(_FakeClient()) as session:
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response(
                            "acp_read_text_file", '{"path": "/tmp/x"}'
                        ),
                        make_text_response("this editor cannot read files"),
                    ]
                ),
                library.ACPToolRuntime(session),
            )
        ):
            assert use_tool() == "this editor cannot read files"


# ============================================================================
# Cancellation
# ============================================================================


def test_a_cancelled_turn_stops_before_the_next_tool_call():
    """There is no checkpoint to reach: the flag is read at each step of the turn.

    `ACPSessionReporter` reads it before a completion and before a tool call, which
    together bracket every step the loop takes.
    """
    ran: list[int] = []

    @Tool.define
    def record(x: int) -> str:
        """Record a number."""
        ran.append(x)
        return "recorded"

    tool_call = DecodedToolCall(
        tool=record,
        bound_args=record.__signature__.bind(1),
        id="call_1",
        name="record",
    )
    with _session(_FakeClient()) as session:
        session.cancel.set()
        with pytest.raises(library.SessionCancelled):
            with handler(library.ACPSessionReporter(session)):
                call_tool(tool_call)

    assert ran == [], "a cancelled turn must not start another tool"


def test_cancellation_is_not_swallowed_by_the_retryer():
    """`SessionCancelled` derives from `BaseException` for exactly this reason.

    `TenacityRetryer.call_tool` catches `Exception`-derived tool failures and reports
    them to the model; a cancellation caught that way would be described to the model
    as a broken tool instead of ending the turn.
    """

    @Tool.define
    def cancel_now() -> str:
        """Cancels the turn."""
        raise library.SessionCancelled

    tool_call = DecodedToolCall(
        tool=cancel_now,
        bound_args=cancel_now.__signature__.bind(),
        id="call_1",
        name="cancel_now",
    )
    with pytest.raises(library.SessionCancelled):
        with handler(_stack()):
            call_tool(tool_call)


def test_an_ordinary_tool_failure_is_still_caught_by_the_retryer():
    """The counterpart: a normal tool failure must keep reaching the model.

    Driven through a real skill call rather than by invoking `call_tool` directly,
    because on the real stack the tool result has to answer an advertised call --
    `HistoryBuilder` rejects one that answers nothing.
    """
    client = _FakeClient()

    @Skill.define
    def use_tool() -> str:
        """Call `explode`."""

    with _session(client) as session:
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response("explode", "{}"),
                        make_text_response("the tool failed"),
                    ]
                ),
                library.ACPSessionReporter(session),
            )
        ):
            assert use_tool() == "the tool failed"

    assert "failed" in _statuses(client.updates)


def test_a_cancelled_prompt_answers_cancelled_rather_than_failing():
    """`session/cancel` arrives on the loop while the turn runs on a worker thread.

    The whole path, end to end: the notification sets a flag, the worker reads it at
    its next step, and the prompt it interrupted is the thing that reports it -- as a
    `cancelled` stop reason, which is an answer, not an error.

    The mock answers with a tool call and then with text, so a cancellation that was
    missed ends the turn normally and fails the assertion rather than hanging.
    """

    class _Slow(ObjectInterpretation):
        """A model slow enough that a cancellation can land while it is thinking."""

        def __init__(self, responses):
            self.responses = list(responses)

        @implements(completion)
        def _completion(self, messages=None, **kwargs):
            time.sleep(0.3)
            return (
                self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
            )

    client = _FakeClient(permission="allow_once")

    async def drive():
        server = await _serve(client)
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        turn = asyncio.ensure_future(
            server.prompt(session_id=session_id, prompt=[acp.text_block("go")])
        )
        await asyncio.sleep(0.05)  # let the worker get as far as the model
        await server.cancel(session_id)
        response = await turn
        await server.close_session(session_id)
        return response

    with handler(
        _stack(
            _Slow(
                [
                    make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
                    make_text_response("not cancelled after all"),
                ]
            )
        )
    ):
        response = asyncio.run(drive())

    assert response.stop_reason == "cancelled"
    # Everything the turn produced reached the editor before the turn answered, and
    # the call the model got as far as announcing is not left spinning.
    assert _statuses(client.updates)[-1] == "failed"


def test_a_cancelled_turn_leaves_no_tool_call_spinning():
    """An announced call the turn never finished must be given a terminal status.

    ACP's `ToolCallStatus` has no `cancelled`, so `failed` is the only way to say it.
    Left as `in_progress`, the editor draws a row that spins for the rest of the
    session -- and there is no later update that would ever clear it.
    """
    client = _FakeClient()
    with _session(client) as session:
        reporter = session.reporter
        reporter.begin_turn()
        reporter._start("call_1", "add_numbers", status="pending")
        assert _statuses(client.updates) == []
        reporter.abandon()

    assert _statuses(client.updates) == ["failed"]
    assert reporter._open == {}, "an abandoned call must not be abandoned twice"


def test_updates_reach_the_editor_before_the_response_on_every_path():
    """ "MAY send update notifications before responding, but MUST do so before the
    final response" -- which the success path got right and the others did not.

    A slow editor is the case that separates the two: the writer task is a task like
    any other, so a queued notification only beats the response if the turn waits.
    """
    order: list[str] = []
    running: list[typing.Any] = []

    class _SlowEditor(_FakeClient):
        async def session_update(self, session_id, update, **kw):
            await asyncio.sleep(0.02)
            order.append("update")
            await super().session_update(session_id, update, **kw)

    class _CancelledMidTurn(ObjectInterpretation):
        """A turn that says three things and is then cancelled, from the worker."""

        @implements(completion)
        def _completion(self, messages=None, **kw):
            for text in "abc":
                running[0].notify(acp.update_agent_message_text(text))
            raise library.SessionCancelled

    client = _SlowEditor()

    async def drive():
        server = await _serve(client)
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        running.append(server.sessions[session_id])
        # Opening the session announced its commands; this test is about the ordering
        # of the *prompt's* updates, so let that one land and forget it.
        await server.sessions[session_id].flush()
        order.clear()
        response = await server.prompt(
            session_id=session_id, prompt=[acp.text_block("go")]
        )
        order.append("response")
        await server.close_session(session_id)
        return response

    with handler(_stack(_CancelledMidTurn())):
        response = asyncio.run(drive())

    assert response.stop_reason == "cancelled"
    # The requirement is an ordering, not a count: everything the turn produced
    # reached the editor, and all of it before the answer.
    assert order[-1] == "response"
    assert set(order[:-1]) == {"update"}
    assert order.count("update") >= 3


# ============================================================================
# Why a turn ended
#
# ACP answers every prompt with one of five stop reasons, and two of the ones
# this agent can reach describe an ending an editor cannot infer for itself: a
# reply cut off at the token limit, and one the provider refused. Reporting
# either as `end_turn` tells the user they were answered when they were not.
#
# `max_turn_requests` is the fifth, and this agent never says it: nothing here
# caps how many times a turn may go back to the model.
# ============================================================================


def _one_turn(client: _FakeClient, responses):
    """Drive one prompt through the real server and return its response."""

    async def drive():
        server = await _serve(client)
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        response = await server.prompt(
            session_id=session_id, prompt=[acp.text_block("go")]
        )
        await server.close_session(session_id)
        return response

    with handler(_stack(MockCompletionHandler(responses))):
        return asyncio.run(drive())


def test_an_ordinary_answer_is_end_turn():
    assert _one_turn(_FakeClient(), [make_text_response("done")]).stop_reason == (
        "end_turn"
    )


def test_a_reply_cut_off_at_the_token_limit_is_max_tokens():
    """`finish_reason="length"` means the answer is a fragment, not an answer."""
    response = _one_turn(_FakeClient(), [_stopped("half a sen", "length")])
    assert response.stop_reason == "max_tokens"


def test_a_reply_the_provider_refused_is_refusal():
    response = _one_turn(
        _FakeClient(), [_stopped("I can't help with that.", "content_filter")]
    )
    assert response.stop_reason == "refusal"


def test_the_context_gauge_is_pushed_as_the_context_fills():
    """`usage_update` is how full the window is *now*, not what the turn cost."""
    client = _FakeClient()
    response = make_text_response(
        "ok",
        usage={"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010},
    )
    response.model = "gpt-4o-mini"  # one litellm knows the context size of

    with _session(client) as session:
        session.reporter.begin_turn()
        session.reporter._account(response)

    (update,) = [u for u in client.updates if u.session_update == "usage_update"]
    assert update.used == 1000
    assert update.size > 1000, "the denominator is the model's real context window"


def test_a_model_litellm_does_not_know_still_gets_a_gauge():
    """litellm's table does not cover every model, and a gauge is not worth losing.

    This is the case that made the feature invisible in practice: a real model behind
    a gateway litellm has not catalogued, silently skipped. A roughly-right gauge
    beats an empty space, so an assumed size stands in.
    """
    client = _FakeClient()
    response = make_text_response(
        "ok", usage={"prompt_tokens": 900, "completion_tokens": 5, "total_tokens": 905}
    )
    response.model = "openrouter/z-ai/glm-5.3-flash"  # litellm: "isn't mapped yet"

    with _session(client) as session:
        session.reporter.begin_turn()
        session.reporter._account(response)

    (update,) = [u for u in client.updates if u.session_update == "usage_update"]
    assert (update.used, update.size) == (900, library.ASSUMED_CONTEXT_SIZE)


def test_the_gauge_survives_the_streaming_path_every_real_turn_takes():
    """The tests above call `_account` directly, which skips how it is ever reached.

    `ACPSessionReporter.completion` adds ``stream=True`` to every request, so what
    `_report_context` reads is not a provider's response but whatever
    `litellm.stream_chunk_builder` reassembles from the chunks. Both the fields it
    needs come from there -- including a `usage` the chunks do not carry, which the
    builder counts locally off `messages` -- and a gauge that worked on a settled
    response but not on that one would be a gauge nobody ever sees.
    """
    client = _FakeClient()
    with _session(client) as session:
        session.reporter.begin_turn()
        with handler(coproduct(_StreamingModel("Hello", " world"), session.reporter)):
            completion(messages=[{"role": "user", "content": "how far along am i"}])

    assert [
        u.content.text
        for u in client.updates
        if u.session_update.endswith("message_chunk")
    ] == ["Hello", " world"], "the deltas should have been reported as they arrived"
    (gauge,) = [u for u in client.updates if u.session_update == "usage_update"]
    assert gauge.used > 0, "counted off the request, since no chunk carried a usage"
    assert gauge.size == library.ASSUMED_CONTEXT_SIZE


def test_the_stream_asks_the_provider_to_report_its_own_usage():
    """Streams carry no usage block unless asked, and the gauge is made of one.

    Unasked, `stream_chunk_builder` falls back to tokenizing the request locally --
    close enough to draw, but blind to cache reads and to the provider's own
    accounting. This is the request that turns the estimate into a measurement.
    """
    provider = _StreamingModel("hi")
    with _session(_FakeClient()) as session:
        session.reporter.begin_turn()
        with handler(coproduct(provider, session.reporter)):
            completion(messages=[{"role": "user", "content": "hello"}])

    assert provider.requests[-1]["stream_options"] == {"include_usage": True}


def test_a_provider_that_reports_its_usage_is_believed_over_the_estimate():
    """The counted-locally number is a fallback, and one the user should not be shown
    when the provider has said what it actually charged for.
    """
    client = _FakeClient()
    reported = litellm.types.utils.Usage(
        prompt_tokens=4321, completion_tokens=7, total_tokens=4328
    )
    with _session(client) as session:
        session.reporter.begin_turn()
        model = _StreamingModel("hi", reports=reported)
        with handler(coproduct(model, session.reporter)):
            completion(messages=[{"role": "user", "content": "hello"}])

    (gauge,) = [u for u in client.updates if u.session_update == "usage_update"]
    assert gauge.used == 4321, "a locally counted estimate would be far smaller"


def test_a_known_model_is_measured_rather_than_assumed():
    """The assumption is a fallback, not the answer: a catalogued model uses its own.

    Checked with a model whose window differs from the assumed one, since the two
    coinciding -- `gpt-4o-mini`'s is exactly 128k -- proves nothing either way.
    """
    measured = library._context_size("openrouter/anthropic/claude-sonnet-5")
    assert measured == 200_000 != library.ASSUMED_CONTEXT_SIZE


def test_a_turns_token_usage_is_reported():
    """The counts are already in the responses; ACP has a field for them."""
    response = _one_turn(
        _FakeClient(),
        [
            make_text_response(
                "done",
                usage={
                    "prompt_tokens": 11,
                    "completion_tokens": 5,
                    "total_tokens": 16,
                },
            )
        ],
    )
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (11, 5)
    assert response.usage.total_tokens == 16


def test_usage_counts_the_turn_and_not_the_conversation():
    """The reporter outlives the turn, so what it counts has to be reset per turn."""
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}

    async def drive():
        server = await _serve(_FakeClient())
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        seen = [
            (
                await server.prompt(
                    session_id=session_id, prompt=[acp.text_block(f"q{i}")]
                )
            ).usage
            for i in range(3)
        ]
        await server.close_session(session_id)
        return seen

    with handler(
        _stack(MockCompletionHandler([make_text_response("ok", usage=usage)]))
    ):
        seen = asyncio.run(drive())

    assert [u.total_tokens for u in seen] == [12, 12, 12]


# ============================================================================
# The session's channel to the editor
# ============================================================================


def test_a_dropped_notification_does_not_deadlock_the_end_of_the_turn():
    """A turn ends by awaiting `updates.join()`, so the writer must mark every item.

    Without that, a single notification the editor refused would hang the end of this
    turn and every turn after it -- which is exactly the bug this test was written
    for, after a live run sat for two minutes with the answer already delivered.
    """
    client = _FakeClient()
    sent: list[str] = []

    async def flaky(session_id, update, **kw):
        if update.content.text == "b":
            raise RuntimeError("the editor refused this one")
        sent.append(update.content.text)

    client.session_update = flaky

    with _session(client) as session:
        for text in "abc":
            session.notify(acp.update_agent_message_text(text))

    assert sent == ["a", "c"]


def test_notifications_keep_the_order_they_were_produced_in():
    client = _FakeClient()
    with _session(client) as session:
        for i in range(50):
            session.notify(acp.update_agent_message_text(str(i)))
    assert [u.content.text for u in client.updates] == [str(i) for i in range(50)]


def test_the_cancel_flag_is_thread_safe():
    """It is set on the loop thread and read on the worker thread."""
    with _session(_FakeClient()) as session:
        assert isinstance(session.cancel, threading.Event)


def test_notifying_from_the_loop_thread_is_delivered_not_lost():
    """`load_session` notifies and then waits, from the loop's own thread.

    Deferring every enqueue through `call_soon_threadsafe` made that sequence wait on
    an empty queue and return before anything had been put on it, so a reloaded
    session arrived with no conversation at all.
    """
    client = _FakeClient()

    async def notify_then_join():
        session = library.ACPSession(
            agent=_Bot("s1"),
            client=typing.cast(typing.Any, client),
            client_capabilities=schema.ClientCapabilities(),
        )
        for text in ("asked", "answered"):
            session.notify(acp.update_agent_message_text(text))
        await asyncio.wait_for(session.updates.join(), timeout=5)
        typing.cast(asyncio.Task, session.writer).cancel()

    asyncio.run(notify_then_join())
    assert [u.content.text for u in client.updates] == ["asked", "answered"]


# ============================================================================
# An editor that does not answer
#
# Nothing in ACP obliges a client to answer a request, and the commonest one --
# a permission prompt -- is answered by a person, on no schedule. So the agent
# waits indefinitely, and these pin the thing that makes waiting safe: the user
# can always end it. Found by driving the real server with a script that read
# stdout but never replied, which wedged the turn permanently.
# ============================================================================


class _SilentClient(_FakeClient):
    """An editor that accepts requests and never answers them."""

    async def request_permission(self, session_id, tool_call, options, **kw):
        self.asked.append(tool_call.title)
        await asyncio.Event().wait()  # never returns

    async def read_text_file(self, session_id, path, line=None, limit=None, **kw):
        await asyncio.Event().wait()


def test_an_unanswered_request_waits_rather_than_giving_up():
    """No clock on the editor: a permission prompt is a dialog in front of a person.

    A deadline would eventually tell the model the editor never answered while the
    dialog was still on screen, and refuse the call out from under the user reaching
    for Allow. So the wait is unbounded and `session/cancel` is what ends it -- which
    is also what every other ACP agent does.

    A second is many `poll_interval` slices, so a reinstated bound would have to be
    implausibly long to survive this.
    """
    raised: list[BaseException] = []

    with _session(_SilentClient()) as session:

        def wait_on_the_editor() -> None:
            try:
                session.call(session.client.read_text_file("s1", "/tmp/x"))
            except BaseException as e:  # noqa: BLE001 - recorded, then asserted on
                raised.append(e)

        waiter = threading.Thread(target=wait_on_the_editor, daemon=True)
        waiter.start()
        waiter.join(timeout=1.0)
        assert waiter.is_alive(), "the wait should still be in progress, not given up"

        session.cancel.set()  # the only thing that ends it
        waiter.join(timeout=5)
        assert not waiter.is_alive(), "cancelling should have released the wait"

    assert isinstance(raised[0], library.SessionCancelled)


def test_cancelling_interrupts_a_pending_request():
    """`session/cancel` must work while the turn waits on the editor.

    This is the case where a user most wants to cancel -- a permission prompt they
    cannot or will not answer -- so a wait that ignored the flag until the next step
    of the turn would ignore them exactly when it mattered. With nothing bounding that
    wait, it is the only way out.
    """
    with _session(_SilentClient()) as session:
        threading.Timer(0.2, session.cancel.set).start()
        started = time.monotonic()
        with pytest.raises(library.SessionCancelled):
            session.call(session.client.read_text_file("s1", "/tmp/x"))
        assert time.monotonic() - started < 5


def test_a_cancelled_permission_prompt_does_not_run_the_tool():
    """Silence is not consent: a tool whose prompt was never approved must not run.

    `SessionCancelled` derives from `BaseException`, so it stops the turn here rather
    than being handed to the model as a tool that happened to fail.
    """
    ran: list[int] = []

    @Tool.define
    def record(x: int) -> str:
        """Record a number."""
        ran.append(x)
        return "recorded"

    @Skill.define
    def use_tool() -> str:
        """Call `record`."""

    with _session(_SilentClient()) as session:
        threading.Timer(0.2, session.cancel.set).start()
        with handler(
            _stack(
                MockCompletionHandler([make_tool_call_response("record", '{"x": 1}')]),
                library.ACPPermissionGate(session),
            )
        ):
            with pytest.raises(library.SessionCancelled):
                use_tool()

    assert ran == [], "a tool must not run without an answer to its prompt"


# ============================================================================
# What a prompt is made of
#
# A prompt is a list of content blocks, not a string: the user's words plus
# whatever their editor attached. What an agent may be sent is governed by the
# `PromptCapabilities` it advertised, so these two things have to agree.
# ============================================================================


def test_text_and_resource_links_are_the_baseline_and_are_accepted():
    """Both are baseline: an agent must handle them with nothing negotiated.

    A resource link is a *reference*, so it becomes the path rather than the file's
    contents -- the model can then read it through the editor and see unsaved
    changes, where inlining here would freeze a stale copy.
    """
    text = library._prompt_text(
        [
            acp.text_block("look at "),
            acp.resource_link_block(name="x.py", uri="file:///tmp/x.py"),
        ]
    )
    assert "look at" in text
    assert "file:///tmp/x.py" in text


def test_a_block_the_agent_cannot_read_is_refused_rather_than_dropped():
    """Silently discarding an attachment is the failure that looks like success.

    The user attaches a screenshot, the agent answers confidently about a picture
    nobody read, and nothing anywhere reports a problem.
    """
    with pytest.raises(acp.RequestError):
        library._prompt_text(
            [acp.text_block("what is this?"), acp.image_block("iVBOR", "image/png")]
        )


def test_an_empty_prompt_is_refused():
    with pytest.raises(acp.RequestError):
        library._prompt_text([acp.text_block("   ")])


def test_advertised_prompt_capabilities_match_what_the_agent_can_actually_read():
    """The agent must not claim a capability whose blocks `_prompt_text` refuses.

    A client "MUST adapt its interface according to PromptCapabilities", so claiming
    `image` is a promise that an attached screenshot will be looked at. This pins the
    two together: adding a claim here without teaching `_prompt_text` to read it
    fails, and so does the reverse.
    """

    caps = _advertised().prompt_capabilities
    claimed = {
        "image": caps.image,
        "audio": caps.audio,
        "embedded_context": caps.embedded_context,
    }
    samples = {
        "image": acp.image_block("iVBOR", "image/png"),
        "audio": acp.audio_block("AAAA", "audio/wav"),
        "embedded_context": acp.resource_block(
            acp.embedded_text_resource("file:///tmp/x.py", "print(1)")
        ),
    }
    for name, is_claimed in claimed.items():
        try:
            library._prompt_text([samples[name]])
        except acp.RequestError:
            readable = False
        else:
            readable = True
        assert bool(is_claimed) == readable, (
            f"prompt capability {name!r} is advertised as {is_claimed!r} but "
            f"_prompt_text {'reads' if readable else 'refuses'} that block"
        )


# ============================================================================
# What the agent claims it can do
# ============================================================================


def _advertised() -> schema.AgentCapabilities:
    async def initialize():
        server = library.EffectfulACPAgent(_Bot, skill_name="respond")
        response = await server.initialize(protocol_version=acp.PROTOCOL_VERSION)
        return response.agent_capabilities

    caps = asyncio.run(initialize())
    assert caps is not None
    return caps


def test_every_capability_gated_method_it_implements_is_advertised():
    """Implementing one without claiming it is as broken as the reverse.

    A client "MUST verify that the Agent supports this capability" before using it,
    so an unadvertised method is one no conforming client will ever call.
    `close_session` was exactly that: dead code, and every session's writer task left
    running for the life of the process because nothing ever closed one.
    """
    caps = _advertised()
    implemented = {
        "load_session": caps.load_session,
        "close_session": (
            caps.session_capabilities or schema.SessionCapabilities()
        ).close,
    }
    for method, claimed in implemented.items():
        assert getattr(library.EffectfulACPAgent, method, None) is not None
        assert claimed is not None and claimed is not False, (
            f"{method} is implemented but not advertised, so it is unreachable"
        )


def test_it_claims_no_mcp_transport_since_it_connects_to_none():
    caps = _advertised().mcp_capabilities or schema.McpCapabilities()
    assert not (caps.http or caps.sse or caps.acp)


# ============================================================================
# The knobs a session offers the user
#
# Modes, config options and slash commands: three features that cost an agent
# almost nothing, that an editor renders for free, and that stay invisible
# until the agent describes itself.
# ============================================================================


_MODELS = ("openrouter/anthropic/claude-opus-5", "openrouter/anthropic/claude-fable-5")


def _in_session(
    body,
    client: _FakeClient | None = None,
    caps: schema.ClientCapabilities | None = None,
    **kwargs,
):
    """Open a session, run ``await body(server, session_id, open_response)``, close it.

    Everything happens on one loop because a session captures the running one when it
    is built -- a helper that opened a session under its own `asyncio.run` and handed
    it back would hand back a session whose loop is already closed, and whose writer
    task died with it.

    `caps` are the editor's, negotiated at `initialize` and read by the tools; they
    have to be given here rather than set afterwards, since a session copies them when
    it is built.
    """

    async def drive():
        server = library.EffectfulACPAgent(_Bot, skill_name="respond", **kwargs)
        server.on_connect(typing.cast(typing.Any, client or _FakeClient()))
        await server.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            **({"client_capabilities": caps} if caps else {}),
        )
        opened = await server.new_session(cwd=_CWD, mcp_servers=[])
        try:
            return await body(server, opened.session_id, opened)
        finally:
            await server.close_session(opened.session_id)

    return asyncio.run(drive())


# ============================================================================
# Cancelling a turn while a tool is actually running
#
# The tests further up drive the pieces: the flag is read before a tool call,
# the retryer does not swallow the exception, `abandon` closes an open row.
# None of them cancels a turn *mid-tool* through `prompt`, which is the moment
# a user actually reaches for the button and the one with the most to go wrong
# -- a worker thread inside a tool, holding the session lock, unwinding through
# three handlers and a thread boundary.
# ============================================================================


class _HangingTerminal(_FakeClient):
    """An editor whose terminal never exits, and which records what was released."""

    def __init__(self):
        super().__init__()
        self.created: list[str] = []
        self.released: list[str] = []

    async def create_terminal(self, session_id, command, args=None, **kw):
        self.created.append(command)
        return schema.CreateTerminalResponse(terminal_id="term-1")

    async def wait_for_terminal_exit(self, session_id, terminal_id, **kw):
        await asyncio.Event().wait()  # the command never finishes

    async def terminal_output(self, session_id, terminal_id, **kw):
        return schema.TerminalOutputResponse(output="", truncated=False)

    async def release_terminal(self, session_id, terminal_id, **kw):
        self.released.append(terminal_id)
        return schema.ReleaseTerminalResponse()


async def _until(predicate, timeout: float = 5.0) -> bool:
    """Wait for `predicate`, rather than sleeping and hoping. True if it came true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


def _cancelled_mid_tool(client, name: str, arguments: str, caps, after=None):
    """Start a turn, wait until a tool is genuinely running, then cancel it.

    Waiting on `ACPSessionReporter.running` rather than on the clock: it holds the id
    of the call in progress, so this cancels during tool execution by construction
    instead of by a sleep that might land anywhere.
    """
    mock = MockCompletionHandler(
        [
            make_tool_call_response(name, arguments),
            make_text_response("the cancelled turn should never reach this"),
            make_text_response("a later turn"),
        ]
    )

    async def body(server, session_id, opened):
        session = server.sessions[session_id]
        session.mode_id = library.AUTO  # no permission prompt in the way
        turn = asyncio.create_task(
            server.prompt(session_id=session_id, prompt=[acp.text_block("go")])
        )
        assert await _until(lambda: session.reporter.running is not None), (
            "the tool never started, so this would not be cancelling during one"
        )
        await server.cancel(session_id=session_id)
        response = await asyncio.wait_for(turn, timeout=10)
        return response, (await after(server, session_id) if after else None)

    with handler(_stack(mock)):
        return _in_session(body, client, caps=caps)


_READS = schema.ClientCapabilities(
    fs=schema.FileSystemCapabilities(read_text_file=True)
)


def test_cancelling_during_a_tool_call_ends_the_turn_rather_than_the_process():
    """The exception is raised in a worker thread, inside a tool, under three handlers.

    It has to survive all of that: `TenacityRetryer` must not catch it and report a
    broken tool, `asyncio.to_thread` must carry a `BaseException` back across the
    thread boundary, and `prompt` must turn it into a stop reason rather than let it
    out to the connection as a failed request.
    """
    client = _SilentClient()  # accepts fs/read_text_file and never answers
    (response, _) = _cancelled_mid_tool(
        client, acp_read_text_file.__name__, '{"path": "/tmp/x.txt"}', _READS
    )
    assert response.stop_reason == "cancelled"


def test_a_tool_cancelled_midway_is_not_left_spinning_in_the_editor():
    """`abandon` runs in `prompt`'s `finally`, which is what makes this true however
    the turn ended. A row left `in_progress` never gets another update.
    """
    client = _SilentClient()
    _cancelled_mid_tool(
        client, acp_read_text_file.__name__, '{"path": "/tmp/x.txt"}', _READS
    )
    assert _statuses(client.updates)[-1] == "failed"


def test_a_session_cancelled_mid_tool_still_answers_the_next_prompt():
    """The worst outcome is not a crash but a wedge: the turn holds `session.lock`, so
    an unwind that lost it would leave the session silently dead to every later prompt.
    """

    async def after(server, session_id):
        return await asyncio.wait_for(
            server.prompt(
                session_id=session_id, prompt=[acp.text_block("still there?")]
            ),
            timeout=10,
        )

    (_, second) = _cancelled_mid_tool(
        _SilentClient(),
        acp_read_text_file.__name__,
        '{"path": "/tmp/x.txt"}',
        _READS,
        after=after,
    )
    assert second.stop_reason == "end_turn"


def test_cancelling_a_running_command_still_releases_its_terminal():
    """ACP has the client hold a terminal until the agent releases it, so an unwind
    that skipped the release would leak one per cancelled command.

    It survives because the release goes out through `ACPSession.detach`: sent
    regardless of how the turn ended, and never itself cancelled.
    """
    client = _HangingTerminal()
    (response, _) = _cancelled_mid_tool(
        client,
        acp_run_terminal_command.__name__,
        '{"command": "sleep", "args": ["600"]}',
        schema.ClientCapabilities(terminal=True),
    )
    assert response.stop_reason == "cancelled"
    assert client.created == ["sleep"]
    assert client.released == ["term-1"], "a cancelled command must not leak a terminal"


def test_cancelling_during_terminal_creation_still_releases_it(monkeypatch):
    """The narrower window: the cancel is already set when the worker reaches the
    `terminal/create` call itself.

    `ACPSession.call` schedules the request before it reads the flag, so the create
    is *sent* -- the editor allocates a terminal and answers -- but the wait raises
    before the agent ever learns the terminal id, and the tool's `finally` cannot
    release an id it never had. The `orphan` callback on the create is what turns
    that answer into a release instead of a leak.

    This interleaving is forced here rather than raced for, because racing is
    exactly how it hid: on an idle machine the worker reaches the (protected) exit
    wait before a 10ms poll can cancel, and only a loaded CI runner ever lost --
    deterministically enough to fail every Linux build while every laptop passed.
    """
    orig_call = library.ACPSession.call

    def cancel_lands_first(self, coro, **kwargs):
        # The user's cancel overtook the worker between announcing the tool
        # call and asking for the terminal.
        if "create_terminal" in getattr(coro, "__qualname__", ""):
            self.cancel.set()
        return orig_call(self, coro, **kwargs)

    monkeypatch.setattr(library.ACPSession, "call", cancel_lands_first)

    client = _HangingTerminal()
    mock = MockCompletionHandler(
        [
            make_tool_call_response(
                acp_run_terminal_command.__name__,
                '{"command": "sleep", "args": ["600"]}',
            ),
            make_text_response("the cancelled turn should never reach this"),
        ]
    )

    async def body(server, session_id, opened):
        session = server.sessions[session_id]
        session.mode_id = library.AUTO
        turn = asyncio.create_task(
            server.prompt(session_id=session_id, prompt=[acp.text_block("go")])
        )
        response = await asyncio.wait_for(turn, timeout=10)
        assert await _until(lambda: client.released), (
            "the orphaned create's answer was never turned into a release"
        )
        return response

    with handler(_stack(mock)):
        response = _in_session(
            body, client, caps=schema.ClientCapabilities(terminal=True)
        )
    assert response.stop_reason == "cancelled"
    assert client.created == ["sleep"]
    assert client.released == ["term-1"], "a terminal created under a cancel leaked"


def test_a_new_session_offers_its_modes():
    """Advertised per session, in the response that opens it -- not a capability."""

    async def body(server, session_id, opened):
        return opened.modes

    modes = _in_session(body)
    assert modes is not None
    assert modes.current_mode_id == library.ASK
    assert [m.id for m in modes.available_modes] == [
        library.ASK,
        library.AUTO,
        library.PLAN,
    ]
    assert all(m.name and m.description for m in modes.available_modes)


def test_auto_mode_runs_tools_without_asking():
    """Picking a mode *is* the user answering these prompts in advance."""
    client = _FakeClient()

    @Skill.define
    def use_tool() -> str:
        """Call `add_numbers`."""

    with _session(client) as session:
        session.mode_id = library.AUTO
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response("add_numbers", '{"a": 1, "b": 2}'),
                        make_text_response("3"),
                    ]
                ),
                library.ACPPermissionGate(session),
            )
        ):
            assert use_tool() == "3"

    assert client.asked == [], "Auto mode must not prompt"


def test_plan_mode_refuses_to_change_anything():
    """The refusal reaches the model as that call's result, so it can say why."""
    client = _FakeClient(files={"/tmp/x.py": "print(1)\n"})

    @Skill.define
    def use_tool() -> str:
        """Write to `/tmp/x.py` with `acp_write_text_file`."""

    with _session(client, _FS) as session:
        session.mode_id = library.PLAN
        with handler(
            coproduct(
                _stack(
                    MockCompletionHandler(
                        [
                            make_tool_call_response(
                                "acp_write_text_file",
                                '{"path": "/tmp/x.py", "content": "print(2)\\n"}',
                            ),
                            make_text_response("I would change it, but Plan mode"),
                        ]
                    )
                ),
                session.intp,
            )
        ):
            assert use_tool() == "I would change it, but Plan mode"

    assert client.files["/tmp/x.py"] == "print(1)\n", "Plan mode must not write"
    assert client.asked == [], "a mode is a decision already taken; do not re-ask"


def test_plan_mode_still_allows_reading():
    """A mode that blocked everything would just be a broken agent."""
    client = _FakeClient(files={"/tmp/x.py": "print(1)\n"})
    with _session(client, _FS) as session:
        session.mode_id = library.PLAN
        with handler(coproduct(_stack(), session.intp)):
            assert acp_read_text_file("/tmp/x.py") == "print(1)\n"
    assert client.asked == []


def test_an_unknown_mode_is_refused():
    async def body(server, session_id, opened):
        with pytest.raises(acp.RequestError):
            await server.set_session_mode(session_id=session_id, mode_id="wishful")
        await server.set_session_mode(session_id=session_id, mode_id=library.PLAN)
        return server.sessions[session_id].mode_id

    assert _in_session(body) == library.PLAN


def _option(options, config_id):
    return next(o for o in options if o.id == config_id)


def test_the_mode_is_offered_as_a_config_option_as_well_as_in_modes():
    """A client that reads config options ignores `modes` and hides its mode picker.

    "Clients that support configOptions MUST use them exclusively and ignore the
    legacy modes field" -- so offering only a model would take the mode picker away
    from exactly the clients that render pickers best.
    """

    async def body(server, session_id, opened):
        return opened.modes, opened.config_options

    modes, options = _in_session(body)
    assert modes is not None, "still sent, for clients that do not read the list"
    mode = _option(options, "mode")
    assert mode.category == "mode"
    assert mode.current_value == library.ASK
    assert [o.value for o in mode.options] == [library.ASK, library.AUTO, library.PLAN]


def test_no_model_picker_when_there_is_nothing_to_pick():
    """A select listing one choice is a control that does nothing."""

    async def body(server, session_id, opened):
        return opened.config_options

    assert [o.id for o in _in_session(body)] == ["mode"]


def test_the_model_picker_leads_with_the_configured_default():
    async def body(server, session_id, opened):
        return opened.config_options

    options = _in_session(body, models=_MODELS)
    assert [o.id for o in options] == ["mode", "model"]
    option = _option(options, "model")
    assert (option.id, option.type, option.category) == ("model", "select", "model")
    assert option.current_value == library.INHERIT_MODEL
    assert [o.value for o in option.options] == [library.INHERIT_MODEL, *_MODELS]


def test_choosing_a_model_answers_with_every_option_again():
    """A client redraws its controls from the response, so it is the whole list."""

    async def body(server, session_id, opened):
        response = await server.set_config_option(
            config_id="model", session_id=session_id, value=_MODELS[1]
        )
        return response, server.sessions[session_id].model

    response, chosen = _in_session(body, models=_MODELS)
    assert _option(response.config_options, "model").current_value == _MODELS[1]
    assert chosen == _MODELS[1]


# ============================================================================
# The picked model reaching the request
#
# Storing the choice is half of it; `ACPSessionConfig` is the other half, and
# these say why it is four lines of its own rather than a second
# `LiteLLMConfigurer` installed in `ACPSession.intp`.
# ============================================================================


def test_the_chosen_model_names_itself_in_the_request():
    """A picker that does not reach `completion` is a label, not a setting."""
    provider = _RecordingModel()
    with _session(_FakeClient()) as session:
        session.model = _MODELS[0]
        with handler(coproduct(provider, library.ACPSessionConfig(session))):
            completion(messages=[])
    assert provider.requests[-1]["model"] == _MODELS[0]


def test_inheriting_the_configured_model_names_no_model_at_all():
    """`INHERIT_MODEL` must leave the request alone rather than substitute a guess.

    `LiteLLMConfigurer` defaults to ``model="gpt-4o"`` and merges its config *under*
    the request, so one installed here to say nothing would say that instead --
    quietly overruling whatever model the launcher was configured with, in the one
    case the user asked for it to be left alone.
    """
    provider = _RecordingModel()
    with _session(_FakeClient()) as session:
        assert session.model == library.INHERIT_MODEL
        with handler(coproduct(provider, library.ACPSessionConfig(session))):
            completion(messages=[])
    assert "model" not in provider.requests[-1]


def test_switching_models_mid_session_needs_no_new_handler_stack():
    """`ACPSession.intp` is built once and cached; `session.model` changes whenever.

    So the model has to be read per request, from the session, which is what this
    handler does and what a `LiteLLMConfigurer` cannot: that binds its config at
    construction, so the picker would stop working after the stack was first built.
    The same interpretation object serves both requests below, as `intp` would.
    """
    provider = _RecordingModel()
    with _session(_FakeClient()) as session:
        intp = coproduct(provider, library.ACPSessionConfig(session))
        for chosen in _MODELS:
            session.model = chosen
            with handler(intp):
                completion(messages=[])
    assert [request["model"] for request in provider.requests] == list(_MODELS)


def test_the_sessions_model_outranks_the_one_the_launcher_configured():
    """Naming the model above `LiteLLMConfigurer` is what makes it the one sent.

    That merge lets a value already on the request stand, which is the property the
    whole arrangement rests on -- and the reason this handler needs to do nothing but
    set a key.
    """
    provider = _RecordingModel()
    with _session(_FakeClient()) as session:
        session.model = _MODELS[1]
        launcher = coproduct(provider, LiteLLMConfigurer(model="configured/at-launch"))
        with handler(coproduct(launcher, library.ACPSessionConfig(session))):
            completion(messages=[])
    assert provider.requests[-1]["model"] == _MODELS[1]


def test_the_mode_can_be_set_through_the_config_option_too():
    """Both ways in must land in the same place: a client picks one, not both."""

    async def body(server, session_id, opened):
        response = await server.set_config_option(
            config_id="mode", session_id=session_id, value=library.PLAN
        )
        return response, server.sessions[session_id].mode_id

    response, mode_id = _in_session(body)
    assert mode_id == library.PLAN
    assert _option(response.config_options, "mode").current_value == library.PLAN


def test_an_unknown_option_or_value_is_refused_rather_than_ignored():
    """A control that silently does nothing is worse than one that reports it can't."""

    async def body(server, session_id, opened):
        for config_id, value in (("model", "not-on-offer"), ("temperature", "0.7")):
            with pytest.raises(acp.RequestError):
                await server.set_config_option(
                    config_id=config_id, session_id=session_id, value=value
                )
        return server.sessions[session_id].model

    assert _in_session(body, models=_MODELS) == library.INHERIT_MODEL


def test_opening_a_session_announces_its_slash_commands():
    """The editor cannot offer `/clear` until it is told the command exists."""
    client = _FakeClient()

    async def body(server, session_id, opened):
        await server.sessions[session_id].flush()

    _in_session(body, client)
    announced = [
        u for u in client.updates if u.session_update == "available_commands_update"
    ]
    assert [c.name for c in announced[0].available_commands] == [
        "clear",
        "status",
        "mode",
    ]
    assert all(c.description for c in announced[0].available_commands)


def test_a_command_that_takes_an_argument_advertises_a_hint():
    """`AvailableCommandInput` is what the editor shows after the name as you type."""
    client = _FakeClient()

    async def body(server, session_id, opened):
        await server.sessions[session_id].flush()

    _in_session(body, client)
    announced = next(
        u for u in client.updates if u.session_update == "available_commands_update"
    )
    commands = {c.name: c for c in announced.available_commands}
    assert commands["clear"].input is None, "takes no argument, so promises none"
    hint = commands["mode"].input
    assert hint is not None
    assert hint.model_dump()["hint"] == "ask | auto | plan"


def test_the_mode_command_switches_and_says_so_on_both_channels():
    """A client reads either `modes` or the config options, so tell it both ways."""
    client = _FakeClient()

    async def body(server, session_id, opened):
        session = server.sessions[session_id]
        answer = server._command(session, "/mode plan")
        await session.flush()
        return answer, session.mode_id

    answer, mode_id = _in_session(body, client)
    assert mode_id == library.PLAN
    assert "Plan" in answer
    kinds = _kinds(client.updates)
    assert "current_mode_update" in kinds
    assert "config_option_update" in kinds
    pushed = next(
        u for u in client.updates if u.session_update == "config_option_update"
    )
    assert _option(pushed.config_options, "mode").current_value == library.PLAN


def test_the_mode_command_with_no_argument_reports_the_choices():
    client = _FakeClient()

    async def body(server, session_id, opened):
        return server._command(server.sessions[session_id], "/mode")

    answer = _in_session(body, client)
    assert "Ask" in answer
    assert all(f"/mode {m.id}" in answer for m in library.SESSION_MODES)


def test_an_unknown_mode_from_the_command_is_reported_not_applied():
    client = _FakeClient()

    async def body(server, session_id, opened):
        session = server.sessions[session_id]
        return server._command(session, "/mode wishful"), session.mode_id

    answer, mode_id = _in_session(body, client)
    assert "wishful" in answer
    assert mode_id == library.ASK


def _prompted(client: _FakeClient, text: str, responses=None, **kwargs):
    """Send one prompt and return what the model saw and what the session became."""
    mock = MockCompletionHandler(responses or [make_text_response("the model ran")])

    async def body(server, session_id, opened):
        session = server.sessions[session_id]
        session.agent.__history__.extend(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        )
        for name, value in kwargs.items():
            setattr(session, name, value)
        response = await server.prompt(
            session_id=session_id, prompt=[acp.text_block(text)]
        )
        return response, list(session.agent.__history__)

    with handler(_stack(mock)):
        response, history = _in_session(body, client)
    said = [u for u in client.updates if u.session_update == "agent_message_chunk"]
    return response, history, mock, said


def test_a_slash_command_arrives_as_an_ordinary_prompt_and_is_answered_here():
    """ACP has no separate method for a command: recognising the prefix is all of it.

    Answered without a model, because both commands are about the session rather than
    about anything a model would know -- paying for a round trip to be told the
    working directory would be an odd way to spend the user's money.
    """
    response, history, mock, said = _prompted(_FakeClient(), "/clear")

    assert response.stop_reason == "end_turn"
    assert mock.call_count == 0, "a command must not cost a model round trip"
    assert "forgotten" in said[-1].content.text


def test_clearing_leaves_no_note_about_clearing_in_the_history():
    """The one message that must not survive forgetting the conversation."""
    _, history, _, _ = _prompted(_FakeClient(), "/clear")
    assert history == []


def test_status_reports_what_the_session_is_actually_using():
    _, history, _, said = _prompted(
        _FakeClient(), "/status", mode_id=library.PLAN, model=_MODELS[0]
    )
    assert "Plan" in said[-1].content.text
    assert _MODELS[0] in said[-1].content.text
    assert _CWD in said[-1].content.text
    assert history, "/status changes nothing"


def test_an_unknown_command_says_so_instead_of_asking_the_model():
    _, _, mock, said = _prompted(_FakeClient(), "/summon a demon")
    assert mock.call_count == 0
    assert "/summon" in said[-1].content.text


def test_a_prompt_that_merely_contains_a_slash_still_reaches_the_model():
    """Only a leading slash is a command; `a/b` is a word."""
    _, _, mock, _ = _prompted(_FakeClient(), "does a/b work?")
    assert mock.call_count == 1


# ============================================================================
# The plan, as a tool whose whole effect is on the screen
# ============================================================================


def test_the_model_can_show_the_user_a_plan():
    """`AgentPlanUpdate` needs no capability: the client's `plan` gates the
    *incremental* `plan_update`, which is why an editor that advertises nothing
    still renders this one."""
    client = _FakeClient()
    with _session(client) as session:
        with handler(library.ACPToolRuntime(session)):
            answer = acp_update_plan(
                [
                    library.PlanStep("Read the file", "high", "completed"),
                    library.PlanStep("Change it", "high", "in_progress"),
                    library.PlanStep("Run the tests", "medium", "pending"),
                ]
            )

    (update,) = [u for u in client.updates if u.session_update == "plan"]
    assert [e.content for e in update.entries] == [
        "Read the file",
        "Change it",
        "Run the tests",
    ]
    assert [e.status for e in update.entries] == [
        "completed",
        "in_progress",
        "pending",
    ]
    assert [e.priority for e in update.entries] == ["high", "high", "medium"]
    # The model is told what the user is now looking at.
    assert "3 step" in answer and "1 of them completed" in answer


def test_a_plan_step_speaks_the_protocols_own_vocabulary():
    """`PlanStep` is not `acp.schema.PlanEntry`, but its words have to be its words.

    Restating them would let the protocol add a status this agent kept refusing to
    send, and the failure would be a runtime validation error inside a notification
    rather than anything a type checker saw.
    """
    fields = {f.name: f.type for f in dataclasses.fields(library.PlanStep)}
    assert fields["priority"] is schema.PlanEntryPriority
    assert fields["status"] is schema.PlanEntryStatus


def test_the_plan_the_model_is_shown_has_no_protocol_metadata_in_it():
    """Which is the reason `PlanStep` exists rather than `acp.schema.PlanEntry`.

    Every ACP type carries `_meta`, a free-form object reserved for implementations to
    attach things to. Tool parameters become a *strict* JSON Schema, and strict schemas
    list every property as required -- so using the wire type would oblige the model to
    invent a value for a field documented as one nobody may assume anything about.
    """
    parameters = _tool_schema(acp_update_plan)
    step = parameters["$defs"]["PlanStep"]
    assert "_meta" not in step["properties"]
    assert set(step["required"]) == {"content", "priority", "status"}

    wire = _tool_schema(_plan_tool_taking(schema.PlanEntry))["$defs"]["PlanEntry"]
    assert "_meta" in wire["required"], "the wire type would demand it of the model"


def test_the_plan_is_replaced_whole_rather_than_appended_to():
    """Each call is the plan now, which is what the model's tool description says."""
    client = _FakeClient()
    with _session(client) as session:
        with handler(library.ACPToolRuntime(session)):
            acp_update_plan([library.PlanStep("One", "low", "pending")])
            acp_update_plan([library.PlanStep("One", "low", "completed")])

    plans = [u for u in client.updates if u.session_update == "plan"]
    assert [len(p.entries) for p in plans] == [1, 1]
    assert [p.entries[0].status for p in plans] == ["pending", "completed"]


def test_the_plan_tool_is_offered_to_the_assistant():
    names = {
        t.__name__ for t in _tools_in_scope(assistant.Assistant().prompt.__context__)
    }
    assert "acp_update_plan" in names


# ============================================================================
# Asking the user a question, as a form in their editor
#
# The client capability nothing used until now. Unlike the permission prompt,
# this one is the model's to reach for -- so the interesting cases are the
# three answers a form can come back with, and the two ways it must not
# interact with the gate that prompts about tool calls.
# ============================================================================


_FORMS = schema.ClientCapabilities(
    elicitation=schema.ElicitationCapabilities(
        form=schema.ElicitationFormCapabilities()
    )
)


def _ask(client: _FakeClient, caps: schema.ClientCapabilities | None = _FORMS, **kw):
    """Call `acp_ask_user` inside a session, and hand back what it returned."""
    with _session(client, caps, **kw) as session:
        with handler(library.ACPToolRuntime(session)):
            return acp_ask_user(
                "I can fix this two ways.",
                [
                    library.AskField(
                        name="approach",
                        title="Which approach?",
                        kind="choice",
                        choices=["narrow the type", "widen the caller"],
                    ),
                    library.AskField(
                        name="note", title="Anything else?", required=False
                    ),
                ],
            )


def test_asking_the_user_goes_through_the_editor():
    """The answers come back as the tool's result, which is what the model reads."""
    client = _FakeClient(
        elicitation=schema.AcceptElicitationResponse(
            action="accept",
            content={"approach": "narrow the type", "note": "keep it small"},
        )
    )
    answer = _ask(client)

    (message, mode) = client.elicited[0]
    assert message == "I can fix this two ways."
    assert isinstance(mode, schema.ElicitationFormSessionMode)
    assert "narrow the type" in answer and "keep it small" in answer


def test_the_form_asks_for_exactly_what_the_field_list_described():
    """The fields are a closed mapping onto the primitives ACP allows.

    A choice carries its options as titled `oneOf` rather than a bare `enum`, because
    the title is what the editor puts on the control.
    """
    client = _FakeClient()
    _ask(client)
    schema_ = client.elicited[0][1].requested_schema

    assert set(schema_.properties or {}) == {"approach", "note"}
    choice = (schema_.properties or {})["approach"]
    assert [o.const for o in choice.one_of or []] == [
        "narrow the type",
        "widen the caller",
    ]
    # Only the field that said so is required.
    assert schema_.required == ["approach"]


def test_a_boolean_field_is_offered_as_a_boolean():
    """A yes/no should be a checkbox, not a text box the user types "yes" into."""
    client = _FakeClient(
        elicitation=schema.AcceptElicitationResponse(
            action="accept", content={"proceed": False}
        )
    )
    with _session(client, _FORMS) as session:
        with handler(library.ACPToolRuntime(session)):
            answer = acp_ask_user(
                "This deletes the migration.",
                [library.AskField(name="proceed", title="Go ahead?", kind="boolean")],
            )

    prop = (client.elicited[0][1].requested_schema.properties or {})["proceed"]
    assert isinstance(prop, schema.ElicitationBooleanPropertySchema)
    # Rendered as a word rather than as `False`, which reads like a missing answer.
    assert "no" in answer.lower()


def test_a_declined_question_is_reported_to_the_model_not_raised():
    """Refusing to answer is an answer, and not a broken tool.

    The distinction from `cancel` below is the whole reason this is not just another
    permission prompt: the model should carry on and say what it could not decide.
    """
    client = _FakeClient(
        elicitation=schema.DeclineElicitationResponse(action="decline")
    )
    answer = _ask(client)
    assert "declined" in answer.lower()
    assert "not ask again" in answer.lower() or "do not ask" in answer.lower()


def test_a_dismissed_question_cancels_the_turn():
    """Dismissing the form is the same gesture as dismissing a permission prompt."""
    client = _FakeClient(elicitation=schema.CancelElicitationResponse(action="cancel"))
    with pytest.raises(library.SessionCancelled):
        _ask(client)


def test_an_accept_with_nothing_filled_in_still_names_every_field():
    """`content` is optional even on an accept, and a blank is not a vanished field.

    Keyed by what was asked rather than by what came back, so the model can tell the
    difference between "they left it blank" and "I never asked".
    """
    client = _FakeClient(
        elicitation=schema.AcceptElicitationResponse(action="accept", content=None)
    )
    answer = _ask(client)
    assert "approach" in answer and "note" in answer
    assert "blank" in answer


def test_an_editor_that_cannot_show_a_form_is_told_so_not_crashed_into():
    """Default capabilities claim no elicitation, which must not be an AttributeError.

    Poolside advertises `elicitation.form`, but VS Code's client advertises no
    elicitation at all -- so this is the common case, not the exotic one.
    """
    client = _FakeClient()
    with _session(client, schema.ClientCapabilities()) as session:
        assert session.elicitation_capabilities.form is None
        with handler(library.ACPToolRuntime(session)):
            with pytest.raises(NotImplementedError):
                acp_ask_user("Which one?", [library.AskField("a", "A")])

    assert client.elicited == [], "it must not ask an editor that cannot answer"


def test_a_form_with_no_fields_is_refused_before_the_editor_sees_it():
    """A dialog the user can only dismiss is worse than no dialog."""
    client = _FakeClient()
    with _session(client, _FORMS) as session:
        with handler(library.ACPToolRuntime(session)):
            with pytest.raises(ValueError):
                acp_ask_user("Well?", [])
    assert client.elicited == []


def test_the_form_is_attached_to_the_tool_call_row_it_belongs_to():
    """Otherwise the editor draws a dialog with no visible cause.

    `toolCallId` is how a client renders the form inside the row for the call that
    asked, which is the same thing `show_terminal` does for a terminal.
    """

    @Skill.define
    def use_tool() -> str:
        """Ask the user something with `acp_ask_user`."""

    client = _FakeClient()
    with _session(client, _FORMS) as session:
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response(
                            "acp_ask_user",
                            '{"message": "Which?", "fields": [{"name": "a", '
                            '"title": "A", "kind": "text"}]}',
                        ),
                        make_text_response("thanks"),
                    ]
                ),
                library.ACPToolRuntime(session),
                session.reporter,
            )
        ):
            assert use_tool() == "thanks"

    (_, mode) = client.elicited[0]
    started = [u for u in client.updates if u.session_update == "tool_call"]
    assert started, "the call should have been announced"
    assert mode.tool_call_id == started[0].tool_call_id
    assert mode.session_id == session.session_id


def test_asking_the_user_needs_no_permission_prompt():
    """A dialog asking permission to open a dialog, answered by the same person.

    The gate exists to put a question in front of the user before a tool runs, so for
    the one tool whose only effect *is* to ask them, it defeats itself.
    """

    @Skill.define
    def use_tool() -> str:
        """Ask the user something with `acp_ask_user`."""

    client = _FakeClient()
    with _session(client, _FORMS) as session:
        assert session.mode_id == library.ASK
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response(
                            "acp_ask_user",
                            '{"message": "Which?", "fields": [{"name": "a", '
                            '"title": "A", "kind": "text"}]}',
                        ),
                        make_text_response("ok"),
                    ]
                ),
                library.ACPToolRuntime(session),
                library.ACPPermissionGate(session),
            )
        ):
            assert use_tool() == "ok"

    assert client.asked == [], "asking the user needs no permission to ask the user"
    assert client.elicited, "and the question itself must still have been asked"


def test_asking_is_allowed_in_plan_mode():
    """Plan mode changes nothing, and a question changes nothing.

    It is also where a clarifying question is worth most, so `acp_ask_user` must not
    be mistaken for a mutating tool.
    """
    assert acp_ask_user.__name__ not in library.MUTATING_TOOLS

    @Skill.define
    def use_tool() -> str:
        """Ask the user something with `acp_ask_user`."""

    client = _FakeClient()
    with _session(client, _FORMS) as session:
        session.mode_id = library.PLAN
        with handler(
            _stack(
                MockCompletionHandler(
                    [
                        make_tool_call_response(
                            "acp_ask_user",
                            '{"message": "Which?", "fields": [{"name": "a", '
                            '"title": "A", "kind": "text"}]}',
                        ),
                        make_text_response("ok"),
                    ]
                ),
                library.ACPToolRuntime(session),
                library.ACPPermissionGate(session),
            )
        ):
            assert use_tool() == "ok"

    assert client.elicited, "Plan mode must still be able to ask"


def test_cancelling_interrupts_a_pending_question():
    """The wait is unbounded, so it has to be interruptible. See `ACPSession.call`."""

    class _SilentAboutForms(_FakeClient):
        async def create_elicitation(self, message, mode, **kw):
            await asyncio.sleep(3600)

    client = _SilentAboutForms()
    with _session(client, _FORMS) as session:
        threading.Timer(0.2, session.cancel.set).start()
        with handler(library.ACPToolRuntime(session)):
            with pytest.raises(library.SessionCancelled):
                acp_ask_user("Which?", [library.AskField("a", "A")])


def test_the_ask_tool_is_offered_to_the_assistant():
    names = {
        t.__name__ for t in _tools_in_scope(assistant.Assistant().prompt.__context__)
    }
    assert "acp_ask_user" in names


# ============================================================================
# Sessions, and replaying a loaded one
# ============================================================================


def _user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


_SYSTEM = {"role": "system", "content": [{"type": "text", "text": "# Harness"}]}


def _load(history: list[dict]) -> _FakeClient:
    """Open a session, give it `history`, reload it, and return what the client saw.

    Driven through `load_session` itself rather than a replay helper, because the
    requirement is about what reaches the client *before the response* -- which is
    the wait for the queue to drain, not the formatting.
    """
    client = _FakeClient()

    async def drive():
        server = await _serve(client)
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        server.sessions[session_id].agent.__history__.extend(history)
        await server.load_session(cwd=_CWD, session_id=session_id)
        await server.close_session(session_id)

    asyncio.run(drive())
    return client


def test_the_session_id_is_the_agents_id():
    """That identity is what makes a conversation persistent.

    With a persistence handler installed, the agent checkpoints under
    `Agent.__agent_id__`; if the server tracked sessions under some id of its own,
    reopening one would restore nothing.
    """

    async def drive():
        server = await _serve(_FakeClient())
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        agent_id = server.sessions[session_id].agent.__agent_id__
        await server.close_session(session_id)
        return session_id, agent_id

    session_id, agent_id = asyncio.run(drive())
    assert session_id == agent_id


def test_forking_copies_a_conversation_without_entangling_it(tmp_path):
    """A fork is for trying a second approach without losing the first."""

    async def drive():
        server = await _serve(_FakeClient())
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        source = server.sessions[session_id]
        source.mode_id, source.model = library.PLAN, "some/model"
        source.title = "the original"
        await server.prompt(session_id=session_id, prompt=[acp.text_block("hello")])

        forked = await server.fork_session(session_id=session_id, cwd=_CWD)
        fork = server.sessions[forked.session_id]
        before = list(fork.agent.__history__)
        # A turn in the fork must leave the source alone.
        await server.prompt(
            session_id=forked.session_id, prompt=[acp.text_block("only in the fork")]
        )
        listed = await server.list_sessions()
        result = (
            forked.session_id != session_id,
            len(before),
            len(source.agent.__history__),
            len(fork.agent.__history__),
            (fork.mode_id, fork.model, fork.title),
            [s.title for s in listed.sessions],
        )
        await server.close_session(forked.session_id)
        await server.close_session(session_id)
        return result

    with (
        handler(_stack(MockCompletionHandler([make_text_response("ok")]))),
        _persisted(tmp_path),
    ):
        distinct, copied, source_len, fork_len, settings, titles = asyncio.run(drive())

    assert distinct, "a fork gets its own id"
    assert copied == source_len, "it starts as a copy of the source"
    assert fork_len > source_len, "and diverges from there"
    assert settings == (library.PLAN, "some/model", "the original (fork)")
    assert sorted(titles) == ["the original", "the original (fork)"]


def test_forking_is_advertised():
    caps = _advertised().session_capabilities
    assert caps is not None and caps.fork is not None


def test_replay_returns_both_halves_in_order():
    """ACP requires the whole conversation back, chronologically."""
    client = _load(
        [
            _SYSTEM,
            _user("what is the capital of France?"),
            {"role": "assistant", "content": "Paris."},
            _user("and of Spain?"),
            {"role": "assistant", "content": "Madrid."},
        ]
    )

    replayed = _conversation(client.updates)
    assert _kinds(replayed) == [
        "user_message_chunk",
        "agent_message_chunk",
        "user_message_chunk",
        "agent_message_chunk",
    ]
    assert [u.content.text for u in replayed] == [
        "what is the capital of France?",
        "Paris.",
        "and of Spain?",
        "Madrid.",
    ]


def test_replay_skips_the_system_message():
    assert _conversation(_load([_SYSTEM]).updates) == []


def test_replay_does_not_invent_a_message_for_a_turn_that_only_called_tools():
    """An assistant turn that only called tools has `content: None`.

    Rendered as JSON that became the literal string ``"null"``, which passed the
    "skip the empty ones" guard and put the word into the editor for every such turn
    a reloaded session replayed.
    """
    client = _load(
        [
            _user("read x.py"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "acp_read_text_file",
                            "arguments": '{"path": "/tmp/x.py"}',
                        },
                    }
                ],
            },
        ]
    )
    assert "null" not in [
        getattr(getattr(u, "content", None), "text", None)
        for u in _conversation(client.updates)
    ]


def test_replay_brings_back_tool_calls_and_their_results():
    """ "The Agent MUST replay the entire conversation", and most of it is not prose.

    A turn that read three files says nothing at all in text. Replaying only the
    prose reproduces a conversation in which the agent sat silent and then knew
    things.
    """
    client = _load(
        [
            _user("read x.py"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "acp_read_text_file",
                            "arguments": '{"path": "/tmp/x.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "print(1)\n"},
            {"role": "assistant", "content": "It prints 1."},
        ]
    )

    replayed = _conversation(client.updates)
    assert _kinds(replayed) == [
        "user_message_chunk",
        "tool_call",
        "tool_call_update",
        "agent_message_chunk",
    ]
    start = replayed[1]
    assert start.title == "acp_read_text_file"
    assert start.status == "completed", "a stored call is one that already ran"
    assert start.raw_input == {"path": "/tmp/x.py"}
    assert "print(1)" in replayed[2].content[0].content.text


def test_a_closed_session_is_rebuilt_rather_than_reused():
    """Closing a session ends its writer, so the server must forget it.

    A closed session left in the table would accept notifications that nothing
    delivers, and `load_session` -- which ends by waiting for the queue to empty --
    would hang instead of replaying.
    """
    client = _FakeClient()

    async def drive():
        server = await _serve(client)
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        await server.close_session(session_id)
        assert session_id not in server.sessions

        # Reopening is `session/load`'s job, and works because the agent reads its own
        # history back under the same id.
        server._open_session(session_id, _CWD, None).agent.__history__.append(
            {"role": "assistant", "content": "Paris."}
        )
        await asyncio.wait_for(
            server.load_session(cwd=_CWD, session_id=session_id), timeout=10
        )
        await server.close_session(session_id)

    asyncio.run(drive())
    assert [u.content.text for u in _conversation(client.updates)] == ["Paris."]


def test_the_editors_working_directory_is_kept_for_the_session():
    """`cwd` "MUST be used for the session regardless of where the Agent was spawned"."""

    async def drive():
        server = await _serve(_FakeClient())
        session_id = (
            await server.new_session(
                cwd="/work/thing",
                additional_directories=["/work/vendor"],
                mcp_servers=[],
            )
        ).session_id
        opened = server.sessions[session_id].roots
        # Reopening from another window may re-root the same conversation.
        await server.load_session(cwd="/work/moved", session_id=session_id)
        reopened = server.sessions[session_id].roots
        await server.close_session(session_id)
        return opened, reopened

    opened, reopened = asyncio.run(drive())
    assert opened == ("/work/thing", "/work/vendor")
    assert reopened == ("/work/moved",)


def test_mcp_servers_are_ignored_rather_than_refused(capsys):
    """Refusing them made the agent unusable in any editor with MCP configured.

    stdio transport is baseline and has no capability to decline, so a conforming
    client sends its configured servers on every `session/new`. Failing the request
    over that helps nobody; ignoring it silently hides it, so it goes to stderr --
    which is free, since `serve` gives the protocol its own descriptor.
    """

    async def drive():
        server = await _serve(_FakeClient())
        response = await server.new_session(
            cwd=_CWD, mcp_servers=[{"name": "everything", "command": "npx"}]
        )
        await server.close_session(response.session_id)
        return response.session_id

    assert asyncio.run(drive())
    assert "MCP" in capsys.readouterr().err


def test_a_prompt_for_a_session_that_was_never_opened_is_refused():
    """Opening one here would turn a typo into a silently fresh conversation.

    Which the user reads as an agent that forgot everything, with nothing anywhere
    reporting a problem.
    """

    async def drive():
        server = await _serve(_FakeClient())
        with pytest.raises(acp.RequestError):
            await server.prompt(session_id="never-opened", prompt=[acp.text_block("?")])
        return server

    server = asyncio.run(drive())
    assert server.sessions == {}, "a refused prompt must not leave a session behind"


def test_cancelling_an_unknown_session_does_nothing_at_all():
    """A notification has nowhere to report an error -- but must not open a session."""

    async def drive():
        server = await _serve(_FakeClient())
        await server.cancel("never-opened")
        return server

    assert asyncio.run(drive()).sessions == {}


def test_a_session_is_one_conversation_across_many_turns():
    """The second prompt must see the first, and the tenth must see all nine.

    The failure this guards against does not look like an error: every turn answers
    fine, and the agent has simply forgotten everything, which reads as "each message
    got its own session". A whole conversation is one `Agent` instance, kept in the
    session table, so the thing to pin is that a later request carries the earlier
    exchanges.
    """
    mock = MockCompletionHandler([make_text_response("ok")])

    async def drive():
        server = await _serve(_FakeClient())
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        agents = []
        for text in ("first", "second", "third"):
            await server.prompt(session_id=session_id, prompt=[acp.text_block(text)])
            agents.append(id(server.sessions[session_id].agent))
        history = list(server.sessions[session_id].agent.__history__)
        await server.close_session(session_id)
        return agents, history

    with handler(_stack(mock)):
        agents, history = asyncio.run(drive())

    assert len(set(agents)) == 1, "every turn must run on the one agent"
    # Each request carries everything before it, so they grow strictly.
    lengths = [len(messages) for messages in mock.received_messages]
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1], lengths
    assert [m.get("role") for m in history].count("user") == 3
    assert [m.get("role") for m in history].count("assistant") == 3


def test_two_prompts_on_one_session_do_not_overlap():
    """A client may send a second prompt before the first has returned.

    `acp.Connection._process_message` dispatches every request as its own task and
    does not await it, so nothing upstream serialises them. Two turns at once would
    put two worker threads on one agent's history.
    """
    live, peak = 0, 0

    class _Slow(ObjectInterpretation):
        @implements(completion)
        def _completion(self, messages=None, **kw):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            time.sleep(0.05)
            live -= 1
            return make_text_response("ok")

    async def drive():
        server = await _serve(_FakeClient())
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        await asyncio.gather(
            *(
                server.prompt(session_id=session_id, prompt=[acp.text_block(f"q{i}")])
                for i in range(3)
            )
        )
        await server.close_session(session_id)

    with handler(_stack(_Slow())):
        asyncio.run(drive())

    assert peak == 1, f"{peak} turns ran at once; prompts must be serialised"


# ============================================================================
# Remembering that a session existed
#
# `session/list` is what lets an editor offer conversations from before it was
# last closed. It needs more than the agent histories the persister keeps -- a
# `SessionInfo` must name the directory the session was opened on -- so these
# run with a real persistence handler installed.
# ============================================================================


@contextlib.contextmanager
def _persisted(tmp_path):
    """A `SQLitePersister`, which is what makes a session index exist at all.

    Enter it *outside* `_stack` when the test depends on checkpointing: the persister
    hooks `call_agent` and has to see the call, which means being installed after
    `AgentLoop` -- which is where `harness(persist_db=...)` puts it in production. A
    test that only needs the index can install it anywhere, since nothing else answers
    `_checkpoint_connection`.
    """
    from effectful.handlers.llm.harness.durability.persistence import SQLitePersister

    with handler(SQLitePersister(tmp_path / "sessions.db")):
        yield


def test_listing_is_advertised_only_when_there_is_an_index(tmp_path):
    """A capability that is a claim about state, not about code.

    Answering "no sessions" without one would be worse than not answering: a client
    reconciles its own history against the reply and would forget every session it
    knew about.
    """
    assert (
        _advertised().session_capabilities or schema.SessionCapabilities()
    ).list is None
    with _persisted(tmp_path):
        caps = _advertised().session_capabilities
        assert caps is not None and caps.list is not None
        assert caps.resume is not None, "resume needs no index and is always offered"


def test_listing_without_an_index_is_refused_rather_than_answered_empty():
    async def body(server, session_id, opened):
        with pytest.raises(acp.RequestError):
            await server.list_sessions()

    _in_session(body)


def test_sessions_are_listed_newest_first_with_their_directory_and_title(tmp_path):
    async def drive():
        server = await _serve(_FakeClient())
        for i, cwd in enumerate(("/work/a", "/work/b", "/work/a")):
            opened = await server.new_session(cwd=cwd, mcp_servers=[])
            server.sessions[opened.session_id].agent.__history__.clear()
            await server.prompt(
                session_id=opened.session_id,
                prompt=[acp.text_block(f"question number {i}")],
            )
            await server.close_session(opened.session_id)
        everything = await server.list_sessions()
        just_a = await server.list_sessions(cwd="/work/a")
        return everything, just_a

    with (
        _persisted(tmp_path),
        handler(_stack(MockCompletionHandler([make_text_response("ok")]))),
    ):
        everything, just_a = asyncio.run(drive())

    assert [s.cwd for s in everything.sessions] == ["/work/a", "/work/b", "/work/a"]
    assert [s.title for s in everything.sessions] == [
        "question number 2",
        "question number 1",
        "question number 0",
    ]
    assert all(s.updated_at for s in everything.sessions)
    assert everything.next_cursor is None, "one page holds them all"
    assert [s.cwd for s in just_a.sessions] == ["/work/a", "/work/a"], "cwd narrows it"


def test_listing_pages_without_dropping_or_repeating_a_session(tmp_path):
    """Keyed on the last row handed out, so a write mid-paging cannot hide one."""

    async def drive():
        server = await _serve(_FakeClient(), page_size=2)
        for i in range(5):
            opened = await server.new_session(cwd=f"/work/{i}", mcp_servers=[])
            server.sessions[opened.session_id].title = f"session {i}"
            library.SessionIndex.record(server.sessions[opened.session_id])
            await server.close_session(opened.session_id)
        pages, cursor = [], None
        while True:
            page = await server.list_sessions(cursor=cursor)
            pages.append([s.title for s in page.sessions])
            cursor = page.next_cursor
            if cursor is None:
                break
        return pages

    with _persisted(tmp_path):
        pages = asyncio.run(drive())

    assert all(len(p) <= 2 for p in pages), pages
    seen = [title for page in pages for title in page]
    assert sorted(seen) == [f"session {i}" for i in range(5)]
    assert len(seen) == len(set(seen)), "a paged session must appear exactly once"


def test_a_session_is_titled_after_its_first_prompt_and_the_editor_is_told(tmp_path):
    """Without this, a session list shows the user a column of UUIDs."""
    client = _FakeClient()

    async def drive():
        server = await _serve(client)
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        for text in ("how do I center a div?", "and what about flexbox?"):
            await server.prompt(session_id=session_id, prompt=[acp.text_block(text)])
        title = server.sessions[session_id].title
        await server.close_session(session_id)
        return title

    with (
        _persisted(tmp_path),
        handler(_stack(MockCompletionHandler([make_text_response("ok")]))),
    ):
        title = asyncio.run(drive())

    assert title == "how do I center a div?", "the first prompt names it, not the last"
    announced = [u for u in client.updates if u.session_update == "session_info_update"]
    assert len(announced) == 1, "renaming it every turn would move it under the reader"
    assert announced[0].title == title


def test_a_slash_command_does_not_name_a_session(tmp_path):
    """`/status` is not what the conversation turned out to be about."""

    async def drive():
        server = await _serve(_FakeClient())
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        await server.prompt(session_id=session_id, prompt=[acp.text_block("/status")])
        await server.prompt(session_id=session_id, prompt=[acp.text_block("real one")])
        title = server.sessions[session_id].title
        await server.close_session(session_id)
        return title

    with (
        _persisted(tmp_path),
        handler(_stack(MockCompletionHandler([make_text_response("ok")]))),
    ):
        assert asyncio.run(drive()) == "real one"


def test_a_long_first_prompt_is_shortened_into_a_title():
    assert library._title_from("a\n  b\tc") == "a b c"
    long = library._title_from("word " * 40)
    assert len(long) <= 60 and long.endswith("\u2026")


def test_resuming_restores_a_session_without_replaying_it(tmp_path):
    """The difference from `load_session`: a client that kept its own transcript."""
    client = _FakeClient()

    async def drive():
        server = await _serve(client)
        session_id = (await server.new_session(cwd=_CWD, mcp_servers=[])).session_id
        # A real turn, so the persister checkpoints it: resuming rebuilds the agent
        # under the same id and it reads its own history back.
        await server.prompt(
            session_id=session_id, prompt=[acp.text_block("what is the capital?")]
        )
        await server.close_session(session_id)
        client.updates.clear()
        response = await server.resume_session(session_id=session_id, cwd="/work/moved")
        resumed = server.sessions[session_id]
        await server.close_session(session_id)
        return response, resumed.cwd, list(resumed.agent.__history__)

    with (
        handler(_stack(MockCompletionHandler([make_text_response("Paris.")]))),
        _persisted(tmp_path),
    ):
        response, cwd, history = asyncio.run(drive())

    assert response.modes is not None and response.config_options
    assert cwd == "/work/moved", "resuming re-roots it, as loading does"
    assert history, "the conversation is still there"
    assert _conversation(client.updates) == [], "but it is not replayed"


# ============================================================================
# The contracts the examples tree imposes
# ============================================================================


@pytest.mark.parametrize("name", ["library.py", "assistant.py"])
def test_the_example_brings_no_handler_stack(name):
    """Restated locally because it is the reason this example is shaped as it is.

    The server inherits the launcher's stack rather than assembling one, so a session
    adds only its own handlers on top. `test_handlers_llm_harness_launcher.py`
    checks it across the whole tree; failing here says which file, and why it matters.
    """
    tree = ast.parse((EXAMPLE_DIR / name).read_text())
    bound = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "harness" not in bound


def test_the_library_is_not_mistaken_for_a_script():
    """`example_scripts` identifies a script by a top-level `main`.

    One here would enrol the library in the live example run, which would try to
    launch a module that is not runnable on its own.
    """
    tree = ast.parse((EXAMPLE_DIR / "library.py").read_text())
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body
    )


def test_the_documented_command_lines_actually_parse():
    """The invocations in the docstrings are what a reader copies, so run them.

    Written after the docstrings taught a wrong one: they separated the script's flags
    with ``--``, which argparse hands to the script as a literal argument it then
    rejects. The launcher needs no separator -- it consumes the flags it knows and
    passes the rest through -- so the ``--`` only ever broke the command.
    """
    from effectful.handlers.llm.harness.__main__ import _parse_args

    for command in _documented_commands():
        ns, script_args = _parse_args(command)
        assert "--" not in script_args, (
            f"`--` reaches the script as an argument it will reject: {command}"
        )
        # Every flag the launcher did not claim must be one the example declares.
        _example_parser().parse_args(script_args)


_LAUNCHER = "-m effectful.handlers.llm.harness "


def _documented_commands() -> list[list[str]]:
    """The launcher command lines shown in the example's docstrings, as argv.

    Continuations are joined first: the docstrings wrap with a trailing backslash,
    which reaches this as one or two literal backslashes depending on the quoting.
    """
    commands: list[list[str]] = []
    for path in ("library.py", "assistant.py"):
        text = re.sub(r"\\+\n\s*", " ", (EXAMPLE_DIR / path).read_text())
        commands.extend(
            line.split(_LAUNCHER, 1)[1].split()
            for line in text.splitlines()
            if _LAUNCHER in line
        )
    assert commands, "the docstrings should show how to run this"
    return commands


def _example_parser() -> argparse.ArgumentParser:
    """The example's own parser, rebuilt from its declared flags.

    ``add_help=False`` so that ``_actions`` holds only what the example declares,
    which is what the test below compares against.
    """
    return argparse.ArgumentParser(allow_abbrev=False, add_help=False)


def test_the_example_declares_the_flags_this_file_claims_it_does():
    """`_example_parser` is a copy, so pin it to the original.

    A flag added to `assistant.main` without being added there would be accepted by
    the test above for the wrong reason: unparsed, because never documented.
    """
    declared = {
        node.args[0].value
        for node in ast.walk(ast.parse((EXAMPLE_DIR / "assistant.py").read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and isinstance(node.args[0], ast.Constant)
    }
    mirrored = {
        action.option_strings[0]
        for action in _example_parser()._actions
        if action.option_strings
    }
    assert declared == mirrored


def test_the_picker_can_be_configured_entirely_from_the_environment(monkeypatch):
    """An editor launches this with a command and an env, and the model it starts on
    already comes from the env -- so the models it can switch to belong beside it.
    """
    monkeypatch.setenv(library.OFFER_MODELS_ENV, f"{_MODELS[0]}, {_MODELS[1]} ,")
    assert library._offered_models() == _MODELS, "split, trimmed, blanks dropped"


def test_an_unset_or_empty_environment_offers_nothing():
    """Not a picker with one empty entry in it: no picker at all."""
    assert library._offered_models() == ()


def test_a_server_told_nothing_about_models_reads_the_environment(monkeypatch):
    """Where the default lives is the point: the editor hands an *environment* to the
    server, so the server is what knows to look in it. A script that wanted a picker
    would otherwise have to parse the variable itself, and spell it right.
    """
    monkeypatch.setenv(library.OFFER_MODELS_ENV, ",".join(_MODELS))
    assert library.EffectfulACPAgent(_Bot).models == _MODELS


def test_a_server_told_there_are_no_models_does_not_consult_the_environment(
    monkeypatch,
):
    """`()` is an answer and `None` is silence; only silence falls back.

    Without that distinction a caller could switch the picker on but never off, since
    the environment would outrank an explicit empty list.
    """
    monkeypatch.setenv(library.OFFER_MODELS_ENV, ",".join(_MODELS))
    assert library.EffectfulACPAgent(_Bot, models=()).models == ()


def test_the_example_leaves_the_models_to_the_server(monkeypatch):
    """The example names no models at all -- it is a script, not a launcher."""
    monkeypatch.setenv(library.OFFER_MODELS_ENV, ",".join(_MODELS))
    served: list[tuple[str, ...]] = []

    class _Recorded(library.EffectfulACPAgent):
        async def serve(self):
            served.append(self.models)

    monkeypatch.setattr(library, "EffectfulACPAgent", _Recorded)
    monkeypatch.setattr(sys, "argv", ["assistant.py"])
    assistant.main()

    assert served == [_MODELS]


def test_a_mistyped_launcher_flag_is_an_error_rather_than_silence(monkeypatch):
    """The launcher passes through what it does not recognise, so the example is the
    only thing left to reject it -- which is why it parses despite declaring no flags.
    """
    monkeypatch.setattr(sys, "argv", ["assistant.py", "--offer-modle", "x"])
    with pytest.raises(SystemExit):
        assistant.main()


def test_serving_takes_the_protocol_channel_away_from_everything_else():
    """Under a served session, fd 1 is the JSON-RPC channel and carries nothing else.

    Checked against the source rather than by serving: a `print` from model-authored
    code under `BuiltinExecutor` would corrupt the stream, and the defence is that
    `serve` hands the real descriptor to the transport and points fd 1 at stderr.
    """
    source = (EXAMPLE_DIR / "library.py").read_text()
    assert "os.dup2(2, 1)" in source
    assert "sys.stdout = sys.stderr" in source


def test_an_idle_editor_is_not_disconnected():
    """`receive_timeout` tears the connection down between *incoming* messages.

    So any value at all is a rule that the user may not think for longer than it
    before their agent vanishes mid-conversation, which a minute of it did here once.

    Checked against the source for the same reason as the test above: `serve` dups
    and rebinds fd 1, which is not something to do inside a test runner.
    """
    tree = ast.parse((EXAMPLE_DIR / "library.py").read_text())
    served = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_agent"
    ]
    assert len(served) == 1, "expected exactly one acp.run_agent call to check"
    assert "receive_timeout" not in {kw.arg for kw in served[0].keywords}
