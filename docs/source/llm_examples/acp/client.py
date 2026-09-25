"""Serve an ACP agent to AG-UI frontends such as CopilotKit.

The bridge runs an Agent Client Protocol agent as a subprocess, plays the part of
its editor, and serves each AG-UI run over HTTP::

    uv run python docs/source/llm_examples/acp/client.py [--port 8000] [--cwd DIR] -- AGENT ...

To serve ``assistant.py`` beside this file, with its model from
``EFFECTFUL_LLM_MODEL``, the agent command is::

    python -m effectful.handlers.llm.harness docs/source/llm_examples/acp/assistant.py

Through the bridge the agent reads and writes files on this machine.

Point an AG-UI ``HttpAgent`` at ``http://localhost:8000/``: a CopilotKit app does so
from its own runtime, since browsers may not call the bridge directly. The agent's
permission requests and forms arrive as AG-UI interrupts: answer a permission with
``resolve(option)`` for one of ``interrupt.metadata.options``, a form with
``resolve(values)`` or ``resolve()`` to decline it, and either with ``cancel()``.

A client that declares ``protocolVersion`` gets AG-UI 1.0. One that does not, such
as the one CopilotKit 1.73 bundles, rejects the cancelled outcome, so a cancelled
turn ends for it with no outcome and ``result.stopReason`` set to ``cancelled``.

The frontend's tools reach the agent as an MCP server at ``/mcp``, named in each
``session/new``. When the agent calls one, the run ends with the call for the
frontend to execute or render, and the next run's tool message is its result.

Only the newest user message is sent, since the ACP session keeps the history.
Sessions live in memory, context is not forwarded, terminals are not offered, a
second run on a busy thread is refused, and only localhost is served.
"""

import argparse
import asyncio
import collections
import collections.abc
import contextlib
import dataclasses
import difflib
import json
import os
import pathlib
import sys
import typing
import uuid

import acp
import acp.interfaces
import acp.schema
import ag_ui.core
import ag_ui.encoder
import fastapi
import fastapi.responses
import fastmcp
import fastmcp.exceptions
import fastmcp.server.dependencies
import fastmcp.server.providers
import fastmcp.tools
import mcp_types
import pydantic
import pydantic.json_schema
import starlette.middleware.trustedhost
import uvicorn


class ThreadBusy(Exception):
    """A run arrived for a thread that another run is streaming."""


@dataclasses.dataclass(eq=False)
class Session:
    """One ACP session, serving one AG-UI thread."""

    id: str
    state: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    turn: "Turn | None" = None
    prompted: str | None = None
    busy: bool = False
    tools: list[ag_ui.core.Tool] = dataclasses.field(default_factory=list)
    """The tools the frontend offered in its latest run."""


@dataclasses.dataclass(eq=False)
class Turn:
    """One ``session/prompt``, which the frontend may see across several runs.

    Its queue holds the agent's updates and questions, then how the prompt ended;
    one run at a time reads it, translating as it goes.
    """

    session: Session
    events: asyncio.Queue = dataclasses.field(default_factory=asyncio.Queue)
    task: asyncio.Task = dataclasses.field(init=False)
    cancelled: asyncio.Task | None = None
    interrupts: dict[str, tuple[ag_ui.core.Interrupt, asyncio.Future]] = (
        dataclasses.field(default_factory=dict)
    )
    exposed: set[str] = dataclasses.field(default_factory=set)
    calls: dict[str, dict[str, typing.Any]] = dataclasses.field(default_factory=dict)
    announced: set[str] = dataclasses.field(default_factory=set)
    finished: set[str] = dataclasses.field(default_factory=set)
    text: str | None = None
    reasoning: str | None = None
    parent: str | None = None

    def unanswered(self) -> list[ag_ui.core.Interrupt]:
        """The interrupts shown to the frontend that still await an answer."""
        return [
            interrupt
            for interrupt, answer in self.interrupts.values()
            if interrupt.id in self.exposed and not answer.done()
        ]

    def reports_frontend_call(self, call_id: str) -> bool:
        """Whether an ACP tool call is the agent's report of a frontend tool call.

        The agent reports its MCP calls over ACP too; the frontend sees the MCP
        call itself, so showing the report would show the call twice.
        """
        title = self.calls.get(call_id, {}).get("title") or ""
        return any(
            title == tool.name or title.startswith(f"{tool.name}(")
            for tool in self.session.tools
        )

    def attach(self) -> None:
        """Begin streaming this turn into a new run."""
        self.text = self.reasoning = self.parent = None

    def translate(
        self, item: typing.Any
    ) -> collections.abc.Iterator[ag_ui.core.BaseEvent]:
        """The AG-UI events one queued item stands for."""
        match item:
            case acp.schema.AgentMessageChunk(
                content=acp.schema.TextContentBlock(text=text)
            ) if text:
                yield from self._end_reasoning()
                if self.text is None:
                    self.text = self.parent = str(uuid.uuid4())
                    yield ag_ui.core.TextMessageStartEvent(
                        message_id=self.text, role="assistant"
                    )
                yield ag_ui.core.TextMessageContentEvent(
                    message_id=self.text, delta=text
                )
            case acp.schema.AgentThoughtChunk(
                content=acp.schema.TextContentBlock(text=text)
            ) if text:
                yield from self._end_text()
                if self.reasoning is None:
                    self.reasoning = str(uuid.uuid4())
                    yield ag_ui.core.ReasoningStartEvent(message_id=self.reasoning)
                    yield ag_ui.core.ReasoningMessageStartEvent(
                        message_id=self.reasoning, role="reasoning"
                    )
                yield ag_ui.core.ReasoningMessageContentEvent(
                    message_id=self.reasoning, delta=text
                )
            case (
                acp.schema.ToolCallStart()
                | acp.schema.ToolCallProgress()
                | acp.schema.ToolCallUpdate()
            ):
                call_id = item.tool_call_id
                call = self.calls.setdefault(call_id, {})
                # ACP sends only what changed, and `content` replaces what was there.
                call.update(
                    item.model_dump(
                        exclude_none=True, exclude={"session_update", "field_meta"}
                    )
                )
                if self.reports_frontend_call(call_id):
                    return
                if call.get("status") in ("in_progress", "completed", "failed"):
                    yield from self.announce(call_id)
                if call.get("status") in ("completed", "failed"):
                    yield from self._result(call_id, _render(call))
            case ag_ui.core.BaseEvent():
                yield item

    def announce(self, call_id: str) -> collections.abc.Iterator[ag_ui.core.BaseEvent]:
        """Show an ACP tool call once, with its arguments as they now stand."""
        if call_id in self.announced or self.reports_frontend_call(call_id):
            return
        self.announced.add(call_id)
        call = self.calls.get(call_id, {})
        yield from self.show(
            call_id,
            call.get("title") or "tool",
            json.dumps(call.get("raw_input") or {}, default=str),
        )

    def show(
        self, call_id: str, name: str, arguments: str
    ) -> collections.abc.Iterator[ag_ui.core.BaseEvent]:
        """A tool call, after closing the open messages."""
        yield from self.close()
        if self.parent is None:
            self.parent = str(uuid.uuid4())
        yield ag_ui.core.ToolCallStartEvent(
            tool_call_id=call_id, tool_call_name=name, parent_message_id=self.parent
        )
        yield ag_ui.core.ToolCallArgsEvent(tool_call_id=call_id, delta=arguments)
        yield ag_ui.core.ToolCallEndEvent(tool_call_id=call_id)

    def close(self) -> collections.abc.Iterator[ag_ui.core.BaseEvent]:
        """End the open text and reasoning messages."""
        yield from self._end_text()
        yield from self._end_reasoning()

    def finish(self) -> collections.abc.Iterator[ag_ui.core.BaseEvent]:
        """Close everything, and say so for shown calls that never finished."""
        yield from self.close()
        for call_id in [c for c in self.calls if c in self.announced]:
            yield from self._result(
                call_id, "The turn ended before this call finished."
            )

    def _result(
        self, call_id: str, content: str
    ) -> collections.abc.Iterator[ag_ui.core.BaseEvent]:
        if call_id not in self.finished:
            self.finished.add(call_id)
            yield ag_ui.core.ToolCallResultEvent(
                message_id=str(uuid.uuid4()), tool_call_id=call_id, content=content
            )

    def _end_text(self) -> collections.abc.Iterator[ag_ui.core.BaseEvent]:
        if (message_id := self.text) is not None:
            self.text = None
            yield ag_ui.core.TextMessageEndEvent(message_id=message_id)

    def _end_reasoning(self) -> collections.abc.Iterator[ag_ui.core.BaseEvent]:
        if (message_id := self.reasoning) is not None:
            self.reasoning = None
            yield ag_ui.core.ReasoningMessageEndEvent(message_id=message_id)
            yield ag_ui.core.ReasoningEndEvent(message_id=message_id)


class FrontendTool(fastmcp.tools.Tool):
    """One of a thread's frontend tools, as the agent sees it over MCP."""

    turn: typing.Annotated[
        pydantic.json_schema.SkipJsonSchema[pydantic.InstanceOf[Turn] | None],
        pydantic.Field(exclude=True),
    ]
    pending: typing.Annotated[
        pydantic.json_schema.SkipJsonSchema[pydantic.InstanceOf[dict]],
        pydantic.Field(exclude=True),
    ]

    async def run(self, arguments: dict[str, typing.Any]) -> fastmcp.tools.ToolResult:
        """Show the call in the thread's run, and return what the next run answers."""
        turn = self.turn
        if turn is None or turn.task.done() or turn.cancelled is not None:
            raise fastmcp.exceptions.ToolError("no turn is running to show this call")
        call = ag_ui.core.ToolCall(
            id=str(uuid.uuid4()),
            type="function",
            function=ag_ui.core.FunctionCall(
                name=self.name, arguments=json.dumps(arguments)
            ),
        )
        answer = self.pending[call.id] = asyncio.get_running_loop().create_future()
        try:
            turn.events.put_nowait(call)
            await asyncio.wait({answer, turn.task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            del self.pending[call.id]
        if not answer.done():
            raise fastmcp.exceptions.ToolError(
                "the turn ended before the frontend answered"
            )
        message: ag_ui.core.ToolMessage = answer.result()
        text = message.content
        if not isinstance(text, str):
            text = "".join(p.text for p in text if isinstance(p, ag_ui.core.TextPart))
        return fastmcp.tools.ToolResult(
            content=[mcp_types.TextContent(type="text", text=text)],
            is_error=bool(message.error),
        )


class ThreadTools(fastmcp.server.providers.Provider):
    """The frontend tools of the thread an agent's MCP request names."""

    def __init__(
        self, threads: dict[str, Session], pending: dict[str, asyncio.Future]
    ) -> None:
        super().__init__()
        self.threads = threads
        self.pending = pending

    async def _list_tools(self) -> list[fastmcp.tools.Tool]:
        # Built again for each request, a call's included, so `turn` is current.
        headers = fastmcp.server.dependencies.get_http_headers()
        session = self.threads.get(headers.get(FrontendTools.HEADER.lower(), ""))
        if session is None:
            return []
        return [
            FrontendTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters or {"type": "object"},
                turn=session.turn,
                pending=self.pending,
            )
            for tool in session.tools
        ]


class FrontendTools(fastmcp.FastMCP):
    """Each AG-UI thread's frontend tools, as an MCP server at `url`."""

    HEADER: typing.ClassVar[str] = "X-AG-UI-Thread"
    """Names the AG-UI thread on the agent's requests."""

    def __init__(self, url: str, threads: dict[str, Session]) -> None:
        self.url = url
        self.pending: dict[str, asyncio.Future] = {}
        super().__init__("ag-ui", providers=[ThreadTools(threads, self.pending)])

    def config(self, thread_id: str) -> acp.schema.HttpMcpServer:
        """This server, as named in `thread_id`'s ``session/new``."""
        header = acp.schema.HttpHeader(name=self.HEADER, value=thread_id)
        return acp.schema.HttpMcpServer(
            type="http", name="ag-ui", url=self.url, headers=[header]
        )

    def reply(self, messages: list[ag_ui.core.Message]) -> None:
        """Return the frontend's tool messages to the calls awaiting them."""
        for message in messages:
            if isinstance(message, ag_ui.core.ToolMessage):
                answer = self.pending.get(message.tool_call_id)
                if answer is not None and not answer.done():
                    answer.set_result(message)


@dataclasses.dataclass(eq=False)
class Bridge:
    """An ACP client that serves its agent as an AG-UI agent."""

    capabilities: typing.ClassVar[acp.schema.ClientCapabilities] = (
        acp.schema.ClientCapabilities(
            fs=acp.schema.FileSystemCapabilities(
                read_text_file=True, write_text_file=True
            ),
            elicitation=acp.schema.ElicitationCapabilities(
                form=acp.schema.ElicitationFormCapabilities()
            ),
        )
    )
    """What this client does for the agent: files and forms."""

    cwd: str
    server: FrontendTools | None = None
    """The frontend's tools, named to the agent in each ``session/new``."""

    agent: acp.schema.AgentCapabilities = dataclasses.field(
        default_factory=acp.schema.AgentCapabilities
    )
    threads: dict[str, Session] = dataclasses.field(default_factory=dict)
    sessions: dict[str, Session] = dataclasses.field(default_factory=dict)
    locks: collections.defaultdict[str, asyncio.Lock] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(asyncio.Lock)
    )
    conn: acp.interfaces.Agent = dataclasses.field(init=False, repr=False)

    # -- The ACP client ------------------------------------------------------

    def on_connect(self, conn: acp.interfaces.Agent) -> None:
        self.conn = conn

    async def start(self) -> None:
        """Introduce this client to the agent, and note what its prompts may carry."""
        response = await self.conn.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            client_capabilities=self.capabilities,
            client_info=acp.schema.Implementation(
                name="effectful-ag-ui", version="0.1.0"
            ),
        )
        if response.protocol_version != acp.PROTOCOL_VERSION:
            raise RuntimeError(f"the agent speaks ACP {response.protocol_version}")
        if response.agent_capabilities is not None:
            self.agent = response.agent_capabilities

    async def session_update(
        self, session_id: str, update: typing.Any, **kwargs: typing.Any
    ) -> None:
        # Never awaits: the SDK runs each notification as its own task, so an
        # await here could let a later update overtake this one.
        session = self.sessions.setdefault(session_id, Session(session_id))
        match update:
            case acp.schema.UserMessageChunk():
                pass
            case (
                acp.schema.AgentMessageChunk()
                | acp.schema.AgentThoughtChunk()
                | acp.schema.ToolCallStart()
                | acp.schema.ToolCallProgress()
            ):
                if session.turn is not None:
                    session.turn.events.put_nowait(update)
            case _:
                session.state[update.session_update] = update.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )
                if session.turn is not None:
                    session.turn.events.put_nowait(
                        ag_ui.core.StateSnapshotEvent(snapshot=dict(session.state))
                    )

    async def request_permission(
        self,
        session_id: str,
        tool_call: acp.schema.ToolCallUpdate,
        options: list[acp.schema.PermissionOption],
        **kwargs: typing.Any,
    ) -> acp.schema.RequestPermissionResponse:
        interrupt = ag_ui.core.Interrupt(
            id=str(uuid.uuid4()),
            reason="confirmation",
            message=tool_call.title,
            tool_call_id=tool_call.tool_call_id,
            response_schema=acp.schema.PermissionOption.model_json_schema(),
            metadata={
                "options": pydantic.TypeAdapter(
                    list[acp.schema.PermissionOption]
                ).dump_python(options, mode="json", by_alias=True, exclude_none=True)
            },
        )
        answer = await self._ask(session_id, interrupt, tool_call)
        chosen = (
            answer.payload.get("optionId")
            if answer is not None
            and answer.status == "resolved"
            and isinstance(answer.payload, dict)
            else None
        )
        if chosen in {option.option_id for option in options}:
            return acp.schema.RequestPermissionResponse(
                outcome=acp.schema.AllowedOutcome(outcome="selected", option_id=chosen)
            )
        return acp.schema.RequestPermissionResponse(
            outcome=acp.schema.DeniedOutcome(outcome="cancelled")
        )

    async def create_elicitation(
        self, message: str, mode: typing.Any, **kwargs: typing.Any
    ) -> acp.schema.CreateElicitationResponse:
        if not isinstance(mode, acp.schema.ElicitationFormSessionMode):
            raise acp.RequestError.invalid_params(
                {"reason": "only forms within a session are supported"}
            )
        interrupt = ag_ui.core.Interrupt(
            id=str(uuid.uuid4()),
            reason="input_required",
            message=message,
            tool_call_id=mode.tool_call_id,
            response_schema=mode.requested_schema.model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
        )
        answer = await self._ask(mode.session_id, interrupt)
        if answer is None or answer.status == "cancelled":
            return acp.schema.CancelElicitationResponse(action="cancel")
        if isinstance(answer.payload, dict):
            return acp.schema.AcceptElicitationResponse(
                action="accept", content=answer.payload
            )
        return acp.schema.DeclineElicitationResponse(action="decline")

    async def complete_elicitation(
        self, elicitation_id: str, **kwargs: typing.Any
    ) -> None:
        """Nothing to do: only forms are offered, and they complete by answering."""

    async def ext_method(
        self, method: str, params: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        raise acp.RequestError.method_not_found(f"_{method}")

    async def ext_notification(
        self, method: str, params: dict[str, typing.Any]
    ) -> None:
        """Extensions this client does not know are ignored, as ACP asks."""

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: int | None = None,
        limit: int | None = None,
        **kwargs: typing.Any,
    ) -> acp.schema.ReadTextFileResponse:
        try:
            text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            raise acp.RequestError.resource_not_found(path) from None
        lines = text.splitlines(keepends=True)
        start = max((line or 1) - 1, 0)
        stop = None if limit is None else start + limit
        return acp.schema.ReadTextFileResponse(content="".join(lines[start:stop]))

    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: typing.Any
    ) -> acp.schema.WriteTextFileResponse:
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return acp.schema.WriteTextFileResponse()

    async def _no_terminals(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.NoReturn:
        """Refuse the terminal methods, which this client does not offer."""
        raise acp.RequestError.method_not_found("terminal")

    create_terminal = terminal_output = wait_for_terminal_exit = _no_terminals
    kill_terminal = release_terminal = _no_terminals

    # -- The AG-UI agent -----------------------------------------------------

    async def run(
        self, input: ag_ui.core.RunAgentInput
    ) -> collections.abc.AsyncIterator[ag_ui.core.BaseEvent]:
        """Serve one AG-UI run: the thread's turn up to its next stopping point."""
        yield ag_ui.core.RunStartedEvent(
            thread_id=input.thread_id,
            run_id=input.run_id,
            protocol_version=ag_ui.core.PROTOCOL_VERSION,
        )
        try:
            async with self.locks[input.thread_id]:
                session, turn, resend = await self._begin(input)
        except Exception as error:
            code = "THREAD_BUSY" if isinstance(error, ThreadBusy) else None
            yield ag_ui.core.RunErrorEvent(message=_describe(error), code=code)
            return
        streaming = done = False
        try:
            if session.state:
                yield ag_ui.core.StateSnapshotEvent(snapshot=dict(session.state))
            if turn is None or resend:
                done = True
                yield self._finished(
                    input, interrupts=turn.unanswered() if turn is not None else []
                )
                return
            turn.attach()
            streaming = True
            while True:
                item = await turn.events.get()
                if isinstance(item, ag_ui.core.Interrupt):
                    if turn.interrupts[item.id][1].done():
                        continue
                    # A question is shown only after everything queued before it,
                    # and after the tool call it is about.
                    if item.tool_call_id in turn.calls:
                        for event in turn.announce(item.tool_call_id):
                            yield event
                    for event in turn.close():
                        yield event
                    turn.exposed.add(item.id)
                    done = True
                    yield self._finished(input, interrupts=turn.unanswered())
                    return
                if isinstance(item, ag_ui.core.ToolCall):
                    # A frontend tool call: the frontend runs it, and its next
                    # run carries the result.
                    for event in turn.show(
                        item.id, item.function.name, item.function.arguments
                    ):
                        yield event
                    done = True
                    yield self._finished(input)
                    return
                if isinstance(item, acp.schema.PromptResponse | Exception):
                    for event in turn.finish():
                        yield event
                    if session.turn is turn:
                        session.turn = None
                    done = True
                    if isinstance(item, Exception):
                        yield ag_ui.core.RunErrorEvent(message=_describe(item))
                    else:
                        yield self._finished(input, response=item)
                    return
                for event in turn.translate(item):
                    yield event
        finally:
            session.busy = False
            if streaming and not done and turn is not None:
                # Scheduled rather than awaited: an await in a cancelled request
                # would itself be cancelled.
                self._cancel(turn)

    async def _begin(
        self, input: ag_ui.core.RunAgentInput
    ) -> tuple[Session, Turn | None, bool]:
        """Claim the thread for this run, and decide what the run streams.

        Returns the session, its turn (if any), and whether the run only re-sends
        the questions the frontend has not answered yet.
        """
        session = await self._open(input.thread_id)
        session.tools = list(input.tools or [])
        if session.busy:
            raise ThreadBusy(f"thread {input.thread_id!r} is streaming another run")
        session.busy = True
        try:
            turn = session.turn
            for entry in input.resume or []:
                pending = turn.interrupts.get(entry.interrupt_id) if turn else None
                if (
                    turn is None
                    or pending is None
                    or entry.interrupt_id not in turn.exposed
                ):
                    print(
                        f"warning: ignoring the answer to unknown interrupt "
                        f"{entry.interrupt_id!r}",
                        file=sys.stderr,
                    )
                elif not pending[1].done():
                    pending[1].set_result(entry)
            if self.server is not None:
                self.server.reply(input.messages)
            if turn is not None and turn.unanswered():
                return session, turn, True
            message = input.messages[-1] if input.messages else None
            if (
                not input.resume
                and isinstance(message, ag_ui.core.UserMessage)
                and message.id != session.prompted
            ):
                session.prompted = message.id
                if turn is not None:
                    # A turn with no run streaming it was abandoned or has ended.
                    # Waiting on the task via `asyncio.wait` keeps a cancelled
                    # request from cancelling the prompt it shares.
                    await asyncio.wait({self._cancel(turn), turn.task})
                prompts = self.agent.prompt_capabilities
                blocks = _blocks(message, bool(prompts and prompts.image))
                turn = Turn(session) if blocks else None
                session.turn = turn
                if turn is not None:
                    turn.task = asyncio.create_task(self._prompt(turn, blocks))
            return session, turn, False
        except BaseException:
            session.busy = False
            raise

    async def _open(self, thread_id: str) -> Session:
        """The session serving `thread_id`, created on first use."""
        if (session := self.threads.get(thread_id)) is None:
            servers: list[typing.Any] = []
            transports = self.agent.mcp_capabilities
            if self.server is not None and transports and transports.http:
                servers.append(self.server.config(thread_id))
            response = await self.conn.new_session(cwd=self.cwd, mcp_servers=servers)
            # `setdefault`, because updates sent while the session was being made
            # have already created it.
            session = self.sessions.setdefault(
                response.session_id, Session(response.session_id)
            )
            self.threads[thread_id] = session
        return session

    async def _ask(
        self,
        session_id: str,
        interrupt: ag_ui.core.Interrupt,
        tool_call: acp.schema.ToolCallUpdate | None = None,
    ) -> ag_ui.core.ResumeEntry | None:
        """Ask the frontend `interrupt` and wait for its answer; `None` if cancelled."""
        session = self.sessions.get(session_id)
        turn = session.turn if session is not None else None
        if turn is None or turn.task.done():
            return None
        if turn.cancelled is not None:
            # So the agent hears `session/cancel` before this answer.
            await asyncio.wait({turn.cancelled})
            return None
        answer = asyncio.get_running_loop().create_future()
        turn.interrupts[interrupt.id] = (interrupt, answer)
        if tool_call is not None:
            turn.events.put_nowait(tool_call)
        turn.events.put_nowait(interrupt)
        return await answer

    async def _prompt(self, turn: Turn, blocks: list[typing.Any]) -> None:
        """Send the turn's prompt, and queue how it ended."""
        outcome: acp.schema.PromptResponse | Exception
        try:
            outcome = await self.conn.prompt(session_id=turn.session.id, prompt=blocks)
        except Exception as error:
            outcome = error
        for _, answer in turn.interrupts.values():
            if not answer.done():
                answer.set_result(None)
        # Last in the queue: the SDK handles every update of the turn before
        # `prompt` returns, and `session_update` queues them without waiting.
        turn.events.put_nowait(outcome)

    def _cancel(self, turn: Turn) -> asyncio.Task:
        """Cancel `turn` once: ``session/cancel`` first, then answer its questions."""

        async def cancel() -> None:
            if not turn.task.done():
                with contextlib.suppress(Exception):
                    await self.conn.cancel(session_id=turn.session.id)
            for _, answer in turn.interrupts.values():
                if not answer.done():
                    answer.set_result(None)

        if turn.cancelled is None:
            turn.cancelled = asyncio.ensure_future(cancel())
        return turn.cancelled

    def _finished(
        self,
        input: ag_ui.core.RunAgentInput,
        *,
        interrupts: collections.abc.Sequence[ag_ui.core.Interrupt] = (),
        response: acp.schema.PromptResponse | None = None,
    ) -> ag_ui.core.RunFinishedEvent:
        """The event ending a run, in the protocol version the client speaks."""
        outcome: (
            ag_ui.core.RunFinishedInterruptOutcome
            | ag_ui.core.RunFinishedCancelledOutcome
            | None
        ) = None
        result: typing.Any = None
        if interrupts:
            outcome = ag_ui.core.RunFinishedInterruptOutcome(
                interrupts=list(interrupts)
            )
        elif response is not None and response.stop_reason != "cancelled":
            result = response.model_dump(mode="json", by_alias=True, exclude_none=True)
        elif response is not None and input.protocol_version is not None:
            outcome = ag_ui.core.RunFinishedCancelledOutcome()
        elif response is not None:
            # A client from before AG-UI versioning rejects the cancelled outcome,
            # and CopilotKit drops a run's stream at the first event it rejects.
            print(
                "warning: this client predates the cancelled outcome, so the stopped "
                "run is reported as finished with result.stopReason 'cancelled'",
                file=sys.stderr,
            )
            result = {"stopReason": "cancelled"}
        return ag_ui.core.RunFinishedEvent(
            thread_id=input.thread_id,
            run_id=input.run_id,
            outcome=outcome,
            result=result,
        )


def _blocks(message: ag_ui.core.UserMessage, images: bool) -> list[typing.Any]:
    """A user message as prompt blocks, skipping the parts the agent cannot take."""
    parts = (
        [ag_ui.core.TextPart(text=message.content)]
        if isinstance(message.content, str)
        else message.content
    )
    blocks: list[typing.Any] = []
    for part in parts:
        match part:
            case ag_ui.core.TextPart(text=text):
                if text:
                    blocks.append(acp.text_block(text))
            case ag_ui.core.ImagePart(
                source=ag_ui.core.DataSource(value=data, mime_type=mime)
            ) if images:
                blocks.append(acp.image_block(data, mime))
            case _:
                print(
                    f"warning: skipping a {part.type} part the agent cannot take",
                    file=sys.stderr,
                )
    return blocks


def _render(call: dict[str, typing.Any]) -> str:
    """A tool call's result as text: its text and diffs, else its raw output."""
    parts: list[str] = []
    for item in call.get("content") or []:
        match item:
            case {"type": "content", "content": {"type": "text", "text": text}}:
                parts.append(text)
            case {"type": "diff", "path": path, "new_text": new}:
                old = item.get("old_text") or ""
                diff = difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    path,
                    path,
                )
                parts.append("".join(diff))
    if not parts and call.get("raw_output") is not None:
        parts.append(json.dumps(call["raw_output"], default=str))
    return "\n".join(parts)


def _describe(error: BaseException) -> str:
    """An error as a message for the frontend, with a JSON-RPC error's details."""
    if isinstance(error, acp.RequestError) and error.data:
        return f"{error}: {json.dumps(error.data, default=str)}"
    return str(error) or type(error).__name__


def make_app(bridge: Bridge) -> fastapi.FastAPI:
    """AG-UI at ``/`` and the frontend's tools at ``/mcp``, for callers on this machine."""
    tools = (
        bridge.server.http_app(path="/mcp", stateless_http=True)
        if bridge.server
        else None
    )
    app = fastapi.FastAPI(lifespan=tools.lifespan if tools else None)
    # A page whose domain was rebound to this machine is same-origin, so the browser
    # lets it call; its Host header gives it away.
    app.add_middleware(
        starlette.middleware.trustedhost.TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1"],
    )

    @app.post("/")
    async def run(
        input: ag_ui.core.RunAgentInput,
    ) -> fastapi.responses.StreamingResponse:
        encoder = ag_ui.encoder.EventEncoder()
        return fastapi.responses.StreamingResponse(
            (encoder.encode(event) async for event in bridge.run(input)),
            media_type=encoder.get_content_type(),
        )

    if tools is not None:
        app.mount("/", tools)
    return app


async def serve(bridge: Bridge, command: list[str], port: int) -> None:
    """Run the agent, and serve it until either stops."""
    async with acp.spawn_agent_process(
        bridge,
        *command,
        env=dict(os.environ),
        transport_kwargs={"stderr": None},
        use_unstable_protocol=True,
    ) as (_, process):
        bridge.server = FrontendTools(f"http://127.0.0.1:{port}/mcp", bridge.threads)
        await bridge.start()
        config = uvicorn.Config(
            make_app(bridge), host="127.0.0.1", port=port, timeout_graceful_shutdown=1
        )
        server = uvicorn.Server(config)
        serving = asyncio.create_task(server.serve())
        exited = asyncio.create_task(process.wait())
        await asyncio.wait({serving, exited}, return_when=asyncio.FIRST_COMPLETED)
        server.should_exit = True
        await serving
        exited.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8000, help="port on 127.0.0.1")
    parser.add_argument(
        "--cwd",
        type=os.path.abspath,
        default=os.getcwd(),
        help="the directory the agent works in",
    )
    parser.add_argument("agent", nargs="+", help="the agent's command line, after --")
    args = parser.parse_args()
    asyncio.run(serve(Bridge(args.cwd), args.agent, args.port))


if __name__ == "__main__":
    main()
