"""The long-lived half of the ACP server: the connection, the sessions and their state.

`library.py` holds everything built from the harness -- the editor tools, the handlers
that translate between the harness and the protocol, the slash commands. This module
holds what has to outlive an edit to any of that: the connection, `EffectfulACPAgent`
and each `ACPSession`, and reading a prompt, which has to agree with the capabilities
advertised once per connection. It imports nothing from the harness or from `library`
at its top, and reaches `library` through `_library` on every use, which is what lets
the launcher's ``--autoreload`` re-run all of that code while the editor stays
connected; this module itself is kept, and an edit to it needs ``/restart``.

## Threads

The harness is synchronous -- `litellm.completion` blocks, and the completion loop is
a `while` -- while ACP is asyncio. So a prompt runs in a worker thread
(`asyncio.to_thread`), and `ACPSession` is the only thing that crosses back:

- **Notifications** are enqueued (`ACPSession.notify`) and drained by one writer task
  per session, so they cannot interleave on the wire.
- **Requests** block the worker until the editor answers (`ACPSession.call`).

`asyncio.to_thread` copies the caller's `contextvars`, and the handler stack lives in
one (`effectful.internals.runtime.INTERPRETATION`), so the worker inherits whatever
stack was installed around the server and adds this session's handlers on top.

## What the protocol asks for that an effect does not supply

The three translations are most of it, but not all: ACP also has requirements about
the *conversation*, which `EffectfulACPAgent` is where it meets. The ones with teeth,
each of which has a test:

- A session is opened on a directory (`cwd`), and everything in it -- what the model
  is told it is working on, where a terminal command runs -- is rooted there rather
  than in whatever directory the editor happened to spawn this process from.
- Every prompt is answered with a `StopReason`, and the interesting ones are not
  `end_turn`: a reply cut off at the token limit, one the provider refused, one the
  user cancelled.
- Whatever the turn told the editor, it told it *before* answering; and no tool call
  it announced is left without a terminal status.
- A capability is claimed if and only if it is implemented, in both directions --
  including `session/list`, which is claimed only when there is somewhere to keep
  the answer, since a conversation the agent cannot name is one it cannot reopen.
"""

import asyncio
import base64
import collections.abc
import concurrent.futures
import contextlib
import dataclasses
import datetime
import importlib
import inspect
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import traceback
import typing
import urllib.parse
import urllib.request
import uuid

import acp
import acp.connection
import acp.interfaces
import acp.schema
import fastmcp
import pydantic
from PIL import Image

from effectful.handlers.llm.harness import autoreload
from effectful.ops.semantics import handler
from effectful.ops.types import Interpretation

if typing.TYPE_CHECKING:
    from effectful.handlers.llm import Agent

# This module holds the running server, so an edit to it needs `/restart`.
autoreload.keep(__name__)

LIBRARY = "library"
"""The module holding the reloadable half of the server; see `_library`."""


class _Reporter(typing.Protocol):
    """What `EffectfulACPAgent.prompt` asks of a session's reporter."""

    def begin_turn(self) -> None: ...
    def abandon(self) -> None: ...
    def stop_reason(self) -> acp.schema.StopReason: ...
    def usage(self) -> acp.schema.Usage | None: ...


class _Command(typing.Protocol):
    """What announcing and dispatching a slash command ask of it."""

    @property
    def spec(self) -> acp.schema.AvailableCommand: ...

    @property
    def run(
        self,
    ) -> collections.abc.Callable[["EffectfulACPAgent", "ACPSession", str], str]: ...


class _Reloader(typing.Protocol):
    """What `EffectfulACPAgent._on_reload` asks of the launcher's reloader."""

    def current_class(self, cls: type, /) -> type: ...

    def refresh(self, agent: object, /) -> None: ...


class _Library(typing.Protocol):
    """Everything this module takes from `library`, and so the line between them.

    `library` may change while the server runs; this may not. Each name here is
    re-read on every use and never kept across a reload, which is what makes a
    live edit to what is *behind* it safe:

    * `SESSION_MODES` -- announced when a session opens and again after every
      reload (`EffectfulACPAgent._announce_config`), which also moves a session off
      a mode no longer offered.
    * `session_handlers` -- the handlers a session's turns run under, and the
      reporter among them, rebuilt for every session on reload.
    * `slash_commands` -- announced when a session opens and after every reload,
      and looked up on each dispatch.
    * `message_as_text` -- turns a message into notification text, on each
      ``session/load``.

    Changing *this* -- a name, a signature, the shape of a command or a reporter --
    needs a restart, since the code here that relies on it is not re-run.
    Checked statically against `library` below, and at run time, by name only,
    in `_library`.
    """

    SESSION_MODES: tuple[acp.schema.SessionMode, ...]

    def session_handlers(
        self, session: "ACPSession", /
    ) -> tuple[_Reporter, Interpretation]: ...

    def slash_commands(self) -> collections.abc.Mapping[str, _Command]: ...

    def tool_kind(self, name: str) -> acp.schema.ToolKind | None: ...

    def message_as_text(self, message: typing.Any) -> str: ...


def _library() -> _Library:
    """`library` as it now is, looked up on each use rather than imported.

    So that ``--autoreload`` can re-run `library` while this module, which holds
    the live server, keeps running; see `_Library` for what may change that way.

    Raises:
        TypeError: If `library` no longer defines every name `_Library` lists.
    """
    library = importlib.import_module(LIBRARY)
    # `hasattr` rather than a runtime-checkable `isinstance`, which reads attributes
    # statically and so cannot see the names of a module hmr has re-run.
    _LIBRARY_NAMES = sorted(
        {*_Library.__annotations__}
        | {
            n
            for n, v in vars(_Library).items()
            if inspect.isfunction(v) and n[0] != "_"
        }
    )
    if missing := [name for name in _LIBRARY_NAMES if not hasattr(library, name)]:
        raise TypeError(f"`{LIBRARY}` no longer defines {', '.join(missing)}")
    return typing.cast(_Library, library)


if typing.TYPE_CHECKING:
    # Relative, because the type checker sees this directory as a package and would
    # not find a bare `library`, and `ignore_missing_imports` would make that `Any`
    # -- a check that passes whatever `library` says. Never run: `library` imports
    # this module.
    from . import library as _checked_library

    _: _Library = _checked_library


type ContentBlock = (
    acp.schema.TextContentBlock
    | acp.schema.ImageContentBlock
    | acp.schema.AudioContentBlock
    | acp.schema.ResourceContentBlock
    | acp.schema.EmbeddedResourceContentBlock
)


type ConfigOption = (
    acp.schema.SessionConfigOptionSelect | acp.schema.SessionConfigOptionBoolean
)


# ---------------------------------------------------------------------------
# The knobs a session offers the user
#
# Two protocol features that cost an agent almost nothing and that an editor
# renders for free, both of which stay dead until the agent describes itself:
# a *mode* (`session/set_mode`) and a *config option* (`session/set_config_option`).
# Only `select` options are used, deliberately -- a boolean one is gated behind
# the client's `session.configOptions.boolean` capability, and most clients,
# including VS Code's, advertise no session capabilities at all. The modes
# themselves are `library.SESSION_MODES`, so an edit to them reaches open sessions.
# ---------------------------------------------------------------------------

MODE_OPTION_ID = "mode"
MODEL_OPTION_ID = "model"
THOUGHT_LEVEL_OPTION_ID = "thought_level"
INHERIT_MODEL = ""
"""The `model` option's value meaning "whatever the process was configured with".

An empty string rather than the model's name, because this agent does not know that
name: the model is bound into `LiteLLMConfigurer` by whoever assembled the stack, and
nothing in the protocol layer can see it. Saying "as configured" is honest; naming a
model here would be a guess printed in the user's editor.
"""

INHERIT_THOUGHT_LEVEL = ""
"""The `thought_level` option's value meaning "whatever was configured".

Sentinel rather than a level name because every explicit `reasoning_effort` value
is one some provider rejects -- OpenAI answers `"default"` with a 400 -- so "let
the model decide" is spelled by omitting the parameter, which is what an empty
value does in `ACPSessionConfig.completion`. This matches the launcher, whose
unset `--reasoning-effort` is likewise left out of the request entirely.
"""


def _thought_levels() -> tuple[str, ...] | None:
    """The `reasoning_effort` values litellm says providers accept.

    Read from litellm's canonical ``REASONING_EFFORT`` alias -- not the looser
    `Literal` on `litellm.completion`'s signature, which also admits `"default"`,
    a value OpenAI rejects -- so the option's choices track litellm across
    upgrades, the same way the launcher's `--reasoning-effort` does.

    Returns `None` when the alias cannot be read, which takes the option off
    offer entirely: a control whose every choice errors is worse than no control.
    """
    try:
        from litellm.types.llms.openai import REASONING_EFFORT

        levels = tuple(
            v for v in typing.get_args(REASONING_EFFORT) if isinstance(v, str)
        )
        return levels or None
    except Exception:
        return None


OFFER_MODELS_ENV = "ACP_OFFER_MODELS"
"""Environment variable naming the models the editor's picker should offer.

An environment variable rather than a flag because an editor launches an agent with a
command and an environment, and the model this one *starts* on already comes from the
environment -- the launcher's ``--model`` defaults to ``EFFECTFUL_LLM_MODEL``. Putting
the models it can switch to anywhere else would split one setting across two
mechanisms in the same block of the editor's configuration.
"""


def _offered_models() -> tuple[str, ...]:
    """The picker's models as named in the environment, in order, or none.

    Comma-separated, since a model name may contain ``/``, ``-``, ``.`` and ``:`` but
    never a comma. Blanks are dropped rather than offered: a picker whose list has an
    empty entry in it is worse than no picker, and a trailing comma is the likeliest
    way to write one by accident.
    """
    listed = os.environ.get(OFFER_MODELS_ENV, "").split(",")
    return tuple(model.strip() for model in listed if model.strip())


RESTART_STATE_ENV = "EFFECTFUL_ACP_RESTART_STATE"
"""Names the file a server leaves for the process replacing it; see `restart`."""

FLUSH_TIMEOUT = 5.0
"""How long a turn waits for its queued updates to reach the editor before answering.

A courtesy wait, not a correctness one: ACP requires the updates to be *sent* before
the final response, and this is what makes that a wait rather than a hope. But an
editor that has stopped reading its own pipe should not be able to wedge the turn
trying to tell it so, which is the whole reason for the bound. Nothing a user waits
on is measured by it, so it is short and fixed rather than configurable.
"""


# ---------------------------------------------------------------------------
# Per-session state, and the crossing between the event loop and the harness
# ---------------------------------------------------------------------------


class SessionCancelled(BaseException):
    """Raised in the worker thread when the editor sends ``session/cancel``.

    Raising is how a cancellation noticed deep in the completion loop -- inside a
    tool, between stream chunks, while waiting on the editor -- reaches
    `EffectfulACPAgent.prompt`, which is the only place that may answer the request.

    Deriving from `BaseException` rather than `Exception` is load-bearing.
    `~effectful.handlers.llm.harness.durability.retrying.TenacityRetryer` catches
    `Exception`-derived tool failures and hands them to the model as feedback, and
    retries `Exception`-derived completion failures, so a cancellation raised inside
    the loop would otherwise be swallowed and reported to the model as a broken tool
    rather than stopping the turn.
    """


@dataclasses.dataclass
class ACPSession[A: "Agent"]:
    """Everything the server keeps for one ACP session.

    ACP has no such object -- it addresses sessions by id and leaves the rest to the
    agent -- so this is where the per-session state lives:

    * the `Agent` whose history *is* the conversation, and whose ``__agent_id__`` is
      the session id the editor knows it by;
    * the binding to the editor (`client`, `client_capabilities`), which every call
      back to it needs;
    * the directories the session is *about* (`cwd`, `additional_directories`), which
      ACP calls its root set;
    * the lock that keeps two prompts on one session from running at once (`lock`),
      since they would put two worker threads on one agent's history;
    * the queue of pending notifications and the task draining it (`updates`,
      `notify`, `writer`, `drain`).

    Most of the methods serve the second, because the harness is synchronous and the
    protocol is not: a prompt runs in a worker thread while the connection lives on
    the event loop, so every call back to the editor crosses a thread boundary, and
    that crossing is written here once.

    Construct it on the event loop thread: it captures the running loop and starts
    the writer task on it.
    """

    agent: A
    client: acp.interfaces.Client
    client_capabilities: acp.schema.ClientCapabilities

    cwd: str = ""
    """The session's working directory, as an absolute path.

    ACP requires it of every ``session/new`` and ``session/load``: it "MUST be used
    for the session regardless of where the Agent subprocess was spawned", and is the
    base every relative path in the session resolves against. The agent's own process
    directory is whatever the editor happened to be launched from and is never it,
    which is why nothing here consults `os.getcwd`.
    """

    additional_directories: tuple[str, ...] = ()
    """Further absolute paths the session may work in, beyond `cwd`."""

    mode_id: str = dataclasses.field(
        default_factory=lambda: _library().SESSION_MODES[0].id
    )
    """Which of `library.SESSION_MODES` the user has picked. Read by
    `ACPPermissionGate`."""

    model: str = INHERIT_MODEL
    """The model the user picked, or `INHERIT_MODEL` for the configured one."""

    thought_level: str | None = None
    """The reasoning effort the user picked, sent as litellm `reasoning_effort`.

    `None` inherits: the launcher's `--reasoning-effort` when it set one, else the
    provider's own default. Distinguished from `INHERIT_THOUGHT_LEVEL` (the empty
    option value), which normalizes to `None` wherever the option is read.
    """

    title: str = ""
    """A human-readable name for the conversation, taken from its first prompt."""

    poll_interval: float = 0.1
    """How often a worker thread waiting on the editor re-reads `cancel`."""

    mcp_servers: list[typing.Any] = dataclasses.field(default_factory=list)
    """The MCP servers the editor gave this session, as ACP specs."""

    mcp_client: fastmcp.Client | None = None
    """One client for all of `mcp_servers`, connected while the session is open."""

    mcp_client_config: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    """The FastMCP configuration `mcp_client` was built from."""

    mcp_connecting: asyncio.Task | None = None
    """A connection started outside a request, which the next turn waits for."""

    loop: asyncio.AbstractEventLoop = dataclasses.field(
        default_factory=asyncio.get_running_loop
    )
    cancel: threading.Event = dataclasses.field(default_factory=threading.Event)
    updates: asyncio.Queue = dataclasses.field(default_factory=asyncio.Queue)
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    writer: asyncio.Task | None = None
    reporter: typing.Any = dataclasses.field(init=False, default=None)
    intp: Interpretation = dataclasses.field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Start the one task that drains `updates`.

        It belongs to the session's whole life rather than to a turn: `load_session`
        replays a conversation with no turn in progress, and a turn's own last act is
        to wait for the queue to empty. Starting it per turn would make the first of
        those deliver nothing and the second wait on a queue nobody is reading.
        """
        self.writer = self.loop.create_task(self.drain())
        self.install_handlers()

    @property
    def session_id(self) -> str:
        """The ACP session id this object represents."""
        return self.agent.__agent_id__

    @property
    def fs_capabilities(self) -> acp.schema.FileSystemCapabilities:
        """What the editor will do to files on the agent's behalf.

        ``fs`` is optional in `acp.schema.ClientCapabilities`, and a client that sends
        it as ``null`` means the same thing as one that claims nothing: reading it
        through here answers "no" rather than raising `AttributeError` inside whatever
        tool happened to ask.
        """
        return self.client_capabilities.fs or acp.schema.FileSystemCapabilities()

    @property
    def elicitation_capabilities(self) -> acp.schema.ElicitationCapabilities:
        """Which kinds of question the editor will put to the user for us.

        Optional exactly as `fs` is, and read through here for the same reason: a
        client that sends it as ``null`` means the same thing as one that claims
        nothing, and `acp_ask_user` should be told "no" rather than meet an
        `AttributeError`.

        There is no agent-side capability to match this one, so nothing in
        `EffectfulACPAgent.agent_capabilities` changes: elicitation is something the
        *client* offers, and the claim-iff-implemented rule applies to it only
        inbound -- ask before asking, and take no for an answer.
        """
        return (
            self.client_capabilities.elicitation or acp.schema.ElicitationCapabilities()
        )

    @property
    def roots(self) -> tuple[str, ...]:
        """Every directory this session may work in, `cwd` first.

        ACP calls this the session's effective root set, and requires `cwd` to be part
        of it -- so it is derived here rather than stored, and cannot fall out of step
        with `cwd`.
        """
        return (self.cwd, *self.additional_directories) if self.cwd else ()

    def install_handlers(self) -> None:
        """Build this session's handlers from `library` as it now is.

        `intp` is the stack a prompt on this session runs under, added on top of the
        harness (see `EffectfulACPAgent._answer`); `reporter` is the one handler in it
        that outlives a call, which `EffectfulACPAgent.prompt` reads once the worker is
        done. Called when the session opens and again by `EffectfulACPAgent._on_reload`.
        """
        self.reporter, self.intp = _library().session_handlers(self)

    def notify(self, update: typing.Any) -> None:
        """Enqueue a `session/update` for the writer task. Never blocks.

        Ordering is the reason for the queue. Scheduling each notification as its own
        coroutine would let two of them interleave inside the connection's writer;
        one consumer draining a queue keeps them in the order they were produced.

        The enqueue is immediate when this is already the loop's own thread, and only
        deferred when it is not. `call_soon_threadsafe` unconditionally would defer it
        past the caller's own next await, so a caller that notifies and then waits for
        the queue to empty -- `load_session` does exactly that -- would find an empty
        queue and return before anything had been put on it.
        """
        try:
            on_loop = asyncio.get_running_loop() is self.loop
        except RuntimeError:  # no running loop: definitely a worker thread
            on_loop = False
        if on_loop:
            self.updates.put_nowait(update)
        else:
            self.loop.call_soon_threadsafe(self.updates.put_nowait, update)

    def call[T](
        self,
        coro: typing.Coroutine[typing.Any, typing.Any, T],
        *,
        orphan: collections.abc.Callable[[concurrent.futures.Future], None]
        | None = None,
    ) -> T:
        """Run a client request on the event loop and wait for the editor's answer.

        Used for the requests whose *answer* the worker needs -- a permission
        decision, a file's contents -- as opposed to the notifications above.

        The wait is unbounded and interruptible, which is the pairing the protocol
        asks for. Unbounded because the thing most often waited on here is a human:
        ``session/request_permission`` puts a dialog in front of the user and there is
        no honest deadline for reading it. A bound would eventually tell the model the
        editor never answered while the dialog was still on screen, and then refuse
        the call out from under the user about to approve it. Every other ACP agent
        simply waits, and so does this one.

        Interruptible is what makes waiting forever safe, and it is not optional. A
        plain ``.result()`` parks the worker thread with no way back: that thread holds
        the session lock, so every later prompt blocks behind it, and the turn never
        reaches the points where it reads `cancel` -- before a completion, before a
        tool call, between stream chunks. `session/cancel` would be a lie in exactly
        the case a user most wants it, a prompt they cannot or will not answer.
        Waiting in short slices and re-reading the flag is what makes it true, and it
        leaves the user, rather than a clock, deciding when a silent editor has waited
        long enough.

        Cancellation normally also cancels the request itself -- the polite thing
        for a permission dialog the user has just walked away from. ``orphan`` is
        for the requests where that would *lose* something: a cancelled
        ``terminal/create`` was already sent, the editor allocates the terminal
        and answers, and an agent that cancelled the answer has leaked a terminal
        it never learned the id of. With ``orphan`` given, cancellation leaves the
        request running and attaches the callback to its eventual completion, so
        the caller can dispose of whatever the answer turns out to be.

        Raises:
            SessionCancelled: If the turn was cancelled while waiting.
        """
        return self.wait(
            asyncio.run_coroutine_threadsafe(coro, self.loop), orphan=orphan
        )

    def wait[T](
        self,
        future: concurrent.futures.Future[T],
        *,
        orphan: collections.abc.Callable[[concurrent.futures.Future], None]
        | None = None,
    ) -> T:
        """Wait for `future` as `call` does, abandoning it if the turn is cancelled.

        Raises:
            SessionCancelled: If the turn was cancelled while waiting.
        """
        while True:
            if self.cancel.is_set():
                if orphan is None:
                    future.cancel()
                else:
                    future.add_done_callback(orphan)
                raise SessionCancelled
            with contextlib.suppress(concurrent.futures.TimeoutError):
                return future.result(timeout=self.poll_interval)

    def detach(
        self, coro: typing.Coroutine[typing.Any, typing.Any, typing.Any]
    ) -> None:
        """Send a client request without waiting for -- or ever cancelling -- it.

        For requests that are obligations rather than questions: the answer is
        not needed and the sending must not depend on the turn's fate. Releasing
        a terminal is the canonical case -- it belongs to *whichever* way the
        turn ends, so tying it to an interruptible wait would let the very
        cancellation that ends a command also revoke the release it owes.
        """
        with contextlib.suppress(RuntimeError):  # a loop already shut down
            asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def connect_mcp(self, servers: collections.abc.Sequence[typing.Any]) -> None:
        """Connect to `servers`, replacing any this session had.

        One FastMCP client serves them all, and skips a server that fails to
        connect. Failures are reported on stderr, not to the editor.
        """
        servers = _valid_mcp(servers)
        config = _mcp_config(servers, self.cwd)
        if self.mcp_client is not None and config == self.mcp_client_config:
            self.mcp_servers = servers
            return
        await self.close_mcp()
        self.mcp_servers, self.mcp_client_config = servers, config
        if config["mcpServers"]:
            client = fastmcp.Client(config)
            try:
                await client.__aenter__()
            except Exception:
                print(
                    f"could not connect to MCP servers {list(config['mcpServers'])}:"
                    f"\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                # Closing re-raises the connection's error.
                with contextlib.suppress(Exception):
                    await client.close()
            else:
                self.mcp_client = client
        self.install_handlers()

    async def close_mcp(self, *, force: bool = False) -> None:
        """Release this session's MCP client, if it has one.

        A running turn keeps the connection until it ends, unless `force`.
        """
        client, self.mcp_client = self.mcp_client, None
        if client is None:
            return
        with contextlib.suppress(Exception):
            await client.__aexit__(None, None, None)
        if force or not client.is_connected():
            with contextlib.suppress(Exception):
                await client.close()

    async def flush(self) -> None:
        """Wait until every notification produced so far has reached the editor.

        ACP requires this of both requests that report progress: an agent "MAY send
        update notifications before responding, but MUST do so before the final
        response", and `session/load` must finish streaming the conversation before it
        answers. The queue and its writer are what make that a wait rather than a
        guarantee, so the wait is written once and used by both.

        Bounded by `FLUSH_TIMEOUT`, because it is a *courtesy* wait: the alternative to
        answering a little early is never answering at all, and an editor that has
        stopped reading its own pipe should not be able to wedge the turn that is
        trying to tell it so. Unlike `call`, nobody is being waited *for* here -- the
        updates have already been produced -- so a bound costs the user nothing.
        """
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.updates.join(), timeout=FLUSH_TIMEOUT)

    async def drain(self) -> None:
        """Deliver queued notifications until cancelled. One task per session.

        ``task_done`` is what lets ``updates.join()`` return, and a turn ends by
        awaiting exactly that, so it has to happen even for a notification that failed
        to send -- otherwise one dropped update would deadlock the end of that turn
        and every turn after it.
        """
        while True:
            update = await self.updates.get()
            try:
                await self.client.session_update(self.session_id, update)
            except Exception:
                pass
            finally:
                self.updates.task_done()


_MCP_SERVER: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
    acp.schema.HttpMcpServer
    | acp.schema.SseMcpServer
    | acp.schema.AcpMcpServer
    | acp.schema.McpServerStdio
)


def _valid_mcp(servers: collections.abc.Sequence[typing.Any]) -> list[typing.Any]:
    """ACP MCP server specs, validated, with malformed ones skipped with a note."""
    valid = []
    for server in servers:
        try:
            valid.append(_MCP_SERVER.validate_python(server))
        except pydantic.ValidationError as error:
            print(f"note: skipping malformed MCP server:\n{error}", file=sys.stderr)
    return valid


def _dump_mcp(servers: collections.abc.Sequence[typing.Any]) -> list[typing.Any]:
    """ACP MCP server specs as JSON, for comparing and saving them."""
    return [
        server.model_dump(mode="json", by_alias=True, exclude_none=True)
        for server in servers
    ]


def _mcp_config(
    servers: collections.abc.Sequence[typing.Any], cwd: str
) -> dict[str, typing.Any]:
    """A FastMCP ``mcpServers`` configuration for ACP MCP server specs.

    Stdio servers run in the session's `cwd`. MCP-over-ACP servers are skipped.
    """
    config: dict[str, typing.Any] = {}
    for server in servers:
        if isinstance(server, acp.schema.McpServerStdio):
            config[server.name] = {
                "command": server.command,
                "args": list(server.args),
                "env": {variable.name: variable.value for variable in server.env},
                "cwd": cwd or None,
            }
        elif isinstance(server, acp.schema.HttpMcpServer | acp.schema.SseMcpServer):
            config[server.name] = {
                "url": server.url,
                "headers": {header.name: header.value for header in server.headers},
                "transport": server.type,
            }
        else:
            print(
                f"note: skipping MCP server {server.name!r}: its transport is not "
                f"supported",
                file=sys.stderr,
            )
    return {"mcpServers": config}


class SessionIndex:
    """The sessions this agent knows of, and enough about each one to list it.

    ACP lets a client ask the agent what conversations it has (`session/list`), which
    is how an editor fills a session picker that survives a restart. Answering needs
    more than the agent histories `SQLitePersister` already keeps: `SessionInfo`
    requires the `cwd` a session was opened on, and a useful listing wants a title and
    a time. That is what this table holds.

    It lives *in the persistence database* rather than a file of its own, and exists
    only when persistence does. Both follow from the same observation: a session
    listed here whose history is not there would be an entry for a conversation that
    cannot be reopened. So when no persistence handler is installed there is no index,
    `session/list` is not advertised, and it is not answered -- which is deliberately
    not the same as answering "no sessions". A client reconciles its own history
    against this reply (VS Code's calls `reconcileFromAgent` with the ids it gets
    back), so an empty answer tells it to forget every session it knew about.
    """

    SCHEMA: typing.ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS acp_sessions (
            session_id             TEXT PRIMARY KEY,
            cwd                    TEXT NOT NULL,
            additional_directories TEXT NOT NULL DEFAULT '[]',
            title                  TEXT,
            updated_at             TEXT NOT NULL,
            mode                   TEXT,
            model                  TEXT,
            thought_level          TEXT
        )
    """

    @classmethod
    def open(cls) -> sqlite3.Connection | None:
        """A connection to the index, or `None` if nothing is persisting anything.

        `SQLitePersister` hands back a fresh connection per call and says that is what
        makes it safe from any thread, so this does not hold one.
        """
        from effectful.handlers.llm.harness.durability.persistence import (
            SQLitePersister,
        )

        conn = SQLitePersister._checkpoint_connection()
        if conn is not None:
            with conn:
                conn.execute(cls.SCHEMA)
                # A table made before these columns existed does not get them from
                # ``IF NOT EXISTS``.
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(acp_sessions)")
                }
                for column in ("mode", "model", "thought_level"):
                    if column not in columns:
                        try:
                            conn.execute(
                                f"ALTER TABLE acp_sessions ADD COLUMN {column} TEXT"
                            )
                        except sqlite3.OperationalError as e:
                            # Another process upgraded the table in the meantime.
                            if "duplicate column" not in str(e):
                                raise
        return conn

    @classmethod
    def available(cls) -> bool:
        """Whether there is an index to answer from. Decides what is advertised."""
        return cls.open() is not None

    @classmethod
    def record(cls, session: ACPSession) -> None:
        """Note that this session exists, where it is rooted, and that it just moved.

        Called whenever a session is opened or answers a prompt, so `updated_at`
        orders the listing by when each conversation was last used -- which is the
        order a session picker wants.
        """
        conn = cls.open()
        if conn is None:
            return
        with conn:
            conn.execute(
                """
                INSERT INTO acp_sessions
                    (session_id, cwd, additional_directories, title, updated_at,
                     mode, model, thought_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    cwd = excluded.cwd,
                    additional_directories = excluded.additional_directories,
                    title = COALESCE(excluded.title, acp_sessions.title),
                    updated_at = excluded.updated_at,
                    mode = excluded.mode,
                    model = excluded.model,
                    thought_level = excluded.thought_level
                """,
                (
                    session.session_id,
                    session.cwd,
                    json.dumps(list(session.additional_directories)),
                    session.title or None,
                    datetime.datetime.now(datetime.UTC).isoformat(),
                    session.mode_id,
                    session.model,
                    session.thought_level,
                ),
            )

    @classmethod
    def get(cls, session_id: str) -> dict[str, typing.Any] | None:
        """What was last recorded about `session_id`, or `None`."""
        conn = cls.open()
        if conn is None:
            return None
        row = conn.execute(
            "SELECT cwd, additional_directories, title, mode, model, thought_level "
            "FROM acp_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        cwd, directories, title, mode, model, thought_level = row
        return {
            "cwd": cwd,
            "additional_directories": json.loads(directories),
            "title": title,
            "mode": mode,
            "model": model,
            "thought_level": thought_level,
        }

    @classmethod
    def page(
        cls, cwd: str | None, cursor: str | None, limit: int
    ) -> tuple[list[acp.schema.SessionInfo], str | None]:
        """One page of sessions, newest first, and the cursor for the next.

        Paged by key rather than by offset: the cursor names the last row handed out,
        so a session that is written while the user is paging cannot push a row across
        a page boundary and hide it. The cursor is opaque to the client, which is why
        it can be this -- the two ordering columns, joined.

        Raises:
            RequestError: If there is no index to read (see the class docstring).
        """
        conn = cls.open()
        if conn is None:
            raise acp.RequestError.invalid_request(
                {
                    "reason": (
                        "this agent keeps no session index; it was started without a "
                        "persistence handler, so its sessions end with the process"
                    )
                }
            )
        where, params = ["1 = 1"], []
        if cwd is not None:
            where.append("cwd = ?")
            params.append(cwd)
        if cursor is not None:
            updated_at, _, session_id = cursor.partition("\x1f")
            where.append("(updated_at, session_id) < (?, ?)")
            params += [updated_at, session_id]
        rows = conn.execute(
            f"SELECT session_id, cwd, additional_directories, title, updated_at "  # noqa: S608
            f"FROM acp_sessions WHERE {' AND '.join(where)} "
            f"ORDER BY updated_at DESC, session_id DESC LIMIT ?",
            (*params, limit + 1),
        ).fetchall()
        more = len(rows) > limit
        rows = rows[:limit]
        sessions = [
            acp.schema.SessionInfo(
                session_id=session_id,
                cwd=cwd_,
                additional_directories=json.loads(directories),
                title=title,
                updated_at=updated_at,
            )
            for session_id, cwd_, directories, title, updated_at in rows
        ]
        next_cursor = f"{rows[-1][4]}\x1f{rows[-1][0]}" if more and rows else None
        return sessions, next_cursor


def _title_from(text: str) -> str:
    """A session title taken from the first thing the user said in it.

    The alternative is asking the model for one, which costs a request and a wait
    before the answer the user is actually waiting for. Their own opening line is
    usually what they would have called it anyway.
    """
    line = " ".join(text.split())
    return line if len(line) <= 60 else line[:59].rstrip() + "…"


class Attachment(pydantic.BaseModel):
    """A file or resource attached by reference URI"""

    uri: str

    @pydantic.field_validator("uri")
    @classmethod
    def _as_path(cls, uri: str) -> str:
        """
        A ``file:`` URI becomes the plain path the read tool takes.
        Anything else is passed through unchanged for the agent to interpret.
        """
        parsed = urllib.parse.urlsplit(uri)
        if parsed.scheme == "file":
            return urllib.request.url2pathname(parsed.path)
        return uri


def _prompt_parts(
    prompt: list[ContentBlock],
) -> tuple[str, list[Attachment], list[Image.Image]]:
    """Split a prompt into the arguments the agent's ``prompt`` skill takes.

    Two block kinds are baseline -- every agent must handle them, with no capability
    to negotiate -- and one is claimed (``image``):

    * ``text``, the user's own words, joined into the prose argument.
    * ``resource_link``, a *reference* to a file rather than its contents. It
      becomes an `Attachment` -- the path, so the model can decide to open it with
      `acp_read_text_file`, which reads through the editor and therefore sees
      unsaved changes. Inlining a file here would freeze a stale copy into the
      conversation and pay its tokens on every request after, whether or not the
      answer ever needed it.
    * ``image``, decoded into the `PIL.Image.Image` the skill accepts -- when it
      carries its data. One that is only a URI is refused: nothing here fetches.

    Everything else is refused rather than dropped -- an audio clip, and notably
    ``resource``, a file's contents inlined by the editor. This agent deliberately
    does not claim `embedded_context`, and a conforming client then sends links
    instead (ACP: capabilities not claimed are unsupported, and clients MUST
    restrict prompt content accordingly), which is the whole point: the flat fee
    for attaching a large file becomes one line, and reading it back is bounded
    and on demand. A non-conforming block is refused so the client is told, not
    quietly answered as though the attachment were never there. Silently
    discarding an attachment is the failure mode that looks like success.

    Raises:
        RequestError: If the prompt is empty, or carries a block this cannot read.
    """

    def unreadable(what: str) -> acp.RequestError:
        return acp.RequestError.invalid_params(
            {"reason": f"this agent cannot read {what}"}
        )

    texts: list[str] = []
    attachments: list[Attachment] = []
    images: list[Image.Image] = []
    for block in prompt:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "resource_link":
            attachments.append(Attachment(uri=block.uri))
        elif block.type == "image":
            if not block.data:
                raise unreadable(f"an image that is only a reference ({block.uri})")
            images.append(Image.open(io.BytesIO(base64.b64decode(block.data))))
        else:
            raise unreadable(
                f"a {block.type!r} block; it advertises no prompt capability for one"
            )
    text = "".join(texts).strip()
    if not text and not attachments and not images:
        raise acp.RequestError.invalid_params({"reason": "the prompt is empty"})
    return text, attachments, images


class EffectfulACPAgent[A: Agent](acp.Agent):
    """An ACP server backed by one `Agent` instance per session.

    Parameterised by how to *make* an agent rather than by an agent, so this module
    never has to know about any particular one. `make_agent` is handed the session id
    and returns an agent bearing it as its ``__agent_id__``; an agent class with an
    ``__agent_id__`` field is already such a callable, which is the usual way to pass
    one (see ``assistant.py``).

    The agent must have a ``prompt`` skill -- named for the protocol method it
    answers, ``session/prompt`` -- of the form::

        prompt(user_input: str,
               attachments: Sequence[Attachment] = (),
               images: Sequence[Image.Image] = ()) -> ...

    The contract is assumed, not discovered: `_answer` calls it directly, and the
    capabilities advertised below claim exactly what it accepts. Introspecting each
    agent for what it happens to take would make the advertisement -- sent once at
    ``initialize`` -- depend on an agent that does not exist yet.

    `models` fills the editor's model picker, and defaults to reading `OFFER_MODELS_ENV`
    rather than to nothing. Which side of this module that default lives on is the
    whole question: an editor configures an agent with a command and an environment, so
    it is the *server* that knows to look there, not the script it is serving. Leaving
    it to the caller would put a few lines of environment parsing in every script that
    wanted a picker, and each of them would be a chance to spell it differently.
    """

    make_agent: collections.abc.Callable[[str], A]
    models: tuple[str, ...]
    page_size: int

    client: acp.interfaces.Client
    client_capabilities: acp.schema.ClientCapabilities

    def __init__(
        self,
        make_agent: collections.abc.Callable[[str], A],
        *,
        models: collections.abc.Sequence[str] | None = None,
        page_size: int = 50,
    ):
        self.make_agent = make_agent
        # `None` rather than `()` as the default, because "the caller said nothing" and
        # "the caller said no models" are different answers and only the first should
        # consult the environment. A caller passing `()` has turned the picker off.
        self.models = _offered_models() if models is None else tuple(models)
        self.page_size = page_size
        self.sessions: dict[str, ACPSession[A]] = {}
        # Set by `serve`, and what `restart` needs: the protocol's own stdout, and the
        # state a restarted predecessor left behind.
        self._channel: typing.TextIO | None = None
        self._stdout: asyncio.StreamWriter | None = None
        self._restored: str | None = None
        self._unanswered: set[typing.Any] = set()
        self._restarting: asyncio.Task | None = None
        self._stdin: asyncio.StreamReader | None = None
        self._cwd = os.getcwd()

    @property
    def agent_capabilities(self) -> acp.schema.AgentCapabilities:
        """The capabilities this agent advertises to the editor.

        The editor uses them to decide what to offer the user, and the model uses
        them to decide what to ask the editor to do. Everything claimed here is
        something implemented below, and the reverse also has to hold: a client "MUST
        verify that the Agent supports this capability" before using one, so a method
        this class defines but does not advertise is a method no conforming client
        will ever call. `close_session` was exactly that until this said so, which
        left every session and its writer task alive for the life of the process.

        `prompt_capabilities` claims exactly what the ``prompt`` skill contract
        accepts -- see `initialize` for the argument. `mcp_capabilities` claims
        the transports `ACPSession.connect_mcp` can reach beyond the baseline stdio.
        """
        return acp.schema.AgentCapabilities(
            load_session=True,
            mcp_capabilities=acp.schema.McpCapabilities(http=True, sse=True),
            prompt_capabilities=acp.schema.PromptCapabilities(image=True),
            session_capabilities=acp.schema.SessionCapabilities(
                close=acp.schema.SessionCloseCapabilities(),
                resume=acp.schema.SessionResumeCapabilities(),
                fork=acp.schema.SessionForkCapabilities(),
                # Conditional, because this one is a claim about *state*: without a
                # persistence handler there are no sessions to list, and saying
                # otherwise invites a client to ask a question with no good answer.
                list=acp.schema.SessionListCapabilities()
                if SessionIndex.available()
                else None,
            ),
        )

    def _modes(self, session: ACPSession[A]) -> acp.schema.SessionModeState:
        """The modes on offer and the one in force, for a session response."""
        return acp.schema.SessionModeState(
            current_mode_id=session.mode_id,
            available_modes=list(_library().SESSION_MODES),
        )

    def _config_options(self, session: ACPSession[A]) -> list[ConfigOption]:
        """Every control this session puts in the editor's UI.

        The mode is here *as well as* in `modes`, which looks like saying it twice and
        is not. A client that understands config options "MUST use them exclusively
        and ignore the legacy modes field" -- so the moment this list is non-empty, a
        client that reads it hides its mode picker and looks for an option whose
        category is ``mode`` instead. Offering only the model would therefore take the
        mode picker away from exactly the clients that render pickers best. `modes`
        stays in the response for clients that do not read this list at all.

        The model option appears only when this server was given models to choose
        between: an option listing one choice is a control that does nothing, which is
        worse in a user interface than no control.
        """
        options: list[ConfigOption] = [
            acp.schema.SessionConfigOptionSelect(
                type="select",
                id=MODE_OPTION_ID,
                name="Mode",
                description="How much this agent may do without asking.",
                category="mode",
                current_value=session.mode_id,
                options=[
                    acp.schema.SessionConfigSelectOption(
                        value=mode.id, name=mode.name, description=mode.description
                    )
                    for mode in _library().SESSION_MODES
                ],
            )
        ]
        if self.models:
            options.append(
                acp.schema.SessionConfigOptionSelect(
                    type="select",
                    id=MODEL_OPTION_ID,
                    name="Model",
                    description="Which model answers in this session.",
                    category="model",
                    current_value=session.model,
                    options=[
                        acp.schema.SessionConfigSelectOption(
                            value=INHERIT_MODEL,
                            name="Default",
                            description="Whatever this agent process was started with.",
                        ),
                        *(
                            acp.schema.SessionConfigSelectOption(
                                value=model, name=model
                            )
                            for model in self.models
                        ),
                    ],
                )
            )
        if (levels := _thought_levels()) is not None:
            options.append(
                acp.schema.SessionConfigOptionSelect(
                    type="select",
                    id=THOUGHT_LEVEL_OPTION_ID,
                    name="Thought level",
                    description="How much the model reasons before answering.",
                    category="thought_level",
                    current_value=session.thought_level or INHERIT_THOUGHT_LEVEL,
                    options=[
                        acp.schema.SessionConfigSelectOption(
                            value=INHERIT_THOUGHT_LEVEL,
                            name="Default",
                            description="Whatever this model does by default.",
                        ),
                        *(
                            acp.schema.SessionConfigSelectOption(
                                value=level, name=level
                            )
                            for level in levels
                        ),
                    ],
                )
            )
        return options

    def _announce_commands(self, session: ACPSession[A]) -> None:
        """Tell the editor which ``/name`` commands to offer for this session.

        Sent as a notification once the session exists, which is what the spec
        describes. It races the response to the request that created the session --
        both go out on one pipe from two tasks -- and a client that has not yet learned
        the id may drop it. Nothing is lost that matters: the commands are a
        convenience, and reopening the session announces them again.
        """
        session.notify(
            acp.schema.AvailableCommandsUpdate(
                session_update="available_commands_update",
                available_commands=[
                    command.spec for command in _library().slash_commands().values()
                ],
            )
        )

    def _announce_config(self, session: ACPSession[A]) -> None:
        """Push this session's config options, including the modes now on offer.

        A session whose mode is no longer offered falls back to the first one.
        """
        modes = _library().SESSION_MODES
        if session.mode_id not in {mode.id for mode in modes}:
            session.mode_id = modes[0].id
            session.notify(
                acp.schema.CurrentModeUpdate(
                    session_update="current_mode_update",
                    current_mode_id=session.mode_id,
                )
            )
        session.notify(
            acp.schema.ConfigOptionUpdate(
                session_update="config_option_update",
                config_options=self._config_options(session),
            )
        )

    @property
    def agent_info(self) -> acp.schema.Implementation:
        """The agent's name, title and version, for the editor to display."""
        return acp.schema.Implementation(
            name="effectful", title="effectful.handlers.llm", version="0.4.0"
        )

    def _open_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None,
    ) -> ACPSession[A]:
        """Open this session, or re-point an already open one at these directories.

        Opening on demand is what makes `load_session` work at all: after a restart
        the editor knows a session id that this process has never seen, and the agent
        constructed under it reads its own history back from the checkpoint. Only
        ``session/new`` and ``session/load`` may do it, though -- see `_session`.

        Call it from the event loop thread, since `ACPSession` starts a task there.
        """
        roots = tuple(additional_directories or ())
        if session_id not in self.sessions:
            session = self.sessions[session_id] = ACPSession(
                agent=self.make_agent(session_id),
                client=self.client,
                client_capabilities=self.client_capabilities,
                cwd=cwd,
                additional_directories=roots,
            )
            # A session this process has not seen may be one it recorded before a
            # restart or a crash; its settings come back with it.
            recorded = SessionIndex.get(session_id)
            if recorded is not None:
                session.title = recorded["title"] or ""
                if recorded["model"] is not None:
                    session.model = recorded["model"]
                if recorded.get("thought_level") is not None:
                    session.thought_level = recorded["thought_level"]
                if recorded["mode"] in {m.id for m in _library().SESSION_MODES}:
                    session.mode_id = recorded["mode"]
        else:
            # An editor may reopen a session it still has open, and may do so from a
            # different window onto a different directory. The conversation is the
            # same one; where it is rooted is whatever it was just told.
            session = self.sessions[session_id]
            session.cwd, session.additional_directories = cwd, roots
        return self.sessions[session_id]

    def _session(self, session_id: str) -> ACPSession[A]:
        """This session, which must already be open.

        Every method other than ``session/new`` and ``session/load`` names a session
        the editor believes is open, so an id that is not is a mistake and is answered
        as one. Opening one here instead would turn a typo -- or a prompt sent against
        a session that was never loaded -- into a silently fresh conversation, which
        looks to the user like an agent that forgot everything.

        Raises:
            RequestError: If no session is open under `session_id`.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise acp.RequestError.resource_not_found(session_id)
        return session

    def on_connect(self, conn: acp.interfaces.Client) -> None:
        self.client = conn
        if self._restored is not None:
            try:
                self._resume_after_restart(self._restored)
            except Exception:
                print(f"could not resume:\n{traceback.format_exc()}", file=sys.stderr)
            self._restored = None

    def _observe(self, event: acp.connection.StreamEvent) -> None:
        """Keep count of the editor's requests this server has not yet answered.

        Told of each message after it is written, so a request leaves the count only
        once its answer is on the pipe -- which is what `restart` waits for.
        """
        message = event.message
        if "id" not in message:
            return
        if event.direction is acp.connection.StreamDirection.INCOMING:
            if "method" in message:
                self._unanswered.add(message["id"])
        elif "method" not in message:
            self._unanswered.discard(message["id"])

    def restart(self, requested_by: ACPSession[A]) -> str:
        """Replace this process with a fresh one, keeping the editor and every session.

        ACP has no way for an agent to restart: the editor owns the process, and the
        only reconnection the protocol knows is the editor starting a new one and
        loading its sessions again. So the replacement happens underneath it. Once
        every request is answered -- this one included -- and nothing is left to
        write, the process `exec`s itself with its own command line. That keeps its
        pid and its pipes, so the editor sees a pause and nothing else.

        Each session comes back the way `load_session` brings one back after a crash:
        its history from the persistence checkpoint, its settings from
        `SessionIndex`. Which sessions were open, and what the editor said at
        ``initialize``, are the only things the editor will not say again; they go to
        a file named by `RESTART_STATE_ENV`, which `on_connect` in the new process
        reads. Anything not yet checkpointed is lost, as it would be in a crash.

        Refused unless every other session is idle -- no turn running, no request
        unanswered -- rather than deferred: a deferred restart would say it was
        restarting and then not, for as long as that work took. Once accepted, new
        turns and sessions are refused until the `exec` (`_refuse_while_restarting`),
        so nothing can start while it waits for this request to be answered. And
        refused if the code it would restart into does not import
        (`_check_restartable`), since after the `exec` there is no going back.

        Returns the reply to the user.
        """
        if self._channel is None or self._stdout is None:
            return "Restarting needs the server to be serving over stdio."
        persisting = SessionIndex.available()
        if not persisting:
            return "Restarting needs `--persist-db`, to bring the sessions back."
        if self._restarting is not None:
            return "Already restarting."
        busy = [
            f"`{s.title or s.session_id}`"
            for s in self.sessions.values()
            if s is not requested_by and s.lock.locked()
        ]
        if busy:
            return (
                f"A turn is running in {', '.join(busy)}. Restart once it has ended, "
                f"or cancel it first."
            )
        # This request is one of them.
        if len(self._unanswered) > 1:
            return "The editor is waiting on another request. Restart once it is done."
        if (problem := self._check_restartable()) is not None:
            return f"Not restarting: the new process would fail to start.\n\n{problem}"
        self._restarting = asyncio.get_running_loop().create_task(
            self._restart(requested_by)
        )
        return "Restarting. This session and its conversation carry over."

    def _refuse_while_restarting(self) -> None:
        """Refuse to start anything once `restart` has accepted.

        Raises:
            RequestError: If this process is about to be replaced.
        """
        if self._restarting is not None:
            raise acp.RequestError.invalid_request(
                {"reason": "the agent is restarting; try again in a moment"}
            )

    def _check_restartable(self) -> str | None:
        """Import what the new process will, in a child process; the error, if any.

        Catches what an edit most often breaks -- a syntax error, a bad import -- but
        not what only fails once running. Blocks the event loop while it runs, which
        costs nothing here: `restart` only gets this far when no turn is running.
        """
        check = (
            "import importlib.util, sys\n"
            "import effectful.handlers.llm.harness.__main__\n"
            "import library\n"
            "if len(sys.argv) > 1:\n"
            "    spec = importlib.util.spec_from_file_location('_restart_check', sys.argv[1])\n"
            "    spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        )
        agent_file = (
            inspect.getsourcefile(self.make_agent)
            if isinstance(self.make_agent, type)
            else None
        )
        env = dict(os.environ)
        env.pop(RESTART_STATE_ENV, None)
        paths = [os.path.dirname(os.path.abspath(__file__))]
        if agent_file is not None:
            paths.insert(0, os.path.dirname(agent_file))
        env["PYTHONPATH"] = os.pathsep.join(
            [*paths, *filter(None, [env.get("PYTHONPATH")])]
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", check, *filter(None, [agent_file])],
                cwd=self._cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"```\n{e}\n```"
        if result.returncode != 0:
            return f"```\n{result.stderr.strip()[-2000:]}\n```"
        return None

    async def _restart(self, requested_by: ACPSession[A]) -> None:
        """Wait until nothing is owed to the editor, then `exec` a replacement.

        A failure anywhere short of the `exec` succeeding leaves this process serving,
        and says so to the session that asked.
        """
        reloader = autoreload.current()
        suspended = (
            reloader.suspended() if reloader is not None else contextlib.nullcontext()
        )
        with suspended:
            await self._exec_replacement(requested_by)

    async def _exec_replacement(self, requested_by: ACPSession[A]) -> None:
        """The body of `_restart`, with reloads held off around it."""
        assert self._channel is not None and self._stdout is not None
        assert self._stdin is not None
        path = None
        try:
            # Nothing new starts once `restart` has accepted, so this waits out only
            # the turn that asked and whatever the editor sends meanwhile.
            while requested_by.lock.locked():
                await asyncio.sleep(0.05)
            for session in list(self.sessions.values()):
                await session.updates.join()
            while (
                self._unanswered
                or self._stdout.transport.get_write_buffer_size()
                # Bytes of a message the editor is part-way through sending would
                # be lost with this process; wait for the rest and let it be handled.
                # (Private, and there is no public way to ask.)
                or self._stdin._buffer  # type: ignore[attr-defined]
            ):
                await asyncio.sleep(0.05)
            for session in self.sessions.values():
                SessionIndex.record(session)
            # The editor will not send its MCP servers again, and their processes
            # would outlive the `exec`.
            for session in self.sessions.values():
                await session.close_mcp(force=True)
            state = {
                "client_capabilities": self.client_capabilities.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ),
                "sessions": list(self.sessions),
                "mcp_servers": {
                    session_id: _dump_mcp(session.mcp_servers)
                    for session_id, session in self.sessions.items()
                },
            }
            fd, path = tempfile.mkstemp(prefix="acp-restart-", suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(state, f)
            os.environ[RESTART_STATE_ENV] = path
            # `serve` pointed fd 1 at stderr and gave the protocol a duplicate, which
            # is not inherited across `exec`; the new process has to find it on fd 1.
            self._channel.flush()
            os.dup2(self._channel.fileno(), 1)
            # The command line's relative paths are relative to where it was run.
            os.chdir(self._cwd)
            os.execv(sys.executable, sys.orig_argv)
        except Exception:
            os.dup2(2, 1)
            os.environ.pop(RESTART_STATE_ENV, None)
            for session in self.sessions.values():
                if session.mcp_client is None and session.mcp_servers:
                    await session.connect_mcp(session.mcp_servers)
            if path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(path)
            error = traceback.format_exc()
            print(f"restart failed:\n{error}", file=sys.stderr)
            requested_by.notify(
                acp.update_agent_message_text(
                    f"\n\nRestart failed; still running as before.\n\n```\n{error}```"
                )
            )
        finally:
            self._restarting = None

    def _resume_after_restart(self, path: str) -> None:
        """Reopen the sessions a restarted predecessor listed in `path`."""
        with open(path) as f:
            state = json.load(f)
        os.unlink(path)
        self.client_capabilities = acp.schema.ClientCapabilities.model_validate(
            state["client_capabilities"]
        )
        for session_id in state["sessions"]:
            try:
                recorded = SessionIndex.get(session_id)
                if recorded is None:
                    print(f"note: no record of session {session_id}", file=sys.stderr)
                    continue
                session = self._open_session(
                    session_id, recorded["cwd"], recorded["additional_directories"]
                )
                servers = state.get("mcp_servers", {}).get(session_id)
                if servers:
                    session.mcp_connecting = session.loop.create_task(
                        session.connect_mcp(servers)
                    )
                self._announce_commands(session)
                self._announce_config(session)
            except Exception:
                # The editor will find it closed, and can load it again.
                print(
                    f"could not reopen session {session_id}:\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                dropped = self.sessions.pop(session_id, None)
                if dropped is not None and dropped.writer is not None:
                    dropped.writer.cancel()

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: acp.schema.ClientCapabilities | None = None,
        client_info: acp.schema.Implementation | None = None,
        **kwargs: typing.Any,
    ) -> acp.schema.InitializeResponse:
        """Negotiate: say what this agent can do, and remember what the client can.

        `prompt_capabilities` claims what `_prompt_parts` reads and nothing more --
        the contract: a client "MUST adapt its interface according to
        `PromptCapabilities`", and treats anything not claimed as unsupported. Both
        directions of that rule are used deliberately here:

        * ``image`` is claimed, because the ``prompt`` skill contract takes decoded
          images -- a promise that an attached screenshot will be *looked at*.
        * ``embedded_context`` is not, and its absence is load-bearing: it is what
          makes a conforming client attach a file as a ``resource_link`` -- a
          reference costing a line -- instead of inlining its whole contents into
          a prompt this agent would then be carrying in the conversation, and
          paying for, on every request after. The model reads an attachment
          through `acp_read_text_file` if and when the request needs it, bounded
          and fresh from the editor's buffer. (Poolside's client, for one,
          auto-attaches the active file to every prompt with its full text when
          this is claimed, and degrades to links itself when it is not.)

        The client's own capabilities are consulted by `ACPToolRuntime`: the editor
        tools are offered to the model either way, and one the client cannot service
        reports itself as a failed call rather than being withheld.
        """
        self.client_capabilities = (
            client_capabilities or acp.schema.ClientCapabilities()
        )
        capabilities = self.agent_capabilities
        return acp.schema.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=capabilities,
            agent_info=self.agent_info,
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> acp.schema.NewSessionResponse:
        """Open a session, and with it a fresh agent.

        The session id becomes the agent's ``__agent_id__``, which is what makes the
        conversation persistent: with a persistence handler installed, the agent's
        history and declared fields are checkpointed under that id after every call
        and restored by `load_session` below.

        `cwd` is the directory the user opened, and the session keeps it: it is what
        the model is told it is working on, and where a terminal command runs.
        """
        self._refuse_while_restarting()
        session = self._open_session(str(uuid.uuid4()), cwd, additional_directories)
        await session.connect_mcp(mcp_servers or [])
        SessionIndex.record(session)
        self._announce_commands(session)
        return acp.schema.NewSessionResponse(
            session_id=session.session_id,
            modes=self._modes(session),
            config_options=self._config_options(session),
        )

    @staticmethod
    def _replay(
        history: collections.abc.Iterable[collections.abc.Mapping[str, typing.Any]],
    ) -> collections.abc.Iterator[typing.Any]:
        """The `session/update` notifications that reproduce a stored conversation.

        ACP requires `session/load` to replay "the entire conversation", and a coding
        agent's conversation is mostly not prose: an assistant turn that read three files
        carries no text at all, only tool calls, and dropping those would replay a
        conversation in which the agent sat silent and then knew things. So each stored
        tool call comes back as a completed tool-call row, and each stored tool *result*
        fills in that row's output.

        A generator rather than a method, so it can be read against a history without a
        session, an editor, or an event loop.
        """
        for message in history:
            role, text = message.get("role"), _library().message_as_text(message)
            if role == "user":
                if text:
                    yield acp.update_user_message_text(text)
            elif role == "assistant":
                if text:
                    yield acp.update_agent_message_text(text)
                for raw in message.get("tool_calls") or []:
                    function = raw.get("function") or {}
                    name = function.get("name") or "?"
                    arguments = function.get("arguments")
                    try:
                        raw_input = (
                            json.loads(arguments)
                            if isinstance(arguments, str)
                            else arguments
                        )
                    except ValueError:
                        raw_input = None
                    yield acp.start_tool_call(
                        str(raw.get("id")),
                        name,
                        kind=_library().tool_kind(name),
                        # Completed, because a stored call is one that already ran: the
                        # turn it belonged to is over, whatever became of the call.
                        status="completed",
                        raw_input=raw_input,
                    )
            elif (
                role == "tool" and (call_id := message.get("tool_call_id")) is not None
            ):
                yield acp.update_tool_call(
                    str(call_id),
                    content=[acp.tool_content(acp.text_block(text))],
                )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[typing.Any] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: typing.Any,
    ) -> acp.schema.LoadSessionResponse:
        """Reopen an earlier session and replay it to the editor.

        Constructing the agent under the same id is the whole of the restore:
        `Agent.__history__` reads the checkpoint lazily on first use. The replay is
        required -- the agent MUST stream the *entire* conversation back, and MUST
        wait until it has, because the client may be a different process with no
        other record of it.
        """
        self._refuse_while_restarting()
        session = self._open_session(session_id, cwd, additional_directories)
        await session.connect_mcp(mcp_servers or [])
        SessionIndex.record(session)
        for update in self._replay(session.agent.__history__):
            session.notify(update)
        self._announce_commands(session)
        await session.flush()
        return acp.schema.LoadSessionResponse(
            modes=self._modes(session),
            config_options=self._config_options(session),
        )

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> acp.schema.ResumeSessionResponse:
        """Reopen a session the client already has the transcript of.

        The same restore as `load_session` without the replay: a client resumes rather
        than loads exactly when it kept its own copy of the conversation and wants the
        agent to pick it up, not to be told it again. That is the whole difference, and
        it is why both are advertised -- a client picks one.
        """
        self._refuse_while_restarting()
        session = self._open_session(session_id, cwd, additional_directories)
        # Optional here; without it the session keeps the servers it has.
        if mcp_servers is not None:
            await session.connect_mcp(mcp_servers)
        SessionIndex.record(session)
        self._announce_commands(session)
        return acp.schema.ResumeSessionResponse(
            modes=self._modes(session),
            config_options=self._config_options(session),
        )

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> acp.schema.ForkSessionResponse:
        """Branch a conversation: a new session that starts as a copy of this one.

        For trying a second approach without losing the first. The copy is a copy --
        a new id, a new agent, its own history that happens to begin the same way --
        so a turn in either leaves the other alone. The user's settings come with it,
        since a fork is a continuation and re-picking the mode and model would be a
        chore rather than a choice.

        Reading the source through `_open_session` rather than requiring it open lets
        a session be forked from a list after a restart, when its history lives only
        in the checkpoint. The fork's own history is checkpointed when it first
        answers, as any session's is -- so a fork abandoned before its first turn is
        an empty conversation, not a copy.
        """
        self._refuse_while_restarting()
        source = self._open_session(session_id, cwd, additional_directories)
        fork = self._open_session(str(uuid.uuid4()), cwd, additional_directories)
        # Optional here; without it the fork uses the source's servers.
        await fork.connect_mcp(
            source.mcp_servers if mcp_servers is None else mcp_servers
        )
        fork.agent.__history__.extend(source.agent.__history__)
        fork.mode_id, fork.model = source.mode_id, source.model
        fork.thought_level = source.thought_level
        fork.title = f"{source.title} (fork)" if source.title else ""
        SessionIndex.record(fork)
        self._announce_commands(fork)
        return acp.schema.ForkSessionResponse(
            session_id=fork.session_id,
            modes=self._modes(fork),
            config_options=self._config_options(fork),
        )

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **kwargs: typing.Any,
    ) -> acp.schema.ListSessionsResponse:
        """The conversations this agent knows of, newest first.

        This is what lets an editor offer sessions from before it was last closed.
        `cwd` narrows the answer to one project, which is what a client asks for when
        its window is that project.

        Raises:
            RequestError: If this agent keeps no session index (see `SessionIndex`).
        """
        sessions, next_cursor = SessionIndex.page(cwd, cursor, self.page_size)
        return acp.schema.ListSessionsResponse(
            sessions=sessions, next_cursor=next_cursor
        )

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: typing.Any
    ) -> acp.schema.SetSessionModeResponse:
        """Switch this session's mode, for a client that does not read config options.

        Raises:
            RequestError: If the session is not open, or the mode is not one offered.
        """
        session = self._session(session_id)
        self._set_mode(session, mode_id)
        SessionIndex.record(session)
        return acp.schema.SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: typing.Any
    ) -> acp.schema.SetSessionConfigOptionResponse:
        """Set one of this session's config options, and answer with all of them.

        The response is the whole list rather than an acknowledgement, because a
        client redraws its controls from it -- which is also why setting an unknown
        option is refused rather than ignored: a control that silently does nothing is
        worse than one that reports it cannot.

        Raises:
            RequestError: If the session, the option, or the value is not known.
        """
        session = self._session(session_id)
        if not isinstance(value, str):
            raise acp.RequestError.invalid_params(
                {"reason": f"{config_id!r} takes a string, not {value!r}"}
            )
        if config_id == MODE_OPTION_ID:
            self._set_mode(session, value)
        elif config_id == MODEL_OPTION_ID and self.models:
            if value != INHERIT_MODEL and value not in self.models:
                raise acp.RequestError.invalid_params(
                    {"reason": f"{value!r} is not one of the models on offer"}
                )
            session.model = value
        elif config_id == THOUGHT_LEVEL_OPTION_ID:
            levels = _thought_levels() or ()
            if value != INHERIT_THOUGHT_LEVEL and value not in levels:
                raise acp.RequestError.invalid_params(
                    {"reason": f"{value!r} is not one of the thought levels on offer"}
                )
            session.thought_level = value or None
        else:
            raise acp.RequestError.invalid_params(
                {"reason": f"no such config option: {config_id!r}"}
            )
        SessionIndex.record(session)
        return acp.schema.SetSessionConfigOptionResponse(
            config_options=self._config_options(session)
        )

    def _set_mode(self, session: ACPSession[A], mode_id: str) -> None:
        """Switch a session's mode, however the editor asked for it.

        Both ways in end here: `session/set_mode`, and `session/set_config_option` on
        the ``mode`` option, which is what a client that reads config options sends
        instead.

        Raises:
            RequestError: If `mode_id` is not one of `library.SESSION_MODES`.
        """
        if mode_id not in {mode.id for mode in _library().SESSION_MODES}:
            raise acp.RequestError.invalid_params(
                {"reason": f"no such mode: {mode_id!r}"}
            )
        session.mode_id = mode_id

    async def prompt(
        self,
        session_id: str,
        prompt: list[ContentBlock],
        **kwargs: typing.Any,
    ) -> acp.schema.PromptResponse:
        """Answer one prompt, reporting progress as it goes.

        A prompt is a *list* of content blocks, not a string: the user's typed text,
        plus whatever their editor attached -- referenced files, screenshots.
        `_prompt_parts` splits it into the arguments the agent's ``prompt`` skill
        takes, and rejects what this agent cannot read rather than dropping it
        (see there).

        The skill call is synchronous and can run for minutes, so it goes to a worker
        thread. `asyncio.to_thread` copies this task's context, and the handler stack
        lives in a `ContextVar`, so the worker inherits the ambient stack -- the one
        the module launcher installed -- and adds this session's own handlers to it.

        The lock is what keeps one session to one turn. Nothing upstream provides it:
        `acp.connection` dispatches each request as its own task and does not await
        it, so a client that sends a second prompt before the first has answered would
        otherwise put two worker threads on one agent's history.

        The stop reason is the turn's summary, and the interesting values all come
        from somewhere other than a normal return: `cancelled` is raised out of the
        loop, and `max_tokens` and `refusal` are read by `ACPSessionReporter` off the
        last reply. Only a turn with nothing else to say is `end_turn`.
        """
        self._refuse_while_restarting()
        session = self._session(session_id)
        reloader = autoreload.current()
        with reloader.turn() if reloader is not None else contextlib.nullcontext():
            return await self._turn(session, prompt)

    async def _turn(
        self, session: ACPSession[A], prompt: list[ContentBlock]
    ) -> acp.schema.PromptResponse:
        """The body of `prompt`, with reloads held off around it."""
        async with session.lock:
            if session.mcp_connecting is not None:
                await session.mcp_connecting
                session.mcp_connecting = None
            session.cancel.clear()
            session.reporter.begin_turn()
            try:
                text, attachments, images = _prompt_parts(prompt)
                self._retitle(session, text)
                answered = self._command(session, text)
                if answered is not None:
                    # A command may have changed the session's settings.
                    SessionIndex.record(session)
                if answered is not None:
                    session.notify(acp.update_agent_message_text(answered))
                    return acp.schema.PromptResponse(stop_reason="end_turn")
                answer = await asyncio.to_thread(
                    self._answer, session, text, attachments, images
                )
                # A `str` skill's answer was already streamed to the editor token by
                # token; anything else was decoded from JSON that would have been noise
                # to stream, so it is reported here, once, in its decoded form.
                if not isinstance(answer, str):
                    session.notify(
                        acp.update_agent_message_text(json.dumps(answer, default=str))
                    )
                stop_reason = session.reporter.stop_reason()
            except SessionCancelled:
                stop_reason = "cancelled"
            finally:
                # Whatever happened -- an answer, a cancellation, a skill that raised
                # past this and out to the connection -- the editor is left with no
                # tool call still spinning, and hears everything before it hears the
                # result.
                session.reporter.abandon()
                await session.flush()
            return acp.schema.PromptResponse(
                stop_reason=stop_reason, usage=session.reporter.usage()
            )

    def _retitle(self, session: ACPSession[A], text: str) -> None:
        """Name the session after its first prompt, and tell the editor the name.

        Once, on the first thing the user says: a title that followed the latest
        message would rename the conversation out from under whoever is reading the
        list. A slash command does not name a session either -- ``/status`` is not
        what the conversation is about.

        Sessions are addressed by an opaque id, so without this a session list shows
        the user a column of UUIDs.
        """
        if session.title or text.startswith("/"):
            SessionIndex.record(session)
            return
        session.title = _title_from(text)
        SessionIndex.record(session)
        session.notify(
            acp.schema.SessionInfoUpdate(
                session_update="session_info_update",
                title=session.title,
                updated_at=datetime.datetime.now(datetime.UTC).isoformat(),
            )
        )

    def _command(self, session: ACPSession[A], text: str) -> str | None:
        """Answer `text` here if it is a slash command, or `None` to send it onward.

        A command is an ordinary prompt whose text begins with the name -- ACP has no
        separate method for one -- so recognising the prefix and looking the name up
        in `library.slash_commands` is the whole mechanism; the same table is what
        `_announce_commands` advertises, so a command offered is a command answered.
        All of them run without a model: they are about the session rather than about
        anything the model would know, and paying for a round trip to be told the
        working directory would be an odd way to spend the user's money.

        Both go into the reply as prose rather than into the agent's history, so the
        model never sees the exchange. `/clear` in particular must not: a message
        saying the conversation was forgotten is the one thing that should not survive
        forgetting it.
        """
        if not text.startswith("/"):
            return None
        name, _, argument = text[1:].partition(" ")
        name, argument = name.strip(), argument.strip()
        commands = _library().slash_commands()
        command = commands.get(name)
        if command is None:
            offered = ", ".join(f"`/{c}`" for c in commands)
            return f"Unknown command `/{name}`. Try {offered}."
        return command.run(self, session, argument)

    def _on_reload(self, reloader: "_Reloader") -> None:
        """Serve new code: rebuild each session's handlers, announce them, move its agent."""
        if isinstance(self.make_agent, type):
            self.make_agent = typing.cast(
                collections.abc.Callable[[str], A],
                reloader.current_class(self.make_agent),
            )
        for session in self.sessions.values():
            session.install_handlers()
            self._announce_commands(session)
            self._announce_config(session)
            reloader.refresh(session.agent)

    def _answer(
        self,
        session: ACPSession[A],
        text: str,
        attachments: list[Attachment],
        images: list[Image.Image],
    ) -> typing.Any:
        """Call the agent's ``prompt`` skill under this session's handlers.

        Runs in a worker thread. The call is direct rather than introspected: the
        skill contract is this server's to define (see the class docstring), and a
        nonconforming agent should fail loudly at its first prompt, the way any
        wrong argument list does.

        Installing on top of the ambient stack, rather than assembling one, is what
        lets the launcher decide the model, the retry budget and the persistence: the
        session contributes only its three translations to the editor.
        """
        with handler(session.intp):
            # `Agent` the bound says nothing about a `prompt` skill; the contract
            # is this server's own (class docstring), so the checker is waved off
            # here rather than widened everywhere the type parameter travels.
            return session.agent.prompt(  # type: ignore
                text, attachments=attachments, images=images
            )

    async def cancel(self, session_id: str, **kwargs: typing.Any) -> None:
        """Ask the worker to stop at its next cancellation point.

        A notification, so it must not block: setting the flag is the whole of it, and
        the turn reads it before the next completion, before the next tool call,
        between stream chunks, and while waiting on the editor.

        A notification also has nowhere to report an error, so an id with no session
        behind it is dropped rather than raised on -- and, unlike every other method
        here, must not open one, since cancelling a session that does not exist would
        otherwise create it.
        """
        if session := self.sessions.get(session_id):
            session.cancel.set()

    async def close_session(
        self, session_id: str, **kwargs: typing.Any
    ) -> acp.schema.CloseSessionResponse:
        """Stop this session's turn and its writer, and forget it.

        Dropping the entry matters: the writer task does not survive being cancelled,
        so a session left in the table after this would accept notifications that
        nothing delivers, and the first turn to wait for its queue to empty would wait
        forever. Forgetting it means a later `load_session` under the same id builds a
        working one instead.
        """
        session = self.sessions.pop(session_id, None)
        if session is None:
            return acp.schema.CloseSessionResponse()
        session.cancel.set()
        if session.writer is not None:
            session.writer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.writer
        await session.close_mcp()
        return acp.schema.CloseSessionResponse()

    async def serve(self) -> None:
        """Serve one agent over stdio until the editor disconnects.

        Under the launcher's ``--autoreload`` this adopts the reloader, so edits apply
        on this loop between turns and reach open sessions through `_on_reload`.

        stdout is the protocol, and the harness runs model-authored Python that may print
        to it. So fd 1 is pointed at stderr for the process's lifetime, after handing a
        duplicate of the real one to the transport -- which captures the file descriptor
        when the streams are built and is unaffected by the later rebinding.

        No ``receive_timeout``, deliberately. That parameter bounds how long the
        transport will wait for the *next message from the editor*, and tears the
        connection down when it expires -- so any value at all is a rule that the user
        may not think for longer than it before their agent disappears mid-conversation,
        which is what a minute of it did here once. Sitting silent is what a server
        does; the editor closing the pipe is what ends it, and that arrives as EOF
        rather than as a timeout.

        Nothing in this agent puts a clock on the editor, in fact -- see
        `ACPSession.call`. The two ends wait for each other indefinitely and either may
        walk away, which is the arrangement the protocol actually describes.
        """
        channel = os.fdopen(os.dup(1), "w", buffering=1)
        os.dup2(2, 1)
        sys.stdout = channel
        try:
            reader, writer = await acp.stdio.stdio_streams()
        finally:
            sys.stdout = sys.stderr
        self._channel, self._stdout, self._stdin = channel, writer, reader
        # What the command line's relative paths are relative to; see `_restart`.
        self._cwd = os.getcwd()
        self._restored = os.environ.pop(RESTART_STATE_ENV, None)

        reloader = autoreload.current()
        watching: asyncio.Future | None = None
        if reloader is not None:
            reloader.subscribe(self._on_reload)
            watching = asyncio.ensure_future(reloader.watch())
        try:
            # `run_agent`'s parameters are named from the client's point of view:
            # the stream the client reads is the one this agent writes.
            await acp.run_agent(
                self,
                input_stream=writer,
                output_stream=reader,
                use_unstable_protocol=True,
                observers=[self._observe],
            )
        finally:
            if reloader is not None:
                reloader.unsubscribe(self._on_reload)
            if watching is not None:
                watching.cancel()
            for session in list(self.sessions.values()):
                await session.close_mcp(force=True)
