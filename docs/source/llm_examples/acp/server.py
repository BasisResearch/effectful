"""The long-lived half of the ACP server: the connection, the sessions and their state.

`library.py` holds everything built from the harness -- the editor tools, the handlers
that translate between the harness and the protocol, the slash commands. This module
holds what has to outlive an edit to any of that: the connection, `EffectfulACPAgent`
and each `ACPSession`, and reading a prompt, which has to agree with the capabilities
advertised once per connection. It imports nothing from the harness or from `library`
at its top, and reaches `library` through `_library` on every use, which is what lets
``--autoreload`` re-run all of that code while the editor stays connected.

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

import argparse
import ast
import asyncio
import base64
import collections.abc
import concurrent.futures
import contextlib
import dataclasses
import datetime
import importlib
import importlib.machinery
import importlib.util
import inspect
import io
import json
import linecache
import os
import pathlib
import sqlite3
import symtable
import sys
import threading
import types
import typing
import urllib.parse
import urllib.request
import uuid

import acp
import acp.interfaces
import acp.schema
import pydantic
from PIL import Image

from effectful.internals.runtime import interpreter
from effectful.ops.semantics import coproduct, handler
from effectful.ops.types import INSTANCE_OP_PREFIX, Interpretation

if typing.TYPE_CHECKING:
    from effectful.handlers.llm import Agent

LIBRARY = "library"
"""The module holding the reloadable half of the server; see `_library`."""


def _library() -> types.ModuleType:
    """The tools, handlers and slash commands, as they now are.

    Looked up on each use rather than imported, so that ``--autoreload`` can re-run
    `library` while this module, which holds the live server, keeps running.
    """
    return importlib.import_module(LIBRARY)


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
INHERIT_MODEL = ""
"""The `model` option's value meaning "whatever the process was configured with".

An empty string rather than the model's name, because this agent does not know that
name: the model is bound into `LiteLLMConfigurer` by whoever assembled the stack, and
nothing in the protocol layer can see it. Saying "as configured" is honest; naming a
model here would be a guess printed in the user's editor.
"""

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

    new_code: bool = False
    """Whether `EffectfulACPAgent.reload` has run since this session's last turn."""

    title: str = ""
    """A human-readable name for the conversation, taken from its first prompt."""

    poll_interval: float = 0.1
    """How often a worker thread waiting on the editor re-reads `cancel`."""

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
        done. Called when the session opens and again by `EffectfulACPAgent.reload`.
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
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
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
            updated_at             TEXT NOT NULL
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
                    (session_id, cwd, additional_directories, title, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    cwd = excluded.cwd,
                    additional_directories = excluded.additional_directories,
                    title = COALESCE(excluded.title, acp_sessions.title),
                    updated_at = excluded.updated_at
                """,
                (
                    session.session_id,
                    session.cwd,
                    json.dumps(list(session.additional_directories)),
                    session.title or None,
                    datetime.datetime.now(datetime.UTC).isoformat(),
                ),
            )

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


# ---------------------------------------------------------------------------
# Re-running edited code while serving (``serve(autoreload=True)``)
# ---------------------------------------------------------------------------

HARNESS = "effectful.handlers.llm.harness"


def _harness_imports(file: str | os.PathLike[str]) -> set[str]:
    """The harness modules `file` imports at its top level, with their packages."""
    names: set[str] = set()
    nodes = list(ast.parse(pathlib.Path(file).read_text()).body)
    while nodes:
        node = nodes.pop()
        if isinstance(node, ast.If | ast.Try):
            nodes += node.body + node.orelse + getattr(node, "finalbody", [])
            nodes += [s for h in getattr(node, "handlers", []) for s in h.body]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names |= {node.module, *(f"{node.module}.{a.name}" for a in node.names)}
        elif isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
    modules = {
        ".".join(n.split(".")[:i]) for n in names for i in range(1, n.count(".") + 2)
    }
    return {m for m in modules if m.startswith(f"{HARNESS}.") and _is_module(m)}


def _is_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError):  # ``from module import Name``
        return False


def _kept_modules() -> set[str]:
    """The harness modules this module holds onto, which must never be re-run.

    Everything imported at the top of this file, and at the top of those modules in
    turn: this module's handlers implement their operations, and a re-run would
    define new ones that nothing here handles.
    """
    kept: set[str] = set()
    todo = list(_harness_imports(__file__))
    while todo:
        if (name := todo.pop()) not in kept:
            kept.add(name)
            spec = importlib.util.find_spec(name)
            if spec is not None and spec.origin is not None:
                todo += _harness_imports(spec.origin)
    return kept


def _bound_names(file: pathlib.Path) -> set[str] | None:
    """The names `file` binds at module level, or `None` if they cannot be known.

    A star import binds its module's ``__all__``, or else its public names, as that
    module now stands; one that cannot be looked up makes the answer unknowable.
    """
    source = file.read_text()
    table = symtable.symtable(source, str(file), "exec")
    bound = {
        s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()
    }
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.names[0].name == "*":
            if node.level or node.module not in sys.modules:
                return None
            names = vars(sys.modules[node.module])
            bound |= set(
                names.get("__all__") or [n for n in names if not n.startswith("_")]
            )
    return bound


def _launcher_args() -> argparse.Namespace:
    """The harness launcher's flags, read back from the command line it was run with.

    Not from `sys.argv`, from which the launcher removes them before the script runs.
    """
    from effectful.handlers.llm.harness.__main__ import _parse_args

    if HARNESS not in sys.orig_argv:
        raise RuntimeError(f"autoreload needs the launcher: python -m {HARNESS}")
    ns, _ = _parse_args(sys.orig_argv[sys.orig_argv.index(HARNESS) + 1 :])
    return ns


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

    harness: Interpretation | None = None
    """The handler stack turns run under in place of the ambient one, once `reload` sets it."""

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
        accepts -- see `initialize` for the argument. `mcp_*` is left claiming
        nothing, which is the honest answer for an agent that connects to no MCP
        servers.
        """
        return acp.schema.AgentCapabilities(
            load_session=True,
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
            self.sessions[session_id] = ACPSession(
                agent=self.make_agent(session_id),
                client=self.client,
                client_capabilities=self.client_capabilities,
                cwd=cwd,
                additional_directories=roots,
            )
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

    def _decline_mcp(self, mcp_servers: list[typing.Any] | None) -> None:
        """Note, without refusing, that this agent will not use the editor's MCP servers.

        Agents "SHOULD connect to all MCP servers specified by the Client", and stdio
        transport is baseline -- there is no capability with which to say "none at
        all", so a client with servers configured will send them on every
        ``session/new`` and is behaving correctly in doing so. Failing the request
        over that would make this agent unusable in any editor that has an MCP server
        set up, to no one's benefit; ignoring them silently would hide it. stderr is
        free (`serve` gives the protocol its own descriptor), so it goes there.
        """
        if mcp_servers:
            print(
                f"note: ignoring {len(mcp_servers)} MCP server(s) offered by the "
                f"editor; this agent has no MCP client",
                file=sys.stderr,
            )

    def on_connect(self, conn: acp.interfaces.Client) -> None:
        self.client = conn

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
        with self._current_stack():
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
        self._decline_mcp(mcp_servers)
        session = self._open_session(str(uuid.uuid4()), cwd, additional_directories)
        with self._current_stack():
            SessionIndex.record(session)
        self._announce_commands(session)
        return acp.schema.NewSessionResponse(
            session_id=session.session_id,
            modes=self._modes(session),
            config_options=self._config_options(session),
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
        self._decline_mcp(mcp_servers)
        session = self._open_session(session_id, cwd, additional_directories)
        with self._current_stack():
            SessionIndex.record(session)
            for update in _library()._replay(session.agent.__history__):
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
        self._decline_mcp(mcp_servers)
        session = self._open_session(session_id, cwd, additional_directories)
        with self._current_stack():
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
        self._decline_mcp(mcp_servers)
        source = self._open_session(session_id, cwd, additional_directories)
        fork = self._open_session(str(uuid.uuid4()), cwd, additional_directories)
        with self._current_stack():
            fork.agent.__history__.extend(source.agent.__history__)
            fork.mode_id, fork.model = source.mode_id, source.model
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
        with self._current_stack():
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
        self._set_mode(self._session(session_id), mode_id)
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
        else:
            raise acp.RequestError.invalid_params(
                {"reason": f"no such config option: {config_id!r}"}
            )
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
        session = self._session(session_id)
        async with session.lock:
            session.cancel.clear()
            session.reporter.begin_turn()
            try:
                text, attachments, images = _prompt_parts(prompt)
                with self._current_stack():
                    self._retitle(session, text)
                    answered = self._command(session, text)
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

    def reload(
        self, make_agent: collections.abc.Callable[[str], A], harness: Interpretation
    ) -> None:
        """Serve new code: later turns run under `harness`, and each open session's
        agent is rebuilt by `make_agent`, keeping everything set on the old one.

        Each session's handlers are rebuilt from `library` too, and its slash commands
        and config options announced again, so a command or mode added there is
        offered at once.

        Call it only while no turn is running.
        """
        self.make_agent, self.harness = make_agent, harness
        for session in self.sessions.values():
            session.new_code = True
            session.install_handlers()
            self._announce_commands(session)
            self._announce_config(session)
            old, new = session.agent, make_agent(session.session_id)
            new.__history__ = old.__history__
            # Its fields and whatever the model set on `self`, but not the skills
            # bound to it, which belong to the old class.
            new.__dict__.update(
                (key, value)
                for key, value in vars(old).items()
                if not key.startswith(INSTANCE_OP_PREFIX)
            )
            session.agent = new

    def _current_stack(self) -> contextlib.AbstractContextManager:
        """The stack for work outside a turn that reads persisted state.

        Reading a history back goes through `SQLitePersister`, which ``--autoreload``
        may have re-run since the server started; under the stack it started with, the
        read would find nothing. Without ``--autoreload`` there is only one stack.
        """
        if self.harness is None:
            return contextlib.nullcontext()
        return interpreter(self.harness)

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
        session contributes only its three translations to the editor. Once `reload`
        has set `harness`, the turn runs under that stack *instead of* the ambient
        one: installing it with `handler` would stack it on the launcher's, and
        every forwarding handler would run twice.
        """
        with (
            handler(session.intp)
            if self.harness is None
            else interpreter(coproduct(self.harness, session.intp))
        ):
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
        return acp.schema.CloseSessionResponse()

    async def serve(self, *, autoreload: bool = False) -> None:
        """Serve one agent over stdio until the editor disconnects.

        With `autoreload`, code is re-run as it is edited; see `_reloading`.

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

        async with self._reloading() if autoreload else contextlib.nullcontext():
            # `run_agent`'s parameters are named from the client's point of view: the
            # stream the client reads is the one this agent writes.
            await acp.run_agent(
                self,
                input_stream=writer,
                output_stream=reader,
                use_unstable_protocol=True,
            )

    @contextlib.asynccontextmanager
    async def _reloading(self) -> collections.abc.AsyncIterator[None]:
        """Re-run edited code for as long as this is entered, via hmr.

        hmr re-runs an edited module and whatever depends on it: the agent class's
        file, the modules it imports from its directory, and the harness, whose stack
        is then rebuilt from the launcher's flags. Once no turn is running, `reload`
        moves every session onto the new code. That includes `library`, which this
        module only ever reaches through `_library`. `_kept_modules` are never re-run,
        nor is anything outside those directories. A module that fails to re-run is reported
        on stderr and its previous version kept.
        """
        from reactivity.hmr.core import (
            HMR_CONTEXT,
            BaseReloader,
            ReactiveModule,
            _loader,
        )
        from reactivity.hmr.hooks import post_reload
        from watchfiles import awatch

        import effectful.handlers.llm.harness.__main__ as launcher

        args = _launcher_args()
        assert isinstance(self.make_agent, type), "autoreload needs an agent class"
        path = pathlib.Path(inspect.getfile(self.make_agent)).resolve()
        harness_spec = importlib.util.find_spec(HARNESS)
        assert harness_spec is not None and harness_spec.origin is not None
        harness_dir = pathlib.Path(harness_spec.origin).parent
        watched = (path.parent, harness_dir)
        # Everything under the watched directories was imported before hmr was set up
        # -- the handlers by the launcher, the agent's own imports by the script -- so
        # hmr could not track it. Dropped here, it is imported again through hmr the
        # next time the agent's file or the stack is built. The launcher's stack is
        # left holding the dropped classes, so nothing may run under it from now on:
        # the effect below installs the rebuilt stack before any turn, and the server
        # runs under it.
        kept = _kept_modules() | {__name__, "__main__"}
        for name, module in list(sys.modules.items()):
            file = getattr(module, "__file__", None)
            if (
                file is not None
                and name not in kept
                and not hasattr(module, "__path__")
                and any(pathlib.Path(file).resolve().is_relative_to(d) for d in watched)
            ):
                del sys.modules[name]

        # hmr finds modules only through `sys.path`, which an editable install of
        # effectful need not put the repository on.
        if str(harness_dir.parents[3]) not in sys.path:
            sys.path.append(str(harness_dir.parents[3]))
        reloader = BaseReloader(str(path), [str(path.parent), str(harness_dir)], [])
        # A skill checks its code against `linecache`, which would serve stale source.
        post_reload(linecache.checkcache)

        # The running copy of the agent's file is `__main__`, which hmr cannot re-run.
        sys.modules[path.stem] = module = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(path.stem, _loader, origin=str(path))
        )
        name = self.make_agent.__qualname__
        agent_class = HMR_CONTEXT.derived(lambda: getattr(module, name))
        stack = HMR_CONTEXT.derived(lambda: launcher._build_harness(args))
        # Any re-run of `library` defines this afresh, so reading it is how an edit
        # there -- a new slash command -- reaches `reload`.
        library = HMR_CONTEXT.derived(lambda: _library().session_handlers)

        def forget_removed_names() -> None:
            # hmr re-runs a module into the namespace it had, so a name its file no
            # longer binds -- a deleted function -- would stay defined and, reached
            # from the agent's scope, stay a tool. Pruned for each module in the
            # agent's directory, through hmr's mapping, which keeps its own record
            # of names; repeated because pruning one changes what a star import of
            # it binds.
            pruned = True
            while pruned:
                pruned = False
                for loaded in list(sys.modules.values()):
                    if not isinstance(loaded, ReactiveModule):
                        continue
                    names = vars(loaded)
                    file = names.get("__file__")
                    if "__path__" in names or file is None:
                        continue
                    if pathlib.Path(file).resolve().parent != path.parent:
                        continue
                    if (bound := _bound_names(pathlib.Path(file))) is None:
                        continue
                    proxy = loaded._ReactiveModule__namespace_proxy
                    for key in list(proxy.raw):
                        if key.startswith(("__", "_ReactiveModule__")):
                            continue
                        if key not in bound:
                            del proxy[key]
                            pruned = True

        def install() -> None:
            library()
            forget_removed_names()
            self.reload(agent_class(), stack())

        installed = HMR_CONTEXT.effect(install)

        async def watch() -> None:
            async for events in awatch(*reloader.includes):
                # So no turn sees two versions of the code.
                while any(s.lock.locked() for s in self.sessions.values()):
                    await asyncio.sleep(0.1)
                reloader.on_events(events)

        watching = asyncio.ensure_future(watch())
        try:
            assert self.harness is not None
            with interpreter(self.harness):
                yield
        finally:
            watching.cancel()
            installed.dispose()
