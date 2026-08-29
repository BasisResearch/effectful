"""Serve an `effectful.handlers.llm` `Agent` over the Agent Client Protocol.

The [Agent Client Protocol](https://agentclientprotocol.com) (ACP) is how an editor
-- Zed, VS Code, Obsidian, Emacs -- drives a coding agent: a JSON-RPC conversation
over stdio in which the editor opens a session, sends prompts, and receives a stream
of updates describing what the agent is doing. This module makes an `Agent` speak it.

## The shape of the thing

An ACP server has to answer three questions that the harness already answers as
effects, so almost nothing here is new machinery -- it is three translations:

| ACP concept | effectful concept | Where |
| ----------- | ----------------- | ----- |
| `session/update` notifications | the `completion` and `call_tool` effects | `ACPSessionReporter` |
| `session/request_permission` | intercepting `call_tool` | `ACPPermissionGate` |
| `fs/*`, `terminal/*` and `elicitation/*` client methods | `Tool`s the model may call | `ACPToolRuntime` |

`ACPSessionReporter` is the interesting one, and it is a close sibling of
`~effectful.handlers.llm.harness.observability.rich.RichTerminalRenderer`: both force
`completion` onto the streaming path and re-render the deltas somewhere. One renders
to a terminal, this one to a JSON-RPC pipe.

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
import functools
import io
import json
import os
import sqlite3
import sys
import threading
import typing
import urllib.parse
import urllib.request
import uuid

import acp
import acp.interfaces
import acp.schema
import litellm
import pydantic
import pydantic_core
from PIL import Image

from effectful.handlers.llm import Agent, Encodable, Tool
from effectful.handlers.llm.harness.durability.persistence import SQLitePersister
from effectful.handlers.llm.harness.hooks import (
    PromptInjectingInterpretation,
    ToolCallExecutionError,
    call_system,
    call_tool,
    completion,
)
from effectful.handlers.llm.harness.serialization import (
    DecodedToolCall,
    PromptSection,
    to_content_blocks,
)
from effectful.handlers.llm.harness.synthesis.body import FinalBodySynthesizer
from effectful.handlers.llm.harness.synthesis.snippet import StatefulReplSynthesizer
from effectful.ops.semantics import coproduct, fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Interpretation

type ContentBlock = (
    acp.schema.TextContentBlock
    | acp.schema.ImageContentBlock
    | acp.schema.AudioContentBlock
    | acp.schema.ResourceContentBlock
    | acp.schema.EmbeddedResourceContentBlock
)

type ToolCallContent = (
    acp.schema.ContentToolCallContent
    | acp.schema.FileEditToolCallContent
    | acp.schema.TerminalToolCallContent
)

type ConfigOption = (
    acp.schema.SessionConfigOptionSelect | acp.schema.SessionConfigOptionBoolean
)

type ElicitationProperty = (
    acp.schema.ElicitationStringPropertySchema
    | acp.schema.ElicitationBooleanPropertySchema
    | acp.schema.ElicitationMultiSelectPropertySchema
)

# ---------------------------------------------------------------------------
# The editor's own capabilities, offered to the model as tools
#
# Declared here and given meaning by `ACPToolRuntime`, one instance per session.
# A `Tool` is an `Operation`, so the declaration is the signature and the
# docstring -- the two things the model is shown -- and the body is unreachable
# under any interpretation that handles it. Splitting them that way is what lets
# the tools be plain module-level names: a script imports them to put them in its
# skills' lexical scope, and the session supplies the editor they talk to.
# ---------------------------------------------------------------------------


READ_RESULT_LIMIT_LINES = 1000
"""How many lines a read may return before it is truncated with a notice.

The bound that keeps one tool call from swallowing a conversation's budget: a
read lands in the history and is re-sent with every request after, so an
unbounded read of a large file is a large recurring cost incurred in one call.
The notice tells the model how to page with ``line``/``limit``; a model that
passes its own ``limit`` has chosen a size, and is not second-guessed.
"""


@Tool.define
def acp_read_text_file(
    path: str, line: int | None = None, limit: int | None = None
) -> str:
    """Read a text file and return its contents.

    Reads through the user's editor, so it sees unsaved changes in an open buffer
    -- use it in preference to opening the file yourself. `path` must be
    absolute. `line` is a 1-based line to start from, and `limit` a maximum
    number of lines to return; pass neither to read from the start. A long
    read with no `limit` is truncated after 1000 lines, with a notice saying
    how to read on from where it stopped.
    """
    raise RuntimeError("missing handler")


@Tool.define
def acp_write_text_file(path: str, content: str) -> str:
    """Write `content` to the file at absolute `path`, replacing what is there.

    Writes through the user's editor, so the change lands in the open buffer and
    the user sees it as an ordinary edit they can undo. Read the file first
    unless you are creating it.
    """
    raise RuntimeError("missing handler")


@Tool.define
def acp_run_terminal_command(command: str, args: list[str]) -> str:
    """Run `command` with `args` in a terminal and return its output.

    The terminal is the user's own, so they can watch the command run. Blocks
    until it exits. Prefer this to a shell one-liner assembled by hand: `args`
    are passed to the process directly, with no shell to quote for.
    """
    raise RuntimeError("missing handler")


@pydantic.dataclasses.dataclass
class PlanStep:
    """One step of the plan you are showing the user.

    Deliberately not `acp.schema.PlanEntry`, which is the same three fields plus the
    ``_meta`` every ACP type carries. A tool's parameters are turned into a strict JSON
    Schema, and strict schemas require every property, so the model would be obliged to
    supply a value for a field whose own documentation says implementations must not
    assume anything about it. The two vocabularies below are borrowed from the protocol
    rather than restated, so the part that could drift cannot.
    """

    content: str
    """What you will do, in a few words, as you would say it to them."""

    priority: acp.schema.PlanEntryPriority = "medium"
    status: acp.schema.PlanEntryStatus = "pending"


@Tool.define
def acp_update_plan(steps: list[PlanStep]) -> str:
    """Show the user your plan for a job with several steps, and keep it current.

    The editor renders this as a checklist beside the conversation, so the user can
    see where you are without reading back through it. Send the *whole* plan every
    time -- each call replaces the last -- marking at most one step `in_progress` and
    everything you have finished `completed`.

    Worth doing for work that takes several tool calls, and not worth it otherwise: a
    one-step plan tells the user nothing they did not just ask for.
    """
    raise RuntimeError("missing handler")


@pydantic.dataclasses.dataclass
class AskField:
    """One thing you are asking the user for, rendered as a field in a form.

    ACP describes a form as a JSON Schema, and `_elicitation_property` is where this
    becomes one. Handing the model that schema directly instead -- an
    `acp.schema.ElicitationSchema`, whose ``properties`` is a dict of a seven-way union
    of property types -- would ask it to author JSON Schema as a side errand of asking
    a question, and costs an order of magnitude more of its context to describe. This
    is the small closed subset an editor can actually render.
    """

    name: str
    """The key its answer comes back under. Short, and unique within one question."""

    title: str
    """The label the editor shows beside it, as you would say it to them."""

    description: str = ""
    """Optional detail, for a label that needs more than a few words."""

    kind: typing.Literal["text", "choice", "boolean"] = "text"
    """What kind of answer: free text, one of `choices`, or a yes/no."""

    choices: list[str] = dataclasses.field(default_factory=list)
    """The options, when `kind` is ``choice``. Required for one, ignored otherwise."""

    required: bool = True
    """Whether the user must fill it in before the form can be submitted."""


@Tool.define
def acp_ask_user(message: str, fields: list[AskField]) -> str:
    """Ask the user something, as a small form in their editor, and wait for a reply.

    For a decision that is genuinely theirs to make: two defensible ways to do what
    they asked, a destructive change worth confirming, a preference you cannot read
    off the code. Say in `message` what you are about to do and why the answer
    matters, and keep `fields` to the one or two things you actually need.

    Not for anything you could find out yourself. If the answer is in a file, read
    the file -- asking instead spends the user's attention to save you a tool call,
    and an assistant that asks before every step is worse than one that gets on with
    it and says what it assumed.

    The user may decline, and declining is an answer: you are told so, and should
    then continue without it or explain what you cannot decide. They may also dismiss
    the form, which ends the turn.
    """
    raise RuntimeError("missing handler")


# ---------------------------------------------------------------------------
# The knobs a session offers the user
#
# Two protocol features that cost an agent almost nothing and that an editor
# renders for free, both of which stay dead until the agent describes itself:
# a *mode* (`session/set_mode`) and a *config option* (`session/set_config_option`).
# Only `select` options are used, deliberately -- a boolean one is gated behind
# the client's `session.configOptions.boolean` capability, and most clients,
# including VS Code's, advertise no session capabilities at all.
# ---------------------------------------------------------------------------

ASK, AUTO, PLAN = "ask", "auto", "plan"

SESSION_MODES: tuple[acp.schema.SessionMode, ...] = (
    acp.schema.SessionMode(
        id=ASK,
        name="Ask",
        description="Ask before running each tool.",
    ),
    acp.schema.SessionMode(
        id=AUTO,
        name="Auto",
        description="Run tools without asking. Undo is your editor's.",
    ),
    acp.schema.SessionMode(
        id=PLAN,
        name="Plan",
        description="Read and discuss, but change nothing: no writes, no commands.",
    ),
)

MUTATING_TOOLS = frozenset(
    {acp_write_text_file.__name__, acp_run_terminal_command.__name__}
)
"""What `PLAN` mode refuses: the tools that change the user's editor or machine.

A denylist of the two `ACPToolRuntime` offers, and deliberately not a sandbox. The
harness may also be running model-authored Python -- `exec_code`,
``write_and_run_body`` -- which this says nothing about, because that is the eval
provider's business and the launcher's ``--eval-provider none`` is the switch for it.
Naming the mode "Plan" rather than "read-only" is what keeps that promise honest.
"""

UNGATED_TOOLS = frozenset({acp_ask_user.__name__})
"""What `ACPPermissionGate` lets through without asking: asking the user something.

The gate exists to put a question in front of the user before a tool runs, so
prompting for permission to ask them a question is the one case where it defeats
itself -- a dialog about a dialog, answered by the same person, immediately followed
by the real one.

Safe on its own terms rather than by exception. The tool's only effect is a form on
the user's screen: it reads nothing, changes nothing, and dismissing it already
cancels the turn, so the decision the gate would have offered is one the user still
has. `PLAN` mode leaves it alone for the same reason -- asking changes nothing, and a
session that may not act is exactly where a clarifying question is worth most.
"""

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
class ACPSession[A: Agent]:
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

    mode_id: str = ASK
    """Which of `SESSION_MODES` the user has picked. Read by `ACPPermissionGate`."""

    model: str = INHERIT_MODEL
    """The model the user picked, or `INHERIT_MODEL` for the configured one."""

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

    def __post_init__(self) -> None:
        """Start the one task that drains `updates`.

        It belongs to the session's whole life rather than to a turn: `load_session`
        replays a conversation with no turn in progress, and a turn's own last act is
        to wait for the queue to empty. Starting it per turn would make the first of
        those deliver nothing and the second wait on a queue nobody is reading.
        """
        self.writer = self.loop.create_task(self.drain())

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

    @functools.cached_property
    def reporter(self) -> "ACPSessionReporter":
        """The reporter in `intp`, reachable by name.

        It is the only handler of the three that outlives the call it is installed
        for: it counts a turn's requests and remembers which tool calls it has
        announced, and `EffectfulACPAgent.prompt` reads both once the worker is done.
        """
        return ACPSessionReporter(self)

    @functools.cached_property
    def intp(self) -> Interpretation:
        """The handler stack that runs a prompt on this session.

        Built once per session, and installed on top of whatever stack the process is
        already running under (see `EffectfulACPAgent._answer`). Every handler in it
        closes over this session, which is how three module-level concerns -- the
        tools, the reporting, the permission gate -- reach *this* editor without any
        of them being per-session types.

        `ACPPermissionGate` goes on last, so it is outermost: it must decide about a
        call before `ACPSessionReporter` announces it as running.
        """
        h = coproduct(
            self.reporter,
            ACPToolRuntime(self),
        )
        h = coproduct(h, ACPSessionConfig(self))
        h = coproduct(h, ACPPermissionGate(self))
        return h

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


def _elicitation_property(field: AskField) -> ElicitationProperty:
    """One `AskField` as the JSON Schema property an editor renders a widget from.

    ACP restricts these to primitives, so this is a small closed mapping rather than
    a general schema translation: text, a single choice, a yes/no. `oneOf` carries the
    choices rather than `enum` because its options are titled, and a title is what the
    editor puts on the control.

    A ``choice`` with nothing to choose from would render as an empty dropdown, so it
    degrades to a text field: the model asked for an answer, and a control the user
    cannot use would be a worse way to fail than one they can type into.
    """
    description = field.description or None
    if field.kind == "boolean":
        return acp.schema.ElicitationBooleanPropertySchema(
            type="boolean", title=field.title, description=description
        )
    if field.kind == "choice" and field.choices:
        return acp.schema.ElicitationStringPropertySchema(
            type="string",
            title=field.title,
            description=description,
            one_of=[
                acp.schema.EnumOption(const=choice, title=choice)
                for choice in field.choices
            ],
        )
    return acp.schema.ElicitationStringPropertySchema(
        type="string", title=field.title, description=description
    )


def _elicitation_schema(fields: list[AskField]) -> acp.schema.ElicitationSchema:
    """The whole form: one property per field, and which of them are required."""
    return acp.schema.ElicitationSchema(
        type="object",
        properties={field.name: _elicitation_property(field) for field in fields},
        required=[field.name for field in fields if field.required] or None,
    )


def _answers_as_text(
    fields: list[AskField], content: collections.abc.Mapping[str, typing.Any] | None
) -> str:
    """What the user filled in, as lines the model can read.

    Keyed by the fields that were asked rather than by what came back, so the answers
    arrive in the order they were asked and a field the user left blank is reported as
    blank instead of vanishing -- the model needs to know it asked and got nothing.

    `content` is optional even on an ``accept``, which is the protocol allowing a form
    with nothing in it to be submitted.
    """
    answers = content or {}
    lines = []
    for field in fields:
        value = answers.get(field.name)
        if isinstance(value, bool):
            shown = "yes" if value else "no"
        elif isinstance(value, list):
            shown = ", ".join(str(item) for item in value) or "(nothing selected)"
        elif value is None or value == "":
            shown = "(left blank)"
        else:
            shown = str(value)
        lines.append(f"{field.name}: {shown}")
    return "The user answered:\n" + "\n".join(lines)


@dataclasses.dataclass
class ACPToolRuntime(PromptInjectingInterpretation):
    """Your filesystem is the user's editor, not this process's disk.

    The `acp_read_text_file`, `acp_write_text_file` and `acp_run_terminal_command`
    tools go through the editor the user is sitting in front of, so a read sees
    unsaved changes in an open buffer and a write lands as an edit the user can undo.
    Prefer them to anything you might reach for in code. Every path you pass them must
    be absolute, and should be inside the directories listed below; a terminal command
    runs in the first of those.

    The user is sitting there too, so `acp_ask_user` can put a question to them and
    wait for the answer -- worth it for a decision that is theirs, and not for
    anything you could learn by reading a file.
    """

    session: ACPSession

    def _directories_section(self) -> PromptSection:
        """The session's root set, named. Computed, so it cannot be a docstring.

        This is the whole point of ACP handing `cwd` to `session/new`: without it the
        model has no idea which project it is in, and "use absolute paths" is advice
        it cannot act on.
        """
        roots = self.session.roots
        listed = (
            "\n".join(f"- `{root}`" for root in roots)
            if roots
            else "- (the editor named none; ask the user before assuming a path)"
        )
        return PromptSection(
            type="prompt_section",
            title="Directories this session is about",
            content=to_content_blocks(
                "The user opened this session on the following directories. The first "
                "is the working directory: it is what a relative path would mean, and "
                "it is where a terminal command runs.\n\n" + listed
            ),
        )

    @implements(call_system)
    def call_system(
        self, harness_prompt: PromptSection, agent_prompt: PromptSection
    ) -> typing.Any:
        """Add the session's directories, then the class docstring the base adds.

        Appending before delegating puts them immediately ahead of the docstring's
        section, so the sentence there about "the directories listed below" is
        followed by the list.
        """
        return super().call_system(
            PromptSection(
                type="prompt_section",
                title=harness_prompt["title"],
                content=[*harness_prompt["content"], self._directories_section()],
            ),
            agent_prompt,
        )

    @implements(acp_read_text_file)
    def acp_read_text_file(
        self, path: str, line: int | None = None, limit: int | None = None
    ) -> str:
        """Read through the editor, truncating an unbounded read of a long file.

        Only a read the model left unbounded is truncated: an explicit `limit`
        already asked the editor for a bounded answer, and cutting it further
        would make the parameter mean less than it says. The notice names the
        line to continue from, so paging costs the model no arithmetic.
        """
        if not self.session.fs_capabilities.read_text_file:
            raise NotImplementedError(
                "this editor cannot read files on your behalf; ask the user instead"
            )
        content = self.session.call(
            self.session.client.read_text_file(
                self.session.session_id, path, line=line, limit=limit
            )
        ).content
        if limit is not None:
            return content
        lines = content.splitlines(keepends=True)
        if len(lines) <= READ_RESULT_LIMIT_LINES:
            return content
        start = line or 1
        shown = start + READ_RESULT_LIMIT_LINES - 1
        total = start - 1 + len(lines)
        return "".join(lines[:READ_RESULT_LIMIT_LINES]) + (
            f"\n[truncated: showing lines {start}..{shown} of {total}; call again "
            f"with line={shown + 1} (and a limit, if you like) for the rest]"
        )

    @implements(acp_write_text_file)
    def acp_write_text_file(self, path: str, content: str) -> str:
        if not self.session.fs_capabilities.write_text_file:
            raise NotImplementedError(
                "this editor cannot write files on your behalf; propose the change "
                "to the user as text instead"
            )
        # Read before writing, so the editor can draw a before-and-after rather than
        # just the new text. One extra round trip to the editor, which is local and
        # fast, and it is skipped entirely when there is nothing to compare against.
        self.session.reporter.show_diff(path, content, self._previous_text(path))
        self.session.call(
            self.session.client.write_text_file(self.session.session_id, path, content)
        )
        # ACP's own response is empty, but the tool is declared to return `str` and
        # the model is shown whatever it returns: `null` reads as a call that did
        # not do anything.
        return f"wrote {len(content)} characters to {path}"

    def _previous_text(self, path: str) -> str | None:
        """What is in the file now, for a diff to be drawn against, or `None`.

        `None` covers every reason this can fail to be an answer -- the file is being
        created, the editor will not read on our behalf, the read failed -- because
        none of them is a reason to fail the *write*. A cancellation is not one of
        those: `SessionCancelled` derives from `BaseException` and passes through, so a
        cancelled turn still stops here.
        """
        if not self.session.fs_capabilities.read_text_file:
            return None
        try:
            return self.session.call(
                self.session.client.read_text_file(self.session.session_id, path)
            ).content
        except Exception:
            return None

    @implements(acp_update_plan)
    def acp_update_plan(self, steps: list[PlanStep]) -> str:
        """Replace the plan the editor is showing, and tell the model what it shows.

        Needs no capability check: the client's `plan` capability gates the incremental
        `plan_update` and `plan_removed` notifications, not this one, which is why an
        editor that advertises nothing still renders it.
        """
        self.session.notify(
            acp.update_plan(
                acp.plan_entry(step.content, priority=step.priority, status=step.status)
                for step in steps
            )
        )
        done = sum(step.status == "completed" for step in steps)
        return f"Showing the user {len(steps)} step(s), {done} of them completed."

    @implements(acp_ask_user)
    def acp_ask_user(self, message: str, fields: list[AskField]) -> str:
        """Put a form in front of the user and hand their answers back to the model.

        The wait is the point, and it is unbounded: see `ACPSession.call`, whose
        argument for a permission dialog is this one word for word. A form is a
        question for a person, and a person is allowed to think.

        Three answers come back, and the difference between the last two is the whole
        of why this is not just a permission prompt. ``accept`` is the answers.
        ``decline`` is the user saying they will not answer *this*, which is a fact
        the model should carry on from rather than a broken tool -- so it returns
        normally, with prose saying so. ``cancel`` is the form dismissed, which is
        the same gesture as dismissing a permission prompt and ends the turn.

        Raises:
            NotImplementedError: If the editor renders no forms; reported to the
                model as this call's result, like any other missing capability.
            SessionCancelled: If the user dismissed the form instead of answering it.
        """
        if not self.session.elicitation_capabilities.form:
            raise NotImplementedError(
                "this editor cannot show the user a form; ask your question in your "
                "reply instead, and stop there so they can answer it"
            )
        if not fields:
            raise ValueError(
                "ask for at least one thing; a form with no fields is a dialog the "
                "user can only dismiss"
            )
        response = self.session.call(
            self.session.client.create_elicitation(
                message,
                acp.schema.ElicitationFormSessionMode(
                    session_id=self.session.session_id,
                    # So the editor draws the form in the tool-call row it belongs to,
                    # rather than as a dialog with no visible cause. `None` outside a
                    # tool call, which the field allows.
                    tool_call_id=self.session.reporter.running,
                    requested_schema=_elicitation_schema(fields),
                ),
            )
        )
        if response.action == "cancel":
            raise SessionCancelled
        if response.action == "decline":
            return (
                "The user declined to answer. Do not ask again; either continue "
                "without their answer, saying what you assumed, or explain what you "
                "cannot decide for them."
            )
        if response.action != "accept":
            # `OtherElicitationResponse` exists for actions added after this was
            # written. Reporting the word rather than guessing which of the three it
            # resembles is the only honest thing to do with one.
            return f"The editor answered with an action this agent does not know: {response.action!r}."
        return _answers_as_text(fields, getattr(response, "content", None))

    @implements(acp_run_terminal_command)
    def acp_run_terminal_command(self, command: str, args: list[str]) -> str:
        """Run a command in the user's terminal and return its output and status."""
        if not self.session.client_capabilities.terminal:
            raise NotImplementedError(
                "this editor cannot run terminal commands on your behalf"
            )

        def release_orphan(created: concurrent.futures.Future) -> None:
            # A cancellation that lands *during* `terminal/create` interrupts the
            # wait below before the terminal id is ever known -- but the request
            # was already sent, so the editor allocates a terminal and answers.
            # Without this, that answer is discarded and the terminal leaks: the
            # `finally` cannot release an id the agent never learned. (Exactly
            # this interleaving is routine on a loaded runner, where the user's
            # cancel overtakes a worker that has only just announced the call.)
            if not created.cancelled() and created.exception() is None:
                self.session.detach(
                    self.session.client.release_terminal(
                        self.session.session_id, created.result().terminal_id
                    )
                )

        terminal = self.session.call(
            self.session.client.create_terminal(
                self.session.session_id,
                command,
                args=args,
                # Without this the command runs wherever the editor happened to spawn
                # this process, which is not the project the user opened.
                cwd=self.session.cwd or None,
            ),
            orphan=release_orphan,
        ).terminal_id
        # Hand the terminal to the editor before waiting on it: this is the one thing
        # in the protocol that renders *while* it happens. The alternative -- what this
        # did until now -- is a spinning row for however long the command takes, and
        # then its whole output at once.
        self.session.reporter.show_terminal(terminal)
        try:
            exit_status = self.session.call(
                self.session.client.wait_for_terminal_exit(
                    self.session.session_id, terminal
                )
            )
            output = self.session.call(
                self.session.client.terminal_output(self.session.session_id, terminal)
            )
        finally:
            # `detach`, not `call`: the release is owed however the turn ended,
            # and must not itself be revoked by the cancellation that ended it
            # (nor, on the way out of a *successful* command, may a late cancel
            # be allowed to discard the output by raising here).
            self.session.detach(
                self.session.client.release_terminal(self.session.session_id, terminal)
            )
        status = (
            f"exit code {exit_status.exit_code}"
            if exit_status.signal is None
            else f"killed by {exit_status.signal}"
        )
        return f"[{status}]\n{output.output}"


@dataclasses.dataclass
class ACPSessionConfig(ObjectInterpretation):
    """Apply the session's user-chosen settings to the requests it makes.

    Just the model, for now, and it is one line: `LiteLLMConfigurer` merges its own
    configuration *under* whatever the request already carries -- "the merge below
    lets a value already in `kwargs` stand" -- so a handler installed above it names
    the model by naming it. That is what turns the picker in the editor's UI from a
    label into a setting.

    A second `LiteLLMConfigurer` in `ACPSession.intp` would be the obvious way to do
    that, and cannot be. It binds its configuration at construction, where `intp` is
    built once per session and `model` changes whenever the user touches the picker,
    so the choice would freeze at whatever it was when the stack was first built.
    Reading `session.model` per request is the whole point, and it is also why
    `INHERIT_MODEL` can be honoured at all: `LiteLLMConfigurer` has no way to say
    "no opinion" -- its `model` defaults to ``gpt-4o``, which would silently overrule
    the model the launcher was configured with.

    Separate from `ACPSessionReporter`, which also handles `completion`, because these
    are opposite directions: the reporter watches a request go past and describes it,
    this alters it.
    """

    session: ACPSession

    @implements(completion)
    def completion(self, *args, **kwargs) -> typing.Any:
        if self.session.model:
            kwargs = {**kwargs, "model": self.session.model}
        return fwd(*args, **kwargs)


# ---------------------------------------------------------------------------
# Reporting the agent's activity as session/update notifications
# ---------------------------------------------------------------------------


class _PartialCall(typing.TypedDict):
    """A tool call being assembled from streaming deltas."""

    id: str
    name: str
    args: str


_TOOL_KINDS: dict[str, acp.schema.ToolKind] = {
    # The harness's own tools for running model-authored Python.
    StatefulReplSynthesizer.exec_code.__name__: "execute",
    FinalBodySynthesizer._SubmitSolutionTool.__toolname__: "execute",
    acp_read_text_file.__name__: "read",
    acp_write_text_file.__name__: "edit",
    acp_run_terminal_command.__name__: "execute",
    # `acp_ask_user` is deliberately absent: ACP's `ToolKind` vocabulary has no entry
    # for asking the user something, and `think` -- the agent reasoning -- is a
    # different thing rather than a near fit. It gets no kind at all; see `_tool_kind`.
}


ASSUMED_CONTEXT_SIZE = 128_000
"""What to assume a model's context window is when litellm has no entry for it.

A guess, and the gauge it feeds is only as good as it. The alternative is no gauge at
all for any model litellm has not catalogued -- which is most new ones, and anything
behind a gateway -- and a roughly-right gauge is worth more to someone watching their
context fill than an empty space where one should be.
"""


@functools.cache
def _context_size(model: str) -> int:
    """How many tokens of context `model` has.

    litellm's table is the only source this can consult, and it does not cover every
    model. `ASSUMED_CONTEXT_SIZE` covers the rest, and the guess is said out loud
    once, since a gauge drawn against the wrong denominator is worth knowing about.

    Cached because it is consulted after every completion and the answer never moves.
    """
    try:
        if size := int(litellm.get_model_info(model).get("max_input_tokens") or 0):
            return size
    except Exception:
        pass
    print(
        f"note: litellm does not know the context size of {model!r}; the usage gauge "
        f"assumes {ASSUMED_CONTEXT_SIZE:,} tokens",
        file=sys.stderr,
    )
    return ASSUMED_CONTEXT_SIZE


def _locations(
    raw_input: collections.abc.Mapping[str, typing.Any],
) -> list[acp.schema.ToolCallLocation]:
    """Which file a call is about, said in the field an editor reads for it.

    Editors attribute a turn's work to files -- "these three were edited" -- and offer
    to jump to them. They will guess from the raw arguments if they must (Poolside's
    looks for ``path``, ``file_path``, ``cwd`` and several more), but `locations` is
    where the answer belongs, and it is the only one that can carry a line number.

    Derived from the arguments rather than declared per tool, so a tool added later
    that takes a `path` is located without anyone remembering to do it.
    """
    path = raw_input.get("path")
    if not isinstance(path, str) or not path:
        return []
    line = raw_input.get("line")
    return [
        acp.schema.ToolCallLocation(
            path=path, line=line if isinstance(line, int) else None
        )
    ]


def _tool_kind(name: str) -> acp.schema.ToolKind | None:
    """The ACP category an editor uses to pick an icon for a tool call, if it is known.

    Keyed by the name a tool is *advertised* under, since that is the name that comes
    back in the model's reply. Every key above is read off the tool itself rather than
    written out, because a key that drifts is invisible: the call still runs, the
    editor just draws the wrong icon.

    Names are assigned per request (`_advertised_names`) and may be disambiguated, so
    this is a lookup with a fallback rather than a table that has to be exhaustive.

    `None` rather than ``"other"`` for the fallback, though both are spelled the same
    way in the protocol's own vocabulary. ``kind`` is optional in every message that
    carries it, and omitting it is how ACP says nothing is claimed; ``"other"`` is a
    positive claim about a call, and editors treat it as one -- Poolside gives it the
    terminal icon, the same one it gives ``execute``, so an unclassified call is drawn
    as a shell command. That is how `acp_ask_user`, which runs no commands at all, came
    to look like one.
    """
    return _TOOL_KINDS.get(name)


@dataclasses.dataclass
class ACPSessionReporter(ObjectInterpretation):
    """Translate the agent's activity into `session/update` notifications.

    Forces `completion` onto the streaming path so the editor sees text as it is
    produced rather than in one block at the end, and brackets every `call_tool` with
    the status transitions an editor renders as a tool-call row.
    """

    session: ACPSession

    _open: dict[str, str] = dataclasses.field(default_factory=dict)
    """Tool calls announced to the editor and not yet given a terminal status."""

    _terminals: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    """Terminals a call has opened, which are how that call renders. See `_content`."""

    _diffs: dict[str, list[ToolCallContent]] = dataclasses.field(default_factory=dict)
    """Edits a call has made, likewise. See `_content`."""

    running: str | None = None
    """The id of the call currently executing, for a tool that wants to say so."""

    finish_reason: str | None = None
    """Why the *last* completion of this turn stopped, in the provider's vocabulary."""

    tokens: collections.Counter = dataclasses.field(default_factory=collections.Counter)

    def begin_turn(self) -> None:
        """Forget the last turn. Called by the server before the worker starts.

        The reporter outlives a turn -- it belongs to the session -- but everything it
        counts is per-turn, and `usage` and `stop_reason` would otherwise report the
        whole conversation's totals as this prompt's.
        """
        self._open.clear()
        self._terminals.clear()
        self._diffs.clear()
        self.running = None
        self.finish_reason = None
        self.tokens.clear()

    def _start(self, call_id: str, name: str, **kwargs) -> None:
        """Announce a tool call, at most once per id."""
        if call_id in self._open:
            return
        self._open[call_id] = name
        self.session.notify(
            acp.start_tool_call(call_id, name, kind=_tool_kind(name), **kwargs)
        )

    def _finish(
        self, call_id: str, status: acp.schema.ToolCallStatus, text: str
    ) -> None:
        """Give an announced call a terminal status, so the editor stops waiting."""
        self._open.pop(call_id, None)
        self.session.notify(
            acp.update_tool_call(
                call_id, status=status, content=self._content(call_id, text)
            )
        )
        self._terminals.pop(call_id, None)
        self._diffs.pop(call_id, None)

    def _content(self, call_id: str, text: str) -> list[ToolCallContent]:
        """How this call should render: as what it *did*, if that can be shown.

        A terminal or a diff is not one rendering among several. The editor streams a
        terminal's output live and "continues to display it even after the terminal is
        released", and it draws a diff as a before-and-after; the text the call also
        produced is then the same information a second time -- and on the failing path
        the error is in the terminal, which is where the user is already looking.

        `ToolCallUpdate.content` replaces the collection rather than appending to it,
        so every update for such a call has to carry it again. Composing the content in
        one place is what keeps that from being remembered at each call site.
        """
        if terminals := self._terminals.get(call_id):
            return [acp.tool_terminal_ref(terminal) for terminal in terminals]
        if diffs := self._diffs.get(call_id):
            return list(diffs)
        return [acp.tool_content(acp.text_block(text))]

    def show_diff(self, path: str, new_text: str, old_text: str | None) -> None:
        """Render the call now running as an edit to `path`.

        `old_text` may be `None` -- for a file being created, or an editor that would
        not say what was there before. The editor then draws the new content alone,
        which is less useful than a diff and still better than a line of prose saying
        a write happened.
        """
        if (call_id := self.running) is None:
            return
        self._diffs.setdefault(call_id, []).append(
            acp.tool_diff_content(path, new_text, old_text)
        )
        self.session.notify(
            acp.update_tool_call(call_id, content=self._content(call_id, ""))
        )

    def show_terminal(self, terminal_id: str) -> None:
        """Render the call now running as this live terminal.

        Called by `ACPToolRuntime` the moment the editor hands back a terminal id, so
        the user watches the command run instead of watching a spinner. A no-op when
        no call is running -- the tool is reachable outside a tool call, and a terminal
        with no row to attach to is not an error.
        """
        if (call_id := self.running) is None:
            return
        self._terminals.setdefault(call_id, []).append(terminal_id)
        self.session.notify(
            acp.update_tool_call(call_id, content=self._content(call_id, ""))
        )

    def abandon(self) -> None:
        """Fail every call still open. Called once the turn is over, however it ended.

        A cancelled turn, or one whose skill raised, leaves calls the editor was told
        had started and never hears about again -- rendered as a row that spins for
        the rest of the session. ACP's `ToolCallStatus` has no ``cancelled``, so
        ``failed`` is the only terminal status available to say so.
        """
        for call_id in list(self._open):
            self._finish(call_id, "failed", "the turn ended before this call finished")

    def stop_reason(self) -> acp.schema.StopReason:
        """Why this turn ended, for a turn the model itself brought to a close.

        The two interesting answers are the ones an editor cannot infer: a reply cut
        off at the token limit and a reply the provider refused both arrive as an
        *answer*, and reporting either as `end_turn` tells the user their question was
        answered when it was not.
        """
        if self.finish_reason == "length":
            return "max_tokens"
        if self.finish_reason == "content_filter":
            return "refusal"
        return "end_turn"

    def usage(self) -> acp.schema.Usage | None:
        """This turn's token counts, or `None` if no provider reported any."""
        if not self.tokens:
            return None
        return acp.schema.Usage(
            input_tokens=self.tokens["prompt_tokens"],
            output_tokens=self.tokens["completion_tokens"],
            total_tokens=self.tokens["total_tokens"],
        )

    def _account(self, response: typing.Any) -> None:
        """Record what one completion cost and why it stopped.

        Read off the value handed back rather than the request, so the streamed and
        unstreamed paths are accounted identically: `litellm.stream_chunk_builder`
        rebuilds both fields onto the response it assembles from the chunks.
        """
        choices = getattr(response, "choices", None) or []
        if choices:
            self.finish_reason = getattr(choices[0], "finish_reason", None)
        if usage := getattr(response, "usage", None):
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self.tokens[field] += getattr(usage, field, 0) or 0
        self._report_context(response)

    def _report_context(self, response: typing.Any) -> None:
        """Say how full the model's context is, after each request.

        Not the same number as `usage`, which is what the whole turn *cost* and is
        reported once at the end. This is how much room is left, reported as it fills,
        and an editor draws it as a gauge -- the difference between "that turn was
        expensive" and "you are nearly out of context".

        Everything here is skipped rather than guessed when it cannot be known.
        `size` is required by the protocol, and a denominator nobody measured makes a
        gauge that lies; litellm does not know every model, and the response is the
        only place the model's real name can be read once a picker has changed it.
        """
        usage = getattr(response, "usage", None)
        used = getattr(usage, "prompt_tokens", 0) or 0
        model = getattr(response, "model", None)
        if not used or not model:
            return
        size = _context_size(model)
        cost = None
        with contextlib.suppress(Exception):
            if amount := litellm.completion_cost(completion_response=response):
                cost = acp.schema.Cost(amount=float(amount), currency="USD")
        self.session.notify(
            acp.schema.UsageUpdate(
                session_update="usage_update", used=used, size=int(size), cost=cost
            )
        )

    @implements(completion)
    def completion(self, *args, **kwargs) -> typing.Any:
        """Stream this request, reporting deltas as they arrive.

        Streaming is something this handler *adds* to a request that did not ask for
        it, so it also owns the cost: a broken stream falls back to an ordinary
        unstreamed request rather than failing a call that would have succeeded. The
        retry is safe because a broken stream produced no result to duplicate.

        This is also the turn's meter -- every request the loop makes passes through
        here exactly once -- so it is where what the last reply cost and why it
        stopped are recorded, for `usage` and `stop_reason` to report.
        """
        if self.session.cancel.is_set():
            raise SessionCancelled

        try:
            response = self._streamed(*args, **kwargs)
        except (
            litellm.exceptions.MidStreamFallbackError,
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.Timeout,
        ):
            # Deliberately narrow: a refused request, a bad model name or a rejected
            # response schema fails identically unstreamed, and re-issuing it would
            # only pay for the same error twice.
            response = fwd(*args, **kwargs)
        self._account(response)
        return response

    def _streamed(self, *args, **kwargs) -> typing.Any:
        # `response_format` is None exactly when the skill returns `str` (see
        # `call_assistant`). Any other answer is JSON shaped like the response
        # format, and streaming it would show the editor a `{"value": ...}` wrapper
        # being typed out; the decoded value is reported once, by the server.
        is_prose = kwargs.get("response_format") is None
        # `include_usage` is what makes the gauge report the provider's own numbers.
        # Without it a stream carries no usage block at all, and the counts come from
        # `stream_chunk_builder` tokenizing the request locally -- an estimate that
        # cannot see cache reads or a provider's own accounting. Asking costs one extra
        # chunk, whose `choices` are empty; the loop below appends before it skips
        # those, so it still reaches the builder. A provider that does not understand
        # the option has it dropped rather than refused, since the launcher sets
        # `litellm.drop_params`.
        stream = fwd(
            *args,
            **{
                "stream_options": {"include_usage": True},
                **kwargs,
                "stream": True,
            },
        )

        # Asking for a stream does not guarantee getting one: an inner handler may
        # answer from a cache or a fixture and hand back a settled response, ignoring
        # the flag this handler added. Report that in one go rather than trying to
        # iterate a response object.
        if isinstance(stream, litellm.types.utils.ModelResponse):
            return self._settled(stream, is_prose=is_prose)

        chunks: list[typing.Any] = []
        calls: dict[int, _PartialCall] = {}
        for chunk in stream:
            if self.session.cancel.is_set():
                raise SessionCancelled
            chunks.append(chunk)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            if delta.content and is_prose:
                self.session.notify(acp.update_agent_message_text(delta.content))
            if reasoning := getattr(delta, "reasoning_content", None):
                self.session.notify(acp.update_agent_thought_text(reasoning))

            for fragment in delta.tool_calls or []:
                slot = calls.setdefault(
                    fragment.index, {"id": "", "name": "", "args": ""}
                )
                slot["id"] = getattr(fragment, "id", None) or slot["id"]
                if function := getattr(fragment, "function", None):
                    slot["name"] = function.name or slot["name"]
                    slot["args"] += function.arguments or ""
                if not slot["id"] or not slot["name"]:
                    continue
                self._start(slot["id"], slot["name"], status="pending")
                try:
                    raw_input = pydantic_core.from_json(
                        slot["args"], allow_partial="trailing-strings"
                    )
                except ValueError:
                    raw_input = None
                self.session.notify(
                    acp.update_tool_call(slot["id"], raw_input=raw_input)
                )

        return litellm.stream_chunk_builder(chunks, messages=kwargs.get("messages"))

    def _settled(
        self, response: litellm.types.utils.ModelResponse, *, is_prose: bool
    ) -> typing.Any:
        """Report a response that arrived whole, and hand it back unchanged."""
        choice = response.choices[0]
        if not isinstance(choice, litellm.types.utils.Choices):
            return response
        if (content := choice.message.get("content")) and is_prose:
            self.session.notify(acp.update_agent_message_text(content))
        if reasoning := choice.message.get("reasoning_content"):
            self.session.notify(acp.update_agent_thought_text(reasoning))
        for raw in choice.message.get("tool_calls") or []:
            self._start(str(raw.id), raw.function.name or "?", status="pending")
        return response

    @implements(call_tool)
    def call_tool(self, tool_call: DecodedToolCall) -> typing.Any:
        """Bracket the call with the status transitions an editor renders.

        A failed call arrives here two ways, and both have to end as ``failed``. This
        handler sits *above* `TenacityRetryer` -- `EffectfulACPAgent._answer` installs
        the session's stack on top of the harness's -- and the retryer's whole job is
        to turn a raising tool into that call's result, so on the ordinary path the
        exception never reaches this ``except``: it comes back as a perfectly normal
        return whose `result` is the error. Reporting on the exception alone would
        show the user every failed call as completed, with the traceback rendered as
        its output.
        """
        if self.session.cancel.is_set():
            raise SessionCancelled
        # A call announced while streaming carries only the name, since its
        # arguments were still arriving; now that they are decoded, say what the
        # call actually is.
        self._start(tool_call.id, tool_call.name, status="pending")
        raw_input = _raw_input(tool_call)
        self.session.notify(
            acp.update_tool_call(
                tool_call.id,
                status="in_progress",
                title=_call_title(tool_call.name, raw_input),
                raw_input=raw_input,
                locations=_locations(raw_input),
            )
        )
        # Named while it runs, so a tool that has something to show -- a terminal --
        # can find the row it belongs to. Restored rather than cleared, since a tool
        # may itself call a Skill whose own tool calls nest inside this one.
        outer, self.running = self.running, tool_call.id
        try:
            message, result, is_final = fwd(tool_call)
        except ToolCallExecutionError as e:
            self._finish(tool_call.id, "failed", str(e))
            raise
        finally:
            self.running = outer

        self._finish(
            tool_call.id,
            "failed" if isinstance(result, ToolCallExecutionError) else "completed",
            _as_text(message),
        )
        return (message, result, is_final)


def _as_text(message: typing.Any) -> str:
    """A message's content as a string, however it was encoded.

    `~effectful.handlers.llm.harness.hooks.call_tool` encodes a result into content
    blocks, so it may be a list rather than a string -- an image tool returns one.

    A *missing* content is the empty string, not the JSON below. An assistant turn
    that only called tools has ``content: None``, and rendering that as ``"null"`` put
    the literal word into the editor for every such turn a reloaded session replayed.
    """
    content = message.get("content") if hasattr(message, "get") else None
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            _part_as_text(part) for part in content if isinstance(part, dict)
        )
    return json.dumps(content, default=str)


def _part_as_text(part: collections.abc.Mapping[str, typing.Any]) -> str:
    """One content block as text: its own if it has any, else a note that it exists.

    A tool result is not always text. `call_tool` encodes one that returns an image as
    an ``image_url`` block, and reading only the ``text`` key across the blocks turned
    that into the empty string -- so the editor rendered the call as having produced
    nothing at all, which is the same thing it renders for a tool that printed
    nothing. The model still receives the image either way; this is what stands in for
    it on the screen.

    A placeholder rather than the block itself because the content here is bound for a
    text block. Handing the editor a real `acp.image_block` is a change to
    `ACPSessionReporter._content`, and worth making for a client that renders one --
    VS Code's does not, and would show nothing where this shows ``[image/png]``.
    """
    if (kind := part.get("type")) == "text":
        return part.get("text") or ""
    if kind == "image_url":
        url = part.get("image_url")
        url = url.get("url", "") if isinstance(url, dict) else (url or "")
        media = url[len("data:") :].split(";", 1)[0] if url.startswith("data:") else ""
        return f"[{media or 'image'}]"
    return f"[{kind or 'attachment'}]"


# ---------------------------------------------------------------------------
# Asking the editor's user before running a tool
# ---------------------------------------------------------------------------


def _raw_input(tool_call: DecodedToolCall) -> dict[str, typing.Any]:
    """A call's arguments as the data an editor renders, back from their decoded form.

    Round-tripped through the encoding the model was given rather than read off
    `bound_args`, so what the editor is shown is what the model actually said -- a
    code object comes back as its source, an image as its reference.
    """
    return json.loads(
        pydantic.TypeAdapter(Encodable[DecodedToolCall]).dump_python(
            tool_call, mode="json", context={}
        )["function"]["arguments"]
    )


def _call_title(name: str, raw_input: collections.abc.Mapping[str, typing.Any]) -> str:
    """A one-line description of a call: what it is, and what it was given.

    The bare tool name is not enough, and not only because it is terse. Clients treat
    a title that is *only* an identifier as a placeholder and replace it with phrasing
    of their own -- Poolside tests it against ``/^[a-z][a-z0-9_]*$/`` and, on a match,
    ignores it -- so `exec_code` is discarded and rendered as "Run exec_code", with the
    code nowhere on screen. A title with arguments in it survives that test.

    It also has to carry the arguments because nothing else reliably does. `rawInput`
    is sent on every call, but a client that has tool *output* to show may prefer it:
    Poolside renders a command, then content, then raw input, whichever comes first,
    and every call of ours ends with content. So this is where the arguments are.
    """
    return f"{name}({', '.join(f'{key}={_abbreviate(value)}' for key, value in raw_input.items())})"


def _abbreviate(value: typing.Any, limit: int = 60) -> str:
    """One argument, short enough to sit in a title."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}…"


@dataclasses.dataclass
class ACPPermissionGate(ObjectInterpretation):
    """Ask the editor's user to approve each tool call before it runs.

    A handler rather than a tool, because unlike the editor capabilities above this is
    not something the model calls -- it gates *every* tool call, including the agent's
    own and the harness's ``exec_code``.
    """

    session: ACPSession

    _standing: dict[str, bool] = dataclasses.field(default_factory=dict)

    permission_options: typing.ClassVar[
        collections.abc.Sequence[acp.schema.PermissionOption]
    ] = (
        acp.schema.PermissionOption(
            option_id="allow_once", name="Allow", kind="allow_once"
        ),
        acp.schema.PermissionOption(
            option_id="allow_always", name="Always allow", kind="allow_always"
        ),
        acp.schema.PermissionOption(
            option_id="reject_once", name="Reject", kind="reject_once"
        ),
        acp.schema.PermissionOption(
            option_id="reject_always", name="Always reject", kind="reject_always"
        ),
    )

    @implements(call_tool)
    def call_tool[T](self, tool_call: DecodedToolCall[T]) -> T:
        """Run the call if it is approved; otherwise make the tool itself refuse.

        A `Tool` is an `Operation`, and `call_tool` invokes it, so handling it is all
        a refusal takes: the call then fails the way any raising tool fails, and
        `TenacityRetryer` reports it to the model as that call's result. Forwarding
        rather than answering here is what keeps `HistoryBuilder` in the loop, so the
        declined call is still answered and the conversation stays sendable.
        """
        with handler({tool_call.tool: self._decide(tool_call)}):
            return fwd(tool_call)

    def _decide[T](
        self, tool_call: DecodedToolCall[T]
    ) -> collections.abc.Callable[..., T]:
        """What should run for this call: the tool itself, or a refusal in its place.

        The session's mode is consulted before the user is, since a mode is the user
        having answered these prompts in advance -- that is the whole of what picking
        one means. `UNGATED_TOOLS` comes before even that, for the one tool whose
        whole purpose is to ask the user something.

        Raises:
            SessionCancelled: If the user dismissed the prompt instead of answering it.
        """

        def refused(
            exc: Exception, *args: typing.Any, **kwargs: typing.Any
        ) -> typing.NoReturn:
            raise exc

        # Before the mode, because this is not a decision a mode makes: `UNGATED_TOOLS`
        # is about a tool whose only effect is to ask the user, and Plan mode is as
        # entitled to ask as Auto is.
        if tool_call.name in UNGATED_TOOLS:
            return lambda *a, **k: fwd()
        if self.session.mode_id == AUTO:
            return lambda *a, **k: fwd()
        if self.session.mode_id == PLAN and tool_call.name in MUTATING_TOOLS:
            return functools.partial(
                refused,
                PermissionError(
                    f"The call to `{tool_call.name}` did not run: this session is in "
                    f"Plan mode, which changes nothing in the user's editor. Say what "
                    f"you would do and why; the user can switch to Ask or Auto mode if "
                    f"they want it done."
                ),
            )

        standing = self._standing.get(tool_call.name)
        if standing is True:
            return lambda *a, **k: fwd()
        if standing is False:
            return functools.partial(
                refused,
                PermissionError(
                    f"The call to `{tool_call.name}` did not run: the user declined it earlier. Do not retry it; either continue without it, or explain what you cannot do and why."
                ),
            )

        raw_input = _raw_input(tool_call)
        # No bound on this wait, deliberately: it is a dialog in front of a person.
        # See `ACPSession.call`.
        response = self.session.call(
            self.session.client.request_permission(
                self.session.session_id,
                acp.schema.ToolCallUpdate(
                    tool_call_id=tool_call.id,
                    title=_call_title(tool_call.name, raw_input),
                    kind=_tool_kind(tool_call.name),
                    raw_input=raw_input,
                ),
                options=list(self.permission_options),
            )
        )

        outcome = response.outcome
        # A `cancelled` outcome is the user dismissing the prompt, not rejecting the
        # call; the turn is over either way.
        if outcome.outcome != "selected":
            raise SessionCancelled

        allowed = outcome.option_id.startswith("allow")
        if outcome.option_id.endswith("always"):
            self._standing[tool_call.name] = allowed
        if allowed:
            return lambda *a, **k: fwd()
        return functools.partial(
            refused,
            PermissionError(
                f"The call to `{tool_call.name}` did not run: the user declined it. Do not retry it; either continue without it, or explain what you cannot do and why."
            ),
        )


# ---------------------------------------------------------------------------
# Remembering that a session existed
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


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
        role, text = message.get("role"), _as_text(message)
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
                    kind=_tool_kind(name),
                    # Completed, because a stored call is one that already ran: the
                    # turn it belonged to is over, whatever became of the call.
                    status="completed",
                    raw_input=raw_input,
                )
        elif role == "tool" and (call_id := message.get("tool_call_id")) is not None:
            yield acp.update_tool_call(
                str(call_id),
                content=[acp.tool_content(acp.text_block(text))],
            )


@dataclasses.dataclass(frozen=True)
class SlashCommand:
    """One ``/name`` command: what the editor is told about it, and what runs it.

    The pairing is the point. A command exists in two conversations -- it is
    *advertised* (`available_commands_update`, so the editor can offer it while the
    user types) and it is *dispatched* (`EffectfulACPAgent._command`, when a prompt
    arrives spelling it) -- and holding both halves in one value is what makes
    those agree by construction. Kept as separate lists, adding a command to one
    and forgetting the other would produce an editor offering something answered
    with "Unknown command", and nothing anywhere would fail.
    """

    spec: acp.schema.AvailableCommand
    """The advertisement: name, description, and the input hint, if any."""

    run: collections.abc.Callable[["EffectfulACPAgent", ACPSession, str], str]
    """The behaviour: handed the server, the session and the argument text."""


def _run_clear(server: "EffectfulACPAgent", session: ACPSession, argument: str) -> str:
    session.agent.__history__.clear()
    return "Cleared. I have forgotten the conversation up to here."


def _run_status(server: "EffectfulACPAgent", session: ACPSession, argument: str) -> str:
    roots = "\n".join(f"- `{root}`" for root in session.roots) or "- (none)"
    mode = next(
        (m.name for m in SESSION_MODES if m.id == session.mode_id),
        session.mode_id,
    )
    model = session.model or "as configured at launch"
    return (
        f"**Mode** {mode}\n\n**Model** {model}\n\n"
        f"**Directories**\n{roots}\n\n"
        f"**Messages so far** {len(session.agent.__history__)}"
    )


def _run_mode(server: "EffectfulACPAgent", session: ACPSession, mode_id: str) -> str:
    """``/mode`` with no argument reports; with one, switches.

    The editor's own picker sends `session/set_config_option`, and this sends
    nothing -- it is already inside the agent. What it must do instead is *say* the
    mode changed, on both channels a client might be listening to:
    `current_mode_update` for one that reads `modes`, and the config options for
    one that reads those.
    """
    offered = {mode.id: mode for mode in SESSION_MODES}
    if not mode_id:
        return "\n".join(
            [f"**Mode** {offered[session.mode_id].name}", ""]
            + [f"- `/mode {mode.id}` — {mode.description}" for mode in SESSION_MODES]
        )
    if mode_id not in offered:
        return (
            f"No such mode `{mode_id}`. Try "
            f"{', '.join(f'`{mode_id}`' for mode_id in offered)}."
        )
    session.mode_id = mode_id
    session.notify(
        acp.schema.CurrentModeUpdate(
            session_update="current_mode_update", current_mode_id=mode_id
        )
    )
    session.notify(
        acp.schema.ConfigOptionUpdate(
            session_update="config_option_update",
            config_options=server._config_options(session),
        )
    )
    return f"Mode is now **{offered[mode_id].name}**. {offered[mode_id].description}"


_SLASH_COMMANDS: dict[str, SlashCommand] = {
    command.spec.name: command
    for command in (
        SlashCommand(
            spec=acp.schema.AvailableCommand(
                name="clear",
                description="Forget the conversation so far, keeping this session open.",
            ),
            run=_run_clear,
        ),
        SlashCommand(
            spec=acp.schema.AvailableCommand(
                name="status",
                description="Show the mode, model and directories this session is using.",
            ),
            run=_run_status,
        ),
        SlashCommand(
            spec=acp.schema.AvailableCommand(
                name="mode",
                description="Switch how much this agent may do without asking.",
                # A command may take an argument, and the hint is what the editor
                # shows after the name while the user is typing it. Derived from
                # `SESSION_MODES`, as everything mode-shaped is.
                input=acp.schema.AvailableCommandInput(
                    acp.schema.UnstructuredCommandInput(
                        hint=" | ".join(mode.id for mode in SESSION_MODES)
                    )
                ),
            ),
            run=_run_mode,
        ),
    )
}
"""Every command, dispatch and advertisement together. See `SlashCommand`."""

SLASH_COMMANDS: tuple[acp.schema.AvailableCommand, ...] = tuple(
    command.spec for command in _SLASH_COMMANDS.values()
)
"""The advertised half of `_SLASH_COMMANDS`, in the shape the notification takes."""


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
            current_mode_id=session.mode_id, available_modes=list(SESSION_MODES)
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
                    for mode in SESSION_MODES
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
                available_commands=list(SLASH_COMMANDS),
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
        return acp.schema.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=self.agent_capabilities,
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
        SessionIndex.record(session)
        for update in _replay(session.agent.__history__):
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
            RequestError: If `mode_id` is not one of `SESSION_MODES`.
        """
        if mode_id not in {mode.id for mode in SESSION_MODES}:
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
                self._retitle(session, text)
                if (answered := self._command(session, text)) is not None:
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
        in `_SLASH_COMMANDS` is the whole mechanism; the same table is what
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
        command = _SLASH_COMMANDS.get(name)
        if command is None:
            return f"Unknown command `/{name}`. Try {', '.join(f'`/{c.name}`' for c in SLASH_COMMANDS)}."
        return command.run(self, session, argument)

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
        return acp.schema.CloseSessionResponse()

    async def serve(self) -> None:
        """Serve one agent over stdio until the editor disconnects.

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

        # `run_agent`'s parameters are named from the client's point of view: the stream
        # the client reads is the one this agent writes.
        await acp.run_agent(
            self,
            input_stream=writer,
            output_stream=reader,
            use_unstable_protocol=True,
        )
