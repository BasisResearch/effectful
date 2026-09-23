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

## Two modules

This is the half of the server that may change while it runs: under
``--autoreload`` an edit here -- a new slash command, a tool, a change to how a
turn is reported -- reaches every open session at its next turn. `server.py` is the
half that holds the running server, and its names are re-exported here.
"""

import collections.abc
import concurrent.futures
import contextlib
import dataclasses
import enum
import functools
import inspect
import json
import sys
import types
import typing

import acp
import acp.schema
import litellm
import pydantic
import pydantic_core
from server import (  # noqa: F401 -- re-exported, so this module is the one to import
    FLUSH_TIMEOUT,
    INHERIT_MODEL,
    MODE_OPTION_ID,
    MODEL_OPTION_ID,
    OFFER_MODELS_ENV,
    ACPSession,
    Attachment,
    ConfigOption,
    ContentBlock,
    EffectfulACPAgent,
    SessionCancelled,
    SessionIndex,
    _offered_models,
    _prompt_parts,
    _title_from,
)

from effectful.handlers.llm import Agent, Encodable, Tool
from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
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

type ToolCallContent = (
    acp.schema.ContentToolCallContent
    | acp.schema.FileEditToolCallContent
    | acp.schema.TerminalToolCallContent
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


class Mode(enum.StrEnum):
    """How much a session may do without asking; see `ACPPermissionGate`."""

    ASK = "ask"
    AUTO = "auto"
    PLAN = "plan"


SESSION_MODES: tuple[acp.schema.SessionMode, ...] = (
    acp.schema.SessionMode(
        id=Mode.ASK,
        name="Ask",
        description="Ask before running each tool.",
    ),
    acp.schema.SessionMode(
        id=Mode.AUTO,
        name="Auto",
        description="Run tools without asking. Undo is your editor's.",
    ),
    acp.schema.SessionMode(
        id=Mode.PLAN,
        name="Plan",
        description="Read and discuss, but change nothing: no writes, no commands.",
    ),
)
"""The modes a session offers; the first is a new session's, and the fallback for one
whose mode an edit removed.

Here rather than in `server.py` so that an edit reaches open sessions: `reload` pushes
the list as a `config_option_update`. ACP has no update for the legacy `modes` list,
so a client reading only that sees the change from its next session.
"""


MUTATING_TOOLS = frozenset(
    {acp_write_text_file.__name__, acp_run_terminal_command.__name__}
)
"""What `Mode.PLAN` refuses: the tools that change the user's editor or machine.

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
has. `Mode.PLAN` leaves it alone for the same reason -- asking changes nothing, and a
session that may not act is exactly where a clarifying question is worth most.
"""


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

    @implements(call_system)
    def call_system(self, *args, **kwargs) -> typing.Any:
        """After a reload, replace the conversation's system message with this one.

        `HistoryBuilder` keeps the system message of a history's first call, so
        without this an edit to a docstring would reach new sessions only. The
        history is the transaction's buffer here, which adopts the replacement when
        the call commits. Only the turn's first call does it: a nested call on the
        same agent shares the history.
        """
        message = fwd(*args, **kwargs)
        if self.session.new_code:
            self.session.new_code = False
            history = HistoryBuilder.get_history()
            if history and history[0]["role"] == "system":
                history[0] = message
        return message


def session_handlers(
    session: ACPSession,
) -> tuple["ACPSessionReporter", Interpretation]:
    """The handlers a prompt on `session` runs under, and the reporter among them.

    Built for each session rather than per type: every handler closes over the
    session, which is how three module-level concerns -- the tools, the reporting,
    the permission gate -- reach *this* editor. `ACPSession.install_handlers` calls
    this when the session opens, and again after a reload.

    `ACPPermissionGate` goes on last, so it is outermost: it must decide about a call
    before `ACPSessionReporter` announces it as running.
    """
    reporter = ACPSessionReporter(session)
    h = coproduct(reporter, ACPToolRuntime(session))
    h = coproduct(h, ACPSessionConfig(session))
    h = coproduct(h, ACPPermissionGate(session))
    return reporter, h


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
        if self.session.mode_id == Mode.AUTO:
            return lambda *a, **k: fwd()
        if self.session.mode_id == Mode.PLAN and tool_call.name in MUTATING_TOOLS:
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

    run: collections.abc.Callable[[EffectfulACPAgent, ACPSession, str], str]
    """The behaviour: handed the server, the session and the argument text."""

    fn: collections.abc.Callable[..., str]
    """The registered function, whose still being defined keeps the command on offer."""

    @property
    def current(self) -> bool:
        """Whether `fn` is still what its module defines under its name.

        ``--autoreload`` re-runs a module, which registers its commands again but
        unregisters none: not one it no longer defines, and not the old name of one it
        renamed. This is what drops those. A command defined inside a function cannot
        be checked this way, and counts as current.
        """
        if self.fn.__qualname__ != self.fn.__name__:
            return True
        module = sys.modules.get(self.fn.__module__)
        return module is None or vars(module).get(self.fn.__name__) is self.fn


_SLASH_COMMANDS: dict[str, SlashCommand] = {}
"""Every command, dispatch and advertisement together, filled by `register_command`."""


type _Command = collections.abc.Callable[
    typing.Concatenate[EffectfulACPAgent, ACPSession, ...], str
]


@typing.overload
def register_command[F: _Command](fn: F, /) -> F: ...


@typing.overload
def register_command[F: _Command](
    *, name: str | None = None, hint: str | None = None
) -> collections.abc.Callable[[F], F]: ...


def register_command[F: _Command](
    fn: F | None = None, /, *, name: str | None = None, hint: str | None = None
) -> F | collections.abc.Callable[[F], F]:
    """Register `fn` as ``/name``, advertised as an ACP `AvailableCommand`.

    The name defaults to the function's; the description is the docstring's first
    paragraph. `fn` takes ``(server, session)`` then any number of positional `str`
    or `StrEnum` parameters, optionally ``| None``, filled from the argument text split
    on whitespace, the last taking the remainder; missing trailing ones take their
    defaults. The hint shows each as ``<required>`` or ``[optional]``, by name or, for
    an enum, by its values, unless `hint` is given.
    """

    def register(fn: F) -> F:
        command = name or fn.__name__
        # Dispatch reads the name up to the first space after the slash.
        if command.split() != [command] or command.startswith("/"):
            raise ValueError(f"{command!r} is not a slash command name")
        description = " ".join((inspect.getdoc(fn) or "").split("\n\n")[0].split())
        if not description:
            raise TypeError(f"/{command} needs a docstring to describe it")
        params = list(inspect.signature(fn).parameters.values())[2:]
        kinds = [_argument_type(command, param) for param in params]
        required = sum(param.default is param.empty for param in params)
        usage = hint or " ".join(
            _placeholder(param, kind) for param, kind in zip(params, kinds)
        )
        input = None
        if params:
            input = acp.schema.AvailableCommandInput(
                acp.schema.UnstructuredCommandInput(hint=usage)
            )

        def run(server: EffectfulACPAgent, session: ACPSession, argument: str) -> str:
            # maxsplit=-1 for a nullary command splits fully, so any word is too many.
            words = argument.split(maxsplit=len(params) - 1)
            if not params and words:
                return f"`/{command}` takes no argument."
            if not required <= len(words) <= len(params):
                return f"Usage: `/{command} {usage}`"
            values = []
            for param, kind, word in zip(params, kinds, words):
                try:
                    values.append(kind(word))
                except ValueError:
                    # Only an enum's constructor refuses a string.
                    members = typing.cast(type[enum.StrEnum], kind)
                    choices = ", ".join(f"`{member.value}`" for member in members)
                    return f"`{word}` is not a valid {param.name}. Try {choices}."
            return fn(server, session, *values)

        _SLASH_COMMANDS[command] = SlashCommand(
            spec=acp.schema.AvailableCommand(
                name=command, description=description, input=input
            ),
            run=run,
            fn=fn,
        )
        return fn

    return register if fn is None else register(fn)


def _argument_type(command: str, param: inspect.Parameter) -> type[str]:
    """The `str` subclass a command's `param` is parsed as, from its annotation."""
    if param.kind is not param.POSITIONAL_OR_KEYWORD:
        raise TypeError(f"/{command}: `{param.name}` must be positional")
    if param.annotation is param.empty:
        return str
    union = isinstance(param.annotation, types.UnionType)
    kinds = [
        kind
        for kind in (typing.get_args(param.annotation) if union else [param.annotation])
        if kind is not type(None)
    ]
    if len(kinds) == 1 and isinstance(kinds[0], type) and issubclass(kinds[0], str):
        return kinds[0]
    raise TypeError(f"/{command}: `{param.name}` must be a `str` or a `StrEnum`")


def _placeholder(param: inspect.Parameter, kind: type[str]) -> str:
    """How `param` appears in a command's hint."""
    label = (
        " | ".join(member.value for member in kind)
        if issubclass(kind, enum.StrEnum)
        else param.name
    )
    return f"<{label}>" if param.default is param.empty else f"[{label}]"


@register_command
def clear[A: "Agent"](server: "EffectfulACPAgent[A]", session: ACPSession[A]) -> str:
    """Forget the conversation so far, keeping this session open."""
    session.agent.__history__.clear()
    return "Cleared. I have forgotten the conversation up to here."


@register_command
def status[A: "Agent"](server: "EffectfulACPAgent[A]", session: ACPSession[A]) -> str:
    """Show the mode, model and directories this session is using."""
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


@register_command
def mode[A: "Agent"](
    server: "EffectfulACPAgent[A]", session: ACPSession[A], mode: Mode | None = None
) -> str:
    """Switch how much this agent may do without asking.

    With no argument it lists the modes instead. The editor's own picker sends
    `session/set_config_option`, and this sends nothing -- it is already inside the
    agent. What it must do instead is *say* the mode changed, on both channels a
    client might be listening to: `current_mode_update` for one that reads `modes`,
    and the config options for one that reads those.
    """
    offered = {m.id: m for m in SESSION_MODES}
    if mode is None:
        return "\n".join(
            [f"**Mode** {offered[session.mode_id].name}", ""]
            + [f"- `/mode {m.id}` — {m.description}" for m in SESSION_MODES]
        )
    session.mode_id = mode.value
    session.notify(
        acp.schema.CurrentModeUpdate(
            session_update="current_mode_update", current_mode_id=mode.value
        )
    )
    session.notify(
        acp.schema.ConfigOptionUpdate(
            session_update="config_option_update",
            config_options=server._config_options(session),
        )
    )
    chosen = offered[mode.value]
    return f"Mode is now **{chosen.name}**. {chosen.description}"


def slash_commands() -> dict[str, SlashCommand]:
    """The commands on offer now, by name: those registered and still current.

    Read when the commands are announced and when one is dispatched, rather than
    fixed when this module is imported, so a command registered later -- from the
    agent's own file -- is offered too.
    """
    return {name: c for name, c in _SLASH_COMMANDS.items() if c.current}
