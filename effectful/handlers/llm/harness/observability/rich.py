import ast
import codeop
import collections.abc
import dataclasses
import json
import sys
import threading
import time
import typing

import litellm
import pydantic_core
import rich.console
import rich.live
import rich.markdown
import rich.panel
import rich.segment
import rich.spinner
import rich.styled
import rich.syntax
import rich.text

from effectful.handlers.llm.harness.hooks import (
    Message,
    completion,
)
from effectful.handlers.llm.harness.observability.dump import _message_text
from effectful.handlers.llm.harness.serialization import _BoxedResponse
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class _PartialToolCall(typing.TypedDict):
    """A tool call being assembled from streamed deltas (name + raw JSON args)."""

    name: str
    args: str


class _PartialAssistant(typing.TypedDict):
    """The in-progress assistant turn accumulated from streaming deltas.

    ``tool_calls`` is keyed by each tool call's streaming ``index`` so fragments
    for the same call (which arrive across many chunks) coalesce.
    """

    content: str
    reasoning_content: str
    tool_calls: dict[int, _PartialToolCall]


def _accumulate(
    partial: _PartialAssistant, delta: litellm.types.utils.Delta | None
) -> None:
    """Fold one streaming ``delta`` into the in-progress assistant ``partial``.

    Concatenates ``content`` and ``reasoning_content``, and accumulates each
    streamed tool-call fragment (``function.name`` / ``function.arguments``) into
    ``partial["tool_calls"]`` keyed by the tool call's ``index``.
    """
    if delta is None:
        return
    partial["content"] += delta.content or ""
    # `reasoning_content` is absent (not just None) on deltas that carry none.
    partial["reasoning_content"] += getattr(delta, "reasoning_content", None) or ""
    for tc in delta.tool_calls or []:
        function = getattr(tc, "function", None)
        slot = partial["tool_calls"].setdefault(tc.index, {"name": "", "args": ""})
        if function is not None:
            slot["name"] = function.name or slot["name"]
            slot["args"] += function.arguments or ""


# Panel border colors keyed by message role.
_ROLE_STYLES = {
    "system": "grey50",
    "user": "cyan",
    "assistant": "green",
    "tool": "yellow",
}


# Longest field body rendered in a panel before it is truncated, keeping
# large-but-static messages (notably the system prompt) from dominating the
# frame. Each renderable truncates with its own native mechanism: `Syntax` by
# whole lines (`line_range`); `Markdown` has none, so its source is clipped by
# whole lines.
_MAX_LINES = 40


def _tail_lines(text: str, limit: int | None) -> tuple[str, int]:
    """The last `limit` lines of `text`, and how many were dropped to get there.

    The head-truncation the settled panels use is wrong for a turn still being
    streamed: what a reader is watching is the end, so it is the beginning that
    gives way. The count comes back with the text because the caller is not the
    one that displays it -- `_Tail` is -- and a window that hides a growing
    amount while reporting only its own share of it reports a number that never
    grows.
    """
    lines = text.splitlines()
    if limit is None or len(lines) <= limit:
        return text, 0
    return "\n".join(lines[-limit:]), len(lines) - limit


def _syntax(code: str, lexer: str, *, clip: bool = True) -> rich.console.RenderableType:
    """Syntax-highlight `code` using the terminal palette, truncated to
    `_MAX_LINES` via `Syntax.line_range` (no parsing, safe on partial input).

    ``clip=False`` leaves it whole, for the live panel: there the head is the
    wrong end to keep, and `_Tail` is already bounding the panel by rendered
    rows -- clipping here as well would hide the newest lines and double-count
    them in the tally.
    """
    syntax = rich.syntax.Syntax(
        code,
        lexer,
        theme="ansi_dark",
        word_wrap=True,
        background_color="default",
        line_range=(1, _MAX_LINES) if clip else None,
    )
    total = code.count("\n") + 1
    if not clip or total <= _MAX_LINES:
        return syntax
    note = rich.text.Text(f"… (+{total - _MAX_LINES} more lines)", style="dim")
    return rich.console.Group(syntax, note)


def _render_markdown(text: str, *, clip: bool = True) -> rich.console.RenderableType:
    """Render prose (system/user/assistant content) as Markdown.

    `Markdown` -- unlike `Syntax` -- has no native length limit, so when ``clip``
    the source is truncated to `_MAX_LINES` whole lines first. The live streaming
    panel passes ``clip=False`` so the growing tail stays fully visible.
    """
    lines = text.splitlines()
    if clip and len(lines) > _MAX_LINES:
        text = (
            "\n".join(lines[:_MAX_LINES])
            + f"\n\n*… (+{len(lines) - _MAX_LINES} more lines)*"
        )
    return rich.markdown.Markdown(text, code_theme="ansi_dark")


def _render_reasoning(text: str, *, clip: bool = True) -> rich.console.RenderableType:
    """Render reasoning as dimmed Markdown.

    `Markdown` takes no ``style=``, so `rich.styled.Styled` applies a ``dim``
    base -- the Markdown-compatible analog of the old dim-italic plain text. A
    base ``italic`` interferes with Markdown's own paragraph styling (dropping
    the dim too), so only ``dim`` is used. ``clip`` is forwarded to
    `_render_markdown`.
    """
    return rich.styled.Styled(_render_markdown(text, clip=clip), "dim")


def _render_data(value: typing.Any) -> rich.console.RenderableType:
    """Render an already-parsed JSON value (tool result / structured-output
    answer / tool-call arguments) as pretty, highlighted, line-truncated JSON."""
    return _syntax(json.dumps(value, indent=2), "json")


# Structured-output answers are wrapped by `_BoxedResponse` as `{"value": ...}`
# (call_assistant); the wrapper is display noise. Sourced from the model.
_BOX_FIELD = next(iter(_BoxedResponse.model_fields))


def _render_content(text: str, *, unwrap: bool = False) -> rich.console.RenderableType:
    """Render message content, choosing by shape rather than role: JSON
    objects/arrays (tool results, direct structured-output answers) as pretty
    JSON, everything else (prose, the Markdown system/user prompts) as Markdown.

    When ``unwrap`` (for a direct structured-output answer), a lone
    ``_BoxedResponse`` ``{"value": ...}`` wrapper is stripped to its payload.
    """
    if text.lstrip()[:1] in ("{", "["):
        try:
            value = json.loads(text)
        except ValueError:
            pass
        else:
            if unwrap and isinstance(value, dict) and set(value) == {_BOX_FIELD}:
                value = value[_BOX_FIELD]
            return _render_data(value)
    return _render_markdown(text)


# Half-written lines to set aside at the end of a streamed snippet when looking
# for the statements that have already landed. What stops a prefix from parsing
# is the construct still being typed, which is a line or two deep; past that the
# search (a parse per line) buys nothing, and on a long *non*-Python payload it
# is the difference between 0.6ms and 10.6ms a frame.
_PARTIAL_PREFIX_LINES = 8


def _parses_as_prefix(source: str) -> bool:
    """Whether `source` is a complete Python statement, or the start of one.

    `codeop` is what a REPL asks to tell "run this" from "prompt for more of
    it", which is the same question a half-arrived snippet poses, and the
    counterpart for source of reading JSON with ``allow_partial``.
    """
    try:
        codeop.compile_command(source, symbol="exec")
    except (SyntaxError, OverflowError, ValueError):
        return False
    return True


def _is_python_prefix(text: str) -> bool:
    """Whether `text` could still become Python source carrying a statement.

    Being a valid prefix is necessary but not sufficient: a lone ``{`` is a
    perfectly good start to a dict literal, so JSON-as-a-string would sail
    through on that test alone. The statement guard `_is_python` applies to
    finished source therefore applies here too -- to the longest prefix that
    parses, and where none does yet (a docstring still open holds up everything
    after it) to the one thing already known about the text, which is whether it
    opens a literal or a module.
    """
    # A half-written last line is a syntax error however sound the rest is.
    if not _parses_as_prefix(text) and not _parses_as_prefix(text.rsplit("\n", 1)[0]):
        return False
    lines = text.split("\n")
    for _ in range(_PARTIAL_PREFIX_LINES):
        lines.pop()
        if not lines:
            break
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        return any(not isinstance(node, ast.Expr) for node in tree.body)
    return text.lstrip()[:1] not in ("{", "[")


def _is_python(text: str, *, partial: bool = False) -> bool:
    """Whether `text` looks like a Python source snippet worth highlighting.

    Detects code by *content* rather than schema/field name, so it covers every
    `Encodable` type that serializes Python as a string -- the synthesis
    `SynthesizedFunction.code` field, `exec_code`'s `types.CodeType`
    argument, and any future code-carrying tool -- uniformly. Requires a
    multi-line string that parses as a module with at least one real statement
    (not a lone expression), which excludes prose and JSON-as-string.

    ``partial`` also admits source that is merely *on its way* to that, for a
    snippet still streaming -- see `_is_python_prefix`.
    """
    if "\n" not in text:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return partial and _is_python_prefix(text)
    return any(not isinstance(node, ast.Expr) for node in tree.body)


def _extract_code(args: typing.Any, *, partial: bool = False) -> str | None:
    """Return an embedded Python source string from parsed tool-call arguments.

    Walks nested dicts (a synthesized callable is ``{"implementation":
    {"code": ...}}``; `exec_code` is a flat ``{"code": ...}``) and returns
    the first string value that :func:`_is_python` recognizes. ``partial`` is
    forwarded, and is set while streaming, where the payload is source cut
    mid-line and has to be judged as a prefix rather than as a module.
    """
    if isinstance(args, str):
        return args if _is_python(args, partial=partial) else None
    if isinstance(args, dict):
        for value in args.values():
            found = _extract_code(value, partial=partial)
            if found is not None:
                return found
    return None


def _render_tool_call(
    name: str, args: str, *, streaming: bool, limit: int | None = None
) -> tuple[rich.console.RenderableType, int]:
    """Render one tool call: a ``→ name`` header over its arguments.

    Synthesized code is shown as Python; ordinary arguments as pretty JSON. While
    ``streaming`` the arguments are an incomplete JSON document, read as far as it
    goes so the payload appears in the form it will finally take rather than as
    the raw escaped fragment -- the difference, while a model writes a function,
    between watching the code appear and watching ``\\n``-separated JSON
    word-wrap.

    ``limit`` clips a streamed payload to its last that many lines, and the
    count dropped comes back with the renderable the way `_tail_lines` returns
    it -- for the same two reasons. It has to be *reported*, or the tally of what
    is out of view stalls; and it has to be clipped at the source rather than
    left to `_Tail`, or every frame lays out and highlights the whole payload
    only to throw most of it away, which costs 84ms a frame by the time a model
    has written a thousand lines and grows from there.
    """
    header = rich.text.Text(f"→ {name}", style="bold magenta")
    parsed: typing.Any
    if streaming:
        # Arguments arrive a few characters at a time, so the part worth showing
        # sits inside an unterminated string inside an unterminated object --
        # which `json.loads` can only reject. Reading a *prefix* of a document is
        # what ``allow_partial`` is for, and ``"trailing-strings"`` is the mode
        # that keeps the half-written string rather than dropping it, which here
        # is the whole point. This is jiter, the same parser (and the same use of
        # it) the OpenAI SDK reaches for on its own streamed structured output,
        # via the pydantic this package already depends on rather than a second
        # copy of it. It raises only for a fragment that is not yet a prefix of
        # anything, which falls back to showing the fragment raw.
        try:
            parsed = pydantic_core.from_json(args, allow_partial="trailing-strings")
        except ValueError:
            parsed = None
    else:
        try:
            parsed = json.loads(args)
        except ValueError:
            parsed = None
    hidden = 0
    code = _extract_code(parsed, partial=streaming)
    if code is not None:
        code, hidden = _tail_lines(code, limit)
        body: rich.console.RenderableType = _syntax(code, "python", clip=not streaming)
    elif parsed is not None:
        body = _render_data(parsed)
    else:
        body = _syntax(args, "json") if args else rich.text.Text("…", style="dim")
    return rich.console.Group(header, body), hidden


def _panel_key(message: Message) -> str:
    """Everything `_message_panel` will show for `message`, as one comparable value.

    Messages are told apart by what they *render as* rather than by dict
    equality, because a turn reaches the next request's history through handlers
    that may re-serialize it -- and two dicts differing only in bookkeeping no
    panel displays (a provider's passthrough fields, a key absent one way and
    None the other) would otherwise print the same panel a second time. Observed
    doing exactly that on the decoding-error path, where a rejected turn is
    replayed into the retry's history.
    """
    msg = typing.cast("collections.abc.Mapping[str, typing.Any]", message)
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        calls.append([fn.get("name"), fn.get("arguments")])
    return json.dumps(
        [
            msg.get("role"),
            _message_text(msg.get("reasoning_content")),
            _message_text(msg.get("content")),
            calls,
        ],
        default=str,
    )


def _message_panel(
    message: Message, subtitle: rich.text.Text | None = None
) -> rich.panel.Panel:
    """Render a single completed message as a titled panel.

    Every role (including ``system``) is shown; long field bodies are truncated
    to `_MAX_LINES` lines so the panel stays readable.

    This is also how a *just*-streamed turn settles (see `_completion_live`),
    which is why it takes a ``subtitle``: the turn carries a TTFT the panel
    should keep, and rendering it through this function rather than through
    `_partial_panel` is what makes the settled turn identical to the history
    message it is about to become, instead of the same message rendered two ways
    one frame apart.
    """
    # A loose view for reads of keys not declared across the whole `Message`
    # union (`reasoning_content`, `tool_calls`), which typecheckers infer as
    # `object`; these messages are dynamically built dicts (see `_make_message`).
    msg = typing.cast("collections.abc.Mapping[str, typing.Any]", message)
    role = msg.get("role", "?")
    renderables: list[rich.console.RenderableType] = []
    reasoning = _message_text(msg.get("reasoning_content"))
    if reasoning:
        renderables.append(_render_reasoning(reasoning))
    content = _message_text(msg.get("content"))
    if content:
        renderables.append(_render_content(content, unwrap=role == "assistant"))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        # A settled call is shown whole, so nothing is hidden to report.
        call, _ = _render_tool_call(
            fn.get("name") or "?", fn.get("arguments") or "", streaming=False
        )
        renderables.append(call)
    body = rich.console.Group(*renderables) if renderables else rich.text.Text("")
    return rich.panel.Panel(
        body,
        title=role,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style=_ROLE_STYLES.get(role, "white"),
    )


@dataclasses.dataclass(frozen=True)
class _Tail:
    """`renderable`, cropped to its last `height` *rendered rows*.

    Rows rather than source lines, because the two come apart exactly where it
    matters: streamed tool-call arguments are a single logical line of JSON that
    word-wraps across dozens of rows, so `_MAX_LINES` does not bound them at all.
    Only the console can say how tall something is, so this asks it and then
    keeps the end -- see `_tail_lines` on why the end.

    ``hidden`` is what the caller dropped before handing the renderable over, and
    it is reported together with what this drops. Reporting only the latter is
    what makes the tally stand still: the caller clips the source to a fixed
    window, so past that point everything further this hides is the *same* few
    rows of wrapping overflow, however long the turn grows.
    """

    renderable: rich.console.RenderableType
    height: int
    hidden: int = 0

    def __rich_console__(
        self, console: rich.console.Console, options: rich.console.ConsoleOptions
    ) -> rich.console.RenderResult:
        lines = console.render_lines(self.renderable, options, pad=False)
        hidden = self.hidden
        if len(lines) > self.height:
            # One row of the budget goes to saying what is not shown.
            dropped = len(lines) - (self.height - 1)
            lines = lines[dropped:]
            hidden += dropped
        if hidden:
            yield rich.text.Text(f"… (+{hidden} earlier lines)", style="dim")
            # Saying so costs a row whether or not this was the thing that cut.
            if len(lines) >= self.height:
                lines = lines[len(lines) - self.height + 1 :]
        for row, line in enumerate(lines):
            if row:
                yield rich.segment.Segment.line()
            yield from line


# Rows the live panel leaves to the rest of the screen: its own two borders, the
# line the shell prompt will land on, and one of slack. The panel must stay
# *shorter* than the terminal or `rich.live.Live` cannot erase it -- it rewinds
# the cursor over the previous frame, and a rewind past the top of the screen is
# clamped, which turns every refresh into a fresh copy appended below the last.
_LIVE_MARGIN = 4

# Floor for that budget, for terminals too short (or too lied-about) to give one.
_MIN_LIVE_LINES = 8


def _live_height(console: rich.console.Console) -> int:
    """Rows the live panel may occupy on `console`."""
    return max(_MIN_LIVE_LINES, console.size.height - _LIVE_MARGIN)


def _ttft_subtitle(ttft: float | None) -> rich.text.Text | None:
    """The time the model spent prefilling before its first delta, as a panel
    subtitle -- or None before the first delta has arrived."""
    if ttft is None:
        return None
    return rich.text.Text(f"TTFT {ttft:.2f}s", style="dim")


def _partial_panel(
    partial: _PartialAssistant,
    ttft: float | None = None,
    *,
    streaming: bool = True,
    height: int | None = None,
) -> rich.panel.Panel:
    """Render the in-progress assistant turn as the live panel.

    ``height`` bounds the panel, in rendered rows, to what `_live_height`
    allows; the newest output is kept and the older scrolls out of the window.
    Passing None leaves it unbounded, which is safe only off the live path.

    ``streaming`` is forwarded to `_render_tool_call`.
    """
    # Content and reasoning render as Markdown even mid-stream (Markdown never
    # raises on incomplete text). They are pre-clipped to the row budget purely
    # to keep the work per frame bounded -- `_Tail` is what actually enforces it,
    # and would otherwise re-lay-out a whole long turn only to discard most of
    # it. What that costs is the honesty of `_Tail`'s tally, so the lines taken
    # here are counted and handed over to be reported along with its own.
    limit = None if height is None else min(height, _MAX_LINES)
    hidden = 0
    renderables: list[rich.console.RenderableType] = []
    if partial["reasoning_content"]:
        text, dropped = _tail_lines(partial["reasoning_content"], limit)
        hidden += dropped
        renderables.append(_render_reasoning(text, clip=False))
    if partial["content"]:
        text, dropped = _tail_lines(partial["content"], limit)
        hidden += dropped
        renderables.append(_render_markdown(text, clip=False))
    for _, slot in sorted(partial["tool_calls"].items()):
        call, dropped = _render_tool_call(
            slot["name"], slot["args"], streaming=streaming, limit=limit
        )
        hidden += dropped
        renderables.append(call)
    body: rich.console.RenderableType = (
        rich.console.Group(*renderables)
        if renderables
        else rich.text.Text("…", style="dim")
    )
    if height is not None:
        body = _Tail(body, height, hidden)
    return rich.panel.Panel(
        body,
        title="assistant",
        title_align="left",
        subtitle=_ttft_subtitle(ttft),
        subtitle_align="right",
        border_style="green",
    )


class _PrefillStatus:
    """Live "prefilling…" line shown until the first streamed chunk arrives.

    litellm/provider APIs report no prompt-processing progress, so there is no
    true prefill percentage. Instead this shows the (locally counted) prompt
    size and a ticking elapsed timer, which is what a large prompt's
    time-to-first-token latency actually reflects.

    It is re-rendered by :class:`rich.live.Live`'s background refresh thread
    while the main thread blocks on the first chunk, so the spinner animates and
    the timer ticks on their own -- :meth:`__rich__` recomputes elapsed each call.
    """

    def __init__(self, prompt_tokens: int | None, start: float):
        self._spinner = rich.spinner.Spinner("dots", style="cyan")
        self._prompt_tokens = prompt_tokens
        self._start = start

    def __rich__(self) -> rich.spinner.Spinner:
        elapsed = time.monotonic() - self._start
        size = (
            f"{self._prompt_tokens:,} tokens"
            if self._prompt_tokens is not None
            else "prompt"
        )
        self._spinner.update(
            text=rich.text.Text(f" prefilling {size}… {elapsed:.1f}s", style="cyan")
        )
        return self._spinner


# Transport failures a streamed read can hit that an unstreamed one would not:
# the connection is held open across chunks, so it is exposed to anything that
# disturbs the pool meanwhile. Deliberately narrow -- a refused request, a bad
# model name or a rejected schema fails identically without streaming, and
# re-issuing it would only pay for the same error twice.
#
# Each is listed on its own: litellm's exceptions inherit from *openai's*
# hierarchy rather than from each other, so `litellm.Timeout` is not a
# `litellm.APIConnectionError` and naming one does not cover the other.
_STREAM_TRANSPORT_ERRORS = (
    litellm.exceptions.MidStreamFallbackError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
)


@dataclasses.dataclass(frozen=True)
class RichTerminalRenderer(ObjectInterpretation):
    """Stream `completion` and live-render the message sequence.

    Opt-in debugging handler: forces streaming so that reasoning, generation and
    tool-call arguments appear as they are produced, then reassembles a normal
    ``ModelResponse`` via :func:`litellm.stream_chunk_builder` so the rest of the
    pipeline is unchanged.

    Each message is printed **once**, as a panel, in the order the conversation
    reaches it; only the turn currently being streamed lives inside a
    :class:`rich.live.Live` region, and only that turn is redrawn. The
    alternative -- rebuilding the whole history into the live region on every
    chunk -- is what a terminal cannot do: `rich` erases a frame by rewinding the
    cursor over it, the rewind is clamped at the top of the screen, and a frame
    taller than the screen therefore accumulates one full copy of the
    conversation per refresh. Measured on a three-turn run of
    ``llm_examples/reasoning/countdown.py``, that came to 1.2 MB and 7,208 lines
    of output carrying 335 distinct ones, with the system and user panels
    reprinted 79 times each.
    """

    # Pin the console to the process's original stdout rather than the live
    # ``sys.stdout``. Otherwise, when a nested ``completion`` renders while stdout
    # is redirected -- inside ``exec_code`` (``redirect_stdout``) or ``run_doctests``
    # (doctest's ``_SpoofOut``) -- the rendered panels are captured and fed back
    # into the model's context. ``sys.__stdout__`` is immune to those rebindings.
    console: rich.console.Console = dataclasses.field(
        default_factory=lambda: rich.console.Console(file=sys.__stdout__)
    )

    # Held for the lifetime of a ``Live`` region. See `completion`.
    #
    # Constructed eagerly, in ``__init__``: the one thing this must never do is hand
    # two threads two different locks, and creating it on first use would risk
    # exactly that -- ``functools.cached_property`` is unsynchronized as of 3.12, so
    # concurrent first access can run the factory more than once. ``compare`` and
    # ``repr`` are off so the generated ``__eq__``/``__hash__``/``__repr__`` stay
    # over the configuration this handler carries rather than its internals.
    _live_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    # `_panel_key` of each message already on screen, so that a request carrying
    # the whole conversation prints only what is new in it. Mutated in place (the
    # dataclass is frozen) and only from the live path, which holds
    # ``_live_lock`` for its whole duration.
    _printed: list[str] = dataclasses.field(
        default_factory=list, repr=False, compare=False
    )

    @implements(completion)
    def completion(self, *args, **kwargs) -> typing.Any:
        """Stream and live-render this completion, or -- if another already holds
        the terminal, or if the stream breaks -- let it run unstreamed and print it
        as a settled panel.

        A ``Live`` region owns the console for its duration, and there is one
        console. Concurrent skill calls (the ``asyncio.gather`` +
        ``asyncio.to_thread`` fan-out several examples demonstrate) would otherwise
        open overlapping ``Live`` regions and interleave two redraws into the same
        rows. Acquiring without blocking is what keeps that fan-out parallel: a
        caller that loses the race proceeds immediately down the settled path
        rather than queueing behind the live one.

        The same fallback covers a stream that dies in flight. Streaming here is a
        debugging affordance this handler *adds* to a request that did not ask for
        it, so it also owns the cost: a long-lived streamed read shares a connection
        pool with whatever else the process is doing, and losing that race is not a
        reason to fail a call that would have succeeded unstreamed. The retry is safe
        because a broken stream yields no result to duplicate.
        """
        if self._live_lock.acquire(blocking=False):
            try:
                return self._completion_live(*args, **kwargs)
            except _STREAM_TRANSPORT_ERRORS as e:
                self.console.print(
                    rich.text.Text(
                        f"stream failed ({type(e).__name__}); retrying unstreamed",
                        style="dim",
                    )
                )
            finally:
                self._live_lock.release()
        return self._completion_settled(*args, **kwargs)

    def _completion_settled(self, *args, **kwargs) -> typing.Any:
        """Render a completion that could not take the live region: forward it
        unchanged -- so it is *not* forced onto the streaming path -- and print the
        finished turn as one panel once it lands."""
        response = fwd(*args, **kwargs)
        if isinstance(response, litellm.types.utils.ModelResponse):
            choice = response.choices[0]
            if isinstance(choice, litellm.types.utils.Choices):
                self.console.print(
                    _message_panel(
                        typing.cast(Message, choice.message.model_dump(mode="json"))
                    )
                )
        return response

    def _completion_live(self, *args, **kwargs) -> typing.Any:
        kwargs = {
            **kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream: litellm.CustomStreamWrapper = fwd(*args, **kwargs)

        # The request already carries the full message history as `messages`.
        history: list[Message] = list(kwargs.get("messages") or [])

        chunks: list[litellm.types.utils.ModelResponseStream] = []
        partial: _PartialAssistant = {
            "content": "",
            "reasoning_content": "",
            "tool_calls": {},
        }

        # Count prompt tokens locally to size the prefill wait. `model` is injected
        # downstream by LiteLLMConfigurer, so it may be absent here -- token_counter
        # falls back to a default tokenizer, giving an approximate count.
        try:
            prompt_tokens: int | None = litellm.token_counter(
                model=kwargs.get("model", ""),
                messages=history,
                tools=kwargs.get("tools"),
            )
        except Exception:
            prompt_tokens = None

        # Everything settled goes above the live region, once, and stays there;
        # the region below it holds only the turn being streamed.
        self._print_new(history)

        start = time.monotonic()
        status: _PrefillStatus | None = _PrefillStatus(prompt_tokens, start)
        ttft: float | None = None
        height = _live_height(self.console)

        # `transient` because this region is a view of a turn in flight, not the
        # record of it: it is erased on exit and replaced by the settled panel
        # below, which is the same rendering the turn will keep as history --
        # rather than the row-clipped one the live window necessarily shows.
        # `ellipsis` is the backstop for that clipping: if a panel ever does
        # outgrow the screen, cropping it keeps `Live`'s erase honest, where
        # `visible` would silently turn each refresh into another copy.
        with rich.live.Live(
            status,
            console=self.console,
            transient=True,
            vertical_overflow="ellipsis",
        ) as live:
            for chunk in stream:
                chunks.append(chunk)
                if chunk.choices:
                    _accumulate(partial, chunk.choices[0].delta)
                # The first chunk carrying any content ends prefill; record TTFT
                # and drop the status line in favor of the streaming panel.
                if status is not None and (
                    partial["content"]
                    or partial["reasoning_content"]
                    or partial["tool_calls"]
                ):
                    ttft = time.monotonic() - start
                    status = None
                live.update(
                    status
                    if status is not None
                    else _partial_panel(partial, ttft=ttft, height=height)
                )

        response = litellm.stream_chunk_builder(chunks, messages=kwargs.get("messages"))
        self._print_settled(response, partial, ttft)
        return response

    def _print_new(self, history: collections.abc.Sequence[Message]) -> None:
        """Print the messages of `history` that are not on screen yet.

        Every request carries the whole conversation, of which all but the last
        few messages have already been rendered by an earlier turn, so only the
        new suffix is printed. A history that *diverges* from what was printed
        -- a retry, a different agent, a fresh skill call sharing this console --
        is printed from the point where it does, since nothing after that point
        is on screen.
        """
        keys = [_panel_key(message) for message in history]
        shared = 0
        for printed, key in zip(self._printed, keys):
            if printed != key:
                break
            shared += 1
        for message in history[shared:]:
            self.console.print(_message_panel(message))
        self._printed[:] = keys

    def _print_settled(
        self,
        response: typing.Any,
        partial: _PartialAssistant,
        ttft: float | None,
    ) -> None:
        """Print the finished turn, once, where the live region just was.

        Rendered from the *reassembled* message rather than from `partial`, for
        two reasons: it is the settled form (tool-call arguments are complete
        JSON, so they show as pretty JSON or synthesized code -- the only chance
        a loop-*terminating* tool call gets to settle, since no later request
        will carry it), and it is verbatim what the next request will carry as
        history, so recording it as printed keeps it from being printed twice.
        `partial` is the fallback for a stream that reassembled into something
        unexpected, which is still worth showing.
        """
        message: Message | None = None
        if isinstance(response, litellm.types.utils.ModelResponse):
            choice = response.choices[0]
            if isinstance(choice, litellm.types.utils.Choices):
                message = typing.cast(Message, choice.message.model_dump(mode="json"))
        if message is None:
            self.console.print(_partial_panel(partial, ttft=ttft, streaming=False))
            return
        self.console.print(_message_panel(message, _ttft_subtitle(ttft)))
        self._printed.append(_panel_key(message))
