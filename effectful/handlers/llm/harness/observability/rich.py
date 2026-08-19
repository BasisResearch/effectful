import ast
import collections.abc
import dataclasses
import json
import sys
import time
import typing

import litellm
import rich.console
import rich.live
import rich.markdown
import rich.panel
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


def _syntax(code: str, lexer: str) -> rich.console.RenderableType:
    """Syntax-highlight `code` using the terminal palette, truncated to
    `_MAX_LINES` via `Syntax.line_range` (no parsing, safe on partial input)."""
    syntax = rich.syntax.Syntax(
        code,
        lexer,
        theme="ansi_dark",
        word_wrap=True,
        background_color="default",
        line_range=(1, _MAX_LINES),
    )
    total = code.count("\n") + 1
    if total <= _MAX_LINES:
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


def _is_python(text: str) -> bool:
    """Whether `text` looks like a Python source snippet worth highlighting.

    Detects code by *content* rather than schema/field name, so it covers every
    `Encodable` type that serializes Python as a string -- the synthesis
    `SynthesizedFunction.module_code` field, `exec_code`'s `types.CodeType`
    argument, and any future code-carrying tool -- uniformly. Requires a
    multi-line string that parses as a module with at least one real statement
    (not a lone expression), which excludes prose and JSON-as-string.
    """
    if "\n" not in text:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(not isinstance(node, ast.Expr) for node in tree.body)


def _extract_code(args: typing.Any) -> str | None:
    """Return an embedded Python source string from parsed tool-call arguments.

    Walks nested dicts (a synthesized callable is ``{"implementation":
    {"module_code": ...}}``; `exec_code` is a flat ``{"code": ...}``) and returns
    the first string value that :func:`_is_python` recognizes.
    """
    if isinstance(args, str):
        return args if _is_python(args) else None
    if isinstance(args, dict):
        for value in args.values():
            found = _extract_code(value)
            if found is not None:
                return found
    return None


def _render_tool_call(
    name: str, args: str, *, streaming: bool
) -> rich.console.RenderableType:
    """Render one tool call: a ``→ name`` header over its arguments.

    Synthesized code is shown as Python; ordinary arguments as pretty JSON. While
    ``streaming`` the argument JSON is still partial (unparseable), so the raw
    fragment is highlighted as JSON instead.
    """
    header = rich.text.Text(f"→ {name}", style="bold magenta")
    parsed: typing.Any = None
    if not streaming:
        try:
            parsed = json.loads(args)
        except ValueError:
            parsed = None
    code = _extract_code(parsed)
    if code is not None:
        body: rich.console.RenderableType = _syntax(code, "python")
    elif parsed is not None:
        body = _render_data(parsed)
    else:
        body = _syntax(args, "json") if args else rich.text.Text("…", style="dim")
    return rich.console.Group(header, body)


def _message_panel(message: Message) -> rich.panel.Panel:
    """Render a single completed history message as a titled panel.

    Every role (including ``system``) is shown; long field bodies are truncated
    to `_MAX_LINES` lines so the frame stays readable.
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
        renderables.append(
            _render_tool_call(
                fn.get("name") or "?", fn.get("arguments") or "", streaming=False
            )
        )
    body = rich.console.Group(*renderables) if renderables else rich.text.Text("")
    return rich.panel.Panel(
        body,
        title=role,
        title_align="left",
        border_style=_ROLE_STYLES.get(role, "white"),
    )


def _partial_panel(
    partial: _PartialAssistant, ttft: float | None = None, *, streaming: bool = True
) -> rich.panel.Panel:
    """Render the in-progress (or just-finished) assistant turn as a live panel.

    When ``ttft`` (time-to-first-token, seconds) is known it is shown as a
    subtitle -- how long the model spent prefilling before the first delta.

    ``streaming`` is forwarded to `_render_tool_call`: the caller passes
    ``streaming=False`` for the final frame, once the stream is exhausted and the
    tool-call arguments form complete JSON, so they render as pretty JSON /
    synthesized code rather than the raw partial payload. This is the only chance
    the *terminating* tool call gets to settle -- there is no later `completion`
    to re-render it as a history message.
    """
    # Content and reasoning render as Markdown even mid-stream (Markdown never
    # raises on incomplete text) and are shown in full (clip=False) so the
    # growing tail stays visible.
    renderables: list[rich.console.RenderableType] = []
    if partial["reasoning_content"]:
        renderables.append(_render_reasoning(partial["reasoning_content"], clip=False))
    if partial["content"]:
        renderables.append(_render_markdown(partial["content"], clip=False))
    for _, slot in sorted(partial["tool_calls"].items()):
        renderables.append(
            _render_tool_call(slot["name"], slot["args"], streaming=streaming)
        )
    body = (
        rich.console.Group(*renderables)
        if renderables
        else rich.text.Text("…", style="dim")
    )
    subtitle = (
        rich.text.Text(f"TTFT {ttft:.1f}s", style="dim") if ttft is not None else None
    )
    return rich.panel.Panel(
        body,
        title="assistant",
        title_align="left",
        subtitle=subtitle,
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


def _render_frame(
    history: collections.abc.Sequence[Message],
    partial: _PartialAssistant,
    *,
    status: _PrefillStatus | None = None,
    ttft: float | None = None,
    streaming: bool = True,
) -> rich.console.Group:
    """Build the full frame: one panel per history message, then either the live
    ``status`` line (while prefilling, before any token) or the partial turn."""
    tail = (
        status
        if status is not None
        else _partial_panel(partial, ttft=ttft, streaming=streaming)
    )
    return rich.console.Group(*[_message_panel(m) for m in history], tail)


@dataclasses.dataclass(frozen=True)
class RichTerminalRenderer(ObjectInterpretation):
    """Stream `completion` and live-render the whole message sequence.

    Opt-in debugging handler: forces streaming, redraws the entire (partial)
    message history from scratch on every chunk via :class:`rich.live.Live`
    (reasoning, generation, and tool-call arguments appear as they are produced),
    then reassembles a normal ``ModelResponse`` via
    :func:`litellm.stream_chunk_builder` so the rest of the pipeline is unchanged.
    """

    # Pin the console to the process's original stdout rather than the live
    # ``sys.stdout``. Otherwise, when a nested ``completion`` renders while stdout
    # is redirected -- inside ``exec_code`` (``redirect_stdout``) or ``run_doctests``
    # (doctest's ``_SpoofOut``) -- the rendered panels are captured and fed back
    # into the model's context. ``sys.__stdout__`` is immune to those rebindings.
    console: rich.console.Console = dataclasses.field(
        default_factory=lambda: rich.console.Console(file=sys.__stdout__)
    )

    @implements(completion)
    def _completion(self, *args, **kwargs) -> typing.Any:
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

        start = time.monotonic()
        status: _PrefillStatus | None = _PrefillStatus(prompt_tokens, start)
        ttft: float | None = None

        with rich.live.Live(
            _render_frame(history, partial, status=status),
            console=self.console,
            vertical_overflow="visible",
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
                live.update(_render_frame(history, partial, status=status, ttft=ttft))

            # The args are now complete JSON; re-render settled so the final
            # (loop-terminating) tool call shows as pretty JSON / synthesized
            # code rather than the raw streaming payload.
            live.update(_render_frame(history, partial, ttft=ttft, streaming=False))

        return litellm.stream_chunk_builder(chunks, messages=kwargs.get("messages"))
