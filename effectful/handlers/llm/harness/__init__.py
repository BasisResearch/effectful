"""Handlers that give the types in :mod:`effectful.handlers.llm.types` their meaning.

The :func:`harness` function assembles the standard stack; its constituents are
documented in the submodules and may be recombined or replaced individually:
:mod:`~effectful.handlers.llm.harness.hooks` (the operations and `AgentLoop`), :mod:`~effectful.handlers.llm.harness.legibility` (what
the model sees and may call), :mod:`~effectful.handlers.llm.harness.synthesis` (answering with code),
:mod:`~effectful.handlers.llm.harness.validation` (type checks and doctests), :mod:`~effectful.handlers.llm.harness.execution`
(running model-authored Python), :mod:`~effectful.handlers.llm.harness.durability` (history, retries,
compaction, persistence), :mod:`~effectful.handlers.llm.harness.provision` (the model backend),
:mod:`~effectful.handlers.llm.harness.observability` (rendering, dumps, tracing), :mod:`~effectful.handlers.llm.harness.serialization`
(the wire format), :mod:`~effectful.handlers.llm.harness.autoreload` and :mod:`~effectful.handlers.llm.harness.__main__` (the launcher).

.. rubric:: One Skill call, one conversation turn

The standard call path::

    Python calls a Skill: begin one turn
    bind arguments over the Skill's captured lexical context
    open a transaction over the receiver's committed history (bound Skill),
      or over a fresh history (free Skill)
    render a candidate system message: installed-handler sections, the defining
      module's source, receiver-class and sibling-Skill documentation, and the
      scope tables; a bound conversation retains the first one it committed
    append one user message: Skill name and signature, the formatted docstring,
      and a REPL-session notice when an executor is installed
    model round <-> tool result or validation feedback, repeated
    decode to the declared return type; type-check, execute, run doctests as required
    invalid: feed back and retry within this turn, when a retryer is installed
    accepted: return the value; a committable bound turn records its history,
      then may checkpoint
    exhausted: raise, leaving the turn uncommitted

A *turn* is one Python invocation of a Skill; the model's several responses
within it are *rounds*. The model-facing prompt text says "call" and "turn" for
the same two things.

Which turns see and commit which history is a transaction over the receiver's
messages: :mod:`~effectful.handlers.llm.harness.durability.transaction`.

.. rubric:: Configure the harness deliberately

Handlers implement the mechanics surrounding each Skill turn. These settings
control different layers and should not be conflated::

    from effectful.handlers.llm.harness import harness
    from effectful.ops.semantics import handler


    with handler(
        harness(
            model="openai/gpt-5-mini",
            tool_collection="auto",      # also publish qualifying ordinary callables
            tool_calling="auto",          # JSON when faithful, code otherwise
            tool_choice="auto",           # provider may call a tool or answer directly
            eval_provider="builtin",      # authority for model-authored Python
            type_checker="ty",
            num_retries=5,
        )
    ):
        main()

Every knob is documented on :func:`harness`, including what each value installs
and what it refuses; the launcher's flags on :mod:`effectful.handlers.llm.harness.__main__`. Forcing body
synthesis with ``tool_choice="required"``: :mod:`~effectful.handlers.llm.harness.synthesis.body`.

.. rubric:: Debugging what the model could know

Debug a Skill turn by reconstructing its exact request and environment, not by
guessing what the model "must have known"::

    python -m effectful.handlers.llm.harness path/to/example.py \\
      --dump-system-prompt /tmp/effectful-prompt.md

The dump is the latest candidate system prompt, not a transcript; what it omits
is in :mod:`~effectful.handlers.llm.harness.observability.dump`. Inspect recorded messages as well when a
bound receiver's retained system message may differ from the candidate.

Then ask, in order:

1. Is the relevant handler installed and described under ``# Harness``? Only
   `PromptInjectingInterpretation` subclasses appear there; `MCPTools` does not.
2. Is the defining module source present?
3. Is the value a current argument, a captured lexical binding, or reachable
   through an in-scope Skill-owning object?
4. Is it a ``Tool``/``Skill``, or does it require ``tool_collection="auto"``?
5. Was the capability discovered, and is the chosen tool-calling pathway able
   to encode its signature?
6. Does the REPL have the runtime object even if the static source checker
   lacks a declaration for it?
7. Is a stale system message being confused with the tool set recomputed for a
   later turn?
8. Is a restriction enforced by the executor, or merely absent from the
   advertised Tool list?

Questions the dump cannot answer -- what a failed turn rolled back, which
receiver's history a turn joined, which synthesis route ran and whose tests it
used, whether feedback arrived before commitment -- are answered in
:mod:`~effectful.handlers.llm.harness.durability`, :mod:`~effectful.handlers.llm.harness.legibility` and :mod:`~effectful.handlers.llm.harness.synthesis`.

Extending the stack with handlers of your own:
:mod:`effectful.handlers.llm.harness.hooks`.
"""

import collections.abc
import json
import os
import pathlib
import typing

import tenacity

from effectful.handlers.llm.harness.durability.truncation import (
    DEFAULT_TOOL_OUTPUT_MAX_CHARS,
)
from effectful.ops.semantics import Interpretation, coproduct


def harness(
    *,
    num_retries: int = 5,
    langfuse: bool = False,
    render: bool = False,
    dump_system_prompt: str | os.PathLike[str] | None = None,
    persist_db: str | os.PathLike[str] | None = None,
    eval_provider: typing.Literal["builtin", "restricted", "none"] = "builtin",
    type_checker: typing.Literal["mypy", "ty", "none"] = "ty",
    tool_calling: typing.Literal["auto", "code", "json"] = "auto",
    tool_collection: typing.Literal["none", "explicit", "auto"] = "explicit",
    check_contracts: bool = True,
    max_tool_output_chars: int | None = DEFAULT_TOOL_OUTPUT_MAX_CHARS,
    compaction_soft_tokens: int | None = None,
    compaction_hard_tokens: int | None = None,
    compaction_recent_tokens: int | None = None,
    mcp_config: str
    | os.PathLike[str]
    | collections.abc.Mapping[str, typing.Any]
    | None = None,
    **provider_config,
) -> Interpretation:
    """
    Instantiate the standard `effectful.handlers.llm` handler stack.
    Install it with :func:`~effectful.ops.semantics.handler`::

        with handler(harness(...)):
            ...

    Constructing a `harness` records the configuration; entering it (as a
    context manager, decorator, or via the module CLI) installs the handlers and
    exiting removes them. The handlers, in installation order, are:

    1. `AgentLoop`, the tool pipeline (an extractor chosen by
       ``tool_collection``, a caller chosen by ``tool_calling``) and
       `LiteLLMConfigurer`, the model backend it drives.
    2. `FrameworkDocumenter` -- describe the framework's concepts in the system
       prompt.
    3. `ToolOutputTruncator` -- bound each textual tool result before it enters
       history (unless ``max_tool_output_chars=None``).
    4. `HistoryBuilder` -- accumulate the message history of a call, with
       `ReplCompactor` just inside it or `MiddleCompactor` above it, per the
       ``compaction_*`` thresholds.
    5. `RichTerminalRenderer` -- live-render the streaming history (if ``render``).
    6. `SystemPromptDumper` -- dump the system prompt (if ``dump_system_prompt``).
    7. The ``type_checker`` and the ``eval_provider`` -- check and run
       model-authored Python (each omitted for ``"none"``).
    8. `StatefulReplSynthesizer`, `FinalBodySynthesizer` and
       `ApiReferenceDocumenter` -- answer a call by running a snippet, and by
       synthesizing a function and calling it; and document the `Tool`,
       `Agent` and `Encodable` API that code written there may use. All three
       are omitted when ``eval_provider="none"``: the synthesizers advertise
       tools (``exec_code``, ``write_and_run_body``) that only an executor can
       decode, and without them the model writes no code.
    9. `PydanticSkillArgValidator` -- enforce the pre-conditions a caller
       wrote into a `Skill`'s parameter annotations (if ``check_contracts``).
    10. `TenacityRetryer` -- retry malformed/failing model output (if
        ``num_retries``).
    11. `MCPTools` -- offer the tools of MCP servers (if ``mcp_config``), above
        the tool callers and the retryer, so a request's retries see one catalog.
    12. `SQLitePersister` -- checkpoint a persisted `Agent`'s state/history to
        SQLite after each successful call (if ``persist_db``).
    13. `LangfuseTracer` -- log calls to Langfuse (if ``langfuse``).

    Args:
        num_retries: Attempts for malformed/failing model output (via
            `TenacityRetryer`, which is left out of the stack altogether when
            this is ``0``) and, independently, for transport-level failures
            (via litellm's own ``num_retries``, bound into the request).
        langfuse: Log LLM calls and metadata to Langfuse.
        render: Live-render the streaming message history in the terminal.
        dump_system_prompt: If set, dump the assembled system prompt to this
            Markdown file.
        persist_db: If set, path to a SQLite database used to checkpoint a
            persisted `~effectful.handlers.llm.types.Agent`'s (one with a
            stable ``__agent_id__``) state and history via
            `~effectful.handlers.llm.harness.durability.persistence.SQLitePersister`.
        eval_provider: Which provider runs model-authored Python:
            ``"builtin"`` (`BuiltinExecutor`, the default), ``"restricted"``
            (`RestrictedPythonExecutor`), or ``"none"`` for no executor --
            which also takes both synthesizers out of the stack, so nothing is
            offered that the stack could not then run.
        type_checker: Which handler type-checks model-authored Python before it
            runs: ``"ty"`` (`TyTypeChecker`, the default), ``"mypy"``
            (`MypyTypeChecker`), or ``"none"`` to run generated code unchecked.
        tool_calling: How the model calls the tools in a `Skill`'s lexical
            scope. ``"auto"`` (the default) installs `MixedToolCaller`, which
            picks per tool: schema-constrained JSON arguments for every tool a
            JSON schema can describe faithfully, and the code pathway for the
            rest (generic, variadic, or unadvertisable signatures). ``"code"``
            installs `ExpressionToolCaller`: uniformly, the model writes a
            Python call expression which is type-checked in the Skill's scope
            and evaluated. ``"json"`` is the classic JSON-only pathway with no
            caller at all (polymorphic tools degrade to untyped argument
            schemas there, and unadvertisable ones are skipped with a
            warning). ``"auto"`` and ``"code"`` require an eval provider:
            combining either with ``eval_provider="none"`` raises `ValueError`
            rather than silently degrading.
        tool_collection: Which tools are *collected* from a `Skill`'s lexical
            scope, as opposed to how they are called. ``"explicit"`` (the
            default) installs `LexicalToolExtractor`: the `Tool`/`Skill`
            values in scope, and those held by in-scope `Agent`\\ s. ``"auto"``
            installs `ImplicitToolExtractor` instead, which additionally wraps
            ordinary functions and methods that look deliberately published (see
            `ImplicitToolExtractor._implicit_tool_candidate`) with no
            ``Tool.define`` decorator; it makes naming conventions
            load-bearing (prefix orchestration helpers with ``_`` to keep them
            out of the model's hands). ``"none"`` installs no extractor at
            all: the model sees only the tools the harness itself injects
            (``exec_code``, ``write_and_run_body``), never the surrounding
            scope's.
        check_contracts: Install `PydanticSkillArgValidator`, so a `Skill`'s
            arguments are validated against the pydantic metadata its parameter
            annotations carry. On by default, which makes such an annotation
            mean the same thing whether a person or a model supplied the
            argument. Turning it off leaves a direct Python call unchecked; a
            model-supplied argument is still validated as the tool call is
            decoded, and metadata on a *return* annotation is enforced by the
            decoder either way.
        max_tool_output_chars: Maximum text characters retained in each tool
            result, including the truncation notice. The beginning and end are
            kept. Pass ``None`` to disable truncation.
        compaction_soft_tokens: Approximate token threshold for stale tool-output
            elision. ``None`` (default) disables middle-region compaction.
        compaction_hard_tokens: Token threshold for compaction. With
            ``compaction_soft_tokens``, it is the threshold at which
            `MiddleCompactor` summarizes, and must exceed the soft one. Alone,
            with an eval provider, a request at least this large is nudged and
            forced to call ``exec_code`` with ``compact`` set (`ReplCompactor`).
        compaction_recent_tokens: Approximate size of the recent window, in
            addition to keeping at least the last two rounds. Defaults to a
            quarter of the hard threshold.
        mcp_config: MCP servers whose tools are offered to every `Skill`, as a
            standard ``{"mcpServers": {name: server}}`` configuration mapping or
            a path to a JSON file holding one. Connection lifetime, naming and
            failure handling: :mod:`~effectful.handlers.llm.harness.legibility.mcp`.

    Raises:
        ValueError: If ``tool_calling`` is ``"auto"`` or ``"code"`` and
            ``eval_provider`` is ``"none"``.
    """
    # Imported here rather than at the top, so that importing this package loads no
    # handler, and a stack rebuilt after a handler module is re-imported uses it.
    from effectful.handlers.llm.harness.durability.compaction import (
        CompactionScope,
        MiddleCompactor,
        ReplCompactor,
    )
    from effectful.handlers.llm.harness.durability.persistence import SQLitePersister
    from effectful.handlers.llm.harness.durability.retrying import TenacityRetryer
    from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
    from effectful.handlers.llm.harness.durability.truncation import (
        ToolOutputTruncator,
    )
    from effectful.handlers.llm.harness.execution.builtin import BuiltinExecutor
    from effectful.handlers.llm.harness.execution.restricted import (
        RestrictedPythonExecutor,
    )
    from effectful.handlers.llm.harness.hooks import AgentLoop
    from effectful.handlers.llm.harness.legibility.framework import (
        ApiReferenceDocumenter,
        FrameworkDocumenter,
    )
    from effectful.handlers.llm.harness.legibility.lexical import (
        ImplicitToolExtractor,
        LexicalToolExtractor,
    )
    from effectful.handlers.llm.harness.observability.dump import SystemPromptDumper
    from effectful.handlers.llm.harness.observability.langfuse import LangfuseTracer
    from effectful.handlers.llm.harness.observability.rich import (
        RichTerminalRenderer,
    )
    from effectful.handlers.llm.harness.provision.litellm import (
        LiteLLMConfigurer,
    )
    from effectful.handlers.llm.harness.synthesis.body import (
        FinalBodySynthesizer,
    )
    from effectful.handlers.llm.harness.synthesis.snippet import StatefulReplSynthesizer
    from effectful.handlers.llm.harness.synthesis.toolcall import (
        ExpressionToolCaller,
        MixedToolCaller,
    )
    from effectful.handlers.llm.harness.validation.mypy import MypyTypeChecker
    from effectful.handlers.llm.harness.validation.pydantic import (
        PydanticSkillArgValidator,
    )
    from effectful.handlers.llm.harness.validation.ty import TyTypeChecker

    h: Interpretation = AgentLoop()

    if tool_calling != "json" and eval_provider == "none":
        raise ValueError(
            f'tool_calling="{tool_calling}" has the model answer by writing '
            f"Python, so it needs an eval provider to run what it writes. Pass "
            f'eval_provider="builtin" or "restricted", or tool_calling="json".'
        )

    if tool_calling == "auto":
        h = coproduct(h, MixedToolCaller())
    elif tool_calling == "code":
        h = coproduct(h, ExpressionToolCaller())
    json_only = tool_calling == "json"
    if tool_collection == "explicit":
        h = coproduct(h, LexicalToolExtractor(json_only=json_only))
    elif tool_collection == "auto":
        h = coproduct(h, ImplicitToolExtractor(json_only=json_only))

    h = coproduct(h, LiteLLMConfigurer(num_retries=num_retries, **provider_config))
    h = coproduct(h, FrameworkDocumenter())
    if max_tool_output_chars is not None:
        h = coproduct(h, ToolOutputTruncator(max_tool_output_chars))
    # Inside `HistoryBuilder`, so a reply it rejects is recorded with its feedback.
    # It compacts through the REPL, which needs an eval provider.
    if (
        compaction_soft_tokens is None
        and compaction_hard_tokens is not None
        and eval_provider != "none"
    ):
        h = coproduct(
            h, ReplCompactor(compaction_hard_tokens, CompactionScope.CONVERSATION)
        )
    h = coproduct(h, HistoryBuilder())
    if compaction_soft_tokens is not None and compaction_hard_tokens is not None:
        h = coproduct(
            h,
            MiddleCompactor(
                compaction_soft_tokens,
                compaction_hard_tokens,
                recent_tokens=compaction_recent_tokens,
            ),
        )

    if render:
        h = coproduct(h, RichTerminalRenderer())

    if dump_system_prompt:
        h = coproduct(
            h,
            SystemPromptDumper(path=pathlib.Path(dump_system_prompt)),
        )

    if type_checker == "ty":
        h = coproduct(h, TyTypeChecker())
    elif type_checker == "mypy":
        h = coproduct(h, MypyTypeChecker())

    if eval_provider == "restricted":
        h = coproduct(h, RestrictedPythonExecutor())
    elif eval_provider == "builtin":
        h = coproduct(h, BuiltinExecutor())

    if eval_provider != "none":
        h = coproduct(h, StatefulReplSynthesizer())
        h = coproduct(h, FinalBodySynthesizer())
        h = coproduct(h, ApiReferenceDocumenter())

    if check_contracts:
        h = coproduct(h, PydanticSkillArgValidator())

    if num_retries > 0:
        h = coproduct(h, TenacityRetryer(stop=tenacity.stop_after_attempt(num_retries)))

    if mcp_config is not None:
        import fastmcp

        from effectful.handlers.llm.harness.legibility.mcp import (
            MCPTools,
            background_loop,
        )

        servers = (
            mcp_config
            if isinstance(mcp_config, collections.abc.Mapping)
            else json.loads(pathlib.Path(mcp_config).read_text())
        )
        h = coproduct(
            h, MCPTools(fastmcp.Client(dict(servers)), loop=background_loop())
        )

    if persist_db is not None:
        h = coproduct(h, SQLitePersister(pathlib.Path(persist_db)))

    if langfuse:
        h = coproduct(h, LangfuseTracer())

    return h
