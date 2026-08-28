"""Handlers that give the types in :mod:`effectful.handlers.llm.types` their meaning.

The :func:`harness` function assembles the standard stack; its constituents are
documented in the submodules below and may be recombined or replaced
individually.
"""

import os
import pathlib
import typing

import tenacity

from effectful.handlers.llm.harness.durability.persistence import SQLitePersister
from effectful.handlers.llm.harness.durability.retrying import TenacityRetryer
from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
from effectful.handlers.llm.harness.execution.builtin import BuiltinExecutor
from effectful.handlers.llm.harness.execution.restricted import (
    RestrictedPythonExecutor,
)
from effectful.handlers.llm.harness.hooks import AgentLoop
from effectful.handlers.llm.harness.legibility.framework import FrameworkDocumenter
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
from effectful.handlers.llm.harness.validation.pydantic import PydanticSkillArgValidator
from effectful.handlers.llm.harness.validation.ty import TyTypeChecker
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

    1. `AgentLoop`, the tool pipeline and `LiteLLMConfigurer` -- the agent
       loop, the tools it offers from a `Skill`'s lexical scope, and the model
       backend it drives. The pipeline is a lexical tool *extractor* chosen by
       ``tool_collection`` (`LexicalToolExtractor` for ``"explicit"``,
       `ImplicitToolExtractor` for ``"auto"``, nothing for ``"none"``), plus,
       above it, the tool *caller* it feeds (`MixedToolCaller` for
       ``tool_calling="auto"``, `ExpressionToolCaller` for ``"code"``, none
       for ``"json"``).
    2. `FrameworkDocumenter` -- describe the framework's concepts in the system
       prompt.
    3. `HistoryBuilder` -- accumulate the message history of a call.
    4. `RichTerminalRenderer` -- live-render the streaming history (if ``render``).
    5. `SystemPromptDumper` -- dump the system prompt (if ``dump_system_prompt``).
    6. The ``type_checker`` and the ``eval_provider`` -- check and run
       model-authored Python (each omitted for ``"none"``).
    7. `StatefulReplSynthesizer` and `FinalBodySynthesizer` -- answer a call by
       running a snippet, and by synthesizing a function and calling it. Both
       are omitted when ``eval_provider="none"``: each advertises a tool
       (``exec_code``, ``write_and_run_body``) that only an executor can decode.
    8. `PydanticSkillArgValidator` -- enforce the pre-conditions a caller
       wrote into a `Skill`'s parameter annotations (if ``check_contracts``).
    9. `TenacityRetryer` -- retry malformed/failing model output (if
       ``num_retries``).
    10. `SQLitePersister` -- checkpoint a persisted `Agent`'s state/history to
        SQLite after each successful call (if ``persist_db``).
    11. `LangfuseTracer` -- log calls to Langfuse (if ``langfuse``).

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
            persisted `~effectful.handlers.llm.types.Agent`'s (one
            constructed with an explicit `agent_id`) state and history via
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
            ordinary functions and methods that look deliberately published --
            public name, docstring, complete annotations (see
            `ImplicitToolExtractor._implicit_tool_candidate`) -- with no
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

    Raises:
        ValueError: If ``tool_calling`` is ``"auto"`` or ``"code"`` and
            ``eval_provider`` is ``"none"``.
    """
    h: Interpretation = AgentLoop()

    if tool_calling != "json" and eval_provider == "none":
        raise ValueError(
            f'tool_calling="{tool_calling}" has the model answer by writing '
            f"Python, so it needs an eval provider to run what it writes. Pass "
            f'eval_provider="builtin" or "restricted", or tool_calling="json".'
        )

    # The tool pipeline composes: the caller (added first, so it runs last)
    # transforms the lexical tools that the extractor (added above it, so it
    # runs first and forwards into it) discovered and unioned into `tools`.
    # With a caller installed, the extractor must not pre-filter unadvertisable
    # tools (json_only=False): the caller wraps exactly those. Under
    # tool_collection="none" there is no extractor, so the caller (still
    # installed, per tool_calling) has nothing lexical to wrap and every tool
    # the model sees is one another handler injected.
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
    h = coproduct(h, HistoryBuilder())

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

    if check_contracts:
        h = coproduct(h, PydanticSkillArgValidator())

    if num_retries > 0:
        h = coproduct(h, TenacityRetryer(stop=tenacity.stop_after_attempt(num_retries)))

    if persist_db is not None:
        h = coproduct(h, SQLitePersister(pathlib.Path(persist_db)))

    if langfuse:
        h = coproduct(h, LangfuseTracer())

    return h
