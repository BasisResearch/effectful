"""Run Skill calls with a configurable stack of handlers.

:func:`harness` assembles the standard stack. Install it around ordinary Python
code::

    from effectful.handlers.llm.harness import harness
    from effectful.ops.semantics import handler

    with handler(harness(model="openai/gpt-5-mini")):
        main()

Or launch a script with ``python -m effectful.handlers.llm.harness script.py``.
See :mod:`~effectful.handlers.llm.harness.__main__` for flags.

Choose Tool discovery, Tool calls, and execution separately::

    with handler(
        harness(
            model="openai/gpt-5-mini",
            tool_collection="auto",  # include qualifying ordinary functions
            tool_calling="auto",    # JSON where faithful, Python expressions otherwise
            tool_choice="auto",     # provider may call a Tool or answer directly
            eval_provider="builtin",  # authority for model-authored Python
            type_checker="ty",
            num_retries=5,
        )
    ):
        main()

One Skill call opens one conversation *turn*. The harness builds its request
from the Skill signature, formatted docstring, defining module, lexical scope,
and, for a method, its receiver's committed history. A model response is one
*round*; a Tool result or validation error can start another round in the same
turn. A successful answer is decoded to the Skill's return type. A failed turn
does not commit its messages to the receiver's history.

The standard call path::

    Python calls a Skill: begin one turn
    bind arguments over the Skill's captured lexical context
    open a transaction over the receiver's committed history (bound Skill),
      or over a fresh history (free Skill)
    render a candidate system message: installed-handler sections, defining
      module source, receiver and sibling-Skill docs, and scope tables;
      a bound conversation retains its first committed system message
    append one user message: current Skill signature and formatted docstring
    model round <-> Tool result or validation feedback, repeated as needed
    decode to the declared return type; type-check, execute, run doctests
      where this synthesis route requires them
    invalid: feed back and retry within this turn, if a retryer is installed
    accepted: return the value; a committable bound turn records its history,
      then may checkpoint
    exhausted: raise, leaving this turn's messages uncommitted

.. code-block:: text

    Skill call -> build request -> model round <-> Tool result / retry feedback
                                      |
                                      v
                              check returned value
                              /                  \
                       success                failure
                       return value           raise; do not commit history

Only a bound, committable turn adds its successful messages to its receiver's
history. See :mod:`~effectful.handlers.llm.harness.durability.transaction` for
nested calls and other history rules.

Choose the relevant layer from here:

- :mod:`~effectful.handlers.llm.harness.hooks`: operations and handler extension.
- :mod:`~effectful.handlers.llm.harness.legibility`: what the model sees and which
  Tools are advertised.
- :mod:`~effectful.handlers.llm.harness.synthesis`: model-authored Python.
- :mod:`~effectful.handlers.llm.harness.validation`: contracts and type checks.
- :mod:`~effectful.handlers.llm.harness.execution`: authority to run code.
- :mod:`~effectful.handlers.llm.harness.durability`: history, retries, and
  persistence.
- :mod:`~effectful.handlers.llm.harness.provision`: model requests.
- :mod:`~effectful.handlers.llm.harness.observability`: rendering and traces.

For a confusing model response, start with ``--dump-system-prompt PATH`` and
``--langfuse``. The dump contains the assembled prompt, while Langfuse records
rounds and Tool calls. A bound conversation may retain an earlier system
message; inspect its recorded history when the dump differs from what it saw.
For example::

    python -m effectful.handlers.llm.harness path/to/example.py \\
      --dump-system-prompt /tmp/effectful-prompt.md --langfuse

Check the request in this order:

1. Was the relevant handler installed under ``# Harness``?
2. Was the Skill's module source recovered, and which named values were
   interpolated into the current user message?
3. Was a capability reachable in lexical scope, advertised as a Tool, and
   encodable by the selected ``tool_calling`` route? These are separate checks.
4. Could runtime code reach it even if static checking could not name it?
5. Did a bound receiver retain an earlier system message while Tool discovery
   refreshed for a later round?
6. Did a failed turn discard messages but leave Python side effects or a nested
   checkpoint?
7. Which synthesis route ran, whose tests did it use, and did validation
   feedback reach the model before the Skill returned?
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
    """Assemble the standard handlers for Skill calls.

    Install the result around ordinary Python code::

        from effectful.handlers.llm.harness import harness
        from effectful.ops.semantics import handler

        with handler(harness(model="openai/gpt-5-mini")):
            main()

    The package docstring explains how a Skill call runs. The subpackages
    describe individual handlers and their boundaries. Options below select
    which handlers are installed; extra provider options go to LiteLLM.

    :param num_retries: Maximum attempts for an invalid model answer. The same
        value also configures LiteLLM's transport retries. ``0`` omits the
        answer retry handler.
    :param langfuse: Record Skill, Tool, and model calls in Langfuse.
    :param render: Stream a live terminal view of the conversation.
    :param dump_system_prompt: Write each assembled system prompt to this path.
        This is a prompt dump, not a transcript.
    :param persist_db: SQLite path for checkpointing receivers with a stable
        ``__agent_id__``. See :mod:`~effectful.handlers.llm.harness.durability.persistence`.
    :param eval_provider: ``"builtin"`` runs model-authored Python with normal
        process authority; ``"restricted"`` narrows it; ``"none"`` disables
        code execution and the tools that need it.
    :param type_checker: ``"ty"`` or ``"mypy"`` checks model-authored Python
        when source is recoverable; ``"none"`` skips static checks.
    :param tool_calling: ``"auto"`` uses JSON arguments when a Tool's signature
        fits a schema and checked Python expressions otherwise. ``"code"``
        always uses expressions; ``"json"`` uses JSON only and may omit Tools
        whose signatures cannot be encoded faithfully. ``"auto"`` and
        ``"code"`` require an eval provider.
    :param tool_collection: ``"explicit"`` offers in-scope ``Tool`` and
        ``Skill`` values. ``"auto"`` also offers qualifying ordinary functions
        and methods. ``"none"`` collects none from lexical scope. This
        controls discovery, independently of ``tool_calling``.
    :param check_contracts: Apply Pydantic annotation constraints to Skill
        arguments before each call. Return constraints are checked by decoding
        regardless of this setting.
    :param max_tool_output_chars: Maximum text retained from each Tool result,
        including its truncation notice. ``None`` disables truncation.
    :param compaction_soft_tokens: Approximate threshold for removing stale
        Tool output. ``None`` disables middle-region compaction.
    :param compaction_hard_tokens: Threshold for summarizing older messages
        when a soft threshold is set; it must exceed that threshold. Without a
        soft threshold, an executor-enabled harness instead asks the model to
        compact through its REPL.
    :param compaction_recent_tokens: Approximate recent window to keep during
        middle-region compaction. By default it is a quarter of the hard limit.
    :param mcp_config: An MCP server configuration mapping or a path to a JSON
        file containing one. Its Tools are offered to every Skill.
    :param provider_config: Additional LiteLLM settings, including ``model``
        and ``tool_choice``. ``tool_choice="required"`` forbids a direct answer;
        completion then needs a finalizing Tool such as ``write_and_run_body``.
        With ``"auto"``, an offered Tool may be skipped. ``"required"`` demands
        some Tool, not a particular one; enforce task-specific call requirements
        in application code.
    :raises ValueError: If expression-based Tool calling is selected with
        ``eval_provider="none"``.
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
    h = coproduct(h, FrameworkDocumenter(include_code_api=eval_provider != "none"))
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
