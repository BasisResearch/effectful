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
from effectful.handlers.llm.harness.legibility.lexical import LexicalToolExtractor
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
from effectful.handlers.llm.harness.validation.mypy import MypyTypeChecker
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

    1. `AgentLoop`, `LexicalToolExtractor` and `LiteLLMConfigurer` -- the agent
       loop, the tools it offers from a `Skill`'s lexical scope, and the model
       backend it drives.
    2. `FrameworkDocumenter` -- describe the framework's concepts in the system
       prompt.
    3. `HistoryBuilder` -- accumulate the message history of a call.
    4. `RichTerminalRenderer` -- live-render the streaming history (if ``render``).
    5. `SystemPromptDumper` -- dump the system prompt (if ``dump_system_prompt``).
    6. The ``type_checker`` (`TYPE_CHECKERS`), the ``eval_provider``
       (`EVAL_PROVIDERS`) and `StatefulReplSynthesizer` -- check and run
       model-authored Python.
    7. `FinalBodySynthesizer` -- synthesize a function and call it.
    8. `TenacityRetryer` -- retry malformed/failing model output.
    9. `LexicalReaders` -- expose lexically-scoped tools to the model.
    10. `SQLitePersister` -- checkpoint a persisted `Agent`'s state/history to
        SQLite after each successful call (if ``persist_db``).
    11. `LangfuseTracer` -- log calls to Langfuse (if ``langfuse``).

    Args:
        num_retries: Attempts for malformed/failing model output.
        langfuse: Log LLM calls and metadata to Langfuse.
        render: Live-render the streaming message history in the terminal.
        dump_system_prompt: If set, dump the assembled system prompt to this
            Markdown file.
        persist_db: If set, path to a SQLite database used to checkpoint a
            persisted `~effectful.handlers.llm.types.Agent`'s (one
            constructed with an explicit `agent_id`) state and history via
            `~effectful.handlers.llm.completions.SQLitePersister`.
        eval_provider: Which provider runs model-authored Python -- a key of
            `EVAL_PROVIDERS` (``"unsafe"`` or ``"restricted"``).
        type_checker: Which handler type-checks model-authored Python before it
            runs -- a key of `TYPE_CHECKERS` (``"mypy"``, or ``"none"`` to run
            generated code unchecked).
    """
    h: Interpretation = AgentLoop()
    h = coproduct(h, LexicalToolExtractor())
    h = coproduct(h, LiteLLMConfigurer(**provider_config))
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

    h = coproduct(h, StatefulReplSynthesizer())
    h = coproduct(h, FinalBodySynthesizer())
    h = coproduct(h, TenacityRetryer(stop=tenacity.stop_after_attempt(num_retries)))
    if persist_db is not None:
        h = coproduct(h, SQLitePersister(pathlib.Path(persist_db)))
    if langfuse:
        h = coproduct(h, LangfuseTracer())
    return h
