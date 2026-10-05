"""Run a script or module under the standard effectful LLM handler stack.

Usage: python -m effectful.handlers.llm.harness SCRIPT.py [harness flags]
[script flags], or -m MODULE in place of the path. Harness flags are consumed
here (_parse_args) and build the stack with harness() (_build_harness);
everything else is passed through in sys.argv under the script's own name. A
script runs as __main__ with its directory on sys.path; a module runs via
runpy.run_module.

The model is --model or the EFFECTFUL_LLM_MODEL environment variable;
_provider_config assembles it with --tool-choice and --reasoning-effort into the
LiteLLMConfigurer settings, rewriting an OpenAI GPT-5.4+ model onto the Responses
API when no effort is given. Provider parameters a model rejects are dropped
(litellm.drop_params).

Launcher-only flags: --pdb enters post-mortem with the stack still installed;
--autoreload re-runs edited modules, including the harness, and rebuilds the
stack (effectful.handlers.llm.harness.autoreload.Reloader).
"""

import argparse
import functools
import importlib.util
import os
import pdb
import runpy
import sys
import textwrap
import typing

import litellm

from effectful.handlers.llm.harness import harness
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import handler
from effectful.ops.types import Interpretation


def _reasoning_effort_choices() -> list[str] | None:
    """The ``reasoning_effort`` values a provider will actually accept.

    Read from litellm's canonical ``REASONING_EFFORT`` alias so the CLI choices
    track litellm across upgrades. That alias -- and not the looser ``Literal``
    on ``litellm.completion``'s own signature, which additionally admits
    ``"default"`` -- is the set providers are held to: OpenAI rejects
    ``"default"`` outright with ``Unsupported value: 'reasoning_effort' does not
    support 'default' with this model``. "Let the model decide" is spelled by
    *omitting* the parameter, which is what this flag's ``None`` default does.

    Returns ``None`` (leave the flag unrestricted) if the alias isn't a Literal
    we can read, so a shape change in litellm degrades to accepting any string
    rather than breaking the launcher.
    """
    try:
        from litellm.types.llms.openai import REASONING_EFFORT

        literals = [v for v in typing.get_args(REASONING_EFFORT) if isinstance(v, str)]
        return literals or None
    except Exception:
        return None


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split ``argv`` into harness options and pass-through script flags.

    ``allow_abbrev=False`` is what makes the split honest. With argparse's default,
    any script flag that is a unique prefix of a harness flag is claimed here
    instead of being passed through -- a script's ``--mode`` would be read as this
    parser's ``--model``, silently overwriting the model *and* dropping the flag the
    script needed.
    """
    # Taken out first, so that with ``-m`` there is no script positional left to
    # swallow a bare value among the script's own arguments (``--budget 20``).
    module = None
    if "-m" in argv:
        at = argv.index("-m")
        module, argv = (
            argv[at + 1 : at + 2][0] if argv[at + 1 : at + 2] else "",
            (argv[:at] + argv[at + 2 :]),
        )
    parser = argparse.ArgumentParser(
        prog=f"python -m {__spec__.name}" if __spec__ else None,
        description=textwrap.dedent(__doc__),
        allow_abbrev=False,
    )
    if module is None:
        parser.add_argument("script", help="Path to the script to run")
    parser.add_argument(
        "-m",
        dest="module",
        metavar="MODULE",
        help="Run an installed module instead, e.g. an effectful.handlers.llm.examples one",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=5,
        help=(
            "Attempts for malformed/failing LLM output, and for transport-level "
            "failures (forwarded to litellm as its own num_retries)"
        ),
    )
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Whether to log LLM calls and metadata to Langfuse",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Live-render the streaming message history in the terminal",
    )
    parser.add_argument(
        "--dump-system-prompt",
        type=str,
        default=None,
        metavar="PATH",
        help="Dump the assembled system prompt to this Markdown file",
    )
    parser.add_argument(
        "--tool-choice",
        type=str,
        default="auto",
        choices=["required", "auto", "none"],
        help="Whether to require, allow, or disable tool calls (none means disabled)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=_reasoning_effort_choices(),
        help="Reasoning effort forwarded to litellm.completion; left unset "
        "(the model's own default) when not given",
    )
    parser.add_argument(
        "--eval-provider",
        type=str,
        default="builtin",
        choices=["builtin", "restricted", "none"],
        help="Provider that runs model-authored Python",
    )
    parser.add_argument(
        "--type-checker",
        type=str,
        default="ty",
        choices=["mypy", "ty", "none"],
        help="Handler that type-checks model-authored Python before it runs",
    )
    parser.add_argument(
        "--tool-calling",
        type=str,
        default="auto",
        choices=["auto", "code", "json"],
        help=(
            "How the model calls lexical tools: JSON arguments where a schema "
            "can describe the tool, code for the rest (auto); by writing a "
            "type-checked Python call expression uniformly (code); or JSON "
            "arguments only (json)"
        ),
    )
    parser.add_argument(
        "--tool-collection",
        type=str,
        default="explicit",
        choices=["none", "explicit", "auto"],
        help=(
            "Which tools are collected from a Skill's lexical scope: the "
            "declared Tool/Skill values (explicit); those plus qualifying "
            "plain functions and methods, no Tool.define decorator needed "
            "(auto); or nothing from the scope at all, leaving only the "
            "harness's own tools (none)"
        ),
    )
    parser.add_argument(
        "--no-check-contracts",
        dest="check_contracts",
        action="store_false",
        help=(
            "Do not validate a Skill's arguments against the pydantic metadata "
            "on its parameter annotations"
        ),
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Drop into pdb post-mortem on an unhandled error (like `python -m pdb`)",
    )
    parser.add_argument(
        "--autoreload",
        action="store_true",
        help=(
            "Re-run edited code imported from sys.path directories, including the "
            "harness, while the script runs"
        ),
    )
    parser.add_argument(
        "--persist-db",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Checkpoint persisted Agent state/history to this SQLite database "
            "(installs SQLitePersister)"
        ),
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Offer the tools of the MCP servers in this JSON file, a standard "
            '{"mcpServers": {...}} configuration, to every Skill (installs MCPTools)'
        ),
    )
    ns, rest = parser.parse_known_args(argv)
    if module is not None:
        if not module or ns.module is not None:
            parser.error("-m takes one module name")
        ns.module, ns.script = module, None
    return ns, rest


def _openai_model_needing_responses_api(model: str) -> str | None:
    """`model` without its provider prefix, if it is an OpenAI model that must be
    addressed through the Responses API to use tools; else ``None``.

    OpenAI rejects function tools alongside reasoning on ``/v1/chat/completions``
    for its GPT-5.4-and-later models (*Function tools with reasoning_effort are
    not supported ... use /v1/responses*), and the harness always sends tools.
    The same model reached through another provider, such as
    ``openrouter/openai/gpt-5.4``, is that provider's business and is left alone,
    as is a model litellm does not classify -- another provider's, or one newer
    than the installed litellm knows -- so an unfamiliar name degrades to today's
    behaviour rather than being rewritten on a guess.
    """
    try:
        from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config

        name, provider, *_ = litellm.get_llm_provider(model)
        if provider != "openai":
            return None
        needs = OpenAIGPT5Config.is_model_gpt_5_4_plus_model(
            name
        ) and not OpenAIGPT5Config.is_model_gpt_5_search_model(name)
        return name if needs else None
    except Exception:
        return None


def _provider_config(ns: argparse.Namespace) -> dict[str, typing.Any]:
    """The litellm kwargs `LiteLLMConfigurer` is built from.

    An unset ``--reasoning-effort`` is left out of the request entirely rather
    than forwarded as a sentinel. Every value this parameter takes is one some
    provider rejects -- OpenAI answers ``reasoning_effort`` of ``"default"`` with
    a 400 -- and the only universally safe way to say "whatever the model does by
    default" is to say nothing.

    Saying nothing has one consequence worth naming, because it is not obvious
    and it is why the model string may be rewritten below. litellm routes a chat
    completion to the Responses API only when ``reasoning_effort`` is not None
    (``litellm.main.responses_api_bridge_check``), so omitting the parameter also
    silently opts a GPT-5.4+ model *out* of the endpoint its tool calls require.
    The ``openai/responses/`` prefix is litellm's own way to ask for that
    endpoint directly, and it does not depend on the reasoning parameter -- which
    lets the effort stay at the model's own default instead of being pinned to a
    value nobody chose. An explicit ``--reasoning-effort`` needs none of this: it
    triggers the bridge by itself, and is left to do so.
    """
    model = ns.model
    if ns.reasoning_effort is None and (
        name := _openai_model_needing_responses_api(model)
    ):
        model = f"openai/responses/{name}"

    config: dict[str, typing.Any] = {"model": model, "tool_choice": ns.tool_choice}
    if ns.reasoning_effort is not None:
        config["reasoning_effort"] = ns.reasoning_effort
    return config


def _build_harness(ns: argparse.Namespace) -> Interpretation:
    """The handler stack the parsed harness flags ask for."""
    return harness(
        num_retries=ns.num_retries,
        langfuse=ns.langfuse,
        render=ns.render,
        dump_system_prompt=ns.dump_system_prompt,
        persist_db=ns.persist_db,
        eval_provider=ns.eval_provider,
        type_checker=ns.type_checker,
        tool_calling=ns.tool_calling,
        tool_collection=ns.tool_collection,
        check_contracts=ns.check_contracts,
        mcp_config=ns.mcp_config,
        **_provider_config(ns),
    )


def main(argv: list[str] | None = None) -> None:
    litellm.drop_params = True
    ns, script_args = _parse_args(sys.argv[1:] if argv is None else argv)
    if ns.module is not None:
        try:
            spec = importlib.util.find_spec(ns.module)
        except ModuleNotFoundError:  # a missing parent package
            spec = None
        if spec is None or spec.origin is None:
            raise SystemExit(f"No module named {ns.module}")
        ns.script = spec.origin
        run = functools.partial(runpy.run_module, ns.module, alter_sys=True)
    else:
        # Mirror `python <script>`: put the script's directory on sys.path so it can
        # import sibling modules (e.g. a shared environment definition) by absolute
        # name. `runpy.run_path` runs the file as `__main__` with no package, so
        # relative imports can't work and this dir would otherwise be off the path.
        sys.path.insert(0, os.path.dirname(os.path.abspath(ns.script)))
        run = functools.partial(runpy.run_path, ns.script)
    # The script should see only its own flags, under its own name.
    sys.argv = [ns.script, *script_args]
    if ns.autoreload:
        # Before the first build, so the handler modules load through hmr.
        from effectful.handlers.llm.harness import autoreload

        reloader = autoreload.Reloader(ns.script, lambda: _build_harness(ns))
        reloader.start()
        installed = interpreter(reloader)
    else:
        installed = handler(_build_harness(ns))
    with installed:
        if ns.pdb:
            try:
                run(run_name="__main__")
            except BaseException:
                # Post-mortem while the handler stack is still installed, so live
                # handler/session state is inspectable at the debugger prompt.
                pdb.post_mortem()
        else:
            run(run_name="__main__")


if __name__ == "__main__":
    main()
