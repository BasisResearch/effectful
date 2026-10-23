import ast
import functools
import os
import pathlib
import typing
from collections.abc import Sequence

import pytest
from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion

from effectful.handlers.llm.harness.hooks import Tool, completion
from effectful.handlers.llm.harness.legibility.lexical import _tools_in_scope
from effectful.handlers.llm.harness.provision.any_llm import any_llm_model
from effectful.ops.syntax import ObjectInterpretation, implements

EFFECTFUL_LLM_MODEL = os.environ.get("EFFECTFUL_LLM_MODEL", "gpt-4.1-mini")

_PROVIDER_METADATA = AnyLLM.get_provider_class(
    AnyLLM.split_model_provider(any_llm_model(EFFECTFUL_LLM_MODEL))[0]
).get_provider_metadata()

_HAS_LLM_API_KEY = bool(os.environ.get(_PROVIDER_METADATA.env_key))

requires_llm = pytest.mark.skipif(
    not _HAS_LLM_API_KEY,
    reason=f"No API key configured for model {EFFECTFUL_LLM_MODEL}",
)

# Provider-level: a provider that takes images may still serve a model that does not.
requires_vision = pytest.mark.skipif(
    not _PROVIDER_METADATA.image,
    reason=f"The provider of {EFFECTFUL_LLM_MODEL} does not support vision",
)

# ============================================================================
# Finding the example scripts
#
# Shared by the launcher suite, which checks the examples statically, and the
# examples suite, which runs them. Both need the same answer to "what is an
# example, and what is it called", and neither should be the one that owns it.
# ============================================================================

EXAMPLES_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "docs" / "source" / "llm_examples"
)


@functools.cache
def example_tree(path: pathlib.Path) -> ast.Module:
    """The parsed source of an example, cached across the tests that walk it."""
    return ast.parse(path.read_text(), filename=str(path))


def example_modules() -> list[pathlib.Path]:
    """Every example module, including the shared sibling libraries."""
    return sorted(
        p
        for p in EXAMPLES_DIR.rglob("*.py")
        if "__pycache__" not in p.parts and p.name != "__init__.py"
    )


def example_scripts() -> list[pathlib.Path]:
    """Every example that is a script -- one with a command line of its own.

    The shared sibling libraries (``choreographies/library.py``,
    ``optimization/ds1000_data.py``, ...) are modules the scripts import; they
    define no ``main`` and there is nothing to launch.
    """
    return [
        p
        for p in example_modules()
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "main"
            for node in example_tree(p).body
        )
    ]


def example_option_strings(path: pathlib.Path) -> list[str]:
    """The ``--flags`` an example's parser declares, read off its source.

    Statically, because the parser is built inside ``main()`` and reaching it
    would mean running the example.
    """
    return sorted(
        {
            arg.value
            for node in ast.walk(example_tree(path))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        }
    )


def example_id(path: pathlib.Path) -> str:
    """``basics/conversation`` -- the name an example is known by in a test id."""
    return str(path.relative_to(EXAMPLES_DIR).with_suffix(""))


def example_ids(paths: Sequence[pathlib.Path]) -> list[str]:
    return [example_id(p) for p in paths]


UNIMPLEMENTED_SUBSTRINGS = [
    "infer.JitTrace_ELBO",
    "the event_dim arg",
    "optim.ClippedAdam",
    "infer.TraceMeanField_ELBO",
]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        output = yield
        return output
    except RuntimeError as e:
        if any(s in str(e) for s in UNIMPLEMENTED_SUBSTRINGS):
            pytest.xfail(str(e))
        else:
            raise e


def offered_tools(env, *handlers):
    """Set of Tools the model would be offered for lexical scope `env`
    under the given handlers.

    Replaces the old ``collect_tools`` operation: tool collection now happens as
    `call_assistant` seeds its `tools` set from :func:`_tools_in_scope` and the
    augmenting handlers (``StatefulReplSynthesizer``, ``FinalBodySynthesizer``,
    ...) union more in.
    This installs a capture handler that records the tools `call_assistant`
    ultimately receives.

    Tools are kept by object identity, not by name: two distinct tools that
    share a ``__name__`` (e.g. the same method bound to different instances)
    are both preserved. Callers checking name presence should compare against
    ``{t.__name__ for t in offered_tools(...)}``.
    """
    import contextlib

    from effectful.handlers.llm.harness.hooks import (
        call_assistant,
    )
    from effectful.ops.semantics import handler
    from effectful.ops.syntax import ObjectInterpretation, implements

    captured: set = set()

    class _Capture(ObjectInterpretation):
        @implements(call_assistant)
        def _ca(self, messages_, response_type, env_, tools=frozenset(), **kw):
            captured.update(tools)
            return ({}, [], None)

    with contextlib.ExitStack() as stack:
        stack.enter_context(handler(_Capture()))
        for h in handlers:
            stack.enter_context(handler(h))
        call_assistant([], str, env, _tools_in_scope(env))
    return captured


def skill_tools(skill, *handlers):
    """Set of Tools a `Skill` would offer under the given handlers.

    Mirrors the behaviour of the removed ``Skill.tools`` property: it applies
    the same handler augmentation as :func:`offered_tools` and drops the skill
    itself, matching `AgentLoop`'s ``_tools_in_scope(env) - {skill}``.
    Like :func:`offered_tools`, tools are kept by object identity.
    """
    return {t for t in offered_tools(skill.__context__, *handlers) if t is not skill}


# ============================================================================
# Offline model doubles
#
# The counterpart to the `requires_*` markers above: these stand in for the
# model entirely, so a test using them needs no API key and makes no network
# call. Shared by the provision and observability suites.
# ============================================================================


class MockCompletionHandler(ObjectInterpretation):
    """Mock handler that returns pre-configured completion responses."""

    def __init__(self, responses: list[ChatCompletion]):
        self.responses = responses
        self.call_count = 0
        self.received_messages: list = []

    @implements(completion)
    def _completion(self, messages=None, **kwargs):
        self.received_messages.append(list(messages) if messages else [])
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


def chat_completion(
    choices: Sequence[dict[str, typing.Any]],
    id: str = "test",
    model: str = "test-model",
    usage: dict[str, int] | None = None,
    **fields: typing.Any,
) -> ChatCompletion:
    """A `ChatCompletion` with the fields the tests never vary filled in.

    Each choice's ``index`` and ``finish_reason`` default to its position and
    ``"stop"``.
    """
    return ChatCompletion.model_validate(
        {
            "id": id,
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {"index": i, "finish_reason": "stop", **choice}
                for i, choice in enumerate(choices)
            ],
            "usage": usage,
            **fields,
        }
    )


def make_tool_call_response(
    tool_name: str | Sequence[tuple[str, str]],
    tool_args: str | None = None,
    tool_call_id: str = "call_1",
) -> ChatCompletion:
    """Create a ChatCompletion with one tool call, or several.

    The common case is one call: pass its name and JSON arguments. A turn that
    requests *several* is a shape the harness has rules about -- a finalizing
    call must be alone, a compacting one must not orphan its siblings -- so pass
    a sequence of ``(name, arguments)`` pairs instead to build one. Their ids are
    then numbered ``call_1``, ``call_2``, ... and `tool_call_id` is unused.
    """
    if isinstance(tool_name, str):
        assert tool_args is not None
        calls = [(tool_call_id, tool_name, tool_args)]
    else:
        calls = [
            (f"call_{i}", name, args) for i, (name, args) in enumerate(tool_name, 1)
        ]
    return chat_completion(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                        for call_id, name, args in calls
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    )


def make_text_response(
    content: str, usage: dict[str, int] | None = None
) -> ChatCompletion:
    """Create a ChatCompletion with text content.

    ``usage`` populates the response's token counts. Real responses carry them
    and some handlers read them (`LangfuseTracer` reports them as
    ``usage_details``), so a test that asserts on them has to ask for them
    explicitly.
    """
    return chat_completion(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage=usage,
    )


@Tool.define
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
