import os

import litellm
import pytest
from litellm.files.main import ModelResponse

from effectful.handlers.llm.harness.hooks import Tool, _tools_in_scope, completion
from effectful.ops.syntax import ObjectInterpretation, implements

EFFECTFUL_LLM_MODEL = os.environ.get("EFFECTFUL_LLM_MODEL", "gpt-4o-mini")

_HAS_LLM_API_KEY = litellm.validate_environment(model=EFFECTFUL_LLM_MODEL)[
    "keys_in_environment"
]

requires_llm = pytest.mark.skipif(
    not _HAS_LLM_API_KEY,
    reason=f"No API key configured for model {EFFECTFUL_LLM_MODEL}",
)

requires_vision = pytest.mark.skipif(
    not litellm.supports_vision(model=EFFECTFUL_LLM_MODEL),
    reason=f"Model {EFFECTFUL_LLM_MODEL} does not support vision",
)

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="No API key configured for OpenAI",
)

requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No API key configured for Anthropic",
)

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
    augmenting handlers (``LexicalReaders``, ``PythonRepl``, ...) union more in.
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

    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.call_count = 0
        self.received_messages: list = []

    @implements(completion)
    def _completion(self, messages=None, **kwargs):
        self.received_messages.append(list(messages) if messages else [])
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


def make_tool_call_response(
    tool_name: str, tool_args: str, tool_call_id: str = "call_1"
) -> ModelResponse:
    """Create a ModelResponse with a tool call."""
    return ModelResponse(
        id="test",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        model="test-model",
    )


def make_text_response(
    content: str, usage: dict[str, int] | None = None
) -> ModelResponse:
    """Create a ModelResponse with text content.

    ``usage`` populates the response's token counts. Real responses carry them
    and some handlers read them (`LangfuseTracer` reports them as
    ``usage_details``), but litellm synthesizes a zero-filled ``usage`` when it
    is omitted, so a test that asserts on them has to ask for them explicitly.
    """
    return ModelResponse(
        id="test",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        model="test-model",
        **({"usage": usage} if usage is not None else {}),
    )


@Tool.define
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
