import os

import litellm
import pytest

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

    from effectful.handlers.llm.completions import (
        _tools_in_scope,
        call_assistant,
    )
    from effectful.ops.semantics import handler
    from effectful.ops.syntax import ObjectInterpretation, implements

    captured: set = set()

    class _Capture(ObjectInterpretation):
        @implements(call_assistant)
        def _ca(self, env_, response_type, tools=frozenset(), **kw):
            captured.update(tools)
            return ({}, [], None)

    with contextlib.ExitStack() as stack:
        stack.enter_context(handler(_Capture()))
        for h in handlers:
            stack.enter_context(handler(h))
        call_assistant(env, str, _tools_in_scope(env))
    return captured


def template_tools(template, *handlers):
    """Set of Tools a `Template` would offer under the given handlers.

    Mirrors the behaviour of the removed ``Template.tools`` property: it applies
    the same handler augmentation as :func:`offered_tools` and drops the template
    itself, matching `LiteLLMProvider`'s ``_tools_in_scope(env) - {template}``.
    Like :func:`offered_tools`, tools are kept by object identity.
    """
    return {
        t for t in offered_tools(template.__context__, *handlers) if t is not template
    }
