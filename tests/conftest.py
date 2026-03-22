import os

import pytest

# Model name for live LLM integration tests, configured via environment variable.
# Defaults to "gpt-4o-mini"; override with EFFECTFUL_LLM_MODEL to test other providers.
LLM_MODEL = os.environ.get("EFFECTFUL_LLM_MODEL", "gpt-4o-mini")

# Skip live LLM tests when no provider API key is available.
# litellm resolves the correct key from the environment based on the model name;
# we just check that at least one common provider key is set.
_HAS_LLM_API_KEY = any(
    os.environ.get(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
)

requires_llm = pytest.mark.skipif(
    not _HAS_LLM_API_KEY,
    reason="No LLM provider API key found in environment",
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
