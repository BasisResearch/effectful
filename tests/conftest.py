import os

import pytest

# Model name for LLM integration tests, configured via environment variable.
# All LLM test files import these from here to avoid duplication.
LLM_MODEL = os.environ.get("EFFECTFUL_LLM_MODEL", "")

requires_llm = pytest.mark.skipif(
    not LLM_MODEL, reason="EFFECTFUL_LLM_MODEL environment variable not set"
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
