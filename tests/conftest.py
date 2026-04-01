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
