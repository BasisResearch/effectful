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


def pytest_collection_modifyitems(config, items):
    """Remove auto-collected doctests from LLM template functions.

    Template docstrings contain ``>>>`` examples that serve as LLM prompts
    for the DoctestHandler, not as standalone doctests for pytest to run.
    """
    items[:] = [
        item
        for item in items
        if not (
            type(item).__name__ == "DoctestItem"
            and "test_handlers_llm_doctest" in item.nodeid
        )
    ]
