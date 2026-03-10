import pytest

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
