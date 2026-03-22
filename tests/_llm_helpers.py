"""Shared constants for LLM integration tests.

All LLM test files import from here to avoid duplication.
"""

import os

import pytest

# Model name for LLM integration tests, configured via environment variable.
LLM_MODEL = os.environ.get("EFFECTFUL_LLM_MODEL", "")

requires_llm = pytest.mark.skipif(
    not LLM_MODEL, reason="EFFECTFUL_LLM_MODEL environment variable not set"
)
