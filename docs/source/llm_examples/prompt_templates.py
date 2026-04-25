"""Basic prompt templates and deterministic caching.

Demonstrates:
- ``@Template.define`` for declaring an LLM-backed function
- Non-determinism: calling the same template twice yields different results
- ``functools.cache`` to make a template call deterministic in-process
- ``LiteLLMProvider(caching=True)`` with ``litellm.cache`` for cross-process caching
"""

import argparse
import functools
import os

import litellm
from litellm.caching.caching import Cache

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@Template.define
def limerick(theme: str) -> str:
    """Write a limerick on the theme of {theme}. Do not use any tools."""
    raise NotHandled


@functools.cache
@Template.define
def haiku(theme: str) -> str:
    """Write a haiku on the theme of {theme}. Do not use any tools."""
    raise NotHandled


@Template.define
def haiku_no_cache(theme: str) -> str:
    """Write a haiku on the theme of {theme}. Do not use any tools."""
    raise NotHandled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Basic prompt templates and deterministic caching"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    parser.add_argument(
        "--theme", type=str, default="fish", help="Theme for the poem"
    )
    args = parser.parse_args()

    provider = LiteLLMProvider(model=args.model)

    print("=== Non-deterministic limerick (two independent calls) ===")
    with handler(provider):
        print(limerick(args.theme))
        print("-" * 40)
        print(limerick(args.theme))

    print("\n=== functools.cache: same result on second call ===")
    with handler(provider):
        print(haiku(args.theme))
        print("-" * 40)
        print(haiku(args.theme))

    print("\n=== LiteLLMProvider(caching=True): backed by litellm.cache ===")
    litellm.cache = Cache()
    provider_cached = LiteLLMProvider(model=args.model, caching=True)
    try:
        with handler(provider_cached):
            print(haiku_no_cache(args.theme))
            print("-" * 40)
            print(haiku_no_cache(args.theme))
    finally:
        litellm.cache = None
