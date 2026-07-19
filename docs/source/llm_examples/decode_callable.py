"""Decoding LLM responses into Python objects, including callables.

Demonstrates:
- Primitive type decoding (``int``) from a template that returns a number
- Synthesizing a Python ``Callable`` from a template, executed via
  ``UnsafeEvalProvider`` from ``effectful.handlers.llm.evaluation``
- ``inspect.getsource`` on the synthesized function
"""

import argparse
import inspect
import os
from collections.abc import Callable

from tenacity import stop_after_attempt

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.handlers.llm.evaluation import UnsafeEvalProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@Template.define
def primes(first_digit: int) -> int:
    """Give a prime number with {first_digit} as the first digit. Do not use any tools."""
    raise NotHandled


@Template.define
def count_char(char: str) -> Callable[[str], int]:
    """Write a function which takes a string and counts the occurrances of '{char}'. Do not use any tools."""
    raise NotHandled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode LLM responses to Python objects (incl. callables)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=5,
        help="Number of retries for malformed LLM output",
    )
    parser.add_argument(
        "--first-digit",
        type=int,
        default=6,
        help="First digit of the prime to request",
    )
    parser.add_argument(
        "--char",
        type=str,
        default="a",
        help="Character whose occurrences the synthesized function will count",
    )
    args = parser.parse_args()

    provider = LiteLLMProvider(model=args.model)

    with (
        handler(provider),
        handler(RetryLLMHandler(stop=stop_after_attempt(args.num_retries))),
        handler(UnsafeEvalProvider()),
    ):
        prime = primes(args.first_digit)
        assert type(prime) is int
        print(f"Prime starting with {args.first_digit}: {prime}")

        counter = count_char(args.char)
        assert callable(counter)
        print("\nGenerated function:")
        print(inspect.getsource(counter))
        print(f'counter("banana") == {counter("banana")}')
        print(f'counter("cherry") == {counter("cherry")}')
