"""Decoding LLM responses into Python objects, including callables.

Demonstrates:
- Primitive type decoding (``int``) from a template that returns a number
- Synthesizing a Python ``Callable`` from a template, executed via
  ``UnsafeEvalProvider`` from ``effectful.handlers.llm.evaluation``
- ``inspect.getsource`` on the synthesized function
"""

import argparse
import inspect
from collections.abc import Callable

from effectful.handlers.llm import Template

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@Template.define
def primes(first_digit: int) -> int:
    """Give a prime number with {first_digit} as the first digit. Do not use any tools."""


@Template.define
def count_char(char: str) -> Callable[[str], int]:
    """Write a function which takes a string and counts the occurrances of '{char}'. Do not use any tools."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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

    prime = primes(args.first_digit)
    assert type(prime) is int
    print(f"Prime starting with {args.first_digit}: {prime}")

    counter = count_char(args.char)
    assert callable(counter)
    print("\nGenerated function:")
    print(inspect.getsource(counter))
    print(f'counter("banana") == {counter("banana")}')
    print(f'counter("cherry") == {counter("cherry")}')


if __name__ == "__main__":
    main()
