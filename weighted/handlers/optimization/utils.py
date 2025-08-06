from typing import Any

from effectful.ops.types import Term

from weighted.ops.monoid import Monoid

"""
Parsing utilities for the program transforms.
"""


def parse_terms(value: Term, monoid: Monoid) -> tuple[Any, list[Term]]:
    if not isinstance(value, Term):
        return None, [value]
    mul = value.op
    if not monoid.distributes_with(mul):
        return None, [value]

    return mul, parse_with_op(value, mul)


def parse_with_op(value: Term, op) -> list[Term]:
    if isinstance(value, Term) and value.op is op:
        return sum((parse_with_op(arg, op) for arg in value.args), [])
    else:
        return [value]
