from collections.abc import Collection, Mapping
from typing import Any

from effectful.ops.types import Term
from effectful.ops.weighted.monoid import Monoid

"""
Parsing utilities for the program transforms.
"""


def parse_terms(value: Term, monoid: Monoid) -> tuple[Any, list[Term]]:
    if not isinstance(value, Term):
        return None, [value]
    mul = value.op
    if not distributes_with(monoid, mul):
        return None, [value]

    return mul, parse_with_op(value, mul)


def parse_with_op(value: Term, op) -> list[Term]:
    if isinstance(value, Term) and value.op is op:
        return sum((parse_with_op(arg, op) for arg in value.args), [])
    else:
        return [value]


def partition_streams[K, V](
    streams: Mapping[K, V], include_keys: Collection[K]
) -> tuple[Mapping[K, V], Mapping[K, V]]:
    """
    Partition streams into two dictionaries based on whether their keys
    are in the include_keys set.

    Returns:
        (included_streams, excluded_streams)
    """
    included = {k: v for k, v in streams.items() if k in include_keys}
    excluded = {k: v for k, v in streams.items() if k not in include_keys}
    return included, excluded
