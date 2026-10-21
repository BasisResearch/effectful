"""Representation effects shared by high-level JAX APIs and custom arrays."""

from collections.abc import Callable
from typing import Any

import jax

from effectful.ops.syntax import defop


@defop
def is_custom_array(value) -> bool:
    """Whether a pytree represents one logical array operand."""
    return False


@defop
def to_jax_array(value):
    """Convert a represented array to its dense value; leave other values alone."""
    if is_custom_array(value):
        raise NotImplementedError(
            f"No conversion from {type(value).__name__} to jax.Array available"
        )
    return value


def _id(x):
    return x


def _compose(outer, inner):
    return lambda x: outer(inner(x))


@defop
def upcast[S, T](left: S, right: T) -> Callable[[S | T], Any]:
    """Select one value-preserving constructor for two abstract witnesses.

    Witnesses come from eval_shape. The constructor must accept either source
    representation, produce a shared physical layout, and capture no buffers.
    """
    if type(left) is type(right):
        return _id
    raise NotImplementedError(
        f"No upcast for {type(left).__name__} and {type(right).__name__}"
    )


def upcast_many(*values):
    """Return conversions from abstract alternatives to a common representation."""
    if not values:
        return ()
    common = values[0]
    conversions = [_id]
    for other in values[1:]:
        convert = upcast(common, other)
        conversions = [_compose(convert, previous) for previous in conversions]
        conversions.append(convert)
        common = jax.eval_shape(lambda x: convert(x), common)
    return tuple(conversions)


def dense_arguments(args, kwargs):
    """Densify whole custom operands only at a numerical baseline boundary."""
    return jax.tree.map(to_jax_array, (args, kwargs), is_leaf=is_custom_array)
