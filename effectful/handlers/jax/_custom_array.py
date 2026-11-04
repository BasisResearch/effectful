"""Representation effects shared by high-level JAX APIs and custom arrays."""

from collections.abc import Callable
from typing import Any

import jax

from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, defop, implements

from .lax._control_flow import cond, scan, switch, while_loop

_MISSING = object()


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


def _logical_flatten(value):
    return jax.tree.flatten(value, is_leaf=is_custom_array)


def _conversions(*outputs):
    """Join logical leaves independently of their physical storage pytrees."""
    flat = [_logical_flatten(output) for output in outputs]
    tree = flat[0][1]
    if any(other_tree != tree for _, other_tree in flat[1:]):
        raise TypeError(
            "Control-flow outputs must have the same logical pytree structure"
        )
    casts = [[] for _ in outputs]
    for alternatives in zip(*(leaves for leaves, _ in flat), strict=True):
        for row, convert in zip(casts, upcast_many(*alternatives), strict=True):
            row.append(convert)

    def conversion(row):
        def convert(value):
            leaves, actual_tree = _logical_flatten(value)
            if actual_tree != tree:
                raise TypeError(
                    "Control-flow output changed its logical pytree structure"
                )
            return jax.tree.unflatten(
                tree, [cast(x) for cast, x in zip(row, leaves, strict=True)]
            )

        return convert

    return tuple(conversion(row) for row in casts)


def _branches(branches, operands):
    # Fresh closures keep shape planning from reusing another policy's trace.
    runs = [lambda *xs, fn=fn: fn(*xs) for fn in branches]
    outputs = [jax.eval_shape(run, *operands) for run in runs]
    casts = _conversions(*outputs)

    def converted(run, cast):
        return lambda *xs: cast(run(*xs))

    return tuple(converted(run, cast) for run, cast in zip(runs, casts, strict=True))


class CustomArrayControlFlow(ObjectInterpretation):
    """Negotiate custom-array layouts before forwarding control flow to JAX.

    Install this interpretation explicitly alongside representation-specific
    handlers for is_custom_array, to_jax_array, and upcast.
    """

    @implements(cond)
    def cond(self, pred, true_fun, false_fun, *operands, operand=_MISSING):
        if operand is not _MISSING:
            if operands:
                raise TypeError("Cannot pass both operand and positional operands")
            operands = (operand,)
        true, false = _branches((true_fun, false_fun), operands)
        return fwd(pred, true, false, *operands)

    @implements(switch)
    def switch(self, index, branches, *operands, operand=_MISSING):
        if operand is not _MISSING:
            if operands:
                raise TypeError("Cannot pass both operand and positional operands")
            operands = (operand,)
        return fwd(index, _branches(tuple(branches), operands), *operands)

    @implements(while_loop)
    def while_loop(self, cond_fun, body_fun, init_val):
        run = lambda xs: body_fun(xs)
        initial = jax.eval_shape(_id, init_val)
        output = jax.eval_shape(run, init_val)
        initial_cast, body_cast = _conversions(initial, output)
        init_val = initial_cast(init_val)
        # JAX validates that a single widening reaches a fixed physical layout.
        return fwd(cond_fun, lambda xs: body_cast(run(xs)), init_val)

    @implements(scan)
    def scan(
        self,
        f,
        init,
        xs=None,
        length=None,
        reverse=False,
        unroll=1,
        _split_transpose=False,
    ):
        if any(
            is_custom_array(x) for x in jax.tree.leaves(xs, is_leaf=is_custom_array)
        ):
            raise NotImplementedError(
                "Scanning custom-array inputs is not yet supported"
            )
        sample = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), xs)
        run = lambda c, x: f(c, x)
        initial = jax.eval_shape(_id, init)
        output, ys = jax.eval_shape(run, init, sample)
        if any(
            is_custom_array(x) for x in jax.tree.leaves(ys, is_leaf=is_custom_array)
        ):
            raise NotImplementedError(
                "Stacking custom-array scan outputs is not yet supported"
            )
        initial_cast, carry_cast = _conversions(initial, output)
        init = initial_cast(init)

        def body(carry, x):
            carry, y = run(carry, x)
            return carry_cast(carry), y

        return fwd(
            body,
            init,
            xs,
            length=length,
            reverse=reverse,
            unroll=unroll,
            _split_transpose=_split_transpose,
        )
