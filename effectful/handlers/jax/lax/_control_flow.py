"""Representation joins around original Python control-flow bodies."""

import jax

from effectful.ops.syntax import defop

from .._custom_array import is_custom_array, upcast_many

_MISSING = object()


def _logical_flatten(value):
    return jax.tree.flatten(value, is_leaf=is_custom_array)


def _conversions(*outputs):
    """Join corresponding logical leaves, independent of their storage trees."""
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
    # Fresh closures keep shape planning from reusing a trace from another policy.
    runs = [lambda *xs, fn=fn: fn(*xs) for fn in branches]
    outputs = [jax.eval_shape(run, *operands) for run in runs]
    casts = _conversions(*outputs)

    def converted(run, cast):
        return lambda *xs: cast(run(*xs))

    return tuple(converted(run, cast) for run, cast in zip(runs, casts, strict=True))


@defop
def cond(pred, true_fun, false_fun, *operands, operand=_MISSING):
    if operand is not _MISSING:
        if operands:
            raise TypeError("Cannot pass both operand and positional operands")
        operands = (operand,)
    true, false = _branches((true_fun, false_fun), operands)
    return jax.lax.cond(pred, true, false, *operands)


@defop
def switch(index, branches, *operands, operand=_MISSING):
    if operand is not _MISSING:
        if operands:
            raise TypeError("Cannot pass both operand and positional operands")
        operands = (operand,)
    return jax.lax.switch(index, _branches(tuple(branches), operands), *operands)


@defop
def while_loop(cond_fun, body_fun, init_val):
    run = lambda xs: body_fun(xs)
    initial = jax.eval_shape(lambda x: x, init_val)
    output = jax.eval_shape(run, init_val)
    initial_cast, body_cast = _conversions(initial, output)
    init_val = initial_cast(init_val)
    # Original JAX checks that one widening step reaches a fixed physical layout.
    return jax.lax.while_loop(cond_fun, lambda xs: body_cast(run(xs)), init_val)


@defop
def scan(
    f, init, xs=None, length=None, reverse=False, unroll=1, _split_transpose=False
):
    if any(is_custom_array(x) for x in jax.tree.leaves(xs, is_leaf=is_custom_array)):
        raise NotImplementedError("Scanning custom-array inputs is not yet supported")
    sample = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), xs)
    run = lambda c, x: f(c, x)
    initial = jax.eval_shape(lambda x: x, init)
    output, ys = jax.eval_shape(run, init, sample)
    if any(is_custom_array(x) for x in jax.tree.leaves(ys, is_leaf=is_custom_array)):
        raise NotImplementedError(
            "Stacking custom-array scan outputs is not yet supported"
        )
    initial_cast, carry_cast = _conversions(initial, output)
    init = initial_cast(init)

    def body(carry, x):
        carry, y = run(carry, x)
        return carry_cast(carry), y

    return jax.lax.scan(
        body,
        init,
        xs,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
    )
