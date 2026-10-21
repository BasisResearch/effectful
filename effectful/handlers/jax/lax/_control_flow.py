"""Control-flow effects: wrap callbacks, then delegate execution to JAX."""

import jax

from effectful.ops.syntax import defop

from .._callables import wrap_callable


@defop
def cond(pred, true_fun, false_fun, *operands, **kwargs):
    return jax.lax.cond(
        pred, wrap_callable(true_fun), wrap_callable(false_fun), *operands, **kwargs
    )


@defop
def switch(index, branches, *operands, **kwargs):
    return jax.lax.switch(
        index, tuple(wrap_callable(branch) for branch in branches), *operands, **kwargs
    )


@defop
def while_loop(cond_fun, body_fun, init_val):
    return jax.lax.while_loop(
        wrap_callable(cond_fun), wrap_callable(body_fun), init_val
    )


@defop
def scan(
    f, init, xs=None, length=None, reverse=False, unroll=1, _split_transpose=False
):
    return jax.lax.scan(
        wrap_callable(f),
        init,
        xs,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
    )
