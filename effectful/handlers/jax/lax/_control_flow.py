"""Control-flow effects: wrap callbacks, then delegate execution to JAX."""

import jax

from effectful.ops.syntax import defop

from .._callables import wrap_callable

_native_cond = jax.lax.cond
_native_switch = jax.lax.switch
_native_while_loop = jax.lax.while_loop
_native_scan = jax.lax.scan


@defop
def cond(pred, true_fun, false_fun, *operands, **kwargs):
    return _native_cond(
        pred, wrap_callable(true_fun), wrap_callable(false_fun), *operands, **kwargs
    )


@defop
def switch(index, branches, *operands, **kwargs):
    return _native_switch(
        index, tuple(wrap_callable(branch) for branch in branches), *operands, **kwargs
    )


@defop
def while_loop(cond_fun, body_fun, init_val):
    return _native_while_loop(
        wrap_callable(cond_fun), wrap_callable(body_fun), init_val
    )


@defop
def scan(
    f, init, xs=None, length=None, reverse=False, unroll=1, _split_transpose=False
):
    return _native_scan(
        wrap_callable(f),
        init,
        xs,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
    )
