import typing
from collections.abc import Iterable

import jax

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, jax_getitem, unbind_dims
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Expr, Interpretation, Operation, Term


def _named_dims(term: Expr[jax.Array]) -> tuple[Operation, ...]:
    if not (isinstance(term, Term) and term.op == jax_getitem):
        return ()
    index = term.args[1]
    assert isinstance(index, Iterable)
    return tuple(i.op for i in index if isinstance(i, Term) and not i.args)


def _reduce_axis(array, axis=None, **kwargs) -> jax.Array:
    if axis is None:
        return fwd()

    named_dims = _named_dims(array)
    if not named_dims:
        return fwd()

    bound_arr = bind_dims(array, *named_dims)
    reduced = fwd(bound_arr, axis=axis + len(named_dims), **kwargs)
    return unbind_dims(reduced, *named_dims)


PartialEvalSingleAxisReduce: Interpretation = typing.cast(
    Interpretation,
    {
        op: _reduce_axis
        for op in [
            jnp.sum,
            jnp.prod,
            jnp.min,
            jnp.max,
            jnp.any,
            jnp.all,
            jnp.mean,
            jnp.argmax,
            logsumexp,
        ]
    },
)
"""Partial evaluator for reductions over a single axis.

More efficient than vmapping a single axis reduction over the remaining
dimensions. Multi-axis reductions are still vmapped.

"""


class PartialEvalMultiAxisReduce(ObjectInterpretation):
    @implements(jnp.tensordot)
    def _(a: jax.Array, b: jax.Array, axes=2, **kwargs) -> jax.Array:
        named_a = _named_dims(a)
        named_b = _named_dims(b)
        a_shape = a.shape

        if isinstance(a_shape, Term):
            return fwd()

        if isinstance(axes, int):
            a_dims = range(len(a_shape) - axes, len(a_shape))
            b_dims = range(axes)
        else:
            a_dims = axes[0]
            b_dims = axes[1]

        shifted_a_dims = tuple(d + len(named_a) for d in a_dims)
        shifted_b_dims = tuple(
            d + len(named_a) + len(a_shape) + len(named_b) for d in b_dims
        )
        shifted_axes = (shifted_a_dims, shifted_b_dims)

        bound_a = bind_dims(a, *named_a)
        bound_b = bind_dims(b, *named_b)
        reduced = fwd(bound_a, bound_b, axes=shifted_axes, **kwargs)
        return jax_getitem(
            reduced,
            tuple(i() for i in named_a)
            + tuple(slice(None) for _ in a_shape)
            + tuple(i() for i in named_b),
        )
