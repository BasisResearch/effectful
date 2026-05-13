import functools
import typing

import jax

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import (
    CartesianProduct,
    Max,
    Min,
    Monoid,
    Product,
    Streams,
    Sum,
    outer_stream,
)
from effectful.ops.semantics import coproduct, evaluate, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, implements
from effectful.ops.types import Interpretation, NotHandled, Operation, Term


def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = jnp.repeat(x, y.shape[0], axis=0), jnp.tile(y, (x.shape[0], 1))
    return jnp.hstack([x, y])


LogSumExp = Monoid(name="LogSumExp", identity=jnp.asarray(float("-inf")))


def _jax_args(args):
    """True iff every arg is a concrete :class:`jax.Array` (no Terms)."""
    return all(isinstance(a, jax.Array) and not isinstance(a, Term) for a in args)


class SumKernelJax(ObjectInterpretation):
    @implements(Sum.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.add, args)


class ProductKernelJax(ObjectInterpretation):
    @implements(Product.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.multiply, args)


class MinKernelJax(ObjectInterpretation):
    @implements(Min.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.minimum, args)


class MaxKernelJax(ObjectInterpretation):
    @implements(Max.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.maximum, args)


class LogSumExpKernelJax(ObjectInterpretation):
    @implements(LogSumExp.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.logaddexp, args)


class CartesianProductKernelJax(ObjectInterpretation):
    @implements(CartesianProduct.plus)
    def plus(self, *args):
        # Skip identity ``[()]`` args; short-circuit on zero ``[]``. Both
        # sentinels arrive as Python lists alongside jax-array factors, so
        # check for them explicitly before composing.
        if not any(isinstance(a, jax.Array) for a in args):
            return fwd()
        result = None
        for a in args:
            if a is CartesianProduct.zero:
                return CartesianProduct.zero
            if a is CartesianProduct.identity:
                continue
            if not isinstance(a, jax.Array):
                return fwd()
            result = a if result is None else cartesian_prod(result, a)
        return result if result is not None else CartesianProduct.identity


def _make_array_reduce_class(monoid: Monoid, reductor):
    """Build an :class:`ObjectInterpretation` that implements
    ``monoid.reduce`` for ``jax.Array`` bodies using ``reductor``.
    """

    class _ArrayReduce(ObjectInterpretation):
        @implements(monoid.reduce)
        def reduce(self, body, streams):
            if typeof(body) is not jax.Array:
                return fwd()
            if not streams:
                return monoid.identity

            index = Operation.define(jax.Array)
            for stream_key, stream_body, streams_tail in outer_stream(streams):
                if typeof(stream_body) is not jax.Array:
                    continue
                with handler({stream_key: deffn(unbind_dims(stream_body, index))}):
                    eval_body = evaluate(body)
                    eval_streams_tail = evaluate(streams_tail)
                    assert isinstance(eval_streams_tail, dict)
                    reduce_tail = (
                        monoid.reduce(eval_body, eval_streams_tail)
                        if len(eval_streams_tail) > 0
                        else eval_body
                    )
                    return reductor(bind_dims(reduce_tail, index), axis=0)
            return fwd()

    _ArrayReduce.__name__ = f"{monoid._name}ArrayReduceJax"
    return _ArrayReduce


_ARRAY_REDUCE_CLASSES = [
    _make_array_reduce_class(Sum, jnp.sum),
    _make_array_reduce_class(Product, jnp.prod),
    _make_array_reduce_class(Min, jnp.min),
    _make_array_reduce_class(Max, jnp.max),
    _make_array_reduce_class(LogSumExp, logsumexp),
]


JaxEvaluateIntp = functools.reduce(
    coproduct,
    typing.cast(
        list[Interpretation],
        [
            SumKernelJax(),
            ProductKernelJax(),
            MinKernelJax(),
            MaxKernelJax(),
            LogSumExpKernelJax(),
            CartesianProductKernelJax(),
            *[cls() for cls in _ARRAY_REDUCE_CLASSES],
        ],
    ),
)
"""JAX kernels for plus and reduce. Composes with
:data:`effectful.ops.monoid.EvaluateIntp` to extend evaluation to JAX arrays.
"""
