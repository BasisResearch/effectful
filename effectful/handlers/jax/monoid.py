import dataclasses
import functools
import typing
from typing import Iterable

import jax

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import (
    CartesianProduct,
    Max,
    Min,
    Monoid,
    NormalizeIntp,
    Product,
    Streams,
    Sum,
    outer_stream,
)
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, implements
from effectful.ops.types import Interpretation, NotHandled, Operation, Term


def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    nx, dx = x.shape
    ny, dy = y.shape
    # Broadcast into (nx, ny, dx+dy), then flatten the first two axes
    x_b = jnp.broadcast_to(x[:, None, :], (nx, ny, dx))
    y_b = jnp.broadcast_to(y[None, :, :], (nx, ny, dy))
    return jnp.concatenate([x_b, y_b], axis=-1).reshape(nx * ny, dx + dy)


LogSumExp = Monoid(name="LogSumExp", identity=jnp.asarray(float("-inf")))


def _jax_args(args):
    """True iff ``args`` is non-empty and every arg is a concrete
    :class:`jax.Array` (no Terms).
    """
    return (
        bool(args)
        and any(isinstance(a, jax.Array) for a in args)
        and all(isinstance(a, jax.typing.ArrayLike) for a in args)
    )


class SumPlusJax(ObjectInterpretation):
    @implements(Sum.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.add, args)


class ProductPlusJax(ObjectInterpretation):
    @implements(Product.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.multiply, args)


class MinPlusJax(ObjectInterpretation):
    @implements(Min.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.minimum, args)


class MaxPlusJax(ObjectInterpretation):
    @implements(Max.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.maximum, args)


class LogSumExpPlusJax(ObjectInterpretation):
    @implements(LogSumExp.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.logaddexp, args)


class CartesianProductPlusJax(ObjectInterpretation):
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


ARRAY_REDUCTORS = {
    Sum: jnp.sum,
    Product: jnp.prod,
    Min: jnp.min,
    Max: jnp.max,
    LogSumExp: logsumexp,
}


class ArrayReduce(ObjectInterpretation):
    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if monoid not in ARRAY_REDUCTORS or typeof(body) is not jax.Array:
            return fwd()
        if not streams:
            return monoid.identity

        reductor = ARRAY_REDUCTORS[monoid]
        index = Operation.define(jax.Array)
        for stream_key, stream_body, streams_tail in outer_stream(streams):
            if not issubclass(typeof(stream_body), jax.Array):
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


@Operation.define
def delta(_index: tuple[int, ...], _weight: jax.Array) -> jax.Array:
    raise NotHandled


py_range = range


@Operation.define
def range(*args: int) -> Iterable[jax.Array]:
    raise NotHandled


def _range_start(term: Term):
    assert term.op == range
    if len(term.args) < 2:
        return 0
    return term.args[0]


def _range_stop(term: Term):
    assert term.op == range
    if len(term.args) < 2:
        return term.args[0]
    return term.args[1]


def _range_step(term: Term):
    assert term.op == range
    if len(term.args) < 3:
        return 1
    return term.args[2]


def _is_simple_range(term):
    assert term.op == range
    start = _range_start(term)
    step = _range_step(term)
    return (
        not isinstance(start, Term)
        and start == 0
        and not isinstance(step, Term)
        and step == 1
    )


class ReduceDeltaIndependent(ObjectInterpretation):
    """Eliminate a Delta that has independent, dense index arguments.

    reduce(M, streams, delta((), body)) ≡ reduce(M, streams, body)

    reduce(M, streams ∪ {v: range(N)}, delta(idx' ++ (v(),), body))
    ═══════════════════════════════════════════════════════════════════════════
    reduce(M, streams, delta(idx', bind_dims(body[v() := unbind_dims(streams[v], fv)], fv)))

    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if not (isinstance(body, Term) and body.op == delta):
            return fwd()

        indices, weight = body.args
        if not indices:
            return monoid.reduce(weight, streams)

        head_indices, tail_index = indices[:-1], indices[-1]
        if not (isinstance(tail_index, Term) and tail_index.op in streams):
            return fwd()

        tail_op: Operation = tail_index.op
        tail_stream = streams[tail_op]
        if not (
            isinstance(tail_stream, Term)
            and tail_stream.op == range
            and _is_simple_range(tail_stream)
        ):
            return fwd()

        fresh_op = Operation.define(tail_op)
        indices = jnp.arange(_range_stop(tail_stream))
        if isinstance(indices, jax.Array) and len(indices) == 0:
            return monoid.identity

        fresh_stream = unbind_dims(indices, fresh_op)
        subst_intp = typing.cast(Interpretation, {tail_op: deffn(fresh_stream)})
        fresh_body = bind_dims(handler(subst_intp)(evaluate)(weight), fresh_op)
        fresh_streams = {k: v for (k, v) in streams.items() if k != tail_op}
        return monoid.reduce(delta(head_indices, fresh_body), fresh_streams)


class ReduceDependentRangeMask(ObjectInterpretation):
    """Eliminate a dependent range by masking.

    reduce(M, streams ∪ {u: range(N), v: range(u())}, body)
    ═══════════════════════════════════════════════════════════════════════════
    reduce(M, streams ∪ {u: range(N), v: range(N)}, where(v() < u(), body, M.identity))

    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        stream_vars = set(streams.keys())

        # streams of the form k: range(X)
        simple_ranges = {
            k: v
            for (k, v) in streams.items()
            if isinstance(v, Term) and _is_simple_range(v)
        }
        for u, u_stream in simple_ranges.items():
            if fvsof(u_stream) & stream_vars:
                continue

            for v, v_stream in simple_ranges.items():
                if (
                    isinstance(v_stream, Term)
                    and isinstance(_range_stop(v_stream), Term)
                    and _range_stop(v_stream).op == u
                ):
                    fresh_streams = {
                        a: (u_stream if a == v else b) for (a, b) in streams.items()
                    }

                    # there are other commuting rules for delta that we do not
                    # currently include
                    if isinstance(body, Term) and body.op == delta:
                        fresh_body = delta(
                            body.args[0],
                            jnp.where(v() < u(), body.args[1], monoid.identity),
                        )
                    else:
                        fresh_body = jnp.where(v() < u(), body, monoid.identity)

                    return monoid.reduce(fresh_body, fresh_streams)

        return fwd()


class ReduceRange(ObjectInterpretation):
    """Replace concrete-range stream values with materialized ``jnp.arange``.

    reduce(M, streams ∪ {v: range(a, b, s)}, body)
    ≡ reduce(M, streams ∪ {v: jnp.arange(a, b, s)}, body)

    when ``a``, ``b``, ``s`` are concrete and ``body`` is not a delta term.
    Delegates the actual reduction to whichever handler picks up the
    materialized ``jax.Array`` streams.
    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if isinstance(body, Term) and body.op == delta:
            return fwd()

        new_streams: dict = {}
        any_replaced = False
        for k, v in streams.items():
            if isinstance(v, Term) and v.op == range:
                new_streams[k] = jnp.arange(
                    _range_start(v), _range_stop(v), _range_step(v)
                )
                any_replaced = True
            else:
                new_streams[k] = v

        if not any_replaced:
            return fwd()
        return monoid.reduce(body, new_streams)


NormalizeIntp.extend(
    ArrayReduce(),
    ReduceRange(),
    ReduceDeltaIndependent(),
    ReduceDependentRangeMask(),
    SumPlusJax(),
    ProductPlusJax(),
    MinPlusJax(),
    MaxPlusJax(),
    LogSumExpPlusJax(),
    CartesianProductPlusJax(),
)
