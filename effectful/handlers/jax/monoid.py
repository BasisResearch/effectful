import functools

import jax

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import (
    ArgMax,
    ArgMin,
    CartesianProduct,
    Max,
    Min,
    Monoid,
    Product,
    Streams,
    Sum,
    outer_stream,
)
from effectful.ops.semantics import evaluate, handler, typeof
from effectful.ops.syntax import deffn
from effectful.ops.types import NotHandled, Operation


def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = jnp.repeat(x, y.shape[0], axis=0), jnp.tile(y, (x.shape[0], 1))
    return jnp.hstack([x, y])


LogSumExp = Monoid(name="LogSumExp", identity=jnp.asarray(float("-inf")))


@Sum.plus.register(jax.Array)
def _(*args):
    return functools.reduce(jnp.add, args)


@Product.plus.register(jax.Array)
def _(*args):
    return functools.reduce(jnp.multiply, args)


@Min.plus.register(jax.Array)
def _(*args):
    return functools.reduce(jnp.minimum, args)


@Max.plus.register(jax.Array)
def _(*args):
    return functools.reduce(jnp.maximum, args)


@LogSumExp.plus.register(jax.Array)
def _(*args):
    return functools.reduce(jnp.logaddexp, args)


@CartesianProduct.plus.register(jax.Array)
def _(*args):
    # Skip identity ``[()]`` args; short-circuit on zero ``[]``. Both sentinels
    # arrive as Python lists alongside jax-array factors, so check for them
    # explicitly before composing under :func:`cartesian_prod`.
    result = None
    for a in args:
        if a is CartesianProduct.zero:
            return CartesianProduct.zero
        if a is CartesianProduct.identity:
            continue
        result = a if result is None else cartesian_prod(result, a)
    return result if result is not None else CartesianProduct.identity


# Chain a JAX-typed case onto the tuple plus handler: handle (jax.Array,
# jax.Array) pairs here, fall through to the prior handler otherwise.
_argmin_tuple_prior = ArgMin.plus.dispatch(tuple)
_argmax_tuple_prior = ArgMax.plus.dispatch(tuple)


@ArgMin.plus.register(tuple)
def _(*args):
    if all(isinstance(a[0], jax.Array) and isinstance(a[1], jax.Array) for a in args):
        best_score, best_value = args[0]
        for score, value in args[1:]:
            is_new = score < best_score
            best_score = jnp.where(is_new, score, best_score)
            best_value = jnp.where(is_new, value, best_value)
        return (best_score, best_value)
    return _argmin_tuple_prior(*args)


@ArgMax.plus.register(tuple)
def _(*args):
    if all(isinstance(a[0], jax.Array) and isinstance(a[1], jax.Array) for a in args):
        best_score, best_value = args[0]
        for score, value in args[1:]:
            is_new = score > best_score
            best_score = jnp.where(is_new, score, best_score)
            best_value = jnp.where(is_new, value, best_value)
        return (best_score, best_value)
    return _argmax_tuple_prior(*args)


def register_array_reduce(monoid: Monoid, reductor) -> None:
    """Register ``reductor`` as the JAX array reduction for ``monoid``.

    A backend-specific reducer (e.g. :func:`jnp.sum`) is paired with a monoid so
    that ``monoid.reduce`` over an array-valued body and array-valued streams
    unbinds the indices, evaluates, and applies the reducer along those axes.

    """

    def _reduce_array(body: jax.Array, streams: Streams):
        index = Operation.define(jax.Array)

        if not streams:
            return monoid.identity

        # find and reduce an array stream
        for stream_key, stream_body, streams_tail in outer_stream(streams):
            if typeof(stream_body) != jax.Array:
                continue

            with handler({stream_key: deffn(unbind_dims(stream_body, index))}):
                (eval_body, eval_streams_tail) = (
                    evaluate(body),
                    evaluate(streams_tail),
                )
                assert isinstance(eval_streams_tail, dict)

                reduce_tail = (
                    monoid.reduce(eval_body, eval_streams_tail)
                    if len(eval_streams_tail) > 0
                    else eval_body
                )
                return reductor(bind_dims(reduce_tail, index), axis=0)

        raise NotHandled

    monoid.reduce.register(jax.Array)(_reduce_array)


register_array_reduce(Sum, jnp.sum)
register_array_reduce(Product, jnp.prod)
register_array_reduce(Min, jnp.min)
register_array_reduce(Max, jnp.max)
register_array_reduce(LogSumExp, logsumexp)
